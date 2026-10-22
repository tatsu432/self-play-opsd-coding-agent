"""Exported archives use the selected filesystem and remain recoverable."""

import io
import json
import subprocess
import tarfile
from types import SimpleNamespace

import pytest

from selfplay_opsd.hpc import HpcError
from selfplay_opsd.hpc_readiness import build_archive


def build_source(tmp_path):
    source = tmp_path / "source"
    (source / "docker").mkdir(parents=True)
    (source / "docker/trainer.Dockerfile").write_text("FROM pinned-image\n")
    (source / "uv.lock").write_text("pinned-lock\n")
    return source


def fake_export(args, **kwargs):
    destination = args[args.index("--output") + 1].split("dest=", 1)[1]
    with tarfile.open(destination, "w") as tar:
        for name, value in {
            "manifest.json": [{"Config": "config.json"}],
            "config.json": {"architecture": "amd64", "os": "linux"},
        }.items():
            data = json.dumps(value).encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return subprocess.CompletedProcess(args, 0)


def test_external_archive_storage_and_cache_integrity(tmp_path, monkeypatch):
    source = build_source(tmp_path)
    directory = tmp_path / "external storage/trainer"
    checked = []
    monkeypatch.setattr(
        "selfplay_opsd.hpc_readiness.shutil.disk_usage",
        lambda path: checked.append(path) or SimpleNamespace(free=100 * 1024**3),
    )
    calls = []

    def export(args, **kwargs):
        calls.append(args)
        return fake_export(args, **kwargs)

    monkeypatch.setattr("selfplay_opsd.hpc_readiness.subprocess.run", export)
    archive, metadata = build_archive("trainer", source, directory, "a" * 40, "external-builder")
    assert checked == [directory]
    assert archive.parent == directory
    assert calls[0][calls[0].index("--builder") + 1] == "external-builder"
    assert metadata["platform"] == "linux/amd64"
    assert build_archive("trainer", source, directory, "a" * 40)[0] == archive
    assert len(calls) == 1
    metadata_path = archive.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata | {"source_commit": "different-source"}))
    with pytest.raises(HpcError, match="provenance"):
        build_archive("trainer", source, directory, "a" * 40)
    metadata_path.write_text(json.dumps(metadata))
    archive.write_bytes(b"corrupted")
    with pytest.raises(HpcError, match="checksum"):
        build_archive("trainer", source, directory, "a" * 40)


def test_capacity_and_interrupted_export(tmp_path, monkeypatch):
    source = build_source(tmp_path)
    directory = tmp_path / "external"
    monkeypatch.setattr(
        "selfplay_opsd.hpc_readiness.shutil.disk_usage", lambda _: SimpleNamespace(free=1024)
    )
    monkeypatch.setattr(
        "selfplay_opsd.hpc_readiness.subprocess.run",
        lambda *a, **k: pytest.fail("No build with insufficient disk"),
    )
    with pytest.raises(HpcError, match="60 GiB"):
        build_archive("trainer", source, directory, "a" * 40)
    monkeypatch.setattr(
        "selfplay_opsd.hpc_readiness.shutil.disk_usage",
        lambda _: SimpleNamespace(free=100 * 1024**3),
    )

    def interrupt(args, **kwargs):
        fake_export(args, **kwargs)
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr("selfplay_opsd.hpc_readiness.subprocess.run", interrupt)
    with pytest.raises(subprocess.CalledProcessError):
        build_archive("trainer", source, directory, "a" * 40)
    assert not list(directory.glob("*.tar"))
    assert not list(directory.glob("*.json"))
    monkeypatch.setattr("selfplay_opsd.hpc_readiness.subprocess.run", fake_export)
    assert build_archive("trainer", source, directory, "a" * 40)[0].is_file()
