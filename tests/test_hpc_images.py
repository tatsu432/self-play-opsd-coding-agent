"""Source build semantics and publication failures, without unprivileged host changes."""

import base64
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from selfplay_opsd.core import digest
from selfplay_opsd.hpc_images import BUILD_CHECK_DOCKERFILE, source_build


def script(name):
    spec = importlib.util.spec_from_file_location(name, f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_conversion_resolves_copy_paths_and_uses_root_build_without_su():
    result = source_build(Path.cwd(), Path("docker/trainer.Dockerfile"), "a" * 40)
    definition = result["definition"]
    assert "docs/methods.md /opt/selfplay/methods.md" in definition
    assert "src /opt/selfplay/src" in definition
    assert "%post\nset -eu\nmkdir -p /opt/selfplay\ncd /opt/selfplay" in definition
    assert "su -" not in definition
    assert 'exec selfplay "$@"' in definition
    assert "From: vllm/vllm-openai@sha256:" in definition
    assert ":v0.11.0@" not in definition
    metadata = result["image_metadata"]
    assert result["build_key"] == digest(metadata)
    assert all("__pycache__" not in name for name in metadata["context_files"])
    assert not any(
        name.startswith(("data/", "runs/", ".git/")) for name in metadata["context_files"]
    )


@pytest.mark.parametrize(
    "old,new",
    [
        ("USER root", "USER nobody"),
        ("WORKDIR /opt/selfplay", "WORKDIR ../outside"),
        ("COPY README.md ./", "COPY ../secret ./"),
        ("COPY README.md ./", "ADD README.md ./"),
        ("COPY README.md ./", "ENV BAD=1\nCOPY README.md ./"),
    ],
)
def test_conversion_rejects_unsupported_or_escaping_instructions(tmp_path, old, new):
    recipe = tmp_path / "Dockerfile"
    recipe.write_text(BUILD_CHECK_DOCKERFILE.replace(old, new))
    with pytest.raises(ValueError):
        source_build(Path.cwd(), recipe, "a" * 40, check=True)


def inputs(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    (root / "README.md").write_text("witness\n")
    (root / "uv.lock").write_text("lock\n")
    dockerfile = tmp_path / "check.Dockerfile"
    dockerfile.write_text(BUILD_CHECK_DOCKERFILE)
    result = source_build(root, dockerfile, "a" * 40, check=True)
    target = tmp_path / "images/check.sif"
    target.parent.mkdir()
    definition = target.with_suffix(".def")
    definition.write_text(result["definition"])
    env = {
        "SELFPLAY_PROJECT": str(root),
        "SLURM_JOB_ID": "123",
        "SELFPLAY_IMAGE_METADATA": base64.b64encode(
            json.dumps(result["image_metadata"]).encode()
        ).decode(),
        "SELFPLAY_BUILD_KEY": result["build_key"],
        "SELFPLAY_BUILD_DEFINITION": str(definition),
        "SELFPLAY_IMAGE_TARGET": str(target),
        "SELFPLAY_IMAGE_MANIFEST": str(target.parent / "build-check.json"),
        "SELFPLAY_CACHE": str(tmp_path / "scratch/huggingface"),
    }
    return result, env


@pytest.mark.parametrize("failure", [False, True])
def test_native_build_publishes_only_after_sif_validation_and_cleans_its_scratch(
    tmp_path, monkeypatch, failure
):
    job = script("hpc_job")
    result, env = inputs(tmp_path)
    monkeypatch.setattr(job.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(job.shutil, "disk_usage", lambda _: SimpleNamespace(free=300 * 1024**3))
    monkeypatch.setattr(
        job.subprocess, "check_output", lambda args, **_: "a" * 40 if args[0] == "git" else "1.4.0"
    )
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        if args[:2] == ["apptainer", "build"]:
            assert (kwargs["cwd"] / "README.md").read_text() == "witness\n"
            Path(args[3]).write_bytes(b"sif")
        elif failure:
            raise subprocess.CalledProcessError(1, args)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(job.subprocess, "run", run)
    if failure:
        with pytest.raises(subprocess.CalledProcessError):
            job.source_image(env)
        assert not Path(env["SELFPLAY_IMAGE_TARGET"]).exists()
        assert not Path(env["SELFPLAY_IMAGE_MANIFEST"]).exists()
    else:
        job.source_image(env)
        report = json.loads(Path(env["SELFPLAY_IMAGE_MANIFEST"]).read_text())
        assert report["ready"] and report["build_key"] == result["build_key"]
        assert "docker_image_id" not in report
        job.source_image(env)
        assert len(calls) == 2  # cached SIF is hashed rather than rebuilt
        Path(env["SELFPLAY_IMAGE_TARGET"]).write_bytes(b"corrupt")
        with pytest.raises(ValueError, match="SHA-256"):
            job.source_image(env)
    assert not list((tmp_path / "scratch/image-builds").iterdir())
    assert not list((tmp_path / "images").glob("*.pending.sif"))


def test_source_tampering_is_rejected_before_build(tmp_path, monkeypatch):
    job = script("hpc_job")
    _, env = inputs(tmp_path)
    Path(env["SELFPLAY_BUILD_DEFINITION"]).write_text("tampered")
    monkeypatch.setattr(job.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(job.subprocess, "check_output", lambda *a, **k: "a" * 40)
    monkeypatch.setattr(job.subprocess, "run", lambda *a, **k: pytest.fail("must not build"))
    with pytest.raises(ValueError, match="SHA-256"):
        job.source_image(env)


def test_remote_source_identity_and_feasibility_gate(tmp_path, monkeypatch):
    remote = script("hpc_remote")
    data, _ = inputs(tmp_path)
    payload = data | {"commit": "a" * 40, "operation": "image-build"}
    monkeypatch.setattr(remote, "workspace", lambda _: tmp_path)
    first = remote.prepare_source_image(payload)
    assert remote.prepare_source_image(payload) == first
    remote.precheck(payload)
    with pytest.raises(ValueError, match="identity"):
        remote.prepare_source_image(payload | {"definition": "tamper"})
    metadata = data["image_metadata"] | {"kind": "trainer"}
    payload |= {"image_metadata": metadata, "build_key": digest(metadata)}
    with pytest.raises(ValueError, match="check-only"):
        remote.precheck(payload)


def test_controller_transport_allows_source_preparation_without_docker(monkeypatch):
    from selfplay_opsd.hpc import YaleSSH

    captured = []

    def run(args, **kwargs):
        captured.append(args)
        return subprocess.CompletedProcess(args, 0, '{"definition_path":"/project/image.def"}', "")

    monkeypatch.setattr(subprocess, "run", run)
    assert YaleSSH().call("prepare_source_image", {}) == {"definition_path": "/project/image.def"}
    assert "prepare_source_image" in captured[0]
