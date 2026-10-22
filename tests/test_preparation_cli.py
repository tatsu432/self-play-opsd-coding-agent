"""CLI selections use a fake scheduler and never allocate cluster resources."""

from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from selfplay_opsd import hpc, hpc_readiness


@pytest.mark.parametrize("complete", [0, 1, 2])
def test_backend_stage_advances_without_worker_or_runtime(complete, monkeypatch):
    state = {
        "images": {"worker": {}, "trainer": {"ready": True, "sha256": "a" * 64}},
        "gates": {"single": complete >= 1, "target": complete >= 2},
        "eligible_to_submit_main": False,
        "budget": {},
    }
    calls = []

    class Remote:
        def __init__(self, alias):
            self.alias = alias

        def call(self, operation, payload):
            calls.append((operation, payload))
            if operation == "study":
                return state
            if operation == "status":
                return {"ledger": {}, "attempts": []}
            assert operation == "submit" and payload["operation"] == "backend-check"
            return {"submitted": True}

    monkeypatch.setattr(hpc, "YaleSSH", Remote)
    monkeypatch.setattr(hpc, "snapshot", lambda: {})
    monkeypatch.setattr(hpc, "remote_context", lambda *a: {"storage_root": "/project"})
    monkeypatch.setattr(
        hpc, "run_payload", lambda *a, **kw: {**kw, "run_id": "probe", "gpu_count": 1}
    )
    monkeypatch.setattr(
        "selfplay_opsd.readiness.execution_identity",
        lambda *a, **kw: pytest.fail("Backend stage must not require runtime/data identity"),
    )
    app = typer.Typer()
    hpc_readiness.register(app)
    result = CliRunner().invoke(app, ["prepare-study", "--stage", "backend"])
    assert result.exit_code == 0, result.exception
    submissions = [payload for operation, payload in calls if operation == "submit"]
    if complete < 2:
        assert len(submissions) == 1
        assert submissions[0]["profile"] == (
            "single_gpu_80gb" if complete == 0 else "dual_gpu_80gb"
        )
        assert submissions[0]["readiness_budget"] == 24
    else:
        assert not submissions
        assert '"backend_ready": true' in result.output
        assert '"eligible_to_submit_main": false' in result.output


@pytest.mark.parametrize("selection", ["default", "environment", "flag"])
def test_image_build_root_precedence_and_builder(selection, tmp_path, monkeypatch):
    recorded = []

    class Remote:
        def __init__(self, alias):
            self.alias = alias

        def call(self, operation, payload):
            return (
                {"archive": "/remote/image.tar"}
                if operation == "prepare_image"
                else {"submitted": True}
            )

        def rsync(self, *args):
            pass

    def archive(kind, source, directory, commit, builder):
        recorded.append((directory, builder))
        path = tmp_path / "archive.tar"
        path.write_bytes(b"archive fixture")
        return path, {"archive_sha256": "a" * 64}

    monkeypatch.setattr(hpc, "YaleSSH", Remote)
    monkeypatch.setattr(hpc, "sync", lambda _: {"commit": "a" * 40})
    monkeypatch.setattr(hpc, "run_payload", lambda *a, **kw: {"identity": "a" * 64})
    monkeypatch.setattr(hpc_readiness, "build_archive", archive)
    monkeypatch.delenv("SELFPLAY_IMAGE_BUILD_ROOT", raising=False)
    args = ["images", "build", "--kind", "trainer", "--builder", "external-builder"]
    expected = hpc.ROOT / "runs/hpc/images"
    if selection != "default":
        monkeypatch.setenv("SELFPLAY_IMAGE_BUILD_ROOT", str(tmp_path / "environment"))
        expected = tmp_path / "environment"
    if selection == "flag":
        expected = tmp_path / "explicit root"
        args += ["--build-root", str(expected)]
    app = typer.Typer()
    hpc_readiness.register(app)
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.exception
    assert recorded == [(Path(expected).resolve() / "trainer", "external-builder")]
