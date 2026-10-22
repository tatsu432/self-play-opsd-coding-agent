"""Readiness must fail closed before allocating resources or changing a study."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from selfplay_opsd.config import load_config
from selfplay_opsd.core import ExperimentError
from selfplay_opsd.preflight import check_frozen, freeze_study
from selfplay_opsd.readiness import readiness_report
from selfplay_opsd.storage import atomic_json


def endpoint():
    spec = importlib.util.spec_from_file_location("readiness_remote", "scripts/hpc_remote.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_budget_reserves_atomically_and_keeps_unknown_accounting(tmp_path, monkeypatch):
    remote = endpoint()
    monkeypatch.setattr(remote, "workspace", lambda _: tmp_path)
    monkeypatch.setattr(remote, "capture", lambda _: {"exit_code": 1})
    payload = {
        "study_id": "study",
        "run_id": "run",
        "gpu_count": 1,
        "time": "17:00:00",
        "readiness_budget": 24,
        "operation": "backend-check",
    }

    def reserve(token):
        try:
            remote.reserve_budget(payload, token)
            return True
        except ValueError:
            return False

    with ThreadPoolExecutor(2) as executor:
        assert sorted(executor.map(reserve, ["one", "two"])) == [False, True]
    token = next(iter(remote.refreshed_budget(payload)["reservations"]))
    remote.bind_budget_job(payload, token, "123")
    assert remote.refreshed_budget(payload)["committed_gpu_hours"] == 17
    monkeypatch.setattr(
        remote,
        "capture",
        lambda _: {"exit_code": 0, "stdout": "123|COMPLETED|3600|gres/gpu=1,cpu=8\n"},
    )
    assert remote.refreshed_budget(payload)["committed_gpu_hours"] == 1
    remote.reserve_budget(payload, "remaining")
    assert remote.refreshed_budget(payload)["committed_gpu_hours"] == 18


def test_artifacts_follow_each_operation(tmp_path, monkeypatch):
    remote = endpoint()
    monkeypatch.setattr(remote, "paths", lambda _: {"runs": tmp_path})
    payload = {"study_id": "study", "method": "B", "seed": 2026, "profile": "dual_gpu_80gb"}
    assert (
        remote.artifact_dir(payload | {"operation": "evaluate", "base": True})
        == tmp_path / "study/base"
    )
    assert (
        remote.artifact_dir(payload | {"operation": "backend-check"})
        == tmp_path / "compatibility/dual_gpu_80gb"
    )
    assert (
        remote.artifact_dir(payload | {"operation": "pilot"}) == tmp_path / "study-pilot/B-seed2026"
    )
    assert (
        remote.artifact_dir(payload | {"operation": "runtime-check"})
        == tmp_path / "readiness/study"
    )


def test_continuation_uses_persisted_specification_and_is_bounded(tmp_path, monkeypatch):
    remote = endpoint()
    checkpoint = tmp_path / "runs/study-pilot/D-seed2026/checkpoints/000002"
    checkpoint.mkdir(parents=True)
    (checkpoint / "state.pt").write_bytes(b"joint state")
    metadata = {"iteration": 2, "scientific_hash": "science", "manifests_hash": "data"}
    atomic_json(checkpoint / "metadata.json", metadata)
    atomic_json(
        checkpoint / "complete.json",
        {
            "files": {
                name: hashlib.sha256((checkpoint / name).read_bytes()).hexdigest()
                for name in ("state.pt", "metadata.json")
            }
        },
    )
    atomic_json(checkpoint.parent / "latest.json", {"path": "000002", "iteration": 2})
    run_dir = checkpoint.parents[1]
    atomic_json(run_dir / "state/health.json", {"phase": "stopped"})
    payload = {
        "study_id": "study",
        "operation": "pilot",
        "method": "D",
        "seed": 2026,
        "run_id": "run",
        "identity": "identity",
        "scientific_hash": "science",
        "manifests_hash": "data",
        "target_iterations": 10,
        "time": "02:00:00",
        "continuation_time": "04:00:00",
        "max_continuations": 1,
        "commit": "original-commit",
    }
    ledger = {
        "specification": payload,
        "identity": "identity",
        "attempts": [{"job_id": "123"}],
        "run_dir": str(run_dir),
        "retries": 0,
    }
    atomic_json(tmp_path / "state/run.json", ledger)
    monkeypatch.setattr(remote, "workspace", lambda _: tmp_path)
    monkeypatch.setattr(remote, "paths", lambda _: {"runs": tmp_path / "runs"})
    monkeypatch.setattr(remote, "job_state", lambda _: {"state": "COMPLETED"})
    submitted = []
    monkeypatch.setattr(remote, "submit", lambda p: submitted.append(p) or {"submitted": True})
    result = remote.recover({"run_id": "run", "explicit_continue": True})
    assert result["submitted"]
    assert submitted[0]["commit"] == "original-commit"
    assert submitted[0]["stop_after_iteration"] == 0
    assert submitted[0]["time"] == "04:00:00"
    assert not remote.recover({"run_id": "run", "explicit_continue": True})["submitted"]
    (checkpoint / "metadata.json").write_text("{}")
    with pytest.raises(ValueError, match="verification failed"):
        remote.verified_checkpoint(payload)


def ready_fixture(tmp_path, monkeypatch):
    import selfplay_opsd.preflight as preflight
    import selfplay_opsd.readiness as readiness

    cfg = load_config(Path("configs/yale.yaml"), output_root=tmp_path, study_id="test-ready")
    identity = {"trainer_sha256": "a" * 64, "worker_sha256": "b" * 64, "manifest_hash": "data"}
    monkeypatch.setattr(readiness, "execution_identity", lambda _: identity)
    monkeypatch.setattr(
        readiness,
        "active_events",
        lambda _: [{"kind": "evaluation_complete", "payload": {"split": "dev_monitor"}}],
    )
    monkeypatch.setattr(preflight, "TaskRepository", lambda _: SimpleNamespace(hash="data"))
    root = tmp_path / "readiness/test-ready"
    validation = {
        "ready": True,
        "executed": True,
        "manifest_hash": "data",
        "sandbox_image_digest": "sha256:" + "b" * 64,
        "execution_limits": cfg.scientific_settings()["sandbox"],
    }
    atomic_json(root / "validation.json", validation)
    atomic_json(
        root / "runtime.json",
        {
            "ready": True,
            "runtime_identity": identity,
            "isolation": {"ready": True, "checks": {"isolated": True}},
        },
    )
    for profile, count in (("single_gpu_80gb", 1), ("dual_gpu_80gb", 2)):
        atomic_json(
            tmp_path / "compatibility" / (profile + ".json"),
            {
                "ready": True,
                "runtime_identity": identity,
                "gpu_count": count,
                "hardware_profile": profile,
                "gpu_type": "h100",
                "objective_probes": list("ABCD"),
                "single_process_reference_compared": count > 1,
            },
        )
    atomic_json(tmp_path / "test-ready/base/runtime_identity.json", identity)
    atomic_json(
        tmp_path / "test-ready/base/evaluation_complete.json",
        {"ready": True, "runtime_identity": identity},
    )
    pilot_cfg = cfg.model_copy(
        update={"study_id": "test-ready-pilot", "method": "D", "iterations": 10}
    )
    pilot = tmp_path / "test-ready-pilot/D-seed2026/pilot_report.json"
    atomic_json(
        pilot,
        {
            "runtime_identity": identity,
            "iterations": 10,
            "includes_alignment": True,
            "target_hardware_verified": True,
            "scored_candidates": 4,
            "student_steps": 10,
            "proposer_steps": 10,
            "resumed_from_iteration": 2,
            "pilot_config": pilot_cfg.model_dump(mode="json"),
            "hardware": cfg.hardware.model_dump(),
        },
    )
    return cfg, pilot


def test_freeze_is_idempotent_and_all_methods_share_the_gate(tmp_path, monkeypatch):
    cfg, pilot = ready_fixture(tmp_path, monkeypatch)
    assert readiness_report(cfg, require_freeze=False)["eligible_to_submit_main"]
    assert not readiness_report(cfg)["eligible_to_submit_main"]
    first = freeze_study(cfg, pilot)
    assert freeze_study(cfg, pilot) == first
    for method in "ABCD":
        check_frozen(cfg.model_copy(update={"method": method}), SimpleNamespace(hash="data"))
    altered = cfg.model_copy(deep=True)
    altered.objective.sd_lambda = 0.2
    with pytest.raises(ExperimentError, match="differs"):
        check_frozen(altered, SimpleNamespace(hash="data"))
    validation = tmp_path / "readiness/test-ready/validation.json"
    atomic_json(validation, json.loads(validation.read_text()) | {"executed": False})
    assert not readiness_report(cfg)["eligible_to_submit_main"]
    with pytest.raises(ExperimentError, match="stale"):
        check_frozen(cfg, SimpleNamespace(hash="data"))


def test_main_preview_rejects_missing_runtime_before_scheduler(tmp_path, monkeypatch):
    remote = endpoint()
    monkeypatch.setattr(
        remote, "allowed", lambda _: pytest.fail("Scheduler validation must not run")
    )
    with pytest.raises(ValueError, match="verified allocation worker"):
        remote.preview({"operation": "train", "gpu_count": 2})


def test_controlled_backend_probe_exercises_all_objectives(repository, tmp_path):
    from selfplay_opsd.backend_check import check_backend

    cfg = load_config(
        study_id="probe-fixture",
        hardware_profile="cpu_debug",
        data_root=repository.root,
        output_root=tmp_path,
    )
    report = check_backend(cfg, tmp_path / "probe")
    assert report["objective_probes"] == list("ABCD")
    assert report["checkpoint_roundtrip"]
    assert report["gpu_count"] == 0
    assert (tmp_path / "probe/reference_gradients.pt").is_file()
