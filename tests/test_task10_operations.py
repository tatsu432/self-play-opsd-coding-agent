import copy
import importlib.util
import json

import pytest

from selfplay_opsd.storage import atomic_json, file_hash


def remote_module():
    spec = importlib.util.spec_from_file_location("remote_task10", "scripts/hpc_remote.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_study_reuse_checks_dependencies_and_preserves_original_evidence(tmp_path, monkeypatch):
    module = remote_module()
    monkeypatch.setattr(module, "workspace", lambda payload: tmp_path)
    original = tmp_path / "runs/old"
    evidence = tmp_path / "runs/readiness/old"
    identity = {"scientific_code_hash": "science", "common_hash": "common"}
    common = {"study_id": "old", "iterations": 200, "seed": 2026}
    plan = {
        "hardware": {"gpu_count": 1},
        "checkpoint_every": 1,
        "common_config": common,
        "per_method": {m: {"signal_seconds": 4000} for m in "ABCD"},
    }
    frozen = {
        "runtime_identity": identity,
        "common_config": common,
        "gpu_type": "h200",
        "resource_plan": plan,
    }
    atomic_json(original / "study_frozen.json", frozen)
    atomic_json(original / "resource_plan.json", plan)
    atomic_json(original / "base/runtime_identity.json", identity)
    atomic_json(original / "base/evaluation_complete.json", {"ready": True})
    (original / "base/events").mkdir()
    (original / "base/events/segment.jsonl").write_text('{"original": true}\n')
    atomic_json(
        tmp_path / "runs/old-pilot-h200/D-seed2026/pilot_report.json",
        {"pilot_config": {"study_id": "old-pilot-h200"}},
    )
    for name in ("worker", "runtime", "validation", "allocation-single_gpu_80gb"):
        atomic_json(evidence / f"{name}.json", {"ready": True})
    for method in "ABC":
        atomic_json(
            evidence / "method-smokes" / f"{method}.json", {"method": method, "ready": True}
        )
    source_hash = file_hash(original / "study_frozen.json")
    monkeypatch.setattr(
        module,
        "study_state",
        lambda payload: {"eligible_to_submit_main": True, "reports": {"freeze": frozen}},
    )
    request = {
        "source_study": "old",
        "study_id": "new",
        "runtime_identity": identity,
        "common_config": common | {"study_id": "new"},
        "hardware": {"gpu_count": 1},
        "checkpoint_every": 1,
        "gpu_type": "h200",
        "pilot_suffix": "-pilot-h200",
        "seed": 2026,
    }
    bad = copy.deepcopy(request)
    bad["runtime_identity"]["scientific_code_hash"] = "changed"
    with pytest.raises(ValueError, match="dependency-equivalent"):
        module.reuse_study(bad)
    assert not (tmp_path / "runs/new").exists()
    assert module.reuse_study(request)["ready"]
    assert module.reuse_study(request)["ready"]  # idempotent before freeze
    assert file_hash(original / "study_frozen.json") == source_hash
    report = json.loads((tmp_path / "runs/readiness/new/method-smokes/A.json").read_text())
    assert report["reuse_provenance"]["sha256"] == file_hash(evidence / "method-smokes/A.json")
    assert not (tmp_path / "runs/new-method-smoke").exists()
    atomic_json(tmp_path / "runs/new/study_frozen.json", {})
    with pytest.raises(ValueError, match="already frozen"):
        module.reuse_study(request)


def test_cpu_wandb_verification_requires_no_gpu_and_no_model(tmp_path, monkeypatch):
    from selfplay_opsd.wandb_tools import verify

    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    import httpx

    monkeypatch.setattr(
        httpx, "get", lambda *args, **kwargs: type("Response", (), {"status_code": 404})()
    )
    report = verify(tmp_path)
    assert report["https_reachable"]
    assert report["allocated_gpu_count"] == 0
    assert not report["ready"] and not report["cloud_verified"]
    assert "required_action" in report


def test_verify_does_not_treat_init_as_cloud_delivery(tmp_path, monkeypatch):
    import httpx
    import wandb

    from selfplay_opsd.wandb_tools import verify

    monkeypatch.setenv("WANDB_API_KEY", "secret")
    monkeypatch.setattr(
        httpx, "get", lambda *args, **kwargs: type("Response", (), {"status_code": 200})()
    )

    class Api:
        viewer = type("User", (), {"username": "tatsu432"})()
        default_entity = "authorized"

        def flush(self):
            pass

        def run(self, path):
            return self

        def scan_history(self, **kwargs):
            return []

    class Run:
        url = "https://wandb.ai/authorized/project/run"
        entity, project, id = "authorized", "project", "run"

        def log(self, payload):
            pass

        def finish(self):
            pass

    monkeypatch.setattr(wandb, "Api", lambda **kwargs: Api())
    monkeypatch.setattr(wandb, "init", lambda **kwargs: Run())
    report = verify(tmp_path, timeout=0)
    assert not report["ready"] and report["error_type"] == "TimeoutError"
    assert "secret" not in str(report)


def test_allocation_telemetry_counts_all_attempts_once(tmp_path, monkeypatch):
    import subprocess
    import time

    spec = importlib.util.spec_from_file_location("job_task10", "scripts/hpc_job.py")
    job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job)
    runs = tmp_path / "runs"
    atomic_json(tmp_path / "state/run.json", {"attempts": [{"job_id": "1"}, {"job_id": "2"}]})
    monkeypatch.setattr(
        job.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args,
            0,
            stdout="1|3600|gres/gpu=1,gres/gpu:h200=1|\n2|1800|gres/gpu=1,gres/gpu:h200=1|\n",
        ),
    )
    env = {
        "SELFPLAY_OPERATION": "train",
        "SELFPLAY_RUNS": str(runs),
        "SELFPLAY_STUDY": "study",
        "SELFPLAY_METHOD": "A",
        "SELFPLAY_SEED": "2026",
        "SELFPLAY_RUN_ID": "run",
        "SLURM_JOB_ID": "2",
    }
    output = runs / "study/A-seed2026/state/allocation-samples.jsonl"
    with job.allocation_telemetry(env):
        deadline = time.monotonic() + 2
        while not output.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
    sample = json.loads(output.read_text())
    assert sample["allocated_gpu_hours"] == 1.5


def test_reused_smoke_source_is_not_redirected_by_current_allocation(tmp_path, monkeypatch):
    from selfplay_opsd.config import load_config
    from selfplay_opsd.method_smoke import SMOKE_CHECKS, smoke_config, verify_smoke
    from selfplay_opsd.storage import CheckpointManager

    source_cfg = load_config(study_id="old", hardware_profile="cpu_debug", output_root=tmp_path)
    cfg = source_cfg.model_copy(update={"study_id": "new"}, deep=True)
    expected = smoke_config(source_cfg, "A")
    identity = {"manifest_hash": "manifest"}
    monkeypatch.setattr("selfplay_opsd.readiness.execution_identity", lambda cfg: identity)
    monkeypatch.setenv("SELFPLAY_MAIN_STUDY", "new")
    monkeypatch.setenv("SELFPLAY_GPU_TYPE", "h100")
    checkpoint = CheckpointManager(expected.run_dir).save(
        3,
        {},
        {
            "iteration": 3,
            "scientific_hash": expected.scientific_hash,
            "runtime_identity": identity,
            "manifests_hash": "manifest",
        },
    )
    report = {
        "ready": True,
        "method": "A",
        "runtime_identity": identity,
        "smoke_runtime_identity": identity,
        "common_settings": cfg.scientific_settings(common=True, calibration=True),
        "hardware": cfg.hardware.model_dump(mode="json"),
        "gpu_type": "h100",
        "checks": dict.fromkeys(SMOKE_CHECKS, True),
        "checkpoint_marker_sha256": file_hash(checkpoint / "complete.json"),
    }
    source = tmp_path / "readiness/old/method-smokes/A.json"
    atomic_json(source, report)
    reused = report | {"reuse_provenance": {"source_study": "old", "sha256": file_hash(source)}}
    atomic_json(tmp_path / "readiness/new/method-smokes/A.json", reused)
    assert verify_smoke(cfg, "A") == reused
    atomic_json(source, report | {"ready": False})
    with pytest.raises(Exception, match="provenance"):
        verify_smoke(cfg, "A")
