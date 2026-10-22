"""Readiness must fail closed before allocating resources or changing a study."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import socket
import subprocess
import tempfile
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
    backend_expected = readiness.backend_identity(cfg)
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
            tmp_path / readiness.BACKEND_EVIDENCE_DIRECTORY / (profile + ".json"),
            {
                "ready": True,
                "backend_identity": backend_expected,
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
    from selfplay_opsd.method_smoke import SMOKE_CHECKS, smoke_config, smoke_path
    from selfplay_opsd.storage import file_hash

    for method in "ABC":
        smoke_cfg = smoke_config(cfg, method)
        checkpoint = smoke_cfg.run_dir / "checkpoints/000001"
        atomic_json(
            checkpoint / "metadata.json",
            {
                "iteration": 1,
                "scientific_hash": smoke_cfg.scientific_hash,
                "runtime_identity": identity,
                "manifests_hash": "data",
            },
        )
        atomic_json(
            checkpoint / "complete.json",
            {"files": {"metadata.json": file_hash(checkpoint / "metadata.json")}},
        )
        atomic_json(
            smoke_path(cfg, method),
            {
                "ready": True,
                "method": method,
                "checks": dict.fromkeys(SMOKE_CHECKS, True),
                "runtime_identity": identity,
                "smoke_runtime_identity": identity,
                "common_settings": cfg.scientific_settings(common=True, calibration=True),
                "hardware": cfg.hardware.model_dump(mode="json"),
                "gpu_type": "h100",
                "checkpoint_marker_sha256": file_hash(checkpoint / "complete.json"),
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


def test_all_method_proofs_and_intact_checkpoints_are_required(tmp_path, monkeypatch):
    cfg, _ = ready_fixture(tmp_path, monkeypatch)
    from selfplay_opsd.method_smoke import smoke_config, smoke_path

    path = smoke_path(cfg, "B")
    original = json.loads(path.read_text())
    atomic_json(path, original | {"method": "A"})
    assert not readiness_report(cfg, require_freeze=False)["gates"]["method_B"]["ready"]
    atomic_json(path, original)
    checkpoint = smoke_config(cfg, "B").run_dir / "checkpoints/000001/metadata.json"
    checkpoint.write_text("{}")
    assert not readiness_report(cfg, require_freeze=False)["gates"]["method_B"]["ready"]


def test_main_preview_rejects_missing_runtime_before_scheduler(tmp_path, monkeypatch):
    remote = endpoint()
    monkeypatch.setattr(
        remote, "allowed", lambda _: pytest.fail("Scheduler validation must not run")
    )
    with pytest.raises(ValueError, match="verified allocation worker"):
        remote.preview({"operation": "train", "gpu_count": 2})


def test_legacy_and_stale_backend_evidence_cannot_satisfy_readiness(tmp_path, monkeypatch):
    cfg, _ = ready_fixture(tmp_path, monkeypatch)
    current = tmp_path / "compatibility/backend-v2/single_gpu_80gb.json"
    legacy = tmp_path / "compatibility/single_gpu_80gb.json"
    current.replace(legacy)
    assert not readiness_report(cfg, require_freeze=False)["gates"]["single_gpu_reference"]["ready"]
    stale = json.loads(legacy.read_text())
    stale["backend_identity"]["trainer_sha256"] = "different-image"
    atomic_json(current, stale)
    assert not readiness_report(cfg, require_freeze=False)["gates"]["single_gpu_reference"]["ready"]


def test_remote_backend_evidence_does_not_establish_runtime_readiness(tmp_path, monkeypatch):
    remote = endpoint()
    monkeypatch.setattr(remote, "workspace", lambda _: tmp_path)
    monkeypatch.setattr(remote, "refreshed_budget", lambda _: {})
    identity = {"schema_version": 2, "trainer_sha256": "a" * 64}
    for profile, count in [("single_gpu_80gb", 1), ("dual_gpu_80gb", 2)]:
        atomic_json(
            tmp_path / "runs/compatibility/backend-v2" / (profile + ".json"),
            {
                "ready": True,
                "backend_identity": identity,
                "gpu_count": count,
                "hardware_profile": profile,
                "gpu_type": "h100",
                "objective_probes": list("ABCD"),
                "single_process_reference_compared": count > 1,
            },
        )
    payload = {"study_id": "study", "backend_identity": identity}
    report = remote.study_state(payload)
    assert report["gates"]["single"] and report["gates"]["target"]
    assert not report["gates"]["worker"] and not report["gates"]["runtime"]
    assert not report["eligible_to_submit_main"]
    payload["backend_identity"] = {"schema_version": 1}
    report = remote.study_state(payload)
    assert not report["gates"]["single"] and not report["gates"]["target"]


def test_controlled_backend_probe_exercises_all_objectives(tmp_path, monkeypatch):
    from selfplay_opsd.backend_check import check_backend

    cfg = load_config(
        study_id="probe-fixture",
        hardware_profile="cpu_debug",
        data_root=tmp_path / "does-not-exist",
        output_root=tmp_path,
    )
    monkeypatch.setattr(
        "selfplay_opsd.readiness.TaskRepository",
        lambda _: pytest.fail("Trusted backend probes must not open a dataset"),
    )
    monkeypatch.setattr(
        "selfplay_opsd.sandbox.SandboxBackend.__init__",
        lambda *a, **kw: pytest.fail("Trusted backend probes must not construct a sandbox"),
    )
    report = check_backend(cfg, tmp_path / "probe")
    assert report["objective_probes"] == list("ABCD")
    assert report["checkpoint_roundtrip"]
    assert report["gpu_count"] == 0
    assert (tmp_path / "probe/reference_gradients.pt").is_file()
    assert "runtime_identity" not in report
    assert report["backend_identity"]["schema_version"] == 2
    assert "worker_sha256" not in report["backend_identity"]


def allocation_module():
    spec = importlib.util.spec_from_file_location("repair_job", "scripts/hpc_job.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def allocation_env(tmp_path):
    image = tmp_path / "trainer.sif"
    image.write_bytes(b"pinned trainer fixture")
    return {
        "SELFPLAY_OPERATION": "backend-check",
        "SELFPLAY_RUNS": str(tmp_path / "runs"),
        "SELFPLAY_STUDY": "study",
        "SLURM_JOB_ID": "123",
        "SELFPLAY_CACHE": str(tmp_path / "cache"),
        "SELFPLAY_PROJECT": str(tmp_path / "checkout"),
        "SELFPLAY_TRAINER_SIF": str(image),
        "SELFPLAY_TRAINER_SHA256": hashlib.sha256(image.read_bytes()).hexdigest(),
        "SELFPLAY_METHOD": "D",
        "SELFPLAY_SEED": "2026",
        "SELFPLAY_GPU_COUNT": "1",
        "SELFPLAY_HARDWARE": "single_gpu_80gb",
        "SELFPLAY_SETTINGS": "configs/yale.yaml",
        "SELFPLAY_OUTPUT": "/runs/probe",
        "SLURM_JOB_GPUS": "1",
        "CUDA_VISIBLE_DEVICES": "0",
    }


def gpu_inventory(args):
    if "-x" in args:
        return (
            "<nvidia_smi_log>"
            "<gpu><uuid>GPU-11111111-1111-1111-1111-111111111111</uuid>"
            "<minor_number>1</minor_number></gpu>"
            "<gpu><uuid>GPU-00000000-0000-0000-0000-000000000000</uuid>"
            "<minor_number>0</minor_number></gpu></nvidia_smi_log>"
        )
    assert args[1] == "--query-gpu=index,uuid,pci.bus_id"
    return (
        "0, GPU-00000000-0000-0000-0000-000000000000, 0000:01:00.0\n"
        "1, GPU-11111111-1111-1111-1111-111111111111, 0000:02:00.0\n"
    )


def test_backend_allocation_ignores_broken_worker_and_needs_no_data(tmp_path, monkeypatch):
    job = allocation_module()
    env = allocation_env(tmp_path) | {
        "SELFPLAY_WORKER_SIF": "/missing/worker.sif",
        "SELFPLAY_WORKER_TOKEN": "stale-token",
        "SELFPLAY_WORKER_URL": "invalid",
    }
    monkeypatch.setattr(job.subprocess, "Popen", lambda *a, **k: pytest.fail("No worker broker"))
    monkeypatch.setattr(job, "user_bus_environment", lambda _: pytest.fail("No D-Bus dependency"))
    # Kernel allocation ownership has independent adversarial coverage. This
    # test checks that trusted backend work has no worker/data dependencies.
    monkeypatch.setattr(
        job,
        "allocated_gpus",
        lambda env, report: [{"uuid": "GPU-11111111-1111-1111-1111-111111111111"}],
    )
    monkeypatch.setattr(
        job.subprocess,
        "check_output",
        lambda args, **kw: (
            gpu_inventory(args)
            if args[0] == "nvidia-smi"
            else json.dumps(
                [
                    {
                        "uuid": "11111111-1111-1111-1111-111111111111",
                        "free_bytes": 79 * 2**30,
                        "total_bytes": 80 * 2**30,
                    }
                ]
            )
        ),
    )
    captured = []
    monkeypatch.setattr(
        job.subprocess,
        "run",
        lambda args, **kw: captured.append((args, kw)) or SimpleNamespace(returncode=0),
    )
    assert job.execute(env) == 0
    args, options = captured[0]
    assert "backend-check" in args and "--nv" in args
    assert not any(":/data/prepared" in arg for arg in args)
    assert "SELFPLAY_WORKER_TOKEN" not in options["env"]
    assert "APPTAINERENV_SELFPLAY_WORKER_URL" not in options["env"]
    assert options["env"]["APPTAINERENV_CUDA_VISIBLE_DEVICES"] == (
        "GPU-11111111-1111-1111-1111-111111111111"
    )
    assert options["env"]["APPTAINERENV_CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    proof = json.loads((tmp_path / "runs/readiness/study/gpu-allocation-123.json").read_text())
    assert proof["ready"] and proof["slurm_cuda_visible_devices"] == "0"


@pytest.mark.parametrize("gpu_ids,mask", [("", "0"), ("0,1", "0"), ("1", ""), ("1,1", "0,1")])
def test_gpu_selection_rejects_missing_or_inconsistent_slurm_mask(tmp_path, gpu_ids, mask):
    job = allocation_module()
    env = allocation_env(tmp_path) | {"SLURM_JOB_GPUS": gpu_ids, "CUDA_VISIBLE_DEVICES": mask}
    with pytest.raises(ValueError, match="Slurm GPU allocation/mask"):
        job.allocated_gpus(env)


def test_gpu_selection_rejects_duplicate_physical_device_identity(tmp_path, monkeypatch):
    job = allocation_module()
    original_read = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, *a, **kw: "0::/slurm/job_123/step_batch\n"
        if str(path) == "/proc/self/cgroup"
        else original_read(path, *a, **kw),
    )
    monkeypatch.setattr(
        job,
        "diagnostic_command",
        lambda args, env: {
            "exit_code": 0,
            "stdout": "JobId=123 JobState=RUNNING NumNodes=1 TRES=cpu=2,gres/gpu=1 "
            if args[0] == "scontrol"
            else gpu_inventory(args).replace("<minor_number>1", "<minor_number>0"),
        },
    )
    with pytest.raises(ValueError, match="Duplicate NVIDIA inventory"):
        job.allocated_gpus(allocation_env(tmp_path))


@pytest.mark.parametrize(
    "uuid,used",
    [
        ("00000000-0000-0000-0000-000000000000", 0),
        ("11111111-1111-1111-1111-111111111111", 70 * 2**30),
    ],
)
def test_container_gpu_preflight_rejects_wrong_or_occupied_device(
    tmp_path, monkeypatch, uuid, used
):
    job = allocation_module()
    env = allocation_env(tmp_path)
    monkeypatch.setattr(
        job.subprocess,
        "check_output",
        lambda *a, **kw: json.dumps(
            [{"uuid": uuid, "free_bytes": 80 * 2**30 - used, "total_bytes": 80 * 2**30}]
        ),
    )
    report = tmp_path / "gpu-allocation.json"
    with pytest.raises(ValueError):
        job.verify_container_gpus(
            ["apptainer", "exec", "trainer.sif"],
            env,
            [{"uuid": "GPU-11111111-1111-1111-1111-111111111111"}],
            report,
        )
    assert not json.loads(report.read_text())["ready"]


@pytest.mark.parametrize(
    "proof,expected",
    [
        ({}, False),
        ({"ready": False, "image_sha256": "image", "allocated": [{}]}, False),
        ({"ready": True, "image_sha256": "old-image", "allocated": [{}]}, False),
        ({"ready": True, "image_sha256": "image", "allocated": [{}]}, True),
    ],
)
def test_h200_fallback_requires_clean_gpu_allocation_proof(tmp_path, monkeypatch, proof, expected):
    remote = endpoint()
    monkeypatch.setattr(remote, "workspace", lambda _: tmp_path)
    monkeypatch.setattr(remote, "refreshed_budget", lambda _: {})
    monkeypatch.setattr(remote, "job_state", lambda _: {"state": "FAILED"})
    atomic_json(
        tmp_path / "state/run.json",
        {
            "specification": {
                "study_id": "study",
                "gpu_type": "h100",
                "gpu_count": 1,
                "operation": "backend-check",
                "run_id": "run",
                "image_sha256": "image",
            },
            "attempts": [{"job_id": "123"}],
            "run_dir": str(tmp_path / "run"),
        },
    )
    (tmp_path / "state/run-123.out").write_text("torch.OutOfMemoryError: CUDA out of memory")
    atomic_json(tmp_path / "runs/readiness/study/gpu-allocation-123.json", proof)
    assert remote.study_state({"study_id": "study"})["h100_memory_fit_failure"] is expected


def test_backend_precheck_does_not_consult_runtime_evidence(tmp_path, monkeypatch):
    remote = endpoint()
    env = allocation_env(tmp_path)
    monkeypatch.setattr(remote, "study_state", lambda _: pytest.fail("No worker evidence gate"))
    remote.precheck(
        {
            "operation": "backend-check",
            "gpu_count": 1,
            "image": env["SELFPLAY_TRAINER_SIF"],
            "image_sha256": env["SELFPLAY_TRAINER_SHA256"],
            "worker_image": "/missing/worker.sif",
        }
    )
    with pytest.raises(ValueError, match="trainer SIF"):
        remote.precheck({"operation": "backend-check"})


@pytest.mark.parametrize(
    "operation",
    ["worker-check", "runtime-check", "train", "pilot", "method-smoke", "evaluate", "diagnose"],
)
def test_generated_code_cannot_launch_without_worker(operation):
    with pytest.raises(ValueError, match="allocation worker"):
        endpoint().precheck({"operation": operation})
    with pytest.raises(ValueError, match="allocation worker"):
        allocation_module().execute({"SELFPLAY_OPERATION": operation})


@pytest.mark.parametrize(
    "operation", ["train", "pilot", "method-smoke", "evaluate", "diagnose", "freeze-study"]
)
@pytest.mark.parametrize(
    "gates", [{"worker": False, "runtime": True}, {"worker": True, "runtime": False}]
)
def test_generated_code_rejects_failed_or_stale_runtime(operation, gates, tmp_path, monkeypatch):
    remote = endpoint()
    worker = tmp_path / "worker.sif"
    worker.write_bytes(b"worker fixture")
    monkeypatch.setattr(remote, "study_state", lambda _: {"gates": gates})
    with pytest.raises(ValueError, match="Runtime isolation"):
        remote.precheck(
            {"operation": operation, "worker_image": str(worker), "worker_sha256": "a" * 64}
        )


def test_failed_allocation_probe_prevents_generated_code(tmp_path, monkeypatch):
    job = allocation_module()
    env = allocation_env(tmp_path) | {
        "SELFPLAY_OPERATION": "runtime-check",
        "SELFPLAY_WORKER_SIF": "worker.sif",
    }
    monkeypatch.setattr(job, "broker_args", lambda *args: ["probe"])
    monkeypatch.setattr(
        job.subprocess,
        "Popen",
        lambda *a, **k: SimpleNamespace(poll=lambda: 1, terminate=lambda: None, wait=lambda **_: 1),
    )
    monkeypatch.setattr(
        job.subprocess, "run", lambda *a, **k: pytest.fail("Generated-code command must not start")
    )
    with pytest.raises(RuntimeError, match="isolation/startup failed"):
        job.execute(env)


def test_unknown_operation_fails_at_both_boundaries():
    with pytest.raises(ValueError, match="Unsupported"):
        endpoint().precheck({"operation": "unrecognized"})
    with pytest.raises(ValueError, match="Unsupported"):
        allocation_module().execute({"SELFPLAY_OPERATION": "unrecognized"})


def test_cpu_conversion_retries_are_bounded_without_gpu_requests(tmp_path, monkeypatch):
    remote = endpoint()
    checkout = tmp_path / "checkout"
    (checkout / "configs").mkdir(parents=True)
    (checkout / "configs/yale.yaml").write_text("study_id: study\n")
    paths = {
        "checkout": checkout,
        "state": tmp_path / "state",
        "runs": tmp_path / "runs",
        "dataset": tmp_path / "data",
        "cache": tmp_path / "cache",
    }
    identity = {"commit": "a" * 40, "manifest_sha256": "b" * 64}
    atomic_json(paths["state"] / f"sync-{identity['commit']}.json", identity)
    monkeypatch.setattr(remote, "workspace", lambda _: tmp_path)
    monkeypatch.setattr(remote, "paths", lambda _: paths)
    monkeypatch.setattr(remote, "allowed", lambda _: {"authorized": True})
    monkeypatch.setattr(remote, "precheck", lambda _: None)
    monkeypatch.setattr(remote, "job_state", lambda _: {"state": "NODE_FAIL"})
    calls = []

    def sbatch(args, **_):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, str(120 + len(calls)), "")

    monkeypatch.setattr(remote, "run", sbatch)
    payload = identity | {
        "run_id": "convert",
        "identity": "conversion",
        "operation": "image-convert",
        "image": "",
        "image_sha256": "",
        "settings": "configs/yale.yaml",
        "study_id": "study",
        "method": "D",
        "seed": 2026,
        "gpu_count": 0,
        "profile": "dual_gpu_80gb",
        "gpu_type": "h100",
        "account": "course",
        "partition": "day",
        "cpus": 8,
        "memory_gib": 64,
        "time": "01:00:00",
        "retry_setup": True,
        "image_metadata": {"kind": "worker"},
        "archive_sha256": "c" * 64,
    }
    assert all(remote.submit(payload)["submitted"] for _ in range(3))
    with pytest.raises(ValueError, match="explicit checkpoint resume"):
        remote.submit(payload)
    assert len(calls) == 3
    assert all(not any(arg.startswith("--gpus=") for arg in args) for args in calls)


def test_worker_uses_only_an_existing_user_bus():
    spec = importlib.util.spec_from_file_location("hpc_job", "scripts/hpc_job.py")
    assert spec and spec.loader
    job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job)
    # AF_UNIX addresses have a short platform-specific path bound.
    with tempfile.TemporaryDirectory(prefix="sp-bus-", dir="/tmp") as location:
        directory = Path(location)
        assert job.user_bus_environment({}, directory) == {}
        with socket.socket(socket.AF_UNIX) as bus:
            bus.bind(str(directory / "bus"))
            env = job.user_bus_environment({}, directory)
            assert env["XDG_RUNTIME_DIR"] == str(directory)
            assert env["DBUS_SESSION_BUS_ADDRESS"] == "unix:path=" + str(directory / "bus")
            assert (
                job.user_bus_environment({"DBUS_SESSION_BUS_ADDRESS": "existing"}, directory)[
                    "DBUS_SESSION_BUS_ADDRESS"
                ]
                == "existing"
            )


def test_live_readers_never_observe_partial_worker_reports(tmp_path):
    from selfplay_opsd.worker_runtime import publish_json

    report = tmp_path / "worker.json"
    publish_json(report, {"attempt": 0, "detail": "x" * 65536})

    def writer():
        for attempt in range(1, 10):
            publish_json(report, {"attempt": attempt, "detail": "x" * 65536})

    with ThreadPoolExecutor(1) as executor:
        future = executor.submit(writer)
        while not future.done():
            value = json.loads(report.read_text())
            assert 0 <= value["attempt"] <= 9 and len(value["detail"]) == 65536
        future.result()
    assert json.loads(report.read_text())["attempt"] == 9
