"""Kernel ownership and diagnostic regressions; these do not establish live GPU readiness."""

import importlib.util
import io
import json
import os
import sys
import threading
from pathlib import Path

import pytest


@pytest.fixture
def allocation(monkeypatch):
    spec = importlib.util.spec_from_file_location("allocation_job", "scripts/hpc_job.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    inventory = [
        {"index": 0, "minor": 2, "uuid": "GPU-" + "11111111-1111-1111-1111-111111111111"},
        {"index": 1, "minor": 0, "uuid": "GPU-" + "22222222-2222-2222-2222-222222222222"},
    ]
    raw_read = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, *a, **kw: "0::/slurm/uid_123/job_123/step_batch\n"
        if str(path) == "/proc/self/cgroup"
        else raw_read(path, *a, **kw),
    )

    def diagnostic(args, env):
        if args[0] == "scontrol":
            text = "JobId=123 JobState=RUNNING NumNodes=1 TRES=cpu=2,gres/gpu=1 "
        elif "-q" in args:
            text = (
                "<nvidia_smi_log>"
                + "".join(
                    f"<gpu><uuid>{d['uuid']}</uuid><minor_number>{d['minor']}</minor_number></gpu>"
                    for d in inventory
                )
                + "</nvidia_smi_log>"
            )
        else:
            text = "\n".join(
                f"{d['index']}, {d['uuid']}, 0000:0{d['index']}:00.0" for d in inventory
            )
        return {"exit_code": 0, "stdout": text, "stderr": ""}

    monkeypatch.setattr(module, "diagnostic_command", diagnostic)
    monkeypatch.setattr(module, "device_access", lambda minor: minor == "0")
    return module


def environment():
    return {
        "SLURM_JOB_ID": "123",
        "SLURM_JOB_GPUS": "0",
        "CUDA_VISIBLE_DEVICES": "0",
        "SELFPLAY_GPU_COUNT": "1",
        "SELFPLAY_TRAINER_SHA256": "a" * 64,
    }


def test_divergent_ordinals_resolve_only_kernel_permitted_device(allocation, tmp_path):
    report = tmp_path / "allocation.json"
    devices = allocation.allocated_gpus(environment(), report)
    assert devices[0]["index"] == "1" and devices[0]["minor"] == "0"
    proof = json.loads(report.read_text())
    assert proof["ownership_verified"] and not proof["ready"]
    assert len(proof["devices"]) == 2
    assert proof["slurm_job"]["exit_code"] == 0


def test_extra_device_access_rejects_and_preserves_raw_evidence(allocation, tmp_path, monkeypatch):
    monkeypatch.setattr(allocation, "device_access", lambda _: True)
    report = tmp_path / "allocation.json"
    with pytest.raises(ValueError, match="unexpected GPU count"):
        allocation.allocated_gpus(environment(), report)
    proof = json.loads(report.read_text())
    assert not proof["ready"] and proof["failure_category"] == "allocation"
    assert proof["inventory"]["stdout"] and proof["xml_inventory"]["stdout"]


def test_worker_diagnostics_report_only_current_uid_subid_mappings(allocation, monkeypatch):
    calls = []

    def diagnostic(args, env):
        calls.append(args)
        if args[:2] == ["getent", "subuid"]:
            return {"exit_code": 0, "stdout": "35094:100000:65536\n", "stderr": ""}
        if args[:2] == ["getent", "subgid"]:
            return {"exit_code": 0, "stdout": "35094:100000:65536\n", "stderr": ""}
        return {"exit_code": 0, "stdout": "version\n", "stderr": ""}

    monkeypatch.setattr(allocation, "diagnostic_command", diagnostic)
    report = allocation.worker_diagnostics(environment())
    assert report["diagnostic_only"] and not report["ready"]
    assert report["subid"]["uid_mappings"]["stdout"] == "35094:100000:65536\n"
    assert report["subid"]["gid_mappings"]["stdout"] == "35094:100000:65536\n"
    assert [call for call in calls if call[:2] == ["getent", "subuid"]] == [
        ["getent", "subuid", str(os.getuid())]
    ]
    assert [call for call in calls if call[:2] == ["getent", "subgid"]] == [
        ["getent", "subgid", str(os.getgid())]
    ]


def test_uuid_disagreement_rejects_ordinal_guess(allocation, tmp_path):
    env = environment() | {"SLURM_JOB_GPUS": "GPU-11111111-1111-1111-1111-111111111111"}
    with pytest.raises(ValueError, match="UUIDs disagree"):
        allocation.allocated_gpus(env, tmp_path / "allocation.json")


def test_missing_mask_publishes_failure_before_inventory(allocation, tmp_path):
    report = tmp_path / "allocation.json"
    with pytest.raises(ValueError, match="Missing or inconsistent"):
        allocation.allocated_gpus(environment() | {"CUDA_VISIBLE_DEVICES": ""}, report)
    assert json.loads(report.read_text())["error"]


def test_container_uuid_mismatch_and_occupancy_never_pass(allocation, tmp_path, monkeypatch):
    report = tmp_path / "allocation.json"
    devices = allocation.allocated_gpus(environment(), report)
    for uuid, used, expected in [
        ("wrong", 0, "UUIDs"),
        (devices[0]["uuid"], 3 * 1024**3, "occupied"),
    ]:
        monkeypatch.setattr(
            allocation.subprocess,
            "check_output",
            lambda *a, uuid=uuid, used=used, **kw: json.dumps(
                [{"uuid": uuid, "total_bytes": 80 * 1024**3, "free_bytes": 80 * 1024**3 - used}]
            ),
        )
        with pytest.raises(ValueError, match=expected):
            allocation.verify_container_gpus([], environment(), devices, report)
        assert not json.loads(report.read_text())["ready"]


def test_cgroup_probe_reads_nested_effective_limits_and_rejects_writable_parent(monkeypatch):
    from selfplay_opsd.worker_runtime import RuntimeConfig, isolation_probes

    files = {
        "/proc/self/cgroup": "0::/slurm/job/worker\n",
        "/proc/self/mountinfo": "1 2 0:1 / /sys/fs/cgroup ro - cgroup2 cgroup ro\n",
        "/sys/fs/cgroup/slurm/job/worker/pids.max": "32",
        "/sys/fs/cgroup/slurm/job/pids.max": "max",
        "/sys/fs/cgroup/pids.max": "max",
    }
    monkeypatch.setattr("builtins.open", lambda path: io.StringIO(files[path]))
    monkeypatch.setattr(os.path, "isfile", lambda path: path in files)
    monkeypatch.setattr(os, "access", lambda *a: False)
    probes = isolation_probes(RuntimeConfig())
    namespace = {}
    exec(probes["pids_limit"], namespace)
    assert namespace["check"]()
    exec(probes["cgroup_readonly"], namespace)
    assert namespace["check"]()
    monkeypatch.setattr(
        os, "access", lambda path, mode: path == "/sys/fs/cgroup/slurm/job/cgroup.procs"
    )
    assert not namespace["check"]()


@pytest.mark.parametrize("close_pipes", [False, True])
def test_cancellation_terminates_running_worker_process(close_pipes):
    from selfplay_opsd.worker_runtime import bounded_process

    cancelled = threading.Event()
    timer = threading.Timer(0.2, cancelled.set)
    timer.start()
    try:
        source = (
            "import os,time; "
            + ("os.close(1); os.close(2); " if close_pipes else "")
            + "time.sleep(60)"
        )
        code, _, _, reason = bounded_process(
            [sys.executable, "-c", source],
            b"",
            60,
            4096,
            cancelled=cancelled,
        )
        assert code < 0 and reason == "cancelled"
    finally:
        timer.cancel()


@pytest.mark.parametrize(
    "message,category",
    [
        ("Ambiguous Slurm/NVIDIA GPU mapping", "allocation"),
        ("DBUS_SESSION_BUS_ADDRESS is not set", "site_setup"),
        ("tensor shape is wrong", "application"),
    ],
)
def test_stage_diagnostics_classify_legacy_failure_logs(tmp_path, monkeypatch, message, category):
    spec = importlib.util.spec_from_file_location("diagnostic_remote", "scripts/hpc_remote.py")
    remote = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(remote)
    state = tmp_path / "state"
    state.mkdir()
    (state / "ledger.json").write_text(
        json.dumps(
            {
                "specification": {
                    "study_id": "study",
                    "run_id": "run",
                    "operation": "backend-check",
                    "profile": "single_gpu_80gb",
                },
                "attempts": [{"job_id": "123"}],
            }
        )
    )
    log = state / "run-123.out"
    log.write_text(message)
    monkeypatch.setattr(remote, "job_state", lambda _: {"state": "FAILED"})
    stage = remote.stage_status(tmp_path, "study", {"single": False}, {})["single"]
    assert stage["failure_category"] == category
    assert stage["evidence"] == str(log)


@pytest.mark.parametrize("partition", [None, "scavenge_gpu"])
@pytest.mark.parametrize("operation", ["allocation-check", "worker-diagnose"])
def test_short_diagnostics_do_not_signal_at_start_or_require_worker_data(
    monkeypatch, operation, partition
):
    from selfplay_opsd import hpc

    class Remote:
        def call(self, name, *args):
            if name == "context":
                return {
                    "storage": {
                        "project": {"project_cpsc4880": "/project/cpsc4880"},
                        "scratch": {"scratch_cpsc4880": "/scratch/cpsc4880"},
                    },
                    "slurm": {
                        "associations": {"exit_code": 0, "stdout": "cpsc4880||normal|normal\n"}
                    },
                }
            assert name == "study"
            return {
                "images": {"trainer": {"path": "/trainer.sif", "sha256": "a" * 64}, "worker": {}},
                "reports": {"worker": {}, "freeze": {}},
            }

    monkeypatch.setattr(hpc, "snapshot", lambda: {"commit": "a" * 40, "manifest_sha256": "a" * 64})
    monkeypatch.setattr(
        "selfplay_opsd.readiness.execution_identity",
        lambda *a, **kw: pytest.fail("Diagnostics must not construct data/worker identity"),
    )
    if partition and operation == "worker-diagnose":
        with pytest.raises(hpc.HpcError, match="restricted to trusted diagnostics"):
            hpc.run_payload(
                Remote(),
                "yale-hpc-cpsc4880",
                operation=operation,
                method="D",
                profile="single_gpu_80gb",
                gpu_type="h100",
                account=None,
                image="",
                image_sha256="",
                settings=Path("configs/yale.yaml"),
                partition=partition,
            )
        return
    payload = hpc.run_payload(
        Remote(),
        "yale-hpc-cpsc4880",
        operation=operation,
        method="D",
        profile="single_gpu_80gb",
        gpu_type="h100",
        account=None,
        image="",
        image_sha256="",
        settings=Path("configs/yale.yaml"),
        partition=partition,
    )
    assert payload["time"] == "00:05:00" and payload["signal_seconds"] == 30
    assert payload["cpus"] == 2 and payload["memory_gib"] == 8
    assert not payload["worker_image"] and payload["runtime_identity"] is None
    assert payload["partition"] == (
        partition or ("day" if operation == "worker-diagnose" else "gpu_h100")
    )


@pytest.mark.parametrize(
    "row,charged",
    [
        ("123|CANCELLED by 35094|0||None", 0),
        ("123|CANCELLED by 35094|0||Unknown", 0),
        ("123|CANCELLED by 35094|0|", 1),
        ("123|FAILED|0||2026-10-22T18:00:00", 1),
    ],
)
def test_never_started_cancellation_releases_only_proven_unused_reservation(
    tmp_path, monkeypatch, row, charged
):
    spec = importlib.util.spec_from_file_location("accounting_remote", "scripts/hpc_remote.py")
    remote = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(remote)
    state = tmp_path / "state"
    state.mkdir()
    (state / "budget-study.json").write_text(
        json.dumps(
            {
                "limit": 24,
                "reservations": {"intent": {"job_id": "123", "reserved": 1, "charged": None}},
            }
        )
    )
    monkeypatch.setattr(remote, "workspace", lambda _: tmp_path)
    monkeypatch.setattr(remote, "capture", lambda _: {"exit_code": 0, "stdout": row})
    assert remote.refreshed_budget({"study_id": "study"})["committed_gpu_hours"] == charged
