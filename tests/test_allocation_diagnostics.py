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


def test_cancellation_terminates_running_worker_process():
    from selfplay_opsd.worker_runtime import bounded_process

    cancelled = threading.Event()
    timer = threading.Timer(0.2, cancelled.set)
    timer.start()
    try:
        code, _, _, reason = bounded_process(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            b"",
            60,
            4096,
            cancelled=cancelled,
        )
        assert code < 0 and reason == "cancelled"
    finally:
        timer.cancel()
