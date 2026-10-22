"""Controller checks that prevent unauthorized or duplicate scheduler work."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

from selfplay_opsd.config import load_config
from selfplay_opsd.hpc import HpcError, choose_account, choose_storage


def remote_module():
    spec = importlib.util.spec_from_file_location("hpc_remote", Path("scripts/hpc_remote.py"))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inspection_never_invents_an_account_or_project():
    report = {
        "storage": {"project": {"project_cpsc4880": "/project/cpsc4880"}},
        "slurm": {"associations": {"exit_code": 0, "stdout": "course|gpu_h100|normal|normal\n"}},
    }
    assert choose_storage(report, "yale-hpc-cpsc4880") == "/project/cpsc4880"
    assert choose_account(report, "gpu_h100") == "course"
    with pytest.raises(HpcError):
        choose_storage(report, "yale-hpc-cpsc4710")
    with pytest.raises(HpcError):
        choose_account(report, "gpu_h200")


def test_remote_storage_uses_only_writable_personal_course_directories(tmp_path, monkeypatch):
    remote = remote_module()
    user = "cpsc4880_ts2236"
    project = tmp_path / "project"
    scratch = tmp_path / "scratch"
    (project / user).mkdir(parents=True)
    (scratch / user).mkdir(parents=True)
    (tmp_path / "project_cpsc4880").symlink_to(project, target_is_directory=True)
    (tmp_path / "scratch_cpsc4880").symlink_to(scratch, target_is_directory=True)
    (tmp_path / "project_other").mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(
        remote,
        "run",
        lambda args, **_: subprocess.CompletedProcess(args, 0, user + "\n", ""),
    )
    assert remote.roots() == {
        "project": {"project_cpsc4880": str(project / user)},
        "scratch": {"scratch_cpsc4880": str(scratch / user)},
    }


def test_live_resource_validation_fails_closed(monkeypatch):
    remote = remote_module()

    def fake_run(args, **_):
        if args[:2] == ["id", "-un"]:
            stdout = "cpsc4880_ts2236\n"
        elif args[0] == "sacctmgr":
            stdout = "course|gpu_h100|normal|normal\n"
        elif args[0] == "scontrol":
            stdout = "PartitionName=gpu_h100 MaxTime=2-00:00:00\n"
        elif args[0] == "sinfo" and "%G" in args:
            stdout = "gpu:h100:4\n"
        else:
            stdout = "48|976000|gpu:h100:4\n"
        return subprocess.CompletedProcess(args, 0, stdout, "")

    monkeypatch.setattr(remote, "run", fake_run)
    good = {
        "account": "course",
        "partition": "gpu_h100",
        "gpu_type": "h100",
        "gpu_count": 2,
        "cpus": 8,
        "memory_gib": 128,
        "time": "06:00:00",
    }
    assert remote.allowed(good)["max_gpus_per_node"] == 4
    with pytest.raises(ValueError, match="association"):
        remote.allowed({**good, "account": "not_my_account"})
    with pytest.raises(ValueError, match="single partition node"):
        remote.allowed({**good, "gpu_count": 8})


def test_unknown_sbatch_response_is_not_submitted_twice(tmp_path, monkeypatch):
    remote = remote_module()
    checkout = tmp_path / "checkout"
    state = tmp_path / "state"
    checkout.mkdir()
    (checkout / "configs").mkdir()
    (checkout / "configs/yale.yaml").write_text("study_id: fixture\n")
    (checkout / "scripts").mkdir()
    (checkout / "scripts/yale.slurm").write_text("#!/bin/bash\n")
    state.mkdir()
    image = tmp_path / "trainer.sif"
    image.write_bytes(b"fixture")
    identity = {"commit": "a" * 40, "manifest_sha256": "b" * 64}
    (state / f"sync-{identity['commit']}.json").write_text(json.dumps(identity))
    paths = {
        "checkout": checkout,
        "state": state,
        "runs": tmp_path / "runs",
        "dataset": tmp_path / "prepared",
        "cache": tmp_path / "cache",
    }
    monkeypatch.setattr(remote, "paths", lambda _: paths)
    monkeypatch.setattr(remote, "workspace", lambda _: tmp_path)
    monkeypatch.setattr(remote, "allowed", lambda _: {"authorized": True})
    monkeypatch.setattr(remote, "reconcile_intent", lambda ledger, _: ledger)
    calls = []

    def fake_run(args, **_):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, "unexpected scheduler response", "")

    monkeypatch.setattr(remote, "run", fake_run)
    payload = {
        **identity,
        "run_id": "run1",
        "identity": "logical-identity",
        "image": str(image),
        "image_sha256": "c" * 64,
        "settings": "configs/yale.yaml",
        "gpu_count": 2,
        "profile_gpu_count": 2,
        "gpu_type": "h100",
        "min_vram_gib": 79,
        "operation": "train",
        "method": "B",
        "profile": "dual_gpu_80gb",
        "study_id": "fixture",
        "seed": 2026,
        "account": "course",
        "partition": "gpu_h100",
        "cpus": 8,
        "memory_gib": 128,
        "time": "23:00:00",
    }
    first = remote.submit(payload)
    second = remote.submit(payload)
    assert first["reason"] == second["reason"] == "unreconciled_submission"
    assert len(calls) == 1


def test_recovery_limit_stops_before_checkpoint_load(tmp_path, monkeypatch):
    remote = remote_module()
    monkeypatch.setattr(remote, "workspace", lambda _: tmp_path)
    monkeypatch.setattr(remote, "job_state", lambda _: {"state": "NODE_FAIL"})
    path = tmp_path / "state/run1.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"identity": "same", "attempts": [{"job_id": "123"}], "retries": 2}))
    assert (
        remote.recover({"storage_root": str(tmp_path), "run_id": "run1", "identity": "same"})[
            "reason"
        ]
        == "retry_limit"
    )


def test_recovery_uses_verified_joint_checkpoint_once(tmp_path, monkeypatch):
    remote = remote_module()
    runs = tmp_path / "runs"
    checkpoint = runs / "study/B-seed2026/checkpoints/000010"
    checkpoint.mkdir(parents=True)
    state = checkpoint / "state.pt"
    state.write_bytes(b"joint state fixture")
    (checkpoint / "metadata.json").write_text(
        json.dumps({"iteration": 10, "scientific_hash": "science", "manifests_hash": "data"})
    )
    (checkpoint / "complete.json").write_text(
        json.dumps({"files": {"state.pt": hashlib.sha256(state.read_bytes()).hexdigest()}})
    )
    (checkpoint.parent / "latest.json").write_text(json.dumps({"path": "000010", "iteration": 10}))
    monkeypatch.setattr(remote, "paths", lambda _: {"runs": runs})
    monkeypatch.setattr(remote, "workspace", lambda _: tmp_path)
    monkeypatch.setattr(remote, "job_state", lambda _: {"state": "NODE_FAIL"})
    submitted = []
    monkeypatch.setattr(
        remote,
        "submit",
        lambda payload: submitted.append(payload) or {"submitted": True, "job_id": "456"},
    )
    ledger = tmp_path / "state/run1.json"
    ledger.parent.mkdir()
    ledger.write_text(
        json.dumps(
            {
                "identity": "same",
                "attempts": [{"job_id": "123"}],
                "retries": 0,
                "run_dir": str(checkpoint.parents[1]),
            }
        )
    )
    payload = {
        "storage_root": str(tmp_path),
        "run_id": "run1",
        "identity": "same",
        "study_id": "study",
        "method": "B",
        "seed": 2026,
        "operation": "train",
        "scientific_hash": "science",
        "manifests_hash": "data",
    }
    result = remote.recover(payload)
    assert result["submitted"] is True
    assert submitted[0]["resume"] == str(checkpoint)
    assert remote.recover(payload)["reason"] == "concurrent_recovery"
    state.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="verification failed"):
        remote.verified_checkpoint(payload)


def test_operational_mounts_do_not_change_scientific_hash(monkeypatch):
    baseline = load_config(Path("configs/yale.yaml"))
    monkeypatch.setenv("SELFPLAY_DATA_ROOT", "/data/prepared")
    monkeypatch.setenv("SELFPLAY_OUTPUT_ROOT", "/runs")
    relocated = load_config(Path("configs/yale.yaml"))
    assert relocated.data_root == Path("/data/prepared")
    assert relocated.output_root == Path("/runs")
    assert relocated.scientific_hash == baseline.scientific_hash


def test_slurm_wrapper_runs_the_pinned_image_without_host_venv(tmp_path):
    origin = tmp_path / "origin"
    origin.mkdir()
    subprocess.run(["git", "init", "-q", str(origin)], check=True)
    subprocess.run(["git", "-C", str(origin), "config", "user.name", "Fixture"], check=True)
    subprocess.run(
        ["git", "-C", str(origin), "config", "user.email", "fixture@example.com"], check=True
    )
    (origin / "README").write_text("fixture\n")
    subprocess.run(["git", "-C", str(origin), "add", "README"], check=True)
    subprocess.run(["git", "-C", str(origin), "commit", "-qm", "fixture"], check=True)
    commit = subprocess.run(
        ["git", "-C", str(origin), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    project = tmp_path / commit
    subprocess.run(["git", "clone", "-q", str(origin), str(project)], check=True)
    image = tmp_path / "trainer.sif"
    image.write_bytes(b"pinned fixture image")
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    data = tmp_path / "prepared"
    data.mkdir()
    runs = tmp_path / "runs"
    capture = tmp_path / "apptainer-args.txt"
    binary = tmp_path / "bin"
    binary.mkdir()
    stub = binary / "apptainer"
    stub.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$CAPTURE_ARGS"\n')
    stub.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(binary) + os.pathsep + os.environ["PATH"],
        "CAPTURE_ARGS": str(capture),
        "SLURM_JOB_ID": "123",
        "SELFPLAY_PROJECT": str(project),
        "SELFPLAY_TRAINER_SIF": str(image),
        "SELFPLAY_TRAINER_SHA256": digest,
        "SELFPLAY_DATA": str(data),
        "SELFPLAY_RUNS": str(runs),
        "SELFPLAY_CACHE": str(tmp_path / "cache"),
        "SELFPLAY_OPERATION": "backend-check",
        "SELFPLAY_METHOD": "D",
        "SELFPLAY_HARDWARE": "single_gpu_80gb",
        "SELFPLAY_GPU_COUNT": "1",
        "SELFPLAY_SETTINGS": "configs/yale.yaml",
        "SELFPLAY_STUDY": "fixture",
        "SELFPLAY_SEED": "2026",
        "SELFPLAY_OUTPUT": "/runs/compatibility/single_gpu_80gb",
    }
    result = subprocess.run(
        ["bash", str(Path("scripts/yale.slurm").resolve())], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    args = capture.read_text().splitlines()
    assert "--nv" in args
    assert f"{project}:/workspace:ro" in args
    assert f"{runs}:/runs" in args
    assert "backend-check" in args
    assert "/runs/compatibility/single_gpu_80gb" in args
    capture.unlink()
    stub.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$CAPTURE_ARGS"\nsleep 0.5\n')
    process = subprocess.Popen(
        ["bash", str(Path("scripts/yale.slurm").resolve())],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    for _ in range(100):
        if capture.exists():
            break
        time.sleep(0.01)
    assert capture.exists()
    process.send_signal(signal.SIGUSR1)
    _, stderr = process.communicate(timeout=5)
    assert process.returncode == 0, stderr
    assert (runs / "fixture/D-seed2026/state/stop_requested").is_file()
