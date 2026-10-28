"""Controller checks that prevent unauthorized or duplicate scheduler work."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
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


def test_method_smoke_launch_honors_independent_selection_and_keeps_scientific_args():
    spec = importlib.util.spec_from_file_location("job", Path("scripts/hpc_job.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prefix = ["apptainer", "exec", "pinned.sif"]
    args = [
        "torchrun",
        "-m",
        "selfplay_opsd.cli",
        "method-smoke",
        "--config",
        "D",
        "--settings",
        "/workspace/configs/yale.yaml",
        "--hardware",
        "single_gpu_80gb",
    ]
    original = list(args)
    all_commands = module.method_smoke_commands(prefix, args, "D")
    position = len(prefix) + args.index("--config")
    assert [c[position] for c in all_commands] == list("ABC")
    assert all("--config" not in command for command in all_commands)
    for method in "ABC":
        selected = module.method_smoke_commands(prefix, args, method)
        assert len(selected) == 1
        expected = list(args)
        index = expected.index("--config")
        expected[index : index + 2] = [method]
        assert selected[0] == prefix + expected
    assert args == original


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


def test_cpu_requests_use_individual_nodes_in_heterogeneous_partitions(monkeypatch):
    remote = remote_module()

    def fake_run(args, **_):
        if args[0] == "id":
            stdout = "course_user\n"
        elif args[0] == "sacctmgr":
            stdout = "course||normal|normal\n"
        elif args[0] == "scontrol":
            stdout = "PartitionName=day MaxTime=1-00:00:00\n"
        elif args[-1] == "%G":
            stdout = "(null)\n"
        else:
            assert "-N" in args
            stdout = "48|376832|(null)\n128|512000|(null)\n"
        return subprocess.CompletedProcess(args, 0, stdout, "")

    monkeypatch.setattr(remote, "run", fake_run)
    assert (
        remote.allowed(
            {
                "account": "course",
                "partition": "day",
                "gpu_type": "h100",
                "gpu_count": 0,
                "cpus": 8,
                "memory_gib": 64,
                "time": "04:00:00",
            }
        )["max_gpus_per_node"]
        == 0
    )


def test_finalized_checkout_ignores_only_managed_output_links(tmp_path, monkeypatch):
    remote = remote_module()
    origin = tmp_path / "origin"
    origin.mkdir()
    subprocess.run(["git", "init", "-q", str(origin)], check=True)
    subprocess.run(["git", "-C", str(origin), "config", "user.name", "Fixture"], check=True)
    subprocess.run(
        ["git", "-C", str(origin), "config", "user.email", "fixture@example.com"], check=True
    )
    (origin / "README").write_text("immutable\n")
    (origin / ".gitignore").write_text("runs/\ndata/\n")
    subprocess.run(["git", "-C", str(origin), "add", "."], check=True)
    subprocess.run(["git", "-C", str(origin), "commit", "-qm", "fixture"], check=True)
    commit = subprocess.run(
        ["git", "-C", str(origin), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    bundle = tmp_path / "fixture.bundle"
    subprocess.run(
        ["git", "-C", str(origin), "bundle", "create", str(bundle), "HEAD"],
        check=True,
        capture_output=True,
    )
    dataset = tmp_path / "prepared"
    dataset.mkdir()
    (dataset / "manifest.json").write_text('{"ready": true}\n')
    runs = tmp_path / "runs"
    runs.mkdir()
    paths = {
        "bundle": bundle,
        "checkout": tmp_path / commit,
        "dataset": dataset,
        "runs": runs,
        "state": tmp_path / "state",
    }
    monkeypatch.setattr(remote, "paths", lambda _: paths)
    assert remote.finalize(
        {"commit": commit, "manifest_sha256": remote.sha256(dataset / "manifest.json")}
    )["ready"]
    status = ["git", "-C", str(paths["checkout"]), "status", "--porcelain", "--untracked-files=all"]
    assert not subprocess.run(status, check=True, capture_output=True, text=True).stdout
    (paths["checkout"] / "README").write_text("changed\n")
    assert "README" in subprocess.run(status, check=True, capture_output=True, text=True).stdout


@pytest.mark.parametrize("operation", ["train", "backend-check"])
def test_unknown_sbatch_response_is_not_submitted_twice(tmp_path, monkeypatch, operation):
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
    monkeypatch.setattr(remote, "precheck", lambda _: None)
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
        "operation": operation,
        "method": "B",
        "profile": "dual_gpu_80gb",
        "study_id": "fixture",
        "seed": 2026,
        "account": "course",
        "partition": "gpu_h100",
        "cpus": 8,
        "memory_gib": 128,
        "time": "23:00:00",
        "exclude_nodes": ["a1127u05n01"] if operation == "backend-check" else [],
    }
    first = remote.submit(payload)
    second = remote.submit(payload)
    assert first["reason"] == second["reason"] == "unreconciled_submission"
    assert len(calls) == 1
    assert ("--no-requeue" in calls[0]) == (operation == "backend-check")
    assert ("--exclude=a1127u05n01" in calls[0]) == (operation == "backend-check")


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
        "target_iterations": 200,
        "time": "01:00:00",
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
    (origin / "scripts").mkdir()
    shutil.copyfile("scripts/hpc_job.py", origin / "scripts/hpc_job.py")
    (origin / "README").write_text("fixture\n")
    subprocess.run(["git", "-C", str(origin), "add", "README", "scripts/hpc_job.py"], check=True)
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
    # The shell-wrapper contract is independent of kernel GPU ownership, which
    # is exercised by test_allocation_diagnostics on synthetic device policies.
    python_stub = binary / "python3"
    python_stub.write_text(
        f"#!{sys.executable}\nimport runpy,sys\n"
        "ns=runpy.run_path(sys.argv[1])\n"
        "ns['execute'].__globals__['allocated_gpus']=lambda env,report: "
        "[dict(index='1', uuid='GPU-11111111-1111-1111-1111-111111111111')]\n"
        "ns['main']()\n"
    )
    python_stub.chmod(0o755)
    stub = binary / "apptainer"
    stub.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$@" > "$CAPTURE_ARGS"\n'
        'printf \'[{"uuid":"11111111-1111-1111-1111-111111111111",'
        '"free_bytes":84825604096,"total_bytes":85899345920}]\\n\'\n'
    )
    stub.chmod(0o755)
    gpu_stub = binary / "nvidia-smi"
    gpu_stub.write_text(
        '#!/bin/sh\nif [ "$1" = "-q" ]; then\n'
        "printf '<nvidia_smi_log><gpu><uuid>GPU-11111111-1111-1111-1111-111111111111</uuid>"
        "<minor_number>1</minor_number></gpu></nvidia_smi_log>\\n'\n"
        'else\nprintf "1, GPU-11111111-1111-1111-1111-111111111111, 0000:02:00.0\\n"\nfi\n'
    )
    gpu_stub.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(binary) + os.pathsep + os.environ["PATH"],
        "CAPTURE_ARGS": str(capture),
        "SLURM_JOB_ID": "123",
        "SLURM_JOB_GPUS": "1",
        "CUDA_VISIBLE_DEVICES": "0",
        "SELFPLAY_PROJECT": str(project),
        "SELFPLAY_TRAINER_SIF": str(image),
        "SELFPLAY_TRAINER_SHA256": digest,
        "SELFPLAY_DATA": str(data),
        "SELFPLAY_RUNS": str(runs),
        "SELFPLAY_CACHE": str(tmp_path / "cache"),
        "SELFPLAY_OPERATION": "backend-check",
        "SELFPLAY_WORKER_SIF": "/missing/incidental-worker.sif",
        "SELFPLAY_WORKER_TOKEN": "stale-credential",
        "SELFPLAY_WORKER_URL": "http://invalid-worker:1234",
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
    stub.write_text(stub.read_text() + "sleep 0.5\n")
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
    assert (runs / "fixture/D-seed2026/state/stop_requested.123").is_file()


@pytest.mark.parametrize("operation,expected_iterations", [("train", 200), ("pilot", 7)])
def test_main_preview_uses_frozen_main_budget_and_resource_plan(
    monkeypatch, operation, expected_iterations
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
                "images": {
                    "trainer": {"path": "/trainer.sif", "sha256": "a" * 64},
                    "worker": {"path": "/worker.sif", "sha256": "b" * 64},
                },
                "reports": {
                    "worker": {},
                    "freeze": {
                        "execution_commit": "f" * 40,
                        "resource_plan": {
                            "per_method": {
                                "D": {
                                    "signal_seconds": 5000,
                                    "max_continuations": 6,
                                    "planning_projection_200_hours": 85.0,
                                }
                            }
                        },
                    },
                },
            }

    monkeypatch.setattr(
        hpc, "snapshot", lambda *a: {"commit": "a" * 40, "manifest_sha256": "c" * 64}
    )
    monkeypatch.setattr(
        "selfplay_opsd.readiness.execution_identity", lambda *a: {"manifest_hash": "d" * 64}
    )
    payload = hpc.run_payload(
        Remote(),
        "yale-hpc-cpsc4880",
        operation=operation,
        method="D",
        profile="single_gpu_80gb",
        gpu_type="h200",
        account="cpsc4880",
        image="",
        image_sha256="",
        settings=Path("configs/task08.yaml"),
        iterations=7,
    )
    assert payload["iterations"] == payload["target_iterations"] == expected_iterations
    if operation == "train":
        assert payload["commit"] == "f" * 40
        assert payload["signal_seconds"] == 5000
        assert payload["max_continuations"] == 6
        assert payload["estimated_main_hours"] == 85.0
        assert payload["checkpoint"] == "" and payload["resume"] is None

        class MissingPlan(Remote):
            def call(self, name, *args):
                value = super().call(name, *args)
                if name == "study":
                    value["reports"]["freeze"].pop("resource_plan")
                return value

        with pytest.raises(hpc.HpcError, match="lacks a measured resource plan"):
            hpc.run_payload(
                MissingPlan(),
                "yale-hpc-cpsc4880",
                operation="train",
                method="D",
                profile="single_gpu_80gb",
                gpu_type="h200",
                account="cpsc4880",
                image="",
                image_sha256="",
                settings=Path("configs/task08.yaml"),
            )
    else:
        assert payload["commit"] == "a" * 40
        assert payload["signal_seconds"] == 300
