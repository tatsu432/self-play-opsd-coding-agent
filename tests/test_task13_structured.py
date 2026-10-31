import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "task13_structured", Path(__file__).parents[1] / "scripts/task13_structured.py"
)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
mutation, sites = module.mutation, module.sites


def test_probe_remote_requires_worker_and_bounded_h200(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("task13_remote", Path("scripts/hpc_remote.py"))
    assert spec and spec.loader
    remote = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(remote)
    worker = tmp_path / "worker.sif"
    worker.touch()
    monkeypatch.setattr(
        remote, "study_state", lambda _: {"gates": {"worker": True, "runtime": True}}
    )
    payload = {
        "operation": "task13-probe",
        "account": "cpsc4880",
        "gpu_count": 1,
        "gpu_type": "h200",
        "time": "02:00:00",
        "method": "A",
        "iterations": 10,
        "worker_image": str(worker),
        "worker_sha256": "a" * 64,
    }
    remote.precheck(payload)
    with pytest.raises(ValueError, match="bounded one-H200"):
        remote.precheck(payload | {"time": "23:00:00"})
    with pytest.raises(ValueError, match="worker"):
        remote.precheck(payload | {"worker_image": ""})
    static = payload | {"operation": "task13-static", "method": "B", "time": "06:00:00"}
    remote.precheck(static)
    with pytest.raises(ValueError, match="static comparison"):
        remote.precheck(static | {"time": "23:00:00"})
    counterfactual = payload | {
        "operation": "task13-counterfactual",
        "method": "C",
        "checkpoint": "/runs/study/C-seed2026/checkpoints/000099",
    }
    remote.precheck(counterfactual)
    with pytest.raises(ValueError, match="C checkpoint 99"):
        remote.precheck(
            counterfactual | {"checkpoint": "/runs/study/C-seed2026/checkpoints/000100"}
        )
    from selfplay_opsd.hpc import TASK13_MODELS

    name, revision, representation = TASK13_MODELS["A"]
    model = payload | {
        "operation": "task13-model-probe",
        "probe_model": {
            "name": name,
            "revision": revision,
            "representation": representation,
            "non_thinking": False,
            "backend": "vllm-inference-only",
        },
    }
    remote.precheck(model)
    with pytest.raises(ValueError, match="pinned manifest"):
        remote.precheck(model | {"probe_model": model["probe_model"] | {"revision": "unverified"}})
    replay = payload | {
        "operation": "task13-header-replay",
        "method": "C",
        "gpu_count": 0,
        "time": "01:00:00",
    }
    remote.precheck(replay)
    with pytest.raises(ValueError, match="CPU-only"):
        remote.precheck(replay | {"gpu_count": 1})
    source = tmp_path / "repair-diagnosis-2026-static-v1/9a44124b61147aee02c685d6"
    source.mkdir(parents=True)
    (source / "stopped.json").write_text("{}")
    monkeypatch.setattr(remote, "paths", lambda _: {"runs": tmp_path})
    monkeypatch.setattr(remote, "job_state", lambda _: {"state": "RUNNING"})
    continuation = payload | {
        "operation": "task13-static-continuation",
        "method": "B",
        "time": "06:00:00",
    }
    with pytest.raises(ValueError, match="original clean stopped"):
        remote.precheck(continuation)
    monkeypatch.setattr(remote, "job_state", lambda _: {"state": "COMPLETED", "exit_code": "0:0"})
    remote.precheck(continuation)
    (source / "complete.json").write_text("{}")
    with pytest.raises(ValueError, match="original clean stopped"):
        remote.precheck(continuation)


def test_structured_mutation_builds_a_valid_patch():
    source = "def f(x):\n    return x + 1\n"
    target = next(x for x in sites(source) if "Sub" in x["operations"])
    result = mutation(source, target["path"], "Sub")
    assert result.source == "def f(x):\n    return x - 1\n"
    assert result.patch.startswith("--- a/src/solution.py\n+++ b/src/solution.py\n")


def test_unlisted_structured_operations_are_rejected():
    source = "def f(x):\n    return x + 1\n"
    with pytest.raises(ValueError, match="Unknown structured"):
        mutation(source, ["body", 0, "name"], "increment")


def test_allocated_probe_command_preserves_checkpoint_and_extra_slots():
    spec = importlib.util.spec_from_file_location("task13_job", Path("scripts/hpc_job.py"))
    assert spec and spec.loader
    job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job)
    env = {
        "SELFPLAY_OPERATION": "task13-probe",
        "SELFPLAY_METHOD": "D",
        "SELFPLAY_SETTINGS": "configs/task12-corrected.yaml",
        "SELFPLAY_OUTPUT": "/runs/probe",
        "SELFPLAY_ITERATIONS": "8",
        "SELFPLAY_CHECKPOINT": "/runs/B/checkpoints/000200",
    }
    args = job.task13_command_args(env)
    assert (
        args.count("--checkpoint") == 1
        and args[args.index("--checkpoint") + 1] == env["SELFPLAY_CHECKPOINT"]
    )
    assert args[args.index("--parents") + 1] == "16"
    assert args[args.index("--slot-start") + 1] == "4"
    assert args[args.index("--representation") + 1] == "diff"
    extra_base = job.task13_command_args(env | {"SELFPLAY_METHOD": "B", "SELFPLAY_CHECKPOINT": ""})
    assert (
        "--checkpoint" not in extra_base and extra_base[extra_base.index("--slot-start") + 1] == "4"
    )
    original_checkpoint = job.task13_command_args(env | {"SELFPLAY_ITERATIONS": "10"})
    assert (
        original_checkpoint.count("--checkpoint") == 1 and "--slot-start" not in original_checkpoint
    )
    counter = job.task13_command_args(env | {"SELFPLAY_OPERATION": "task13-counterfactual"})
    assert counter.count("--checkpoint") == 1 and "--slot-start" not in counter
    model = job.task13_command_args(
        env | {"SELFPLAY_OPERATION": "task13-model-probe", "SELFPLAY_METHOD": "C"}
    )
    assert model[:2] == ["python", "/workspace/scripts/task13_probe.py"]
    assert model[model.index("--model-choice") + 1] == "qwen3-8b" and "--checkpoint" not in model
    replay = job.task13_command_args(env | {"SELFPLAY_OPERATION": "task13-header-replay"})
    assert replay[:2] == ["python", "/workspace/scripts/task13_header_replay.py"]
    assert "--checkpoint" not in replay
    continuation = job.task13_command_args(
        env | {"SELFPLAY_OPERATION": "task13-static-continuation"}
    )
    assert "/workspace/scripts/task13_static_continue.py" in continuation
