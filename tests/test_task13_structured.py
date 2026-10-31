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
