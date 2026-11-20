"""Regression coverage for the Task 20 solver control and allocation boundary."""

from __future__ import annotations

import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from selfplay_opsd.core import digest, read_json
from selfplay_opsd.task19_rollout import Repair, Turn
from selfplay_opsd.task20 import Task20Control, Task20Study


class ToyPolicy:
    def __init__(self):
        self.parameter = torch.nn.Parameter(torch.tensor(0.0))
        self.parameters = [self.parameter]
        self.optimizer = torch.optim.AdamW(self.parameters, lr=0.01, weight_decay=0)

    def score(self, prompt, ids, sampling, **kwargs):
        return torch.stack([self.parameter * token for token in ids]), None

    def version(self):
        return digest(float(self.parameter.detach()))

    def state(self):
        return {
            "parameter": self.parameter.detach().clone(),
            "optimizer": self.optimizer.state_dict(),
        }

    def restore(self, state):
        with torch.no_grad():
            self.parameter.copy_(state["parameter"])
        self.optimizer.load_state_dict(state["optimizer"])


def rollout(success, token):
    return Repair(
        task_id="task",
        parent_id="parent",
        seed=1,
        turns=[
            Turn(
                prompt=[17],
                ids=[token],
                logps=[0.0],
                text="sample",
                token_kinds=["assistant"],
                stopped=True,
                history=[],
                result="accepted",
            )
        ],
        success=success,
        public_pass=success,
        hidden_pass=success,
        termination="finish",
        failure="verified" if success else "no_edit",
        verifier_status="ok" if success else "student_failure",
        final_source_hash="mock",
        test_calls=0,
        timings={"generation": 0, "tool": 0, "verifier": 0},
    )


@pytest.mark.parametrize("group_size", [4, 8])
def test_solver_control_only_updates_on_mixed_rollouts(tmp_path, monkeypatch, group_size):
    task = SimpleNamespace(task_id="task")
    repo = SimpleNamespace(
        hash="dataset",
        parents=lambda _: [SimpleNamespace(parent_id="parent")],
        tasks=lambda split: [(task, object())] if split == "solver_train" else [],
    )
    sandbox = SimpleNamespace(identity=lambda: "worker", infrastructure_retries=0)
    cfg = Task20Control(group_size=group_size)
    policy = ToyPolicy()
    study = Task20Study(cfg, "control", policy, sandbox, repo, tmp_path, {}, mode="fixed")
    study.commit({}, None)
    start = study.snapshot

    def forbidden(*args, **kwargs):
        pytest.fail("Solver-only path touched the proposer or teacher")

    monkeypatch.setattr("selfplay_opsd.task19_train.propose", forbidden)
    monkeypatch.setattr("selfplay_opsd.task19_train.privilege", forbidden)
    calls = []

    def repair(*args):
        calls.append(args[-1])
        return rollout(False, 1)

    monkeypatch.setattr("selfplay_opsd.task19_train.repair", repair)
    uniform = study.update(1, [])
    assert uniform["shared"]["optimizer_step"] is False
    assert study.snapshot == start
    assert uniform["learning"]["effective_rl_updates"] == 0

    def mixed(*args):
        calls.append(args[-1])
        success = len(calls) % 2 == 0
        return rollout(success, 1 if success else 0)

    monkeypatch.setattr("selfplay_opsd.task19_train.repair", mixed)
    informative = study.update(1, [])
    assert len(calls) == len(set(calls)) == 2 * group_size
    assert informative["learning"]["mixed_group_tokens"] == group_size
    assert informative["learning"]["effective_rl_updates"] == 1
    assert informative["shared"]["optimizer_step"] is True
    assert informative["shared"]["update_norm"] > 0
    assert informative["shared"]["proposer_gradient_norm"] == 0
    assert informative["opsd"]["gradient_norm"] == 0
    assert study.snapshot != start

    resumed = Task20Study(cfg, "control", ToyPolicy(), sandbox, repo, tmp_path, {}, mode="fixed")
    resumed.restore(study.latest)
    assert resumed.snapshot == study.snapshot
    assert resumed.effective_updates == 1
    assert resumed.episodes == 2 * group_size
    assert resumed.policy.optimizer.state_dict()["state"]


def test_evalplus_compound_edit_preserves_unrelated_lines():
    from selfplay_opsd.mutations import apply_patch

    task20_data = runpy.run_path("scripts/task20_data.py")
    source = "def solve(x):\n    zero = 0\n    one = 1\n    return x + zero + one\n"
    mutation = task20_data["compound"](source, 2026)
    assert mutation.changed_lines == 4
    assert mutation.changed_nodes == 2
    assert apply_patch(mutation.source, mutation.inverse_patch) == source


def test_allocation_dispatch_and_budget_namespace(tmp_path, monkeypatch):
    job = runpy.run_path("scripts/hpc_job.py")
    args = job["task13_command_args"](
        {
            "SELFPLAY_OPERATION": "task20-evaluate",
            "SELFPLAY_METHOD": "control",
            "SELFPLAY_SETTINGS": "configs/task20-yale.yaml",
            "SELFPLAY_OUTPUT": "/runs/study/control-seed2026",
            "SELFPLAY_TASK20_CHECKPOINT_ITERATION": "0",
        }
    )
    assert args[-2:] == ["--checkpoint-iteration", "0"]
    assert job["model_cache_overrides"]("task20-solver")["WANDB_MODE"] == "offline"
    remote = runpy.run_path("scripts/hpc_remote.py")
    monkeypatch.setitem(remote["budget_path"].__globals__, "workspace", lambda _: tmp_path)
    payload = {"study_id": "repair-selfplay-2026-task20-v1", "readiness_budget": 5}
    ledger = remote["budget_path"](payload)
    assert ledger == tmp_path / "state/task20-budget.json"
    payload.update(gpu_count=1, time="01:00:00", run_id="fake", operation="task20-solver")
    remote["reserve_budget"](payload, "trial")
    assert read_json(ledger)["reservations"]["trial"]["reserved"] == 1
    with pytest.raises(ValueError, match="Cannot change"):
        remote["reserve_budget"](payload | {"readiness_budget": 6}, "trial-2")
    assert (
        remote["task20_marker"]({"operation": "task20-evaluate", "task20_checkpoint_iteration": 0})
        == "complete-evaluate-000000.json"
    )


def test_screen_payload_uses_prepared_data_and_explicit_budget(monkeypatch):
    from selfplay_opsd import hpc

    class Remote:
        def call(self, command, payload=None):
            if command in {"inspect", "context"}:
                return {
                    "user": "tester",
                    "storage": {
                        "project": {"project_cpsc4880": "/project"},
                        "scratch": {"scratch_cpsc4880": "/scratch"},
                    },
                    "slurm": {
                        "associations": {"exit_code": 0, "stdout": "cpsc4880||normal|normal\n"}
                    },
                }
            assert command == "study"
            return {
                "images": {
                    "task19-trainer": {"path": "/pinned-trainer.sif", "sha256": "a" * 64},
                    "worker": {"path": "/pinned-worker.sif", "sha256": "b" * 64},
                },
                "reports": {"worker": {}, "freeze": {}},
            }

    roots = []

    def snapshot(root):
        roots.append(root)
        return {"commit": "c" * 40, "manifest_sha256": "d" * 64}

    monkeypatch.setattr(hpc, "snapshot", snapshot)
    monkeypatch.setattr(
        "selfplay_opsd.readiness.execution_identity", lambda *_: {"manifest_hash": "e" * 64}
    )
    inputs = {
        "operation": "task20-screen",
        "method": "control",
        "profile": "single_gpu_80gb",
        "gpu_type": "h200",
        "account": "cpsc4880",
        "image": "",
        "image_sha256": "",
        "settings": Path("configs/task20-yale.yaml"),
    }
    with pytest.raises(hpc.HpcError, match="bounded one-H200"):
        hpc.run_payload(Remote(), "yale-hpc-cpsc4880", **inputs)
    plan = hpc.run_payload(
        Remote(), "yale-hpc-cpsc4880", **inputs, task20_budget_hours=5, task20_wall_minutes=90
    )
    assert roots[-1] == hpc.ROOT / "data/task20/evalplus-v1"
    assert plan["readiness_budget"] == 5
    assert plan["time"] == "01:30:00"
    assert plan["output"].endswith("/screen-seed2026")
    assert plan["image"] == "/pinned-trainer.sif"
    assert plan["scientific_hash"] == Task20Control.read(Path("configs/task20-yale.json")).science


def test_reviewed_solver_gate_requires_training_improvement_and_rescoring_support(tmp_path):
    import json

    remote = runpy.run_path("scripts/hpc_remote.py")
    (tmp_path / "iterations").mkdir()
    (tmp_path / "one-step.json").write_text(
        json.dumps(
            {
                "step": {
                    "optimizer_step": True,
                    "update_norm": 0.1,
                    "weighted_logprob_change": 2,
                    "support_exits": 29,
                }
            }
        )
    )
    for iteration in (1, 2):
        (tmp_path / "iterations" / f"{iteration:06}.json").write_text(
            json.dumps(
                {
                    "metrics": {
                        "shared": {"optimizer_step": True},
                        "rl": {"norm": 0.1, "support_exits": 0},
                    }
                }
            )
        )
    marker = {
        "effective_rl_updates": 2,
        "result": {
            "gate": {"learnable": False},
            "solver_train": {"a": {"before": 8, "after": 11}, "b": {"before": 9, "after": 13}},
        },
    }
    assert remote["task20_solver_evidence"](marker, tmp_path)
    marker["result"]["solver_train"]["b"]["after"] = 9
    assert not remote["task20_solver_evidence"](marker, tmp_path)
