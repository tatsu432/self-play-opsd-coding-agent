"""Independent objective checks and Task 19 protocol/recovery regressions."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from selfplay_opsd.core import digest
from selfplay_opsd.mutations import replace_text
from selfplay_opsd.task19 import Task19Control, centered_rewards, clipped_sum, opsd_sum
from selfplay_opsd.task19_policy import (
    Sampling,
    SharedPolicy,
    sampling_distribution,
    target_logps,
    visibility,
)
from selfplay_opsd.task19_protocol import STUDENT_TOOLS, assistant_history, parse_call, render_call
from selfplay_opsd.task19_train import Study, exact_reference, joint_step


def test_drgrpo_matches_independent_episode_reference():
    rewards = [1.0, 1.0, 0.0, 0.0]
    assert centered_rewards([1.0] * 4).tolist() == [0.0] * 4
    assert centered_rewards(rewards).tolist() == [0.5, 0.5, -0.5, -0.5]
    episodes = [[1.4, 0.7], [1.05], [], [0.6, 1.4, 1.0]]
    vector = torch.tensor([ratio for row in episodes for ratio in row]).log().requires_grad_()
    separated = vector.split([len(row) for row in episodes])
    old = torch.zeros_like(vector, requires_grad=True)
    fixed_budget = 10
    terms = [
        clipped_sum(row, saved, float(advantage), 0.2)
        for row, saved, advantage in zip(
            separated,
            old.split([len(row) for row in episodes]),
            centered_rewards(rewards),
            strict=True,
        )
    ]
    loss = sum(terms) / (len(rewards) * fixed_budget)
    expected = 0.0
    for ratios, reward in zip(episodes, rewards, strict=True):
        advantage = reward - sum(rewards) / len(rewards)
        for ratio in ratios:
            reference = min(ratio * advantage, max(0.8, min(ratio, 1.2)) * advantage)
            expected -= reference / (len(rewards) * fixed_budget)
    assert float(loss.detach()) == pytest.approx(expected)
    loss.backward()
    assert old.grad is None
    assert vector.grad.tolist() == pytest.approx([0, -0.00875, -0.013125, 0, 0.0175, 0.0125])


def test_opsd_detaches_gap_and_gating_preserves_sign():
    gap = torch.tensor([-10.0, 2.0], requires_grad=True)
    logps = torch.tensor([-1.0, -0.5], requires_grad=True)
    opsd_sum(logps, gap, gated=True).backward()
    assert gap.grad is None
    assert logps.grad.tolist() == pytest.approx(
        [5 * torch.sigmoid(torch.tensor(-10.0)), -2 * torch.sigmoid(torch.tensor(2.0))]
    )


def test_sampling_captures_actual_support_and_ppo_zero_ratio():
    logits = torch.tensor([[4.0, 3.0, 2.0, -5.0]], requires_grad=True)
    sample = Sampling(1, 0.7, 2)
    ids, probabilities = sampling_distribution(logits, sample)
    assert ids[0, 0] == 0
    assert probabilities.exp().tolist() == [[1.0, 0.0]]
    selected = target_logps(logits, torch.tensor([1]), sample)
    loss = clipped_sum(selected, torch.tensor([-0.5]), 1.0, 0.2)
    loss.backward()
    assert loss.item() == 0
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_native_call_multiline_empty_and_strict_boundaries():
    action = {
        "path": "src/solution.py",
        "old_text": "    if a:\n        return 3\n",
        "new_text": "",
    }
    native = render_call("replace_text", action)
    assert parse_call("I will edit the small branch.\n" + native, STUDENT_TOOLS) == (
        "replace_text",
        action,
    )
    history = assistant_history(native, ("replace_text", action))
    assert history["reasoning_content"] == ""
    assert history["tool_calls"][0]["function"]["arguments"] == action
    for invalid in (
        native + "\n" + native,
        native + " suffix",
        native.replace("src/solution.py", "tests/test_public.py"),
        native.replace(
            "</function>", "<parameter=path>\nsrc/solution.py\n</parameter>\n</function>"
        ),
        native.replace("return 3", "<|im_end|>"),
    ):
        with pytest.raises(ValueError):
            parse_call(invalid, STUDENT_TOOLS)


def test_reasoning_is_private_even_when_it_contains_tool_markers():
    markers = {"<think>": 1, "</think>": 2, "<tool_call>": 3, "</tool_call>": 4}
    kinds, visible = visibility([7, 3, 8, 4, 2, 3, 9, 4, 10], markers, {10}, True)
    assert kinds == ["reasoning"] * 5 + ["tool"] * 3 + ["assistant"]
    assert visible == [3, 9, 4]


def test_converted_native_reference_span_restores_exact_source():
    before = "def f(n):\n    if n > 10:\n        return 2\n    return 0\n"
    after = before.replace("n > 10", "n >= 10")
    old, new = exact_reference(before, after)
    assert replace_text(before, old, new) == after
    before = "def f(n):\n    return 0\n"
    after = "def f(n):\n    n += 1\n    return 0\n"
    old, new = exact_reference(before, after)
    assert replace_text(before, old, new) == after


class ToyPolicy:
    def __init__(self):
        self.parameters = [torch.nn.Parameter(torch.tensor([1.0, 2.0]))]
        self.optimizer = torch.optim.AdamW(self.parameters, lr=0.01, weight_decay=0)

    def version(self):
        return digest(self.parameters[0].detach().tolist())

    def state(self):
        return SharedPolicy.state(self)

    def restore(self, state):
        SharedPolicy.restore(self, state)

    @property
    def named_parameters(self):
        return [("toy.shared.weight", self.parameters[0])]


def test_joint_update_and_verified_resume(tmp_path):
    cfg = Task19Control()
    repo = SimpleNamespace(hash="dataset", parents=lambda _: [SimpleNamespace(parent_id="p")])
    worker = SimpleNamespace(identity=lambda: "worker")
    policy = ToyPolicy()
    study = Study(cfg, "B", policy, worker, repo, tmp_path, {"code": "test"})
    study.commit({}, None)
    joint_step(policy, [torch.tensor([1.0, -2.0])], [torch.tensor([-3.0, 0.0])], 0.5, 1.0)
    expected = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    optim = torch.optim.AdamW([expected], lr=0.01, weight_decay=0)
    expected.grad = torch.tensor([-0.5, -2.0])
    torch.nn.utils.clip_grad_norm_([expected], 1.0)
    optim.step()
    assert torch.equal(policy.parameters[0], expected)
    assert policy.optimizer.state[policy.parameters[0]]["step"].item() == 1
    assert policy.parameters[0].grad is None
    study.iteration = 1
    study.snapshot = policy.version()
    study.commit({}, None)
    resumed_policy = ToyPolicy()
    resumed = Study(cfg, "B", resumed_policy, worker, repo, tmp_path, {"code": "test"})
    resumed.restore(study.latest)
    assert torch.equal(resumed_policy.parameters[0], expected)
    assert resumed_policy.optimizer.state[resumed_policy.parameters[0]]["step"].item() == 1
    assert not joint_step(resumed_policy, [torch.zeros(2)], [torch.zeros(2)], 0.5, 1.0)[
        "optimizer_step"
    ]
    assert torch.equal(resumed_policy.parameters[0], expected)


def test_frozen_endpoint_does_not_change_scientific_identity():
    cfg = Task19Control(
        sd_scale=0.1, proposer_scale=0.4, qualification_hash="q", calibration_hash="c"
    )
    main = Task19Control.model_validate(cfg.model_dump() | {"endpoint": 672})
    assert cfg.science == main.science
    assert main.milestones() == [0, 32, 168, 336, 504, 672]
    with pytest.raises(ValueError):
        Task19Control(endpoint=512)


def test_isolated_gradient_measurement_preserves_production_gradients(tmp_path):
    policy = ToyPolicy()
    parameter = policy.parameters[0]
    parameter.grad = torch.tensor([0.5, 0.25])
    repo = SimpleNamespace(hash="dataset", parents=lambda _: [SimpleNamespace(parent_id="p")])
    worker = SimpleNamespace(identity=lambda: "worker")
    study = Study(Task19Control(), "D", policy, worker, repo, tmp_path, {})
    rl, sd, reference = study.zeros(), study.zeros(), study.zeros()
    study.accumulate(rl, parameter[0] * 2)
    study.accumulate(sd, -parameter[1] * 3)
    study.accumulate(reference, parameter[0] + parameter[1])
    assert [p.tolist() for p in (rl[0], sd[0], reference[0])] == [
        [2.0, 0.0],
        [0.0, -3.0],
        [1.0, 1.0],
    ]
    assert parameter.grad.tolist() == [0.5, 0.25]
    assert not policy.optimizer.state
    assert parameter.tolist() == [1.0, 2.0]


def test_qualification_resumes_saved_episodes_without_resampling(tmp_path, monkeypatch):
    import selfplay_opsd.task19_qualification as qualification
    from selfplay_opsd.core import OracleSpec, TaskSpec
    from selfplay_opsd.task19_qualification import qualify
    from selfplay_opsd.task19_rollout import Repair
    from selfplay_opsd.task19_train import SegmentStop

    task = TaskSpec(
        task_id="t",
        parent_id="p",
        split="train",
        specification="s",
        initial_source="x",
        entry_point="f",
        public_tests="",
        source_hash="h",
        mutation_seed=1,
    )
    oracle = OracleSpec(
        task_id="t",
        correct_source="x",
        inverse_patch="",
        public_cases=[],
        hidden_cases=[],
        provenance={},
    )
    other_task = task.model_copy(update={"task_id": "a"})
    other_oracle = oracle.model_copy(update={"task_id": "a"})
    calls = []

    def fake_repair(policy, sandbox, cfg, task, oracle, seed):
        calls.append((task.task_id, seed))
        success = len(calls) == 1
        return Repair(
            task_id=task.task_id,
            parent_id="p",
            seed=seed,
            turns=[],
            success=success,
            public_pass=success,
            hidden_pass=success,
            termination="finish",
            failure="",
            verifier_status="ok",
            final_source_hash="h",
            test_calls=0,
            timings={},
        )

    monkeypatch.setattr(qualification, "repair", fake_repair)
    monkeypatch.setattr(qualification, "privilege", lambda *args: None)

    def stop_at_numerical(*args):
        raise RuntimeError("reached numerical")

    monkeypatch.setattr(qualification, "numerical", stop_at_numerical)

    def study(identity):
        obj = SimpleNamespace(
            output=tmp_path,
            cfg=Task19Control(cached_rollout=False),
            snapshot="snap",
            identity=identity,
            proposal_groups=0,
            seen=set(),
            iteration=0,
            policy=None,
            sandbox=None,
            references=lambda: [],
            reference_gradient=lambda _: None,
        )

        def collect(_):
            obj.proposal_groups += 1
            return [], {"t": (task, oracle), "a": (other_task, other_oracle)}, {}

        obj.collect = collect
        return obj

    with pytest.raises(SegmentStop):
        qualify(study({"id": "same"}), lambda: len(calls) >= 1)
    assert len(calls) == 1
    with pytest.raises(ValueError, match="identity differs"):
        qualify(study({"id": "different"}))
    with pytest.raises(RuntimeError, match="reached numerical"):
        qualify(study({"id": "same"}))
    assert len(calls) == 8
    assert len(set(calls)) == 8
    assert [task_id for task_id, _ in calls] == ["t"] * 4 + ["a"] * 4
    diagnostics = []

    def passed_numerical(*args):
        diagnostics.append(True)
        return {"passed": True}

    monkeypatch.setattr(qualification, "numerical", passed_numerical)
    with pytest.raises(SegmentStop, match="before calibration"):
        qualify(study({"id": "same"}), lambda: bool(diagnostics))
    with pytest.raises(SegmentStop, match="before calibration"):
        qualify(study({"id": "same"}), lambda: True)
    assert len(diagnostics) == 1
    (tmp_path / "numerical.json").write_text('{"passed": true, "tampered": true}')
    with pytest.raises(ValueError, match="numerical evidence identity differs"):
        qualify(study({"id": "same"}))
