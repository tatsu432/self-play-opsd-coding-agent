from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_task21 import task_pair

from selfplay_opsd.core import digest, read_json
from selfplay_opsd.storage import atomic_json, file_hash
from selfplay_opsd.task22 import (
    Task22Control,
    Task22Study,
    paired_rewards,
    qualified,
    reference_sidecar,
)
from selfplay_opsd.task22_contract import (
    INITIAL,
    MATCHED_DATA,
    STUDY,
    identity_hash,
    reference_path,
)

ROOT = Path(__file__).resolve().parents[1]


def control(**changes):
    matched = read_json(ROOT / "configs/task21-main.json")
    return Task22Control.model_validate(
        matched | {"study": STUDY, "reference_sha256": "a" * 64} | changes
    )


def test_task22_reward_weights_and_grpo_uniform_groups():
    reference = [torch.tensor([0.0, 1.0])]
    sd = [torch.tensor([0.0, 2.0])]
    scores = paired_rewards([torch.tensor([1.0, 0.0])], sd, reference, 0.25, [True, False] * 2)
    assert scores["C"]["reward"] == 0.5
    assert scores["D"]["cosine"] == pytest.approx(0.5 / 1.25**0.5)
    assert scores["D"]["candidate_norm"] == pytest.approx(1.25**0.5)
    assert scores["weighted_opsd_norm"] == 0.5
    assert scores["difficulty"] == 1
    for success in (True, False):
        scores = paired_rewards([torch.zeros(2)], sd, reference, 0.25, [success] * 4)
        assert scores["C"]["near_zero"] and scores["C"]["reward"] == 0
        assert scores["D"]["reward"] == 1 and scores["D"]["cosine"] == 1
        assert scores["difficulty"] == 0
    scores = paired_rewards([torch.zeros(2)], sd, [-reference[0]], 0.25, [False] * 4)
    assert scores["D"]["cosine"] == -1 and scores["D"]["reward"] == 0


def test_qualification_requires_signal_from_distinct_valid_tasks():
    good = {"within_parent_distinct": 1, "proposer_norm": 0.01}
    invalid_only = {"within_parent_distinct": 0, "proposer_norm": 0.01}
    c = {"comparison": {"C": good, "D": invalid_only}}
    d = {"comparison": {"C": invalid_only, "D": good}}
    assert qualified([c, d])
    assert not qualified([c, c])
    assert not qualified([{"comparison": {"C": good, "D": good}}])


def test_reference_selection_checks_family_overlap_and_uses_native_loss_mask(tmp_path, monkeypatch):
    template, oracle = task_pair()
    rows = []
    for i in range(10):
        key = f"task-{i:02}"
        rows.append(
            (
                template.model_copy(update={"task_id": key, "parent_id": key}),
                oracle.model_copy(update={"task_id": key, "provenance": {"family_id": key}}),
            )
        )
    eval_task, eval_oracle = rows[-1]
    eval_rows = [(eval_task.model_copy(update={"parent_id": "eval-parent"}), eval_oracle)]
    splits = {"feasibility": rows, "primary_final": eval_rows}
    repo = SimpleNamespace(
        hash=MATCHED_DATA,
        root=tmp_path,
        partition_names=lambda: sorted(splits),
        tasks=lambda key: splits.get(key, []),
        parents=lambda _: [],
    )
    sidecar = reference_sidecar(repo)
    assert [row["task_id"] for row in sidecar["candidates"]] == [f"task-{i:02}" for i in range(9)]

    path = tmp_path / "input.json"
    atomic_json(path, sidecar)
    checksum = file_hash(path)
    path = path.rename(reference_path(tmp_path, checksum))
    calls = []

    def tokenize(text, add_special_tokens):
        assert not add_special_tokens
        assert text.startswith("</think>\n\n<tool_call>")
        assert "<function=replace_text>" in text
        calls.append(text)
        return [1, 4, 2, 5, 3, 6]

    def verify(sandbox, source, task, oracle):
        result = source == oracle.correct_source
        return SimpleNamespace(passed=result, semantic_failures=int(not result))

    def encode(history, tools, thinking):
        assert "- 1" not in str(history) and "+ 1" not in str(history)
        return [8, 9]

    monkeypatch.setattr("selfplay_opsd.task22.verify_pdb", verify)
    study = Task22Study.__new__(Task22Study)
    study.cfg = control(reference_sha256=checksum)
    study.repository = repo
    study.execution = {}
    study.sandbox = SimpleNamespace(identity=lambda: "worker")
    study.policy = SimpleNamespace(
        encode=encode,
        tokenizer=SimpleNamespace(encode=tokenize),
        markers={"<think>": 0, "</think>": 1, "<tool_call>": 2, "</tool_call>": 3},
        eos=set(),
    )
    records = study.prepare_references()
    assert len(calls) == len(records["rows"]) == 9
    assert records["rows"][0]["mask"] == [False, False, True, True, True, False]
    assert all(
        row["repaired_verified"] and row["original_semantic_failures"] == 1
        for row in records["rows"]
    )
    path.write_text("{}")
    with pytest.raises(ValueError, match="sidecar differs"):
        study.prepare_references()


def test_reference_gradient_is_refreshed_and_uses_mean_action_loss():
    study = Task22Study.__new__(Task22Study)
    parameter = torch.tensor(0.3, requires_grad=True)

    def score(prompt, targets, sampling, gradients):
        assert gradients and sampling is None
        return torch.nn.functional.logsigmoid(parameter * torch.tensor(targets)), None

    study.policy = SimpleNamespace(parameters=[parameter], score=score)
    references = [([9], [1, 2, 3], [False, True, False]), ([9], [3, 1, 2], [True, True, False])]
    first = study.reference_gradient(references)[0]
    expected = torch.autograd.grad(
        -(
            torch.nn.functional.logsigmoid(2 * parameter)
            + torch.nn.functional.logsigmoid(parameter * torch.tensor([3, 1])).mean()
        )
        / 2,
        parameter,
    )[0]
    assert torch.allclose(first, expected)
    assert first < 0
    parameter.detach().fill_(0.8)
    assert not torch.allclose(study.reference_gradient(references)[0], first)


def test_milestone_guard_preserves_frozen_main_requirement(tmp_path, monkeypatch):
    study = Task22Study.__new__(Task22Study)
    study.cfg = control()
    study.initial = INITIAL
    study.arm = "C"
    study.output = tmp_path
    study.checkpoints = SimpleNamespace(root=tmp_path / "checkpoints")
    with pytest.raises(ValueError, match="prespecified"):
        study.evaluation("evaluate-primary", 9, 128, lambda: False)
    with pytest.raises(ValueError, match="completed main"):
        study.evaluation("evaluate-primary", 20, 512, lambda: False)
    monkeypatch.setattr(
        "selfplay_opsd.task22.CheckpointManager.verify", lambda _: {"episodes": 256}
    )
    with pytest.raises(ValueError, match="declared exposure"):
        study.evaluation("evaluate-primary", 10, 384, lambda: False)


def test_controller_task22_ledger_stays_separate_and_hashes_match(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "task22_hpc_remote", ROOT / "scripts/hpc_remote.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, "workspace", lambda _: tmp_path)
    stage = {
        "study_id": STUDY,
        "operation": "task21-align-qualify",
        "readiness_budget": 95,
        "gpu_count": 1,
        "run_id": "task22",
        "time": "03:00:00",
    }
    assert runner.budget_path(stage).name == "task22-budget.json"
    assert (
        runner.budget_path({"study_id": "repair-selfplay-2026-task21-pdb-v1"}).name
        == "task21-budget.json"
    )
    runner.reserve_budget(stage, "first")
    runner.bind_budget_job(stage, "first", "1234")
    assert read_json(runner.budget_path(stage))["reservations"]["first"]["job_id"] == "1234"
    with pytest.raises(ValueError, match="five H200-hours"):
        runner.reserve_budget(stage, "second")
    assert identity_hash(control().model_dump(mode="json")) == digest(
        control().model_dump(mode="json")
    )
