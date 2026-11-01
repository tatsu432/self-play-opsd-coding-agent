"""Structured proposals retain sampled behavior tokens through validation."""

import json
from contextlib import nullcontext
from types import SimpleNamespace

from selfplay_opsd.config import RunConfig
from selfplay_opsd.policy import Generation
from selfplay_opsd.trainer import SelfPlayTrainer


def test_structured_proposals_preserve_generated_tokens(monkeypatch):
    parent = SimpleNamespace(
        parent_id="p", source="def f(x):\n    return x + 1\n", specification="f"
    )
    cfg = RunConfig(study_id="task14-test", proposer_representation="structured_ast")
    cfg.parents_per_iteration = 1
    cfg.proposals_per_parent = 2
    cfg.hardware.proposer_batch_size = 1
    policy = SimpleNamespace(
        active=lambda _: nullcontext(),
        version=lambda _: "initial",
        encode_history=lambda messages: [1, 2],
        generate=lambda *args: Generation(
            [7, 8], [-0.3, -0.2], json.dumps({"site": 0, "operation": "Sub"})
        ),
        coordinator_call=lambda fn: fn(),
    )
    trainer = object.__new__(SelfPlayTrainer)
    trainer.config = cfg
    trainer.policy = policy
    trainer.parents = [parent]
    trainer.history = {}
    trainer.repository = SimpleNamespace(parent_cases=lambda _: ([], []))
    trainer.sandbox = object()
    trainer.proposal_observer = None
    trainer._detail_times = {"propose": 0.0, "validate": 0.0}
    trainer._proposal_batch_sizes = []
    monkeypatch.setattr(
        "selfplay_opsd.trainer.build_task", lambda *args: (SimpleNamespace(task_id="t"), object())
    )
    monkeypatch.setattr("selfplay_opsd.trainer.check_admission", lambda *args: None)
    proposals, tasks = trainer.proposals(1)
    assert len(tasks) == 1
    assert [p.valid for p in proposals] == [True, False]
    assert proposals[1].reason == "duplicate"
    assert proposals[0].turn.target_ids == [7, 8]
    assert proposals[0].turn.old_logps == [-0.3, -0.2]
