"""Reward normalization remains explicit in the proposer counterfactual."""

import importlib.util
import sys
from pathlib import Path

import pytest

from selfplay_opsd.config import RunConfig
from selfplay_opsd.core import ProposalRecord, TokenTurn


def test_single_valid_reward_is_erased_only_by_standardization(monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    spec = importlib.util.spec_from_file_location(
        "task14_counterfactual", "scripts/task14_counterfactual.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    cfg = RunConfig(study_id="test", parents_per_iteration=1, proposals_per_parent=4)
    turn = TokenTurn(
        prompt_ids=[1],
        target_ids=[2],
        old_logps=[-1],
        history=[],
        raw_text="",
        action="propose",
        parse_status="valid",
    )
    proposals = [
        ProposalRecord(
            proposal_id=str(i),
            parent_id="p",
            slot=i,
            seed=i,
            policy_version="v",
            turn=turn,
            valid=i == 0,
            reason="test",
        )
        for i in range(4)
    ]

    def values(reward, centered):
        return [
            example.advantage
            for example in module.examples_for(proposals, [reward, -1, -1, -1], cfg, centered)
        ]

    assert values(0.2, False) == pytest.approx(values(0.8, False), abs=2e-6)
    assert values(0.2, True) != pytest.approx(values(0.8, True))
