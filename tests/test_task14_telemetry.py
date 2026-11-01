from types import SimpleNamespace

import pytest

from selfplay_opsd.telemetry import aggregate_metrics
from selfplay_opsd.trainer import proposer_reward_diagnostics


def test_proposer_reward_discrimination_is_visible_to_telemetry():
    proposals = [SimpleNamespace(parent_id="p", valid=x) for x in (True, True, False, False)]
    metrics = proposer_reward_diagnostics(proposals, [0.0, 0.8, -1.0, -1.0], 4)
    assert metrics["groups_with_multiple_valid"] == 1
    assert metrics["groups_with_distinct_valid_rewards"] == 1
    assert metrics["advantage_vs_validity_l2_mean"] > 0
    row = aggregate_metrics({"proposer": metrics})
    assert row["proposer/groups_with_multiple_valid"] == 1
    assert row["proposer/advantage_vs_validity_l2_mean"] == pytest.approx(
        metrics["advantage_vs_validity_l2_mean"]
    )


def test_single_valid_reward_is_nearly_validity_only():
    proposals = [SimpleNamespace(parent_id="p", valid=x) for x in (True, False, False, False)]
    result = proposer_reward_diagnostics(proposals, [0.3, -1, -1, -1], 4)
    assert result["groups_with_multiple_valid"] == 0
    assert result["groups_with_distinct_valid_rewards"] == 0
    assert result["advantage_vs_validity_l2_mean"] < 1e-5
