"""Diagnostic calculations must preserve missingness and scientific lineage."""

import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

from selfplay_opsd.mutations import make_patch

SPEC = importlib.util.spec_from_file_location(
    "diagnose_learning", Path(__file__).parents[1] / "scripts/diagnose_learning.py"
)
assert SPEC and SPEC.loader
diagnosis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnosis)


def proposal(patch, reason="valid"):
    return {
        "reason": reason,
        "valid": reason == "valid",
        "turn": {
            "raw_text": json.dumps({"patch": patch}),
            "history": [
                {},
                {"content": json.dumps({"correct_source": "def f(x):\n    return x + 1\n"})},
            ],
        },
    }


def test_singleton_rewards_are_nearly_invariant_but_epsilon_is_not_erased():
    result = diagnosis.compare_advantages([-1, -1, -1, 0.8], [False, False, False, True])
    assert not result["exact_validity_equal"]
    assert 0 < result["l2"] < 1e-5
    assert result["cosine"] == pytest.approx(1, abs=1e-6)
    discriminating = diagnosis.compare_advantages([-1, -1, 0, 1], [False, False, True, True])
    assert discriminating["l2"] > 0.5
    assert discriminating["valid_reward_variance"] > 0


def test_uniform_rewards_have_undefined_cosine_and_zero_advantages():
    result = diagnosis.compare_advantages([-1] * 4, [False] * 4)
    assert result["cosine"] is None
    assert result["nonzero_advantages"] == 0
    assert result["valid_reward_variance"] is None
    assert diagnosis.compare_advantages([0.5] * 4, [True] * 4)["nonzero_advantages"] == 0


def test_duplicates_do_not_get_reexecuted_or_assumed_semantically_invalid():
    patch = make_patch("def f(x):\n    return x + 1\n", "def f(x):\n    return x - 1\n")
    result = diagnosis.proposal_syntax(proposal(patch, "duplicate"))
    assert result["stages"]["constraints"] is True
    assert result["stages"]["novel"] is False
    assert result["stages"]["semantic_failure"] is None
    assert result["sites"] == ["Module.body[0].body[0].value.op:Add->Sub"]


def test_funnel_preserves_unobserved_downstream_stages():
    row = proposal("not a patch")
    result = diagnosis.proposal_syntax(row)
    assert result["stages"]["headers"] is False
    assert result["stages"]["semantic_failure"] is None
    assert result["stages"]["novel"] is True  # admission is recorded, independent of replay
    table = {r["stage"]: r for r in diagnosis.funnel([result])}
    assert table["semantic_failure"]["unknown"] == 1
    assert table["semantic_failure"]["conditional_pass_among_assessed"] is None


def test_context_and_count_errors_can_both_be_detected_without_execution():
    patch = "--- a/src/solution.py\n+++ b/src/solution.py\n@@ -1,3 +1,3 @@\n def f(x):\n-    return y + 1\n+    return x - 1\n"
    result = diagnosis.proposal_syntax(proposal(patch, "Patch context mismatch"))
    assert result["stages"]["hunks"] is False
    assert result["stages"]["context"] is False
    assert set(result["additional_failures"]) == {"Hunk count mismatch", "Patch context mismatch"}
    assert result["stages"]["novel"] is None


def test_actual_funnel_checks_duplicates_before_semantic_execution():
    patch = make_patch("def f(x):\n    return x + 1\n", "def f(x):\n    return x - 1\n")
    rows = []
    for reason in (
        "valid",
        "duplicate",
        "Mutant has no semantic failure (crash/timeout-only mutants invalid)",
    ):
        row = proposal(patch, reason)
        rows.append(row | {"category": diagnosis.category(reason)} | diagnosis.proposal_syntax(row))
    stages = {r["stage"]: r for r in diagnosis.recorded_pipeline(rows)}
    assert stages["novelty"]["reached"] == 3
    assert stages["novelty"]["rejected"] == 1
    assert stages["semantic_execution"]["reached"] == 2
    assert stages["semantic_execution"]["rejected"] == 1
    assert stages["admission"]["passed"] == 1


def test_beta_projection_does_not_treat_zero_of_four_as_zero_probability():
    assert diagnosis.mixed_probability(0, 4, 8, 0.5) > 0
    assert diagnosis.mixed_probability(0, 4, 8, 0.5) == pytest.approx(
        diagnosis.mixed_probability(4, 4, 8, 0.5)
    )
    rows = diagnosis.group_size_projection(Counter({"0": 3, "2": 1}), 4)
    assert len(rows) == 6
    assert rows[1]["rollout_cost_multiple"] == 2
    assert rows[1]["expected_mixed_fraction"] > rows[0]["expected_mixed_fraction"]
    assert rows[1]["expected_mixed_fraction"] / 2 < rows[0]["expected_mixed_fraction"]


def event(kind, iteration, payload, index):
    return {
        "kind": kind,
        "payload": payload,
        "committed": True,
        "wall_timestamp": index,
        "segment_id": str(index),
        "event_index": index,
        "completed_iteration": iteration,
    }


def test_commits_use_rollback_lineage_and_deduplicate_exact_records(tmp_path):
    records = [
        event("iteration_commit", 1, {"iteration": 1, "valid": 1}, 1),
        event("iteration_commit", 2, {"iteration": 2, "valid": 99}, 2),
        event("segment_start", 1, {"rollback_to": 1}, 3),
        event("iteration_commit", 2, {"iteration": 2, "valid": 2}, 4),
    ]
    records.append(records[-1])
    (tmp_path / "events").mkdir()
    (tmp_path / "events/e.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    _, commits = diagnosis.committed_records(tmp_path)
    assert [r["valid"] for r in commits] == [1, 2]
    records.append(event("iteration_commit", 2, {"iteration": 2, "valid": 3}, 5))
    (tmp_path / "events/e.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    with pytest.raises(ValueError, match="Conflicting"):
        diagnosis.committed_records(tmp_path)


def test_missing_and_escaping_shards(tmp_path):
    assert diagnosis.load_shard(tmp_path, "missing", {}) is None
    with pytest.raises(ValueError, match="escapes"):
        diagnosis.load_shard(tmp_path, "../outside", {})


def test_scheduler_accounting_includes_previous_attempt_and_clamps_future_jobs():
    jobs = [
        {"start_epoch": 0, "elapsed_seconds": 3600, "gpus": 1},
        {"start_epoch": 7200, "elapsed_seconds": 3600, "gpus": 2},
    ]
    assert diagnosis.allocated_hours(jobs) == 3
    assert diagnosis.allocated_hours(jobs, 1800) == 0.5
    assert diagnosis.allocated_hours(jobs, 5400) == 1
    assert diagnosis.allocated_hours(jobs, 9000) == 2


def test_parent_contrast_does_not_compare_different_parent_populations():
    def row(early, late):
        return {
            "windows": {
                "early": dict(
                    attempted=4, accepted=early, duplicate=0, syntax_constraints_passed=4
                ),
                "late": dict(
                    attempted=4 if late is not None else 0,
                    accepted=late or 0,
                    duplicate=0,
                    syntax_constraints_passed=4,
                ),
            }
        }

    result = diagnosis.parent_contrast([row(2, 1), row(4, None)], "early", "late")
    assert result["paired_parents"] == 1
    assert result["metrics"]["accepted"]["difference"] == -0.25
    assert result["metrics"]["accepted"]["parent_bootstrap_95_interval"] == [-0.25, -0.25]


def test_tool_observations_are_counted_once_from_history_suffix():
    def turn(history):
        return {
            "action": "read_file",
            "parse_status": "valid",
            "target_ids": [1],
            "old_logps": [-1.0],
            "teacher_logps": [-1.0],
            "history": history,
        }

    initial = [{"role": "user", "content": "read"}]
    exchange = [
        {"role": "assistant", "content": "read"},
        {"role": "tool", "content": '{"ok":true}'},
    ]
    episode = {
        "turns": [turn(initial), turn(initial + exchange), turn(initial + exchange * 2)],
        "termination": "token_limit",
        "reward": 0,
        "test_calls": 0,
    }
    result = diagnosis.episode_metrics([episode], True, {})
    assert result["observations"] == {"read_file:True": 2}
    assert result["opsd"]["all"]["mean_gap"] == 0
    assert result["opsd"]["all"]["mean_gate"] == 0.5
