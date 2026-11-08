"""Synthetic contracts, separate from the real scientific evidence."""

import ast
import json
import subprocess
from pathlib import Path

import pytest

from selfplay_opsd.hpc import HpcError, YaleSSH
from selfplay_opsd.task16_analysis import (
    allocated_hours,
    complete_evaluations,
    compute_comparisons,
    episode_diagnostics,
    failure_category,
    mutation_sites,
    paired_statistics,
    relative_artifact,
    select_checkpoint,
    selection_stability,
)


def event(kind, payload, iteration=0, segment="one"):
    return {
        "kind": kind,
        "payload": payload,
        "completed_iteration": iteration,
        "segment_id": segment,
        "wall_timestamp": 1,
    }


def evaluation_events():
    common = {"split": "final_id", "checkpoint": "checkpoint", "endpoint": "primary"}
    row = common | {"task_id": "t", "parent_id": "p", "success": False}
    return [
        event("evaluation_start", common | {"expected_task_ids": ["t"]}),
        event("evaluation", row),
        event("evaluation_complete", common | {"expected": 1, "completed": 1}),
    ]


def test_complete_denominators_and_attempt_identity():
    es = evaluation_events()
    expected = {"final_id": {"t": "p"}}
    assert len(complete_evaluations(es, expected)) == 1
    with pytest.raises(ValueError, match="Duplicate evaluation task"):
        complete_evaluations(es[:2] + [es[1], es[2]], expected)
    with pytest.raises(ValueError, match="task set mismatch"):
        complete_evaluations(es, {"final_id": {"other": "p", "t": "p"}})
    with pytest.raises(ValueError, match="Unfinished"):
        complete_evaluations(es[:-1], expected)
    broken = [dict(e) for e in es]
    broken[1] = event("evaluation", es[1]["payload"] | {"checkpoint": "wrong"})
    with pytest.raises(ValueError, match="matching start"):
        complete_evaluations(broken, expected)
    broken = [dict(e) for e in es]
    broken[1] = event("evaluation", es[1]["payload"] | {"success": None})
    with pytest.raises(ValueError, match="Incomplete"):
        complete_evaluations(broken, expected)


def test_conflicting_commits_are_not_silently_collapsed():
    commits = [
        event("iteration_commit", {"iteration": 1}, 1),
        event("iteration_commit", {"iteration": 1}, 1, "two"),
    ]
    with pytest.raises(ValueError, match="Conflicting active"):
        complete_evaluations(commits, {})


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ({"success": True, "infrastructure_failure": True}, "success"),
        ({"success": False, "infrastructure_failure": True}, "infrastructure"),
        ({"success": None}, "unknown"),
        (
            {
                "success": False,
                "public_pass": True,
                "hidden_pass": False,
                "termination": "turn_limit",
            },
            "public_pass_hidden_fail",
        ),
        (
            {
                "success": False,
                "termination": "turn_limit",
                "valid_edits": 0,
                "rejected_patches": 1,
            },
            "budget_exhausted",
        ),
        (
            {"success": False, "valid_edits": 0, "rejected_patches": 1, "parse_errors": 1},
            "no_edit_patch_rejected",
        ),
        ({"success": False, "valid_edits": 0, "parse_errors": 1}, "no_edit_parse_failure"),
        ({"success": False, "valid_edits": 1}, "edited_semantic_failure"),
        ({"success": False, "source_changed": True}, "edited_semantic_failure"),
        ({"success": False, "valid_edits": None}, "unknown"),
    ],
)
def test_failure_precedence(row, expected):
    assert failure_category(row) == expected


def test_missing_trajectory_preserves_metrics_but_not_action_counts():
    record = {
        "task_id": "t",
        "parent_id": "p",
        "checkpoint": "base",
        "success": False,
        "status": "student_failure",
        "metrics": {"tokens": 3, "termination": "finish"},
    }
    row = episode_diagnostics(record, None, "hash")
    assert row["tokens"] == 3
    assert row["valid_actions"] is None
    assert row["primary_category"] == "unknown"


def test_tool_observation_is_in_next_history_final_error_unknown():
    record = {
        "task_id": "t",
        "parent_id": "p",
        "checkpoint": "base",
        "success": False,
        "status": "student_failure",
        "endpoint": "base",
        "metrics": {"turns": 2, "tokens": 2, "termination": "finish"},
    }
    turn = {"action": "apply_patch", "parse_status": "tool_error", "target_ids": [1], "history": []}
    episode = {
        "task_id": "t",
        "reward": 0,
        "policy_version": "hash",
        "turns": [
            turn,
            turn
            | {
                "history": [
                    {"role": "tool", "content": json.dumps({"error": "hunk context mismatch"})}
                ]
            },
        ],
    }
    row = episode_diagnostics(record, episode, "initial")
    assert row["context_errors"] == 1 and row["unknown_tool_errors"] == 1
    assert row["rejected_patches"] == 2


def test_allocation_cost_includes_retries_and_clips_at_timestamp():
    allocations = [
        {"job_id": "one", "start": 0, "end": 3600, "gpus": 1},
        {"job_id": "two", "start": 7200, "end": 10800, "gpus": 1},
    ]
    assert allocated_hours(allocations, 9000) == 1.5
    assert allocated_hours(allocations) == 2
    assert allocated_hours(allocations, -1) == 0
    with pytest.raises(ValueError, match="duplicate"):
        allocated_hours([allocations[0], allocations[0]])
    with pytest.raises(ValueError, match="Overlapping"):
        allocated_hours([allocations[0], allocations[1] | {"start": 3500}])


def curve(iteration, successes, episodes=None):
    records = [
        {"task_id": str(i), "parent_id": str(i // 2), "success": bool(s)}
        for i, s in enumerate(successes)
    ]
    parents = [(sum(successes[:2]) / 2), (sum(successes[2:]) / 2)]
    return {
        "iteration": iteration,
        "episodes": iteration if episodes is None else episodes,
        "parent_macro": sum(parents) / 2,
        "records": records,
    }


def test_selection_ties_noise_and_shared_parent_resampling():
    rows = [curve(0, [0, 0, 0, 0]), curve(1, [1, 1, 0, 0]), curve(2, [0, 0, 1, 1])]
    assert select_checkpoint(rows)["iteration"] == 1
    families = {"0": "a", "1": "b"}
    result = selection_stability(rows, families, samples=200)
    assert result == selection_stability(rows, families, samples=200)
    assert result["selected_nonfinal_fraction"] > 0
    assert sum(result["selection_frequency"].values()) == pytest.approx(1)
    with pytest.raises(ValueError, match="No development"):
        select_checkpoint([])
    stats = paired_statistics(rows[0]["records"], rows[1]["records"], families, samples=100)
    assert stats["difference_parent_macro"] == 0.5 and stats["second_only"] == 2
    with pytest.raises(ValueError, match="Paired task"):
        paired_statistics(rows[0]["records"], rows[1]["records"][:-1], families)


def test_budget_comparison_preserves_overshoot_without_extrapolation():
    rows = []
    for method, count in [("A", 256), ("B", 260)]:
        for iteration, episodes in [(0, 0), (1, count)]:
            rows.append(
                curve(iteration, [iteration] * 4, episodes)
                | {
                    "method": method,
                    "seed": 2026,
                    "tokens": episodes,
                    "student_steps": iteration,
                    "allocated_gpu_hours": iteration + 1,
                }
            )
    result = compute_comparisons(rows)
    episode_rows = [r for r in result if r["axis"] == "episodes"]
    assert max(r["budget"] for r in episode_rows) == 256
    row = next(r for r in episode_rows if r["budget"] == 256)
    assert row["b_observed_budget"] == 0  # 260 is not a measured 256-episode endpoint.


def test_ast_change_tracks_operator_parent_location_and_negative_constants():
    changes = mutation_sites(
        ast.parse("def f(x):\n return x + 1\n"), ast.parse("def f(x):\n return x - 1\n")
    )
    assert changes == [{"path": ".body[0].body[0].value.op", "line": 2, "change": "Add->Sub"}]
    changes = mutation_sites(ast.parse("x = 0"), ast.parse("x = -1"))
    assert changes[0]["change"] == "Constant->UnaryOp"


def test_relative_artifact_rejects_escape(tmp_path):
    for name in ("../secret", "/secret"):
        with pytest.raises(ValueError):
            relative_artifact(tmp_path, name)
    (tmp_path / "link").symlink_to(tmp_path.parent)
    with pytest.raises(ValueError):
        relative_artifact(tmp_path, "link/secret")


def test_selective_fetch_excludes_weights_and_validates_paths(tmp_path, monkeypatch):
    captured = []

    def fake_run(args, **kwargs):
        names = Path(
            next(a.split("=", 1)[1] for a in args if a.startswith("--files-from="))
        ).read_text()
        captured.append((args, names))
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    ssh = YaleSSH()
    ssh.fetch("/runs/x", tmp_path, trajectory_paths=["trajectories/segment/task.json"])
    assert captured[0][1] == "trajectories/segment/task.json\n"
    assert "--no-links" in captured[0][0]
    for name in (
        "../secret",
        "checkpoints/state.pt",
        "trajectories/../x.json",
        "trajectories/s/x.json\nsecret",
    ):
        with pytest.raises(HpcError, match="Unsafe"):
            ssh.fetch("/runs/x", tmp_path, trajectory_paths=[name])
    with pytest.raises(HpcError, match="cannot include"):
        ssh.fetch("/runs/x", tmp_path, full=True, trajectory_paths=[])


def test_initial_mutant_evidence_requires_unchanged_verified_source():
    from selfplay_opsd.task16_analysis import initial_mutant_observations

    features = [{"task_id": "t", "initial_public_failure": None, "initial_hidden_failure": None}]
    observed = {
        "task_id": "t",
        "source_changed": False,
        "infrastructure_failure": False,
        "public_pass": True,
        "hidden_pass": False,
    }
    assert initial_mutant_observations(features, [observed]) == {"t": {(True, False)}}
    assert features[0]["initial_public_failure"] is False
    assert features[0]["initial_hidden_failure"] is True
    assert not initial_mutant_observations([], [observed | {"source_changed": True}])
    assert not initial_mutant_observations([], [observed | {"infrastructure_failure": True}])
    with pytest.raises(ValueError, match="inconsistent"):
        initial_mutant_observations(features, [observed, observed | {"public_pass": False}])


def test_missing_checkpoint_does_not_change_selector(tmp_path):
    from selfplay_opsd.storage import CheckpointManager

    rows = [curve(0, [0, 0, 0, 0]), curve(1, [1, 1, 1, 1]), curve(2, [0, 0, 1, 1])]
    selected = select_checkpoint(rows)
    assert selected["iteration"] == 1
    assert not (tmp_path / "000001").exists()
    # An unavailable selected checkpoint must not be replaced by a poorer one.
    with pytest.raises(FileNotFoundError):
        CheckpointManager.verify(tmp_path / "000001")
    assert select_checkpoint(rows) is selected
