import json

from selfplay_opsd.scientific_health import summarize


def test_health_report_uses_committed_work_and_explicit_allocated_hours(tmp_path):
    events = tmp_path / "events"
    events.mkdir()
    payload = {
        "iteration": 1,
        "attempted": 4,
        "valid": 2,
        "episodes": 8,
        "student_steps": 1,
        "proposer_steps": 1,
        "episodes_path": "missing-episodes.json",
        "proposals_path": "missing-proposals.json",
        "metrics": {
            "proposer": {"invalid_reasons": {"bad_patch": 2}, "duplicate_count": 0},
            "student": {
                "success_group_mix": {"0": 1, "1": 1},
                "success_numerator": 1,
                "gradient_interaction": {"sampled": True, "rl_norm": 0, "lambda_sd_norm": 1},
                "grad_norm": 1,
            },
            "alignment": {
                "scored_count": 2,
                "scores": [{"near_zero": True}, {"near_zero": False}],
                "near_zero_fraction": 0.5,
            },
        },
    }
    record = {
        "kind": "iteration_commit",
        "payload": payload,
        "committed": True,
        "wall_timestamp": 1,
        "segment_id": "segment",
        "event_index": 1,
        "completed_iteration": 1,
    }
    (events / "segment.jsonl").write_text(json.dumps(record) + "\n")
    report = summarize(tmp_path)
    assert report["valid_fraction"] == 0.5
    assert report["outcome_groups"] == {"all_zero": 1, "all_one": 0, "mixed": 1}
    assert report["sampled_zero_rl_gradients"] == 1
    assert report["valid_proposals_per_allocated_gpu_hour"] is None
    assert report["raw_episode_shards_available"] == 0
    assert summarize(tmp_path, 0.5)["valid_proposals_per_allocated_gpu_hour"] == 4


def test_regression_windows_and_missing_shards(tmp_path):
    from selfplay_opsd.scientific_health import regression

    events = tmp_path / "events"
    events.mkdir()

    def event(kind, iteration, payload, index):
        return {
            "kind": kind,
            "payload": payload,
            "committed": True,
            "wall_timestamp": index,
            "segment_id": "s",
            "event_index": index,
            "completed_iteration": iteration,
        }

    def commit(iteration, groups):
        return {
            "iteration": iteration,
            "attempted": 16,
            "valid": sum(groups.values()),
            "episodes": 4 * sum(groups.values()),
            "student_steps": 1,
            "episodes_path": "missing",
            "proposals_path": "missing",
            "metrics": {
                "proposer": {"invalid_reasons": {"bad": 2}},
                "student": {
                    "success_group_mix": groups,
                    "success_numerator": 1,
                    "gradient_interaction": {"sampled": False},
                },
            },
        }

    records = [
        event("iteration_commit", 20, commit(20, {"0": 1}), 1),
        event("iteration_commit", 21, commit(21, {"1": 1}), 2),
    ]
    for index, (iteration, success) in enumerate([(20, True), (40, False), (60, True)], 3):
        records.append(
            event(
                "evaluation",
                iteration,
                {
                    "split": "dev_monitor",
                    "iteration": iteration,
                    "task_id": "task",
                    "parent_id": "parent",
                    "success": success,
                    "episode_path": None,
                    "metrics": {},
                },
                index,
            )
        )
        records.append(
            event(
                "evaluation_complete",
                iteration,
                {"split": "dev_monitor", "completed": 1, "expected": 1},
                index + 10,
            )
        )
    records.append(
        event(
            "evaluation",
            80,
            {
                "split": "dev_monitor",
                "iteration": 80,
                "task_id": "task",
                "parent_id": "parent",
                "success": False,
                "episode_path": None,
                "metrics": {},
            },
            30,
        )
    )
    (events / "s.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    report = regression(tmp_path, tmp_path / "out")
    assert report["windows"][0]["zero_rl_signal_steps"] == 1
    assert report["windows"][1]["zero_rl_signal_steps"] == 0
    assert report["windows"][1]["mixed"] == 1
    assert report["initial_success_transitions"][0]["first_failure"] == 40
    assert report["initial_success_transitions"][0]["recovered"] is True
    assert report["incomplete_evaluation_rows"] == 1
    assert "80" not in report["initial_success_transitions"][0]["results"]
    assert report["missing_training_shards"] == {"episodes": 2, "proposals": 2}
    assert report["local_latest_checkpoint"] is None
    assert report["local_checkpoint_payload_present"] is False


def test_regression_excludes_superseded_iteration(tmp_path):
    from selfplay_opsd.scientific_health import regression

    events = tmp_path / "events"
    events.mkdir()

    def record(kind, payload, iteration, index, segment):
        return {
            "kind": kind,
            "payload": payload,
            "committed": True,
            "wall_timestamp": index,
            "segment_id": segment,
            "event_index": index,
            "completed_iteration": iteration,
        }

    def payload(valid):
        return {
            "iteration": 1,
            "attempted": 16,
            "valid": valid,
            "episodes": 4 * valid,
            "student_steps": 1,
            "episodes_path": "missing",
            "proposals_path": "missing",
            "metrics": {
                "proposer": {},
                "student": {
                    "success_group_mix": {"0": valid},
                    "success_numerator": 0,
                    "gradient_interaction": {"sampled": False},
                },
            },
        }

    records = [
        record("iteration_commit", payload(2), 1, 1, "a"),
        record("segment_start", {"rollback_to": 0}, 0, 2, "b"),
        record("iteration_commit", payload(1), 1, 3, "b"),
    ]
    (events / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    result = regression(tmp_path, tmp_path / "out")
    assert result["committed_iterations"] == 1
    assert result["windows"][0]["valid"] == 1


def test_regression_distinguishes_escaped_patch_and_duplicate_ownership(tmp_path):
    from selfplay_opsd.scientific_health import regression

    (tmp_path / "events").mkdir()
    source = "def f():\n    return 1\n"
    patch = "\n--- a/src/solution.py\n+++ b/src/solution.py\n@@ -1,2 +1,2 @@\n def f():\n-    return 1\n+    return 2\n"
    escaped = patch.replace("\n", "\\n")
    turn = {
        "action": "apply_patch",
        "parse_status": "tool_error",
        "raw_text": json.dumps({"action": "apply_patch", "patch": escaped}),
        "history": [
            {
                "role": "tool",
                "content": json.dumps({"ok": True, "path": "src/solution.py", "content": source}),
            }
        ],
    }
    (tmp_path / "eval.json").write_text(json.dumps([{"task_id": "t", "turns": [turn]}]))

    def proposal(valid, key, reason):
        return {"valid": valid, "dedup_key": key, "parent_id": "p", "reason": reason}

    (tmp_path / "p1.json").write_text(
        json.dumps([proposal(True, "x", "valid"), proposal(False, "x", "duplicate")])
    )
    (tmp_path / "p2.json").write_text(json.dumps([proposal(False, "x", "duplicate")]))

    def ev(kind, iteration, payload, index):
        return {
            "kind": kind,
            "payload": payload,
            "committed": True,
            "wall_timestamp": index,
            "segment_id": "s",
            "event_index": index,
            "completed_iteration": iteration,
        }

    def commit(it):
        return {
            "iteration": it,
            "attempted": 2,
            "valid": 1,
            "episodes": 0,
            "student_steps": 0,
            "episodes_path": "missing",
            "proposals_path": f"p{it}.json",
            "metrics": {
                "proposer": {},
                "student": {
                    "success_group_mix": {},
                    "success_numerator": 0,
                    "gradient_interaction": {"sampled": False},
                },
            },
        }

    rows = [
        ev("iteration_commit", 1, commit(1), 1),
        ev("iteration_commit", 2, commit(2), 2),
        ev(
            "evaluation",
            20,
            {
                "split": "dev_monitor",
                "iteration": 20,
                "task_id": "t",
                "parent_id": "p",
                "success": False,
                "episode_path": "eval.json",
                "metrics": {},
            },
            3,
        ),
    ]
    rows.append(
        ev("evaluation_complete", 20, {"split": "dev_monitor", "completed": 1, "expected": 1}, 4)
    )
    (tmp_path / "events" / "s.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    report = regression(tmp_path, tmp_path / "out")
    assert report["duplicate_ownership_in_available_shards"] == {
        "previous_iteration": 1,
        "same_iteration": 1,
    }
    evaluation = report["evaluations"][0]
    assert evaluation["failed_patch_newlines"]["literal_backslash_n"] == 1
    assert evaluation["normalized_patch_diagnostic"]["applies"] == 1
    assert evaluation["failure_categories"] == {"double_escaped_patch": 1}
