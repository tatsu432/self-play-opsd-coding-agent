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
