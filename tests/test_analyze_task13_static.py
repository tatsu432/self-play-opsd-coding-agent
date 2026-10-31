import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "static_summary", Path("scripts/analyze_task13_static.py")
)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_static_summary_counts_committed_work_and_only_complete_evaluations():
    iteration = {
        "kind": "iteration_commit",
        "payload": {
            "iteration": 1,
            "student_steps": 1,
            "valid": 2,
            "episodes": 8,
            "metrics": {
                "train": {"student_steps": 1, "proposer_steps": 0},
                "student": {
                    "success_group_mix": {"0": 1, "2": 1},
                    "success_numerator": 2,
                    "tokens": 20,
                },
            },
        },
    }
    evaluation = {
        "kind": "evaluation",
        "segment_id": "s",
        "completed_iteration": 1,
        "payload": {"split": "dev_monitor", "iteration": 1, "parent_id": "p", "success": True},
    }
    complete = {
        "kind": "evaluation_complete",
        "segment_id": "s",
        "completed_iteration": 1,
        "payload": {"split": "dev_monitor", "expected": 1, "completed": 1},
    }
    result = module.summarize([iteration, evaluation, complete, evaluation])
    assert (result["student_steps"], result["total_mixed_groups"]) == (1, 1)
    assert result["development"][0]["successes"] == 1
    assert result["incomplete_evaluation_records_excluded"] == 1
    complete["payload"]["expected"] = 2
    with pytest.raises(ValueError, match="completion count"):
        module.summarize([evaluation, complete])
