import importlib.util
from pathlib import Path

import pytest


def test_static_exposure_and_allocation_accounting():
    spec = importlib.util.spec_from_file_location(
        "task14_static_analysis", Path("scripts/analyze_task14_static.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metrics = {
        "student": {"cumulative_episodes": 80, "cumulative_generated_tokens": 900},
        "train": {"student_steps": 3},
    }
    events = [
        {"kind": "iteration_commit", "payload": {"iteration": 10, "metrics": metrics}},
        {
            "kind": "evaluation",
            "segment_id": "s",
            "payload": {
                "split": "dev_monitor",
                "iteration": 10,
                "task_id": "t",
                "parent_id": "p",
                "success": True,
            },
        },
        {
            "kind": "evaluation_complete",
            "segment_id": "s",
            "completed_iteration": 10,
            "wall_timestamp": 1300,
            "payload": {"split": "dev_monitor", "completed": 1, "expected": 1},
        },
    ]
    rows = module.evaluations(events, [{"start": 1000, "end": 2000, "gpus": 1}])
    assert rows[0]["episodes"] == 80 and rows[0]["student_steps"] == 3
    assert rows[0]["allocated_gpu_hours"] == pytest.approx(300 / 3600)
    assert (
        module.paired_interval(rows[0]["tasks"], rows[0]["tasks"])["mean_parent_macro_difference"]
        == 0
    )
