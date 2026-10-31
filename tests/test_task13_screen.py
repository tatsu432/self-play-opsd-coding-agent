import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "task13_screen", Path(__file__).parents[1] / "scripts/analyze_task13_screen.py"
)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
analyze = module.analyze


def record(task_id, rewards):
    return {
        "task_id": task_id,
        "successes": sum(rewards),
        "episodes": [{"reward": reward} for reward in rewards],
    }


def test_screen_counts_mixed_subsets_and_validates_completeness():
    rows = [record("one", [1] + [0] * 7), record("zero", [0] * 8)]
    result = analyze(rows, expected_tasks=2)
    assert result["mixed_eight_tasks"] == 1
    assert result["mean_mixed_probability_four_without_replacement"] == 0.25
    assert result["paired_eight_minus_four"] == 0.25
    with pytest.raises(ValueError, match="incomplete"):
        analyze(rows)
    rows[0]["successes"] = 2
    with pytest.raises(ValueError, match="success count"):
        analyze(rows, expected_tasks=2)
