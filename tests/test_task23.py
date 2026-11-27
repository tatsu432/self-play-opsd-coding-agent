"""Regression tests for paired two-checkpoint statistics and follow-up wiring."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture
def script_modules(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    __import__("analyze_task23")
    __import__("hpc_job")
    return sys.modules


def test_family_bootstrap_resamples_two_seeds_together(script_modules):
    paired = script_modules["analyze_task23"].paired_across_seeds
    observed = {}
    for seed, offset in ((2026, 0), (2027, 1)):
        outcomes = {}
        for method, delta in (("A", offset), ("B", 1 - offset)):
            outcomes[method] = {
                "primary": (
                    [
                        {"task_id": f"t{i}", "parent_id": f"p{i}", "success": bool((i + delta) % 2)}
                        for i in range(4)
                    ],
                    None,
                )
            }
        observed[seed] = outcomes
    result = paired(observed, {f"p{i}": f"f{i // 2}" for i in range(4)}, ("B", "A"), "primary")
    assert result["difference_pp"] == 0
    assert result["ci95_pp"] == [0, 0]
    assert (result["tasks"], result["families"]) == (4, 2)


def test_followup_wiring_reuses_the_pdb_allocation_wrapper(script_modules):
    args = script_modules["hpc_job"].task13_command_args(
        {
            "SELFPLAY_OPERATION": "task21-followup",
            "SELFPLAY_METHOD": "relaxed-2026-B",
            "SELFPLAY_SETTINGS": "configs/task21-main.yaml",
            "SELFPLAY_OUTPUT": "/runs/task23/relaxed-2026-B",
            "SELFPLAY_TASK22_EXPOSURE": "8",
        }
    )
    assert args == [
        "python",
        "/workspace/scripts/task23_experiment.py",
        "--output",
        "/runs/task23/relaxed-2026-B",
        "--limit",
        "8",
    ]
