"""Summarize eight-rollout student screens without executing generated code."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from selfplay_opsd.storage import atomic_json, file_hash


def analyze(records: list[dict], *, expected_tasks: int = 32) -> dict:
    if len(records) != expected_tasks or len({r["task_id"] for r in records}) != len(records):
        raise ValueError("Screen is incomplete or contains duplicate tasks")
    outcomes = []
    for row in records:
        rewards = [episode["reward"] for episode in row["episodes"]]
        if len(rewards) != 8 or any(r not in (0, 1) for r in rewards):
            raise ValueError("Each task needs eight binary-reward episodes")
        if sum(rewards) != row["successes"]:
            raise ValueError("Recorded success count differs from episode rewards")
        outcomes.append(rewards)
    groups = list(itertools.combinations(range(8), 4))
    fractions4 = np.array(
        [
            sum(0 < sum(row[i] for i in group) < 4 for group in groups) / len(groups)
            for row in outcomes
        ]
    )
    mixed8 = np.array([0 < sum(row) < 8 for row in outcomes], dtype=float)
    first4 = np.array([0 < sum(row[:4]) < 4 for row in outcomes], dtype=float)
    rng = np.random.default_rng(2026)
    indices = rng.integers(0, expected_tasks, size=(10000, expected_tasks))
    difference = (mixed8 - fractions4)[indices].mean(axis=1)
    return {
        "tasks": expected_tasks,
        "episodes_per_task": 8,
        "success_counts": [int(sum(row)) for row in outcomes],
        "mixed_first_four_tasks": int(first4.sum()),
        "mixed_eight_tasks": int(mixed8.sum()),
        "mean_mixed_probability_four_without_replacement": float(fractions4.mean()),
        "mixed_fraction_eight": float(mixed8.mean()),
        "paired_eight_minus_four": float((mixed8 - fractions4).mean()),
        "paired_task_bootstrap_95_interval": np.quantile(difference, [0.025, 0.975]).tolist(),
        "interpretation": "Descriptive on screened static-pool tasks; four averages all 70 subsets of the same eight outcomes, not independent resampling or a training effect.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("screen", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(json.loads(args.screen.read_text()))
    result["screen_sha256"] = file_hash(args.screen)
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
