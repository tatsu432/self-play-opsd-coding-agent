"""Analyze paired Task 14 proposer probes without executing candidate code."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from analyze_task13_probe import compare, load_probe, summary

from selfplay_opsd.data import TaskRepository
from selfplay_opsd.storage import atomic_json
from selfplay_opsd.structured_mutations import sites


def structured_diversity(probe, repository):
    parents = {p.parent_id: p for p in repository.parents("train")}
    locations, operators = set(), Counter()
    for row in probe["rows"]:
        if not row["valid"]:
            continue
        action = json.loads(row["raw_text"])
        site = sites(parents[row["parent_id"]].source)[action["site"]]
        locations.add((row["parent_id"], tuple(site["path"])))
        operators[action["operation"]] += 1
    return {"distinct_parent_locations": len(locations), "operators": dict(operators)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diff", type=Path, required=True)
    parser.add_argument("--structured", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diff-gpu-hours", type=float, required=True)
    parser.add_argument("--structured-gpu-hours", type=float, required=True)
    args = parser.parse_args()
    if min(args.diff_gpu_hours, args.structured_gpu_hours) <= 0:
        raise ValueError("Actual allocated GPU hours must be positive")
    first, second = load_probe(args.diff), load_probe(args.structured)
    if (
        first["manifest"]["study"] != "repair-diagnosis-2026-task14-probe-v1"
        or second["manifest"]["study"] != "repair-diagnosis-2026-task14-probe-v1"
    ):
        raise ValueError("Inputs are not Task 14 prospective probes")
    if (
        first["manifest"]["representation"] != "diff"
        or second["manifest"]["representation"] != "structured"
    ):
        raise ValueError("Expected diff and structured arms")
    first_summary, second_summary = summary(first), summary(second)
    for result, hours in (
        (first_summary, args.diff_gpu_hours),
        (second_summary, args.structured_gpu_hours),
    ):
        result["allocated_gpu_hours"] = hours
        result["unique_verified_per_gpu_hour"] = result["unique_verified_parent_mutations"] / hours
    report = {
        "diff": first_summary,
        "structured": second_summary,
        "paired": compare(first, second, resamples=2000),
        "structured_diversity": structured_diversity(second, TaskRepository(args.data)),
        "inputs": {"diff": first["input_sha256"], "structured": second["input_sha256"]},
    }
    atomic_json(args.output, report)


if __name__ == "__main__":
    main()
