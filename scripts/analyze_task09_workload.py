"""Project Task 09 costs from a shared A/B/C allocation and resumed D pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from selfplay_opsd.storage import active_events, all_events, read_json
from selfplay_opsd.workload import project_method


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_root", type=Path, help="Fetched fetch-study output directory")
    parser.add_argument("--suite-seconds", type=float, required=True)
    parser.add_argument("--pilot-seconds", type=float, required=True)
    parser.add_argument("--baseline-seconds", type=float, required=True)
    args = parser.parse_args()
    if min(args.suite_seconds, args.pilot_seconds, args.baseline_seconds) <= 0:
        parser.error("Allocated seconds must be positive reconciled Slurm accounting")
    root = args.study_root
    prefix = "repair-selfplay-2026-efficiency-v1"
    readiness = root / "readiness/method-smokes"
    records = {}
    spans = {}
    fixtures = {}
    previous_end = 0.0
    for method in "ABC":
        run = root / f"{prefix}-method-smoke/{method}-seed2026"
        records[method] = active_events(run)
        fixture = read_json(readiness / f"{method}.json")["optimizer_mechanics_fixture"]
        fixtures[method] = fixture["seconds"]
        start = min(event["wall_timestamp"] for event in all_events(run))
        if start < previous_end or fixture["ended_at"] < start:
            raise ValueError("Method execution intervals overlap or have invalid timestamps")
        previous_end = fixture["ended_at"]
        spans[method] = fixture["ended_at"] - start
    remaining = args.suite_seconds - sum(spans.values())
    if remaining < 0:
        raise ValueError("Method spans exceed reconciled suite allocation")
    # Shared startup, interprocess gaps and shutdown are charged once, evenly.
    allocations = {method: spans[method] + remaining / 3 for method in "ABC"}
    pilot = root / f"{prefix}-pilot-h200/D-seed2026"
    projections = {
        "D": project_method(
            active_events(pilot),
            allocated_seconds=args.pilot_seconds,
            baseline_seconds=args.baseline_seconds,
        )
    }
    for method in "ABC":
        projections[method] = project_method(
            records[method],
            allocated_seconds=allocations[method],
            baseline_seconds=args.baseline_seconds,
            acceptance_only_seconds=fixtures[method],
            diagnostic_seconds_per_valid=projections["D"]["diagnostic_seconds_per_valid_p90"],
        )
    print(json.dumps({"suite_allocated_seconds": allocations, "per_method": projections}, indent=2))


if __name__ == "__main__":
    main()
