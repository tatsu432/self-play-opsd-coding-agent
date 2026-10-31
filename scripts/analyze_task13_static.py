"""Summarize committed static-control iterations and complete development evaluations."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from selfplay_opsd.storage import active_events, atomic_json


def summarize(events: list[dict]) -> dict:
    iterations = []
    completed_evaluations = []
    pending: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for event in events:
        payload = event["payload"]
        if event["kind"] == "iteration_commit":
            metrics = payload["metrics"]
            mix = metrics["student"]["success_group_mix"]
            iterations.append(
                {
                    "iteration": payload["iteration"],
                    "student_step": metrics["train"]["student_steps"],
                    "proposer_step": metrics["train"]["proposer_steps"],
                    "student_stepped": payload["student_steps"],
                    "tasks": payload["valid"],
                    "episodes": payload["episodes"],
                    "successful_episodes": metrics["student"]["success_numerator"],
                    "mixed_groups": sum(
                        int(count) for successes, count in mix.items() if 0 < int(successes) < 4
                    ),
                    "generated_tokens": metrics["student"]["tokens"],
                }
            )
        elif event["kind"] == "evaluation" and payload.get("split") == "dev_monitor":
            pending[(event["segment_id"], payload["iteration"])].append(payload)
        elif event["kind"] == "evaluation_complete" and payload.get("split") == "dev_monitor":
            key = (event["segment_id"], event["completed_iteration"])
            records = pending.pop(key, [])
            if len(records) != payload["completed"] or payload["expected"] != len(records):
                raise ValueError("Development evaluation completion count differs")
            if any(r["success"] is None for r in records):
                raise ValueError("Development evaluation has missing outcomes")
            parents: dict[str, list[bool]] = defaultdict(list)
            for record in records:
                parents[record["parent_id"]].append(record["success"])
            completed_evaluations.append(
                {
                    "iteration": key[1],
                    "tasks": len(records),
                    "successes": sum(r["success"] for r in records),
                    "parent_macro": sum(sum(v) / len(v) for v in parents.values()) / len(parents),
                }
            )
    if len({r["iteration"] for r in iterations}) != len(iterations):
        raise ValueError("Duplicate active iteration records")
    return {
        "iterations": iterations,
        "development": completed_evaluations,
        "total_tasks": sum(r["tasks"] for r in iterations),
        "total_episodes": sum(r["episodes"] for r in iterations),
        "total_mixed_groups": sum(r["mixed_groups"] for r in iterations),
        "student_steps": sum(r["student_stepped"] for r in iterations),
        "proposer_steps": iterations[-1]["proposer_step"] if iterations else 0,
        "total_generated_tokens": sum(r["generated_tokens"] for r in iterations),
        "incomplete_evaluation_records_excluded": sum(map(len, pending.values())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Directory for one static-control arm")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not (args.run / "events").is_dir():
        raise ValueError("Static arm event directory is missing")
    result = summarize(active_events(args.run))
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
