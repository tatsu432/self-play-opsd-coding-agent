"""Publish scalar and count evidence from a reconciled Task 22 analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import wandb

from selfplay_opsd.core import read_json
from selfplay_opsd.storage import file_hash


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--entity", default="tatsu432s")
    parser.add_argument("--project", default="self-play-opsd-coding-agent")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    report = read_json(args.summary)
    if report["unreconciled_reservations"]:
        raise ValueError("Reconcile all allocations before publishing results")
    if not report["incremental_h200_hours"]:
        raise ValueError("Task 22 has no reconciled GPU evidence to publish")
    endpoints = []
    for arm, row in report["endpoints"].items():
        for split, scores in row["scores"].items():
            endpoints.append(
                [
                    arm,
                    split,
                    row["episodes"],
                    row["optimizer_steps"],
                    scores["successes"],
                    scores["count"],
                    row["training_h200_hours"],
                    row["end_to_end_h200_hours"],
                ]
            )
    contrasts = [
        [
            name,
            split,
            row["difference_pp"],
            *row["ci95_pp"],
            row["gains"],
            row["losses"],
            row["tasks"],
            row["families"],
        ]
        for name, groups in report["paired_2026"].items()
        for split, row in groups.items()
    ]
    mechanism = [
        [name, arm, row["tasks"], counts["near_zero"], counts["informative"]]
        for name, row in report["alignment"].items()
        for arm, counts in row["counts"].items()
    ]
    record = {
        "endpoints": wandb.Table(
            columns=[
                "arm",
                "split",
                "episodes",
                "steps",
                "successes",
                "tasks",
                "training_h200_hours",
                "end_to_end_h200_hours",
            ],
            data=endpoints,
        ),
        "paired_contrasts": wandb.Table(
            columns=[
                "contrast",
                "split",
                "delta_pp",
                "ci_low_pp",
                "ci_high_pp",
                "gains",
                "losses",
                "tasks",
                "families",
            ],
            data=contrasts,
        ),
        "alignment": wandb.Table(
            columns=["source", "reward", "tasks", "near_zero", "informative"], data=mechanism
        ),
        "incremental_h200_hours": report["incremental_h200_hours"],
        "task21_and_task22_h200_hours": report["combined_task21_task22_h200_hours"],
    }
    if not args.publish:
        print(
            f"{len(endpoints)} endpoint rows, {len(contrasts)} contrast rows, {len(mechanism)} alignment rows"
        )
        return
    checksum = file_hash(args.summary)
    with wandb.init(
        entity=args.entity,
        project=args.project,
        id=f"task22-{checksum[:12]}",
        name="Task 22 verified aggregates",
        group="task22",
        resume="allow",
        config={"aggregate_sha256": checksum},
    ) as run:
        run.log(record)
        print(run.url)


if __name__ == "__main__":
    main()
