"""Publish only reconciled, aggregate seed-2027 comparison results."""

from __future__ import annotations

import argparse
from pathlib import Path

import wandb

from selfplay_opsd.core import read_json
from selfplay_opsd.storage import file_hash


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    report = read_json(args.summary)
    if report["unreconciled_reservations"] or report["replication_h200_hours"] <= 0:
        raise ValueError("Publication requires reconciled allocated compute")
    endpoints = [
        [arm, split, val["successes"], val["count"]]
        for arm, splits in report["seed2027"].items()
        for split, val in splits.items()
    ]
    comparisons = [
        [
            name,
            split,
            val["difference_pp"],
            *val["ci95_pp"],
            val["gains"],
            val["losses"],
            val["tasks"],
        ]
        for name, splits in report["paired_2027"].items()
        for split, val in splits.items()
    ]
    if not args.publish:
        print(f"{len(endpoints)} endpoint rows, {len(comparisons)} comparison rows")
        return
    checksum = file_hash(args.summary)
    with wandb.init(
        entity="tatsu432s",
        project="self-play-opsd-coding-agent",
        id=f"task22-seed2027-{checksum[:12]}",
        name="Task 22 seed 2027 verified aggregates",
        group="task22",
        resume="allow",
        config={"aggregate_sha256": checksum},
    ) as run:
        run.log(
            {
                "endpoints": wandb.Table(
                    columns=["arm", "split", "successes", "tasks"], data=endpoints
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
                    ],
                    data=comparisons,
                ),
                "replication_h200_hours": report["replication_h200_hours"],
            }
        )
        print(run.url)


if __name__ == "__main__":
    main()
