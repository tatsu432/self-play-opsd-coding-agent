"""Analyze verified seed-2027 C/D receipts against the frozen matched PDB runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from analyze_task22 import PRIMARY, alignment_summary, contrasts, load_run, primary_figure

from selfplay_opsd.core import read_json
from selfplay_opsd.storage import atomic_json
from selfplay_opsd.task21 import PdbRepository
from selfplay_opsd.task22_contract import INITIAL_2027


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--budget", type=Path, required=True)
    parser.add_argument(
        "--historical", type=Path, default=Path("runs/task21/matched-analysis/results.json")
    )
    parser.add_argument(
        "--prior", type=Path, default=Path("runs/task22/final-analysis/summary.json")
    )
    parser.add_argument("--data", type=Path, default=Path("data/task21/pdb-solver-v1"))
    parser.add_argument("--artifacts", type=Path, default=Path("runs/hpc/yale-hpc-cpsc4880"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    budget = read_json(args.budget)
    allocations = list(budget["reservations"].values())
    if any(row["charged"] is None for row in allocations):
        raise ValueError("All replication allocations must be reconciled")
    records = {
        name: load_run(spec, args.artifacts, allocations, INITIAL_2027)
        for name, spec in read_json(args.manifest).items()
    }
    if set(records) != {"C-2027", "D-2027"}:
        raise ValueError("Expected both matched seed-2027 arms")
    historical = read_json(args.historical)["runs"]
    repo = PdbRepository(args.data)
    families = {
        task.parent_id: oracle.provenance["family_id"]
        for split in repo.partition_names()
        for task, oracle in repo.tasks(split)
    }
    rows = {"Initial": historical["B-2027"]["outcomes"]["0"]}
    for arm in ("control", "A", "B"):
        rows["Control" if arm == "control" else arm] = historical[f"{arm}-2027"]["outcomes"]["512"]
    for arm in ("C", "D"):
        rows[arm] = records[f"{arm}-2027"]["outcomes"]["512"]
    comparisons = contrasts(rows, families)
    endpoints = {
        arm: {
            split: {"successes": sum(t["success"] for t in tasks), "count": len(tasks)}
            for split, tasks in outcomes.items()
        }
        for arm, outcomes in rows.items()
    }
    if any(row[PRIMARY]["count"] != 64 for row in endpoints.values()):
        raise ValueError("Primary benchmark must contain all 64 tasks")
    args.output.mkdir(parents=True, exist_ok=True)
    primary_figure(args.output, rows, comparisons, 512, seed=2027)
    prior = read_json(args.prior)
    report = {
        "seed2027": endpoints,
        "paired_2027": comparisons,
        "seed2026": prior["endpoints"],
        "paired_2026": prior["paired_2026"],
        "alignment": {
            name: alignment_summary(row["paired_batches"]) for name, row in records.items()
        },
        "replication_h200_hours": sum(row["charged"] for row in allocations),
        "training_h200_hours": {name: row["gpu_hours"] for name, row in records.items()},
        "end_to_end_h200_hours": {
            name: row["end_to_end_h200_hours"] for name, row in records.items()
        },
        "unreconciled_reservations": 0,
    }
    atomic_json(args.output / "summary.json", report)


if __name__ == "__main__":
    main()
