"""Summarize reconciled Task 21 benchmark, training, and paired evaluation evidence."""

from __future__ import annotations

import argparse
import os
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from analyze_task19 import allocations_before, paired
from analyze_task20 import figures

from selfplay_opsd.core import read_json
from selfplay_opsd.storage import atomic_json
from selfplay_opsd.task21 import PdbRepository


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Cannot score an empty measured stage")
    return {
        "verified": sum(row["success"] for row in rows),
        "count": len(rows),
        "public": sum(row["public_pass"] for row in rows),
        "hidden": sum(row["hidden_pass"] for row in rows),
        "accepted_edits": sum(row["accepted_edits"] for row in rows),
        "rejected_edits": sum(row["rejected_edits"] for row in rows),
        "mean_turns": statistics.mean(row["turns"] for row in rows),
        "median_turns": statistics.median(row["turns"] for row in rows),
        "tokens": sum(row["tokens"] for row in rows),
        "failures": dict(Counter(row["failure"] for row in rows)),
    }


def stage_evidence(spec: dict, reservations: dict) -> dict:
    path = Path(spec["path"])
    stage = spec["stage"]
    marker = read_json(path / f"complete-{stage}.json")
    identity = marker["identity"]
    if spec["identity"] != identity or spec["checkpoint_sha256"] != marker["checkpoint_sha256"]:
        raise ValueError("Local records differ from scheduler-verified endpoint identity")
    owners = set(spec["controller_runs"])
    allocations = [
        row
        for row in reservations.values()
        if row["run_id"] in owners and row["operation"] != "task21-evaluate"
    ]
    if {row["run_id"] for row in allocations} != owners or any(
        row.get("charged") is None for row in allocations
    ):
        raise ValueError("Run has missing or unreconciled training allocation")
    iterations = [read_json(p) for p in sorted((path / "iterations").glob("*.json"))]
    snapshots = {row["episodes"]: row["snapshot"] for row in iterations}
    curves, outcomes = [], {}
    for file in sorted((path / "evaluation").glob("*/*/summary.json")):
        summary = read_json(file)
        exposure, split = summary["episodes"], summary["split"]
        if summary["identity"] != identity or summary["snapshot"] != snapshots.get(exposure):
            raise ValueError("Evaluation summary differs from its checkpoint")
        rows = summary["tasks"]
        if (
            len(rows) != summary["count"]
            or sum(row["success"] for row in rows) != summary["verified"]
        ):
            raise ValueError("Evaluation summary counts differ from task records")
        if split == "dev_monitor":
            curves.append(
                {
                    "episodes": exposure,
                    "success": summary["verified"] / summary["count"],
                    "tokens": next(
                        (
                            row["metrics"].get("student_tokens", 0)
                            for row in iterations
                            if row["episodes"] == exposure
                        ),
                        0,
                    ),
                    "h200_hours": allocations_before(allocations, summary["recorded_at"]),
                }
            )
        else:
            outcomes.setdefault(str(exposure), {})[split] = rows
    return {
        "identity": identity,
        "endpoint": marker["episodes"],
        "curves": curves,
        "outcomes": outcomes,
        "training": [row["metrics"] for row in iterations if row["batch"] is not None],
        "gpu_hours": sum(row["charged"] for row in allocations),
        "result": marker["result"],
    }


def plot_difficulty(path: Path, output: Path) -> None:
    samples = read_json(path)["rows"]
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in samples:
        groups[row["task_id"]].append(row)
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.3), layout="constrained")
    for size, axis in zip((4, 8), axes, strict=True):
        outcomes = Counter(
            sum(row["success"] for row in rows if row["slot"] < size) for rows in groups.values()
        )
        axis.bar(range(size + 1), [outcomes[n] for n in range(size + 1)], color="#356999")
        axis.set(
            xlabel=f"Verified repairs per group of {size}",
            ylabel="Training parent count",
            xticks=range(size + 1),
            title=f"{len(groups)} matched parents",
        )
    figure.savefig(output / "difficulty.pdf")
    plt.close(figure)


def analyze(args) -> None:
    repo = PdbRepository(args.data)
    manifest = read_json(args.manifest)
    budget = read_json(args.budget)
    families = {
        task.parent_id: repo.parent_oracle(task.parent_id).provenance["family_id"]
        for split in repo.partition_names()
        for task, _ in repo.tasks(split)
    }
    args.output.mkdir(parents=True, exist_ok=True)
    if args.screen:
        plot_difficulty(args.screen, args.output)
    records = {
        spec["label"]: stage_evidence(spec, budget["reservations"]) for spec in manifest["runs"]
    }
    by_seed = defaultdict(dict)
    for spec in manifest["runs"]:
        if spec["stage"] == "main":
            by_seed[str(spec["seed"])][spec["label"]] = records[spec["label"]]
    contrasts = {}
    endpoints = {}
    common_compute = {}
    for seed, arms in sorted(by_seed.items()):
        if (
            len(
                {
                    (row["identity"]["initial"], row["identity"]["data"], row["endpoint"])
                    for row in arms.values()
                }
            )
            != 1
        ):
            raise ValueError("Within-seed matched arms differ in initialization, data, or exposure")
        endpoints[seed] = {}
        contrasts[seed] = {}
        if all(record["curves"] for record in arms.values()):
            shared_hours = min(
                max(row["h200_hours"] for row in record["curves"]) for record in arms.values()
            )
            common_compute[seed] = {
                "h200_hours_limit": shared_hours,
                "last_observed": {
                    arm: max(
                        (row for row in record["curves"] if row["h200_hours"] <= shared_hours),
                        key=lambda row: row["h200_hours"],
                        default=None,
                    )
                    for arm, record in arms.items()
                },
            }
        for arm, record in arms.items():
            end = record["outcomes"].get(str(record["endpoint"]), {})
            endpoints[seed][arm] = {split: summarize(rows) for split, rows in end.items()}
            baseline = record["outcomes"].get("0", {})
            contrasts[seed][f"{arm}-base"] = {
                split: paired(end[split], baseline[split], families)
                for split in end.keys() & baseline.keys()
            }
        for old, new in combinations(arms, 2):
            before = arms[old]["outcomes"].get(str(arms[old]["endpoint"]), {})
            after = arms[new]["outcomes"].get(str(arms[new]["endpoint"]), {})
            contrasts[seed][f"{new}-{old}"] = {
                split: paired(after[split], before[split], families)
                for split in after.keys() & before.keys()
            }
        directory = args.output / f"seed-{seed}"
        directory.mkdir(exist_ok=True)
        figures(directory, {"arms": arms, "final": endpoints[seed]})
    pilots = {
        spec["label"]: records[spec["label"]]
        for spec in manifest["runs"]
        if spec["stage"] in {"solver", "shared-pilot"}
    }
    if pilots:
        directory = args.output / "learning-signal"
        directory.mkdir(exist_ok=True)
        figures(directory, {"arms": pilots, "final": {}})
    charged = [row["charged"] for row in budget["reservations"].values()]
    atomic_json(
        args.output / "results.json",
        {
            "runs": records,
            "endpoint_scores": endpoints,
            "paired_differences_within_seed": contrasts,
            "common_compute": common_compute,
            "allocated_h200_hours": sum(value for value in charged if value is not None),
            "unreconciled_allocations": sum(value is None for value in charged),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data", "manifest", "budget", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--screen", type=Path)
    os.umask(0o077)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
