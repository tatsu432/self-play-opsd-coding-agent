"""Analyze comparable Task 20 records and plot measured checkpoints only."""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze_task19 import allocations_before, paired

from selfplay_opsd.core import read_json
from selfplay_opsd.data import TaskRepository
from selfplay_opsd.storage import CheckpointManager, atomic_json, file_hash
from selfplay_opsd.task20 import Task20Control


def analyze(
    root: Path, control: Task20Control, data: TaskRepository, allocations: list[dict]
) -> dict:
    marker = read_json(root / "complete-main.json")
    identity = marker["identity"]
    checkpoint = root / "checkpoints" / marker["checkpoint"]
    if (
        identity["data"] != data.hash
        or identity["science"] != control.science
        or marker["episodes"] != control.endpoint
        or file_hash(checkpoint / "complete.json") != marker["checkpoint_sha256"]
    ):
        raise ValueError("Main completion does not match the frozen comparison")
    metadata = CheckpointManager.verify(checkpoint)
    if metadata["identity"] != identity or metadata["snapshot"] != marker["snapshot"]:
        raise ValueError("Checkpoint metadata mismatch")
    metrics = [read_json(path) for path in sorted((root / "iterations").glob("*.json"))]
    by_episodes = {row["episodes"]: row for row in metrics}
    curves, outcomes = [], {}
    for path in sorted((root / "evaluation").glob("*/*/summary.json")):
        summary = read_json(path)
        exposure, split = summary["episodes"], summary["split"]
        if (
            summary["identity"] != identity
            or summary["snapshot"] != by_episodes[exposure]["snapshot"]
        ):
            raise ValueError("Evaluation snapshot differs from the checkpoint")
        expected = {task.task_id for task, _ in data.tasks(split)}
        actual = [row["task_id"] for row in summary["tasks"]]
        if len(set(actual)) != len(actual) or set(actual) != expected:
            raise ValueError("Evaluation coverage differs from the frozen tasks")
        if split == "dev_monitor":
            training = by_episodes[exposure]["metrics"]
            curves.append(
                {
                    "episodes": exposure,
                    "success": summary["verified"] / summary["count"],
                    "tokens": training.get("student_tokens", 0),
                    "h200_hours": allocations_before(allocations, summary["recorded_at"]),
                }
            )
        else:
            outcomes.setdefault(str(exposure), {})[split] = summary["tasks"]
    return {
        "identity": identity,
        "curves": curves,
        "outcomes": outcomes,
        "training": [row["metrics"] for row in metrics if row["batch"] is not None],
        "gpu_hours": sum(row["charged"] for row in allocations),
    }


def figures(output: Path, result: dict) -> None:
    arms = result["arms"]
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.4), layout="constrained")
    for arm, record in arms.items():
        for axis, x, label in zip(
            axes,
            ("episodes", "tokens", "h200_hours"),
            ("Attempted training episodes", "Sampled training tokens", "Allocated H200-hours"),
            strict=True,
        ):
            axis.plot(
                [row[x] for row in record["curves"]],
                [row["success"] for row in record["curves"]],
                marker="o",
                label=arm,
            )
            axis.set(xlabel=label, ylabel="Verified development repair rate", ylim=(0, 1))
    axes[0].legend()
    figure.savefig(output / "learning-and-efficiency.pdf")
    plt.close(figure)
    figure, axes = plt.subplots(2, 3, figsize=(11, 6.6), layout="constrained")
    for arm, record in arms.items():
        rows = record["training"]
        episodes = [row["attempted_episodes"] for row in rows]
        series = (
            ("Mixed-reward fraction", [row["learning"]["mixed_reward_fraction"] for row in rows]),
            ("Effective solver updates", [row["learning"]["effective_rl_updates"] for row in rows]),
            ("Adapter update norm", [row["shared"]["update_norm"] for row in rows]),
            ("Mean OPSD gate", [row["opsd"]["mean_gate"] for row in rows]),
            (
                "Valid-only distinct reward groups",
                [row["proposer"]["distinct_reward_groups"] for row in rows],
            ),
            ("Rollout/rescore log-prob drift", [row["rl"]["behavior_drift"] for row in rows]),
        )
        for axis, (title, values) in zip(axes.flat, series, strict=True):
            axis.plot(episodes, values, label=arm)
            axis.set(xlabel="Attempted training episodes", ylabel=title)
    axes[0, 0].legend()
    figure.savefig(output / "training-signal.pdf")
    plt.close(figure)
    final = result["final"]
    splits = sorted({split for arm in final.values() for split in arm})
    if not splits:
        return
    figure, axes = plt.subplots(1, len(splits), figsize=(4.8 * len(splits), 3.4), squeeze=False)
    for axis, split in zip(axes[0], splits, strict=True):
        values = {arm: row[split] for arm, row in final.items() if split in row}
        axis.bar(values.keys(), [row["verified"] / row["count"] for row in values.values()])
        axis.set(title=split.replace("_", " "), ylabel="Verified repair rate", ylim=(0, 1))
    figure.tight_layout()
    figure.savefig(output / "generalization.pdf")
    plt.close(figure)
    figure, axes = plt.subplots(1, len(splits), figsize=(5.2 * len(splits), 3.8), squeeze=False)
    for axis, split in zip(axes[0], splits, strict=True):
        values = {arm: row[split] for arm, row in final.items() if split in row}
        categories = sorted(
            {category for row in values.values() for category in row["failures"]} - {"verified"}
        )
        if not categories:
            axis.set(title=split.replace("_", " "), ylabel="Failure rate", ylim=(0, 1))
            continue
        totals = [0.0] * len(values)
        for category in categories:
            rates = [row["failures"].get(category, 0) / row["count"] for row in values.values()]
            axis.bar(values.keys(), rates, bottom=totals, label=category.replace("_", " "))
            totals = [a + b for a, b in zip(totals, rates, strict=True)]
        axis.set(title=split.replace("_", " "), ylabel="Failure rate", ylim=(0, 1))
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "verification-failures.pdf")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--budget", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = TaskRepository(args.data)
    control = Task20Control.read(args.control)
    specs = read_json(args.manifest)
    budget = read_json(args.budget)["reservations"]
    arms = {}
    for arm, item in specs.items():
        controller_runs = set(item["controller_runs"])
        allocations = [row for row in budget.values() if row["run_id"] in controller_runs]
        if {row["run_id"] for row in allocations} != controller_runs or any(
            row.get("charged") is None for row in allocations
        ):
            raise ValueError("Run allocation is missing or unreconciled")
        arms[arm] = analyze(Path(item["path"]), control, data, allocations)
        if arms[arm]["identity"]["arm"] != arm:
            raise ValueError("Analysis arm does not match its checkpoint")
    if not arms or len({row["identity"]["initial"] for row in arms.values()}) != 1:
        raise ValueError("Main arms must share one initial policy")
    families = {
        parent.parent_id: parent.family_id
        for split in ("final_id", "external")
        for parent in data.parents(split)
    }
    endpoint = str(control.endpoint)
    comparisons = {}
    for old, new in combinations(arms, 2):
        a = arms[old]["outcomes"].get(endpoint, {})
        b = arms[new]["outcomes"].get(endpoint, {})
        comparisons[f"{new}-{old}"] = {
            split: paired(b[split], a[split], families) for split in sorted(a.keys() & b.keys())
        }
    base = arms.get("control", {}).get("outcomes", {}).get("0", {})
    if base:
        for arm, record in arms.items():
            scored = record["outcomes"].get(endpoint, {})
            comparisons[f"{arm}-base"] = {
                split: paired(scored[split], base[split], families)
                for split in sorted(scored.keys() & base.keys())
            }
    final = {
        arm: {
            split: {
                "verified": sum(row["success"] for row in rows),
                "count": len(rows),
                "failures": dict(Counter(row["failure"] for row in rows)),
            }
            for split, rows in record["outcomes"].get(endpoint, {}).items()
        }
        for arm, record in arms.items()
    }
    result = {
        "arms": arms,
        "endpoint": control.endpoint,
        "paired": comparisons,
        "final": final,
        "missing_main_arms": sorted({"control", "A", "B"} - arms.keys()),
    }
    atomic_json(args.output / "results.json", result)
    figures(args.output, result)


if __name__ == "__main__":
    main()
