"""Analyze the frozen Task 19 common endpoint and actual Slurm accounting."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from selfplay_opsd.core import read_json
from selfplay_opsd.data import TaskRepository
from selfplay_opsd.storage import CheckpointManager, atomic_json, file_hash
from selfplay_opsd.task19 import ARMS, PILOT, Task19Control


def evaluated_milestones(control: Task19Control) -> list[int]:
    """Exclude milestones skipped when the completed pilot reached 32 episodes."""
    return [count for count in control.milestones() if count == 0 or count >= PILOT]


SPLITS = ("final_id", "external")
LABELS = {"final_id": "MBPP+ ID", "external": "HumanEval+ external"}
COLORS = dict(
    zip(
        ("base", *ARMS),
        ("#69767e", "#386cb0", "#7b5c9e", "#008b82", "#c46d31", "#ab4566"),
        strict=True,
    )
)
FAILURES = (
    "verified",
    "public_pass_hidden_fail",
    "semantic_or_runtime_after_edit",
    "truncation",
    "rejected_edit",
    "invalid_tool",
    "no_edit",
)


def interval(values: dict[str, list[float]]) -> list[float]:
    rng = np.random.default_rng(2026)
    groups = [np.array(values[key], dtype=float) for key in sorted(values)]
    counts = np.array([len(row) for row in groups])
    totals = np.array([row.sum() for row in groups])
    indices = rng.integers(0, len(groups), size=(2000, len(groups)))
    means = totals[indices].sum(axis=1) / counts[indices].sum(axis=1)
    return np.percentile(means, [2.5, 97.5]).tolist()


def paired(new: list[dict], old: list[dict], families: dict[str, str]) -> dict:
    prior = {row["task_id"]: row for row in old}
    current = {row["task_id"]: row for row in new}
    if len(prior) != len(old) or len(current) != len(new) or current.keys() != prior.keys():
        raise ValueError("Paired evaluation task coverage differs")
    differences: dict[str, list[float]] = defaultdict(list)
    for task, row in current.items():
        if prior[task]["parent_id"] != row["parent_id"]:
            raise ValueError("Paired task parent differs")
        group = families[row["parent_id"]]
        differences[group].append(100 * (int(row["success"]) - int(prior[task]["success"])))
    return {
        "difference_pp": sum(sum(x) for x in differences.values()) / len(current),
        "ci95_pp": interval(differences),
        "tasks": len(current),
        "families": len(differences),
    }


def allocations_before(items: list[dict], timestamp: float) -> float:
    total = 0.0
    for item in items:
        hours = item["charged"]
        if not hours:
            continue
        started = datetime.fromisoformat(item["allocation_start"])
        if started.tzinfo is None:
            started = started.replace(tzinfo=ZoneInfo("America/New_York"))
        total += min(hours, max(0.0, (timestamp - started.timestamp()) / 3600))
    return total


def outcomes(root, arm, endpoint, expected, repository, science):
    exposure = 0 if arm == "base" else endpoint
    complete = root / ("complete-base.json" if arm == "base" else f"complete-{endpoint}.json")
    marker = read_json(complete)
    if marker["identity"]["data"] != repository.hash or marker["identity"]["science"] != science:
        raise ValueError("Evaluation identity does not match the common freeze")
    if arm != "base":
        checkpoint = root / "checkpoints" / marker["checkpoint"]
        if file_hash(checkpoint / "complete.json") != marker["checkpoint_sha256"]:
            raise ValueError("Endpoint checkpoint checksum differs")
        metadata = CheckpointManager.verify(checkpoint)
        if metadata["snapshot"] != marker["snapshot"] or metadata["total_episodes"] != endpoint:
            raise ValueError("Endpoint checkpoint scientific exposure differs")
    result = {}
    for split in SPLITS:
        summary = read_json(root / "evaluation" / f"{exposure:04d}" / split / "summary.json")
        if summary["identity"] != marker["identity"] or summary["snapshot"] != marker["snapshot"]:
            raise ValueError("Final evaluation snapshot differs from the designated checkpoint")
        rows = summary["tasks"]
        if len(rows) != len(expected[split]) or {row["task_id"] for row in rows} != expected[split]:
            raise ValueError("Final evaluation coverage is incomplete")
        result[split] = rows
    return result, marker["identity"]


def figures(output, table, comparisons, curves, training, final):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout="constrained", sharey=True)
    for axis, key, xlabel in zip(
        axes,
        ("episodes", "tokens", "allocated_hours"),
        ("Attempted episodes", "Generated student tokens", "Allocated H200-hours"),
        strict=True,
    ):
        for arm in ARMS:
            rows = curves[arm]
            axis.plot(
                [r[key] for r in rows],
                [r["repair_rate"] for r in rows],
                marker="o",
                label=arm,
                color=COLORS[arm],
            )
        axis.set_xlabel(xlabel)
        axis.set_ylim(0, 1)
        axis.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_ylabel("Development repair success")
    axes[-1].legend(frameon=False)
    save(figure, output, "01-learning-curves")

    figure, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    for axis, split in zip(axes, SPLITS, strict=True):
        for index, arm in enumerate(("base", *ARMS)):
            row = table[arm][split]
            point = row["rate"]
            lo, hi = row["ci95"]
            axis.errorbar(
                index,
                point,
                yerr=[[max(0, point - lo)], [max(0, hi - point)]],
                fmt="o",
                color=COLORS[arm],
                capsize=3,
            )
            if arm != "base":
                delta = comparisons[f"{arm}_minus_base"][split]["difference_pp"]
                axis.annotate(
                    f"{delta:+.0f} pp",
                    (index, hi),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )
        axis.set_xticks(range(6), ("Base", *ARMS))
        axis.set_title(LABELS[split])
        axis.set_ylabel("Verified repair rate")
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.set_ylim(0, 1.04)
    save(figure, output, "02-generalization")

    figure, axes = plt.subplots(2, 2, figsize=(9, 6), layout="constrained")
    for arm in ("control", "A", "B"):
        rows = training[arm]
        episodes = [row["attempted_episodes"] for row in rows]
        values = [
            [r["opsd"]["rl_cosine"] for r in rows],
            [r["opsd"]["mean_gap"] for r in rows],
            [r["opsd"]["mean_gate"] for r in rows],
            [r["student"]["reasoning_tokens"] / max(1, r["student"]["tool_tokens"]) for r in rows],
        ]
        for axis, y in zip(axes.flat, values, strict=True):
            axis.plot(
                episodes,
                [np.nan if value is None else value for value in y],
                label=arm,
                color=COLORS[arm],
                alpha=0.85,
            )
    for axis, title in zip(
        axes.flat,
        (
            "RL/OPSD gradient cosine",
            "Mean signed teacher gap",
            "Mean gate weight",
            "Reasoning tokens per tool token",
        ),
        strict=True,
    ):
        axis.set_title(title)
        axis.set_xlabel("Attempted episodes")
    axes[1, 1].legend(frameon=False)
    save(figure, output, "03-opsd")

    figure, axes = plt.subplots(2, 2, figsize=(9, 6), layout="constrained")
    for arm in ("B", "C", "D"):
        rows = training[arm]
        episodes = [row["attempted_episodes"] for row in rows]
        values = [
            [
                x["proposer"]["distinct_reward_groups"] / max(1, x["proposer"]["groups"])
                for x in rows
            ],
            [x["proposer"]["valid"] / max(1, x["proposer"]["attempted"]) for x in rows],
            [x["proposer"]["mean_alignment"] for x in rows],
            [x["proposer"]["mean_solve_rate"] for x in rows],
        ]
        for axis, y in zip(axes.flat, values, strict=True):
            axis.plot(
                episodes, [np.nan if v is None else v for v in y], label=arm, color=COLORS[arm]
            )
    for axis, title in zip(
        axes.flat,
        (
            "Groups with distinct valid rewards",
            "Valid proposals",
            "Mean defined alignment cosine",
            "Mean student solve rate",
        ),
        strict=True,
    ):
        axis.set_title(title)
        axis.set_xlabel("Attempted episodes")
    axes[1, 1].legend(frameon=False)
    save(figure, output, "04-proposer")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    palette = ("#007d71", "#8bb171", "#496c91", "#b69bb9", "#d59266", "#994d5b", "#bdbdbd")
    for axis, split in zip(axes, SPLITS, strict=True):
        arms = ["base", *ARMS]
        counts = [Counter(row["failure"] for row in final[arm][split]) for arm in arms]
        totals = np.array([sum(c.values()) for c in counts])
        bottom = np.zeros(len(arms))
        for kind, color in zip(FAILURES, palette, strict=True):
            heights = np.array([c[kind] for c in counts]) / totals
            axis.bar(
                range(len(arms)), heights, bottom=bottom, label=kind.replace("_", " "), color=color
            )
            bottom += heights
        axis.set_title(LABELS[split])
        axis.set_xticks(range(len(arms)), ("Base", *ARMS))
        axis.yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    save(figure, output, "05-failure-taxonomy")


def save(figure, output, name):
    for kind in ("pdf", "png"):
        figure.savefig(output / f"{name}.{kind}", dpi=180)
    plt.close(figure)


def complexity_summary(rows):
    result = {}
    for field in ("source_lines", "specification_chars", "ast_nodes"):
        ordered = sorted(rows, key=lambda row: (row[field], row["task_id"]))
        size = len(ordered)
        result[field] = [
            {
                "min": group[0][field],
                "max": group[-1][field],
                "tasks": len(group),
                "verified": sum(row["success"] for row in group),
            }
            for index in range(3)
            if (group := ordered[index * size // 3 : (index + 1) * size // 3])
        ]
    return result


def alignment_followup(rows, curve):
    samples = []
    for previous, next_check in zip(curve[:-1], curve[1:], strict=True):
        alignments = [
            row["proposer"]["mean_alignment"]
            for row in rows
            if previous["episodes"] < row["attempted_episodes"] <= next_check["episodes"]
            and row["proposer"]["mean_alignment"] is not None
        ]
        if alignments:
            samples.append(
                {
                    "next_milestone": next_check["episodes"],
                    "mean_alignment": float(np.mean(alignments)),
                    "development_change_pp": 100
                    * (next_check["repair_rate"] - previous["repair_rate"]),
                }
            )
    x, y = (
        [sample[name] for sample in samples] for name in ("mean_alignment", "development_change_pp")
    )
    corr = float(np.corrcoef(x, y)[0, 1]) if len(x) >= 3 and np.std(x) and np.std(y) else None
    return {"milestone_intervals": samples, "descriptive_correlation": corr}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/protocol-v3"))
    parser.add_argument(
        "--manifest", type=Path, required=True, help="Local artifact directories and controller IDs"
    )
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--budget", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = Task19Control.read(args.control)
    if cfg.endpoint is None:
        raise ValueError("Analysis requires the frozen main endpoint")
    dataset = TaskRepository(args.data)
    expected = {split: {task.task_id for task, _ in dataset.tasks(split)} for split in SPLITS}
    families = {
        parent.parent_id: parent.family_id or parent.parent_id
        for split in SPLITS
        for parent in dataset.parents(split)
    }
    manifest = read_json(args.manifest)
    budget = read_json(args.budget)
    allocations = list(budget["reservations"].values())
    if any(row["charged"] is None for row in allocations):
        raise ValueError("Final analysis requires reconciled Slurm charges")
    if set(manifest) != {"base", *ARMS}:
        raise ValueError("Manifest requires the base and all five arms")
    final, identities, curves, training, table = {}, {}, {}, {}, {}
    for arm, record in manifest.items():
        root = Path(record["path"])
        final[arm], identities[arm] = outcomes(
            root, arm, cfg.endpoint, expected, dataset, cfg.science
        )
        items = [row for row in allocations if row["run_id"] in record["controller_runs"]]
        if not items:
            raise ValueError(f"No accounted H200 allocation for {arm}")
        endpoint_hours = sum(row["charged"] for row in items)
        table[arm] = {"allocated_hours": endpoint_hours, "wandb": None}
        if (root / "telemetry.json").is_file():
            table[arm]["wandb"] = read_json(root / "telemetry.json").get("run_url")
        for split, rows in final[arm].items():
            sampled: dict[str, list[float]] = defaultdict(list)
            for row in rows:
                sampled[families[row["parent_id"]]].append(float(row["success"]))
            n = len(rows)
            solved = sum(row["success"] for row in rows)
            table[arm][split] = {
                "successes": solved,
                "tasks": n,
                "families": len(sampled),
                "rate": solved / n,
                "ci95": interval(sampled),
                "generated_tokens": sum(row["tokens"] for row in rows),
                "inference_seconds": sum(row["timings"]["generation"] for row in rows),
                "verification_seconds": sum(row["timings"]["verifier"] for row in rows),
                "failure_counts": dict(Counter(row["failure"] for row in rows)),
                "complexity": complexity_summary(rows),
            }
        if arm == "base":
            continue
        committed = [read_json(path) for path in sorted((root / "iterations").glob("*.json"))]
        training[arm] = [row["metrics"] for row in committed if row["metrics"]]
        by_exposure = {
            row["episodes"]: row["metrics"].get("student_tokens", 0) for row in committed
        }
        curves[arm] = []
        for count in evaluated_milestones(cfg):
            summary = read_json(root / "evaluation" / f"{count:04d}" / "dev_monitor/summary.json")
            curves[arm].append(
                {
                    "episodes": count,
                    "tokens": by_exposure[count],
                    "repair_rate": summary["verified"] / summary["count"],
                    "allocated_hours": allocations_before(items, summary["recorded_at"]),
                }
            )
        table[arm]["training_verified_repairs_per_allocated_hour"] = (
            sum(row["student"]["verified"] for row in training[arm]) / endpoint_hours
        )
    shared = ("study", "science", "data", "execution", "initial", "worker")
    if any(identities[arm][key] != identities["base"][key] for arm in ARMS for key in shared):
        raise ValueError("Arm initialization or common identity differs from base")
    comparisons = {
        f"{new}_minus_{old}": {
            split: paired(final[new][split], final[old][split], families) for split in SPLITS
        }
        for new, old in (
            *((arm, "base") for arm in ARMS),
            ("A", "control"),
            ("B", "control"),
            ("B", "A"),
            ("C", "B"),
            ("D", "C"),
        )
    }
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(
        args.output / "results.json",
        {
            "seed": cfg.seed,
            "endpoint": cfg.endpoint,
            "interval_unit": "parent family conditional on one observed training run",
            "data_hash": dataset.hash,
            "table": table,
            "comparisons": comparisons,
            "learning_curves": curves,
            "alignment_followup": {
                arm: alignment_followup(training[arm], curves[arm]) for arm in ("C", "D")
            },
            "gpu_hours": sum(row["charged"] for row in allocations),
            "identity": identities["base"],
        },
    )
    with (args.output / "benchmark.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "arm",
                "split",
                "repaired",
                "tasks",
                "rate",
                "ci_low",
                "ci_high",
                "tokens",
                "run_h200_hours",
            )
        )
        for arm, values in table.items():
            for split in SPLITS:
                row = values[split]
                writer.writerow(
                    (
                        arm,
                        split,
                        row["successes"],
                        row["tasks"],
                        row["rate"],
                        *row["ci95"],
                        row["generated_tokens"],
                        values["allocated_hours"],
                    )
                )
    figures(args.output, table, comparisons, curves, training, final)


if __name__ == "__main__":
    main()
