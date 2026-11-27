"""Analyze verified Task 21/22 runs and export aggregate-only presentation figures."""

from __future__ import annotations

import argparse
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from analyze_task19 import allocations_before, paired
from scipy.stats import spearmanr

from selfplay_opsd.core import read_json
from selfplay_opsd.storage import atomic_json, file_hash
from selfplay_opsd.task21 import PdbRepository
from selfplay_opsd.task22_contract import INITIAL, MATCHED_DATA

METHODS = ("Initial", "Control", "A", "B", "C", "D")
COLORS = dict(
    zip(METHODS, ("#717985", "#3d5166", "#4896a3", "#167367", "#957135", "#ad523b"), strict=True)
)
CONTRASTS = (("A", "Control"), ("B", "Control"), ("B", "A"), ("C", "B"), ("D", "C"), ("D", "B"))
PRIMARY = "primary_final"


def evidence(receipt: Path, artifacts: Path) -> tuple[dict, Path]:
    row = read_json(receipt)
    root = artifacts / row["run_id"]
    if row["scheduler"]["state"] != "COMPLETED":
        raise ValueError("Analysis requires a completed controller-verified receipt")
    marker = next(
        (
            read_json(path)
            for path in root.glob("complete-*.json")
            if (record := read_json(path))["result"] == row["result"]
            and record["checkpoint_sha256"] == row["checkpoint_sha256"]
        ),
        None,
    )
    if marker is None or any(row.get(key) != value for key, value in marker.items()):
        raise ValueError("Fetched completion marker differs from the controller receipt")
    return row, root


def load_run(spec: dict, artifacts: Path, reservations: list[dict], initial: str = INITIAL) -> dict:
    evaluation_ids = set()
    outcomes: dict[str, dict] = {}
    identities = set()
    for receipt in spec["evaluations"]:
        record, root = evidence(Path(receipt), artifacts)
        identity, exposure = record["identity"], record["episodes"]
        if identity["data"] != MATCHED_DATA or identity["initial"] != initial:
            raise ValueError("Evaluation is not from the matched PDB study")
        identities.add(identity["science"])
        evaluation_ids.add(record["run_id"])
        for key, result in record["result"].items():
            summary = read_json(root / f"evaluation/{exposure:04d}/{key}/summary.json")
            rows = summary["tasks"]
            if (
                summary["identity"] != identity
                or summary["snapshot"] != record["snapshot"]
                or summary["episodes"] != exposure
                or summary["split"] != key
                or len(rows) != result["count"]
                or sum(row["success"] for row in rows) != result["verified"]
            ):
                raise ValueError("Frozen evaluation receipt differs from the fetched task records")
            previous = outcomes.setdefault(str(exposure), {}).setdefault(key, rows)
            if previous != rows:
                raise ValueError("Separate receipts disagree on the same evaluation")
    if len(identities) != 1:
        raise ValueError("Expected one matched science identity per arm")
    train_ids = set(spec["training_run_ids"])
    train = [row for row in reservations if row["run_id"] in train_ids]
    related = [row for row in reservations if row["run_id"] in train_ids | evaluation_ids]
    if (
        any(row["charged"] is None for row in related)
        or {row["run_id"] for row in related} != train_ids | evaluation_ids
    ):
        raise ValueError("Task 22 compute must be reconciled before final analysis")
    iterations = [read_json(path) for path in sorted((root / "iterations").glob("*.json"))]
    metrics, paired_batches, steps_at = [], [], {}
    for iteration in iterations:
        batch = iteration["batch"]
        if batch:
            metrics.append(iteration["metrics"])
            path = root / "batches" / batch["path"]
            if path.is_file():
                if file_hash(path) != batch["sha256"]:
                    raise ValueError("Committed trajectory checksum differs")
                grouped = read_json(path)["proposals"]
                groups = [
                    {
                        row["task"]: row["alignment"]["paired"]
                        for row in group
                        if row["task"] is not None
                        and row["alignment"]
                        and "paired" in row["alignment"]
                    }
                    for group in grouped
                ]
                if any(groups):
                    paired_batches.append(groups)
        steps_at[iteration["episodes"]] = sum(row["shared"]["optimizer_step"] for row in metrics)
    curves = []
    for path in sorted((root / "evaluation").glob("*/dev_monitor/summary.json")):
        row = read_json(path)
        exposure = row["episodes"]
        if row["identity"]["science"] not in identities:
            raise ValueError("Development monitor identity differs")
        iteration = next(
            record
            for record in iterations
            if record["episodes"] == exposure and record["snapshot"] == row["snapshot"]
        )
        curves.append(
            {
                "episodes": exposure,
                "tokens": iteration["metrics"].get("student_tokens", 0),
                "optimizer_steps": steps_at[exposure],
                "h200_hours": allocations_before(train, row["recorded_at"]) if exposure else 0,
                "success": row["verified"] / row["count"],
            }
        )
    return {
        "curves": curves,
        "training": metrics,
        "outcomes": outcomes,
        "paired_batches": paired_batches,
        "gpu_hours": sum(row["charged"] for row in train),
        "end_to_end_h200_hours": sum(row["charged"] for row in related),
    }


def rank_correlation(left: list[float], right: list[float]) -> float | None:
    if len(set(left)) < 2 or len(set(right)) < 2:
        return None
    value = float(spearmanr(left, right).statistic)
    return value if math.isfinite(value) else None


def alignment_summary(batches: list[list[dict]]) -> dict:
    methods = ("difficulty", "C", "D")
    groups = [group for batch in batches for group in batch if group]

    def scores(task: dict) -> list[float]:
        return [task["difficulty"], task["C"]["reward"], task["D"]["reward"]]

    by_success = {str(n): {method: [] for method in methods} for n in range(5)}
    counts = {arm: Counter() for arm in ("C", "D")}
    within, ranking = defaultdict(list), defaultdict(list)
    changes = []
    for batch_number, batch in enumerate(batches):
        # One observation per scored task per frozen batch snapshot.
        tasks = {key: value for group in batch for key, value in group.items()}
        for task_number, task in enumerate(tasks.values()):
            for method, value in zip(methods, scores(task), strict=True):
                by_success[str(task["successes"])][method].append(value)
            for arm in counts:
                counts[arm]["near_zero"] += int(task[arm]["near_zero"])
                counts[arm]["informative"] += int(task[arm]["cosine"] is not None)
            changes.append(
                {
                    "batch": batch_number,
                    "task_position": task_number,
                    "successes": task["successes"],
                    "C": task["C"]["reward"],
                    "D": task["D"]["reward"],
                    "absolute_change": abs(task["D"]["reward"] - task["C"]["reward"]),
                    "rl_norm": task["rl_norm"],
                    "weighted_opsd_norm": task["weighted_opsd_norm"],
                }
            )
    for group in groups:
        values = list(zip(*(scores(task) for task in group.values()), strict=True))
        for method, column in zip(methods, values, strict=True):
            within[method].append(
                {
                    "valid_tasks": len(column),
                    "distinct": len(set(column)),
                    "variance": statistics.pvariance(column),
                }
            )
        for i, j in ((0, 1), (0, 2), (1, 2)):
            value = rank_correlation(list(values[i]), list(values[j]))
            if value is not None:
                ranking[f"{methods[i]} vs {methods[j]}"].append(value)
    return {
        "groups": len(groups),
        "tasks": len(changes),
        "counts": counts,
        "rewards_by_success": by_success,
        "within_parent": within,
        "rank_correlations": ranking,
        "largest_reward_changes": sorted(changes, key=lambda row: -row["absolute_change"])[:5],
    }


def contrasts(rows: dict, families: dict) -> dict:
    result = {}
    for new, old in CONTRASTS:
        if new not in rows or old not in rows:
            continue
        right, left = rows[new], rows[old]
        result[f"{new} − {old}"] = {}
        for split in right.keys() & left.keys():
            prior = {row["task_id"]: row["success"] for row in left[split]}
            current = {row["task_id"]: row["success"] for row in right[split]}
            estimates = paired(right[split], left[split], families)
            result[f"{new} − {old}"][split] = estimates | {
                "before": sum(prior.values()),
                "after": sum(current.values()),
                "gains": sum(value and not prior[key] for key, value in current.items()),
                "losses": sum(not value and prior[key] for key, value in current.items()),
                "relative_improvement": (sum(current.values()) / sum(prior.values()) - 1)
                if any(prior.values())
                else None,
            }
    return result


def finish(fig, out: Path, file: str, *, dpi: int = 220) -> None:
    for extension in ("pdf", "png"):
        fig.savefig(out / f"{file}.{extension}", dpi=dpi)
    plt.close(fig)


def framework(out: Path, *, matched_study: bool = False) -> None:
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    from matplotlib.path import Path as ArrowPath

    fig, ax = plt.subplots(figsize=(10.8, 4.8), layout="constrained")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    nodes = {
        "Proposer": (0.13, 0.76),
        "Executable\nPython task": (0.37, 0.76),
        "Student\nrepair agent": (0.62, 0.76),
        "Environment\nand verifier": (0.87, 0.76),
        "Verified PDB\nreference repairs": (0.13, 0.43),
        ("Difficulty or\nalignment reward" if matched_study else "Alignment\nreward C / D"): (
            0.37,
            0.43,
        ),
        "Privileged\nteacher context": (0.62, 0.43),
        (
            "Dr. GRPO + optional\ngated / ungated OPSD"
            if matched_study
            else "Sparse Dr. GRPO\nDense gated OPSD"
        ): (0.87, 0.43),
        "Shared LoRA update": (0.62, 0.12),
    }
    for label, (x, y) in nodes.items():
        ax.add_patch(
            FancyBboxPatch(
                (x - 0.092, y - 0.067),
                0.184,
                0.134,
                boxstyle="round,pad=.013",
                linewidth=1.3,
                edgecolor="#1d5b55",
                facecolor="#e9f4f1",
            )
        )
        ax.text(x, y, label, ha="center", va="center", fontsize=11, color="#172b32")
    connections = (
        ((0.235, 0.76), (0.263, 0.76)),
        ((0.476, 0.76), (0.514, 0.76)),
        ((0.726, 0.76), (0.764, 0.76)),
        ((0.87, 0.677), (0.87, 0.515)),
        ((0.726, 0.43), (0.764, 0.43)),
        ((0.235, 0.43), (0.263, 0.43)),
        ((0.62, 0.677), (0.62, 0.515)),
        ((0.367, 0.515), (0.206, 0.675)),
        ((0.378, 0.347), (0.565, 0.20)),
        ((0.90, 0.347), (0.715, 0.174)),
    )
    for start, end in connections:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=14,
                color="#386f72",
            )
        )
    for points, color in (
        ([(0.84, 0.35), (0.84, 0.26), (0.31, 0.26), (0.31, 0.347)], "#775936"),
        ([(0.515, 0.12), (0.014, 0.12), (0.014, 0.72), (0.027, 0.72)], "#386f72"),
    ):
        ax.add_patch(
            FancyArrowPatch(
                path=ArrowPath(points),
                arrowstyle="-|>",
                mutation_scale=12,
                color=color,
                linewidth=1.1,
            )
        )
    ax.text(0.685, 0.279, "Candidate gradient", fontsize=8.5, color="#775936", ha="center")
    ax.set_title("Gradient-guided self-play for Python repair", fontsize=16, loc="left", pad=12)
    finish(fig, out, "01-framework", dpi=300 if matched_study else 220)


def primary_figure(
    out: Path, rows: dict, comparisons: dict, exposure: int, seed: int = 2026
) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.2, 4.1), layout="constrained")
    methods = [name for name in METHODS if PRIMARY in rows.get(name, {})]
    for i, name in enumerate(methods):
        values = rows[name][PRIMARY]
        successes = sum(item["success"] for item in values)
        left.bar(i, 100 * successes / len(values), color=COLORS[name])
        left.text(
            i,
            100 * successes / len(values) + 1.1,
            f"{successes}/{len(values)}",
            ha="center",
            fontsize=9,
        )
    left.set(
        xticks=range(len(methods)),
        xticklabels=methods,
        ylim=(0, 100),
        ylabel="Verified repairs (%)",
        title=f"Seed {seed} · {exposure} attempted episodes",
    )
    chosen = [(label, item[PRIMARY]) for label, item in comparisons.items() if PRIMARY in item]
    for i, (_, row) in enumerate(chosen):
        right.hlines(i, *row["ci95_pp"], color="#2b7169")
        right.plot(row["difference_pp"], i, "o", color="#2b7169")
    right.axvline(0, color="#b0b4ba", linewidth=1)
    right.set(
        yticks=range(len(chosen)),
        yticklabels=[label for label, _ in chosen],
        xlabel="Paired difference (pp)",
        title="95% family bootstrap",
    )
    right.invert_yaxis()
    finish(fig, out, "02-primary")


def curves_figure(out: Path, runs: dict) -> None:
    fig, grid = plt.subplots(2, 2, figsize=(10, 7.1), layout="constrained")
    axes = grid.flatten()
    dimensions = {
        "episodes": "Attempted student episodes",
        "tokens": "Generated student tokens",
        "optimizer_steps": "Optimizer steps",
        "h200_hours": "Training allocation (H200-hours)",
    }
    for axis, (key, label) in zip(axes, dimensions.items(), strict=True):
        for name, record in runs.items():
            method, _, seed = name.partition("-")
            rows = [row for row in record["curves"] if key in row]
            if rows:
                axis.plot(
                    [row[key] for row in rows],
                    [100 * row["success"] for row in rows],
                    marker="o",
                    markersize=4,
                    color=COLORS[method],
                    linestyle="--" if seed == "2027" else "-",
                    label=name,
                )
        axis.set(xlabel=label, ylabel="Development accuracy (%)", ylim=(0, 100))
    axes[0].legend(ncol=2, frameon=False, fontsize=8)
    finish(fig, out, "03-learning-efficiency")


def alignment_figure(out: Path, report: dict) -> None:
    if not report["tasks"]:
        return
    fig, (zero, rewards, diversity) = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for i, arm in enumerate(("C", "D")):
        near_zero = report["counts"][arm]["near_zero"]
        zero.bar(i, 100 * near_zero / report["tasks"], color=COLORS[arm])
        zero.text(
            i,
            100 * near_zero / report["tasks"] + 1,
            f"{near_zero}/{report['tasks']}",
            ha="center",
            fontsize=9,
        )
    zero.set(
        xticks=(0, 1),
        xticklabels=("C", "D"),
        ylim=(0, 110),
        ylabel="Candidate gradients (%)",
        title="Near-zero gradients",
    )
    for arm in ("C", "D"):
        values = [
            (int(n), statistics.mean(row[arm]))
            for n, row in report["rewards_by_success"].items()
            if row[arm]
        ]
        if values:
            rewards.plot(*zip(*values, strict=True), "o", color=COLORS[arm], label=arm)
    rewards.legend(frameon=False)
    rewards.set(
        xticks=range(5),
        ylim=(0, 1.05),
        xlabel="Verified rollouts / 4",
        ylabel="Mean alignment reward",
        title="Rewards on identical tasks",
    )
    for i, method in enumerate(("difficulty", "C", "D")):
        groups = [row for row in report["within_parent"][method] if row["valid_tasks"] > 1]
        if groups:
            count = sum(row["distinct"] > 1 for row in groups)
            diversity.bar(i, 100 * count / len(groups), color=COLORS.get(method, "#687782"))
            diversity.text(
                i, 100 * count / len(groups) + 1, f"{count}/{len(groups)}", ha="center", fontsize=9
            )
    diversity.set(
        xticks=range(3),
        xticklabels=("Difficulty", "C", "D"),
        ylim=(0, 110),
        ylabel="Multi-task parent groups (%)",
        title="Distinct valid rewards",
    )
    finish(fig, out, "04-alignment")


def dynamics_figure(out: Path, runs: dict) -> None:
    fig, (magnitudes, cosines, gates) = plt.subplots(1, 3, figsize=(12, 4.1), layout="constrained")
    for name, record in runs.items():
        arm, _, seed = name.partition("-")
        if seed == "2027":
            continue
        color = COLORS[arm]
        rows = record["training"]
        ratios = [
            (
                row["attempted_episodes"],
                0.1717172194518303 * row["opsd"]["gradient_norm"] / row["rl"]["norm"],
            )
            for row in rows
            if row["rl"]["norm"] and arm != "Control"
        ]
        if ratios:
            magnitudes.plot(*zip(*ratios, strict=True), "o-", markersize=3, color=color, label=arm)
        geometry = [
            (row["attempted_episodes"], row["opsd"]["rl_cosine"])
            for row in rows
            if row["opsd"]["rl_cosine"] is not None
        ]
        if geometry:
            cosines.plot(*zip(*geometry, strict=True), "o-", markersize=3, color=color)
        counts = [sum(row["learning"]["gates"].get(str(i), 0) for row in rows) for i in range(5)]
        if sum(counts):
            gates.plot(range(5), [100 * n / sum(counts) for n in counts], "o-", color=color)
    magnitudes.set(
        xlabel="Episodes", ylabel="Weighted OPSD / RL norm", title="Effective distillation strength"
    )
    magnitudes.axhline(1, color=".65", linewidth=1)
    if magnitudes.get_legend_handles_labels()[0]:
        magnitudes.legend(frameon=False)
    cosines.set(
        xlabel="Episodes", ylabel="Cosine", ylim=(-1.05, 1.05), title="RL and OPSD gradients"
    )
    cosines.axhline(0, color=".65", linewidth=1)
    gates.set(
        xticks=range(5),
        xticklabels=("[0,.2)", "[.2,.4)", "[.4,.6)", "[.6,.8)", "[.8,1]"),
        ylabel="Scored tokens (%)",
        title="Gate weight distribution",
    )
    finish(fig, out, "05-gradient-dynamics")


def failures_figure(out: Path, rows: dict) -> dict:
    categories = (
        "Verified",
        "Public pass / hidden fail",
        "Truncation",
        "Tool or edit error",
        "Other failure",
    )
    colors = ("#198377", "#c0a35d", "#a45440", "#886c97", "#b8c2c9")
    fig, ax = plt.subplots(figsize=(8.5, 3.9), layout="constrained")
    recorded = {}
    for name in METHODS:
        if PRIMARY not in rows.get(name, {}):
            continue
        counts: Counter = Counter()
        for row in rows[name][PRIMARY]:
            label = "Other failure"
            if row["success"]:
                label = "Verified"
            elif row["failure"] == "truncation":
                label = "Truncation"
            elif row["public_pass"] and not row["hidden_pass"]:
                label = "Public pass / hidden fail"
            elif row["failure"] in {"rejected_edit", "invalid_tool"}:
                label = "Tool or edit error"
            counts[label] += 1
        recorded[name] = {
            "count": len(rows[name][PRIMARY]),
            "categories": counts,
            "terminations": Counter(row["termination"] for row in rows[name][PRIMARY]),
            "failure_labels": Counter(row["failure"] for row in rows[name][PRIMARY]),
            "invalid_calls": sum(row["invalid_calls"] for row in rows[name][PRIMARY]),
            "rejected_edits": sum(row["rejected_edits"] for row in rows[name][PRIMARY]),
            "public_pass_hidden_fail": sum(
                row["public_pass"] and not row["hidden_pass"] for row in rows[name][PRIMARY]
            ),
        }
    for i, name in enumerate(recorded):
        total = 0
        for label, color in zip(categories, colors, strict=True):
            count = recorded[name]["categories"][label]
            ax.barh(name, count, left=total, color=color, label=label if i == 0 else None)
            total += count
    ax.invert_yaxis()
    ax.set(xlabel="Primary evaluation tasks", title="Observed failures and verified repairs")
    ax.legend(ncol=3, frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.43))
    finish(fig, out, "06-failure-modes")
    return recorded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task21", type=Path, default=Path("runs/task21/matched-analysis/results.json")
    )
    parser.add_argument("--data", type=Path, default=Path("data/task21/pdb-solver-v1"))
    parser.add_argument("--artifacts", type=Path, default=Path("runs/hpc/yale-hpc-cpsc4880"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--budget", type=Path)
    parser.add_argument("--exposure", type=int, choices=(256, 384, 512), default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    historical = read_json(args.task21)
    known = {
        label.replace("control-", "Control-"): value
        for label, value in historical["runs"].items()
        if label in {f"{arm}-{seed}" for arm in ("control", "A", "B") for seed in (2026, 2027)}
    }
    repo = PdbRepository(args.data)
    families = {
        task.parent_id: oracle.provenance["family_id"]
        for split in repo.partition_names()
        for task, oracle in repo.tasks(split)
    }
    budget = read_json(args.budget) if args.budget else {"reservations": {}}
    reservations = list(budget["reservations"].values())
    live = {
        name: load_run(spec, args.artifacts, reservations)
        for name, spec in (read_json(args.manifest) if args.manifest else {}).items()
    }
    alignments = {
        name: alignment_summary(record["paired_batches"]) for name, record in live.items()
    }
    if args.qualification:
        verified, _ = evidence(args.qualification, args.artifacts)
        report = verified["result"]
        batches = []
        for row in report["batches"]:
            scores = [group for group in row["comparison"]["difficulty"]["within_parent_scores"]]
            batches.append([{key: row["tasks"][key] for key in group} for group in scores])
        alignments["qualification"] = alignment_summary(batches)
    for record in known.values():
        steps = 0
        by_episode = {0: 0}
        for batch in record["training"]:
            steps += int(batch["shared"]["optimizer_step"])
            by_episode[batch["attempted_episodes"]] = steps
        for row in record["curves"]:
            row["optimizer_steps"] = by_episode[row["episodes"]]
    runs = known | live
    if "B-2026" in live:
        historical_b, new_b = known["B-2026"], live["B-2026"]
        runs["B-2026"] = historical_b | {
            "outcomes": historical_b["outcomes"] | new_b["outcomes"],
            "end_to_end_h200_hours": historical_b["gpu_hours"] + new_b["end_to_end_h200_hours"],
        }
    baseline = known["B-2026"]["outcomes"]["0"]
    rows = {
        name: run["outcomes"].get(str(args.exposure), {})
        for name, run in runs.items()
        if name.endswith("-2026") and run["outcomes"].get(str(args.exposure))
    }
    rows = {"Initial": baseline} | {name.removesuffix("-2026"): row for name, row in rows.items()}
    comparisons = contrasts(rows, families)
    two_seed = {
        str(seed): contrasts(
            {arm: known[f"{arm}-{seed}"]["outcomes"]["512"] for arm in ("Control", "A", "B")},
            families,
        )
        for seed in (2026, 2027)
    }
    endpoints = {
        name: {
            "episodes": args.exposure if name != "Initial" else 0,
            "scores": {
                split: {"successes": sum(row["success"] for row in values), "count": len(values)}
                for split, values in outcomes.items()
            },
            "optimizer_steps": sum(
                row["shared"]["optimizer_step"]
                for row in runs.get(f"{name}-2026", {}).get("training", [])
                if row["attempted_episodes"] <= args.exposure
            ),
            "training_h200_hours": (
                runs.get(f"{name}-2026", {}).get("gpu_hours")
                if args.exposure == 512
                else next(
                    (
                        row["h200_hours"]
                        for row in runs.get(f"{name}-2026", {}).get("curves", [])
                        if row["episodes"] == args.exposure
                    ),
                    None,
                )
            ),
            "end_to_end_h200_hours": runs.get(f"{name}-2026", {}).get("end_to_end_h200_hours"),
        }
        for name, outcomes in rows.items()
    }
    args.output.mkdir(parents=True, exist_ok=True)
    framework(args.output)
    primary_figure(args.output, rows, comparisons, args.exposure)
    curves_figure(args.output, runs)
    alignment_figure(args.output, alignments.get("qualification", {"tasks": 0}))
    dynamics_figure(args.output, runs)
    failures = failures_figure(args.output, rows)
    charged = sum(row["charged"] for row in reservations if row["charged"] is not None)
    atomic_json(
        args.output / "summary.json",
        {
            "endpoints": endpoints,
            "paired_2026": comparisons,
            "task21_two_seed_paired": two_seed,
            "task21_two_seed_primary_accuracy": {
                arm: statistics.mean(
                    sum(
                        row["success"] for row in known[f"{arm}-{seed}"]["outcomes"]["512"][PRIMARY]
                    )
                    / len(known[f"{arm}-{seed}"]["outcomes"]["512"][PRIMARY])
                    for seed in (2026, 2027)
                )
                for arm in ("Control", "A", "B")
            },
            "alignment": alignments,
            "failure_modes": failures,
            "incremental_h200_hours": charged,
            "combined_task21_task22_h200_hours": historical["allocated_h200_hours"] + charged,
            "unreconciled_reservations": sum(row["charged"] is None for row in reservations),
            "training_diagnostics": {
                name: [
                    {
                        key: row[key]
                        for key in (
                            "attempted_episodes",
                            "student",
                            "proposer",
                            "rl",
                            "opsd",
                            "shared",
                            "learning",
                            "timing",
                            "gpu_peak_bytes",
                        )
                        if key in row
                    }
                    for row in record["training"]
                ]
                for name, record in runs.items()
            },
        },
    )


if __name__ == "__main__":
    main()
