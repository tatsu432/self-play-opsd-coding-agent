"""Export publication figures and tables from aggregate-only Task 23 statistics."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

from analyze_task22 import COLORS, METHODS, PRIMARY, finish, framework

from selfplay_opsd.core import read_json

SEEDS = (2026, 2027)
SPLITS = {
    "harder_transfer": "PDB-Multi",
    "cross_source": "PDB cross-source",
    "cross_source_bcb_mutants": "BigCodeBench mutants",
    "ood_source": "New source transfer",
    "ood_complexity": "New complexity transfer",
}
CATS = {
    "truncation": "#95655b",
    "public_pass_hidden_fail": "#e0ac67",
    "semantic_or_runtime_after_edit": "#a47155",
    "rejected_edit": "#bd8a7b",
    "invalid_tool": "#817e9c",
    "no_edit": "#9d9ba8",
    "verified": "#287968",
}
MARKERS = ("o", "^")


def mpl():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42}
    )
    return plt


def primary_figure(output: Path, report: dict, compact: bool = False) -> None:
    plt = mpl()
    fig, axes = plt.subplots(
        1,
        1 if compact else 2,
        figsize=(7.1 if compact else 11.4, 3.8),
        squeeze=False,
        layout="constrained",
    )
    ax = axes[0, 0]
    for x, arm in enumerate(METHODS):
        overall = report["primary"][arm]["mean_percent"]
        ax.bar(x, overall, width=0.66, color=COLORS[arm], alpha=0.26)
        for i, seed in enumerate(SEEDS):
            row = report["cells"][f"{seed}-{arm}"]["scores"][PRIMARY]
            offset, marker = (-0.12, 0.12)[i], MARKERS[i]
            ax.scatter(
                x + offset,
                100 * row["verified"] / row["count"],
                color=COLORS[arm],
                marker=marker,
                s=44,
                label=str(seed) if x == 0 else None,
                zorder=3,
            )
        ax.hlines(overall, x - 0.30, x + 0.30, color=COLORS[arm], linewidth=1.8)
    ax.set(
        xticks=range(len(METHODS)),
        xticklabels=METHODS,
        ylim=(0, 62),
        ylabel="Verified pass@1 (%)",
        title="Held-out single-bug repair · 64 shared tasks",
    )
    ax.legend(frameon=False, loc="upper left", ncols=2)
    if not compact:
        ax = axes[0, 1]
        contrasts = report["contrasts"]
        for i, scores in enumerate(contrasts.values()):
            row = scores[PRIMARY]
            low, high = row["ci95_pp"]
            ax.hlines(i, low, high, color="#3f7471", linewidth=1.6)
            ax.plot(row["difference_pp"], i, "o", color="#13574c")
        ax.axvline(0, color=".55", linestyle="--", linewidth=0.9)
        ax.set(
            yticks=range(len(contrasts)),
            yticklabels=contrasts.keys(),
            xlabel="Paired difference (percentage points)",
            title="95% task-family intervals · conditional on checkpoints",
        )
        ax.invert_yaxis()
    finish(fig, output, "presentation-primary" if compact else "02-primary")


def transfer_figure(output: Path, report: dict) -> None:
    plt = mpl()
    observed = report.get("followup", {}).get("ood", {})
    splits = list(SPLITS)[:3] + [key for key in list(SPLITS)[3:] if key in observed]
    fig, axes = plt.subplots(
        1, len(splits), figsize=(3.65 * len(splits), 3.9), sharey=True, layout="constrained"
    )
    for ax, split in zip(axes, splits, strict=True):
        table = observed.get(split)
        for x, arm in enumerate(METHODS):
            rows = [
                table["cells"][f"{seed}-{arm}"]
                if table
                else report["cells"][f"{seed}-{arm}"]["scores"][split]
                for seed in SEEDS
            ]
            values = [100 * row["verified"] / row["count"] for row in rows]
            ax.bar(x, statistics.mean(values), color=COLORS[arm], width=0.65, alpha=0.25)
            for j, v in enumerate(values):
                ax.scatter(
                    x + (-0.14, 0.14)[j],
                    v,
                    marker=MARKERS[j],
                    color=COLORS[arm],
                    s=30,
                    label=str(SEEDS[j]) if x == 0 else None,
                )
        ax.set_title(f"{SPLITS[split]}\n{rows[0]['count']} shared tasks", fontsize=10.5)
        ax.set(xticks=range(6), xticklabels=METHODS, ylim=(0, 110))
        ax.tick_params(axis="x", labelrotation=32)
    axes[0].set_ylabel("Verified pass@1 (%)")
    axes[0].legend(frameon=False, ncols=2, loc="upper left")
    finish(fig, output, "03-transfer")


def development_figure(output: Path, report: dict, compact: bool = False) -> None:
    plt = mpl()
    axes_keys = (
        ("episodes", "Attempted student episodes"),
        ("student_tokens", "Generated student tokens (M)"),
        ("steps", "Optimizer steps"),
        ("h200_hours", "Training-job H200-hours"),
    )
    if compact:
        axes_keys = axes_keys[:1]
    fig, axes = plt.subplots(
        1 if compact else 2,
        1 if compact else 2,
        squeeze=False,
        figsize=(7.2, 3.9) if compact else (10.2, 6.4),
        layout="constrained",
    )
    for ax, (key, label) in zip(axes.flat, axes_keys, strict=True):
        show_seeds = key != "episodes" or not compact
        ax.axhline(
            31.25, color=COLORS["Initial"], linewidth=0.9, linestyle=":", label="Initial mean"
        )
        for arm in METHODS[1:]:
            series = [report["cells"][f"{seed}-{arm}"]["curves"] for seed in SEEDS]
            if show_seeds:
                for i, rows in enumerate(series):
                    ax.plot(
                        [row[key] / (1e6 if key == "student_tokens" else 1) for row in rows],
                        [100 * row["rate"] for row in rows],
                        color=COLORS[arm],
                        alpha=0.65 if key != "episodes" else 0.26,
                        linestyle=("-", "--")[i],
                        marker=MARKERS[i],
                        markersize=3,
                        label=arm if i == 0 and key != "episodes" else None,
                    )
            if key == "episodes":
                values = [{row[key]: row["rate"] for row in rows} for rows in series]
                observed = sorted(values[0].keys() & values[1].keys())
                ax.plot(
                    observed,
                    [100 * statistics.mean(row[x] for row in values) for x in observed],
                    color=COLORS[arm],
                    marker="o",
                    markersize=3.7,
                    linewidth=1.6,
                    label=arm,
                )
        ax.set(xlabel=label, ylabel="Development pass@1 (%)", ylim=(15, 70))
    fig.suptitle("Fixed development set · 24 repeatedly monitored tasks")
    axes[0, 0].legend(frameon=False, ncols=3)
    finish(fig, output, "presentation-development" if compact else "04-development")
    if compact:
        return
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(9.6, 3.6), layout="constrained")
    for ax, seed in zip(axes, SEEDS, strict=True):
        for arm in METHODS[1:]:
            series = report["cells"][f"{seed}-{arm}"]["curves"]
            ax.plot(
                [row["episodes"] for row in series],
                [row["verified"] for row in series],
                color=COLORS[arm],
                marker="o",
                markersize=4,
                label=arm,
            )
        initial = report["cells"][f"{seed}-B"]["curves"][0]["verified"]
        ax.axhline(initial, color=".46", linestyle=":", label="Initial")
        ax.set(xlabel="Attempted student episodes", title=f"Seed {seed}", ylim=(3, 17))
    axes[0].set_ylabel("Verified development repairs / 24")
    axes[0].legend(frameon=False, ncols=3)
    finish(fig, output, "04-development-by-seed")


def dynamics_figure(output: Path, report: dict) -> None:
    plt = mpl()
    fig, axes = plt.subplots(3, 3, figsize=(12, 8.1), sharex=True, layout="constrained")
    metrics = (
        ("rolling_rate", "Student successes · trailing 64 attempts"),
        ("mixed_fraction", "Mixed student groups"),
        ("all_failed_fraction", "All-failed student groups"),
        ("all_successful_fraction", "All-successful student groups"),
        ("valid_proposal_rate", "Valid proposal fraction"),
        ("proposal_group_diversity", "Groups with distinct proposer rewards"),
        ("proposal_mean_reward", "Difficulty proposer reward · Control/A/B"),
        ("proposal_mean_reward", "Alignment proposer reward · C/D"),
        ("cumulative_unique_tasks", "Cumulative unique scored tasks"),
    )
    for idx, (ax, (key, title)) in enumerate(zip(axes.flat, metrics, strict=True)):
        arms = METHODS[1:4] if idx == 6 else METHODS[4:] if idx == 7 else METHODS[1:]
        for arm in arms:
            for i, seed in enumerate(SEEDS):
                rows = report["cells"][f"{seed}-{arm}"]["batches"]
                ax.plot(
                    [row["episode"] for row in rows],
                    [row[key] for row in rows],
                    linestyle=("-", "--")[i],
                    color=COLORS[arm],
                    alpha=0.85,
                    label=arm if i == 0 else None,
                    linewidth=1.1,
                )
                if idx == 0:
                    ax.scatter(
                        [row["episode"] for row in rows],
                        [row["online_rate"] for row in rows],
                        color=COLORS[arm],
                        alpha=0.14,
                        s=9,
                    )
        ax.set_title(title, fontsize=9.25)
    axes[0, 0].legend(frameon=False, ncols=3, fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Attempted student episodes")
    fig.suptitle("Online learning dynamics on evolving generated tasks · solid 2026, dashed 2027")
    finish(fig, output, "05-online-dynamics")


def alignment_figure(output: Path, report: dict, compact: bool = False) -> None:
    plt = mpl()
    fig, axes = plt.subplots(
        1,
        1 if compact else 3,
        squeeze=False,
        layout="constrained",
        figsize=(6.8, 3.7) if compact else (12.6, 4.2),
    )
    sources = [
        (seed, arm, report["cells"][f"{seed}-{arm}"]["alignment"])
        for seed in SEEDS
        for arm in ("C", "D")
    ]
    ax = axes[0, 0]
    for i, (_, _, row) in enumerate(sources):
        whole = row["by_outcome"]["all"]
        for j, arm in enumerate(("C", "D")):
            count = whole["arms"][arm]["near_zero"]
            pos = i + (-0.15, 0.15)[j]
            rate = 100 * count / whole["count"]
            ax.bar(pos, rate, 0.28, color=COLORS[arm], label=arm if i == 0 else None)
            ax.annotate(
                f"{count}/{whole['count']}",
                (pos, rate + 2.5),
                ha="center",
                fontsize=8,
                color=COLORS[arm],
            )
    ax.set(
        xticks=range(len(sources)),
        xticklabels=[f"{seed}\n{arm} trajectory" for seed, arm, _ in sources],
        ylabel="Near-zero candidate gradients (%)",
        ylim=(0, 75),
        title="Same task, rollouts and snapshot",
    )
    ax.legend(frameon=False)
    if not compact:
        ax = axes[0, 1]
        keys = ("all_failed", "mixed", "all_successful")
        for i, key in enumerate(keys):
            rows = [row["by_outcome"][key] for _, _, row in sources if key in row["by_outcome"]]
            n = sum(row["count"] for row in rows)
            for j, arm in enumerate(("C", "D")):
                count = sum(row["arms"][arm]["near_zero"] for row in rows)
                x = i + (-0.15, 0.15)[j]
                ax.bar(x, 100 * count / n, 0.28, color=COLORS[arm])
                ax.annotate(f"{count}/{n}", (x, 100 * count / n + 3), ha="center", fontsize=8)
        ax.set(
            xticks=range(3),
            xticklabels=("All failed", "Mixed", "All passed"),
            ylim=(0, 117),
            title="Gradient sparsity by rollout outcome",
        )
        ax = axes[0, 2]
        for i, (_, _, row) in enumerate(sources):
            for j, arm in enumerate(("C", "D")):
                x = i + (-0.15, 0.15)[j]
                q = row["by_outcome"]["all"]["arms"][arm]["cosine_quantiles"]
                ax.vlines(x, q[1], q[3], color=COLORS[arm], linewidth=3)
                ax.scatter(x, q[2], color=COLORS[arm], s=24)
                distinct = row["distinct_reward_groups"][arm]
                ax.annotate(
                    f"{distinct}/{row['group_count']}",
                    (x, q[3] + 0.016),
                    ha="center",
                    fontsize=7,
                    color=COLORS[arm],
                )
        ax.axhline(0, linestyle=":", color=".55")
        ax.set(
            xticks=range(4),
            xticklabels=[f"{seed}\n{arm} traj." for seed, arm, _ in sources],
            title="Defined cosines · median and IQR\nLabels: groups with distinct rewards",
        )
    finish(fig, output, "presentation-sparsity" if compact else "06-alignment")


def strength_figure(output: Path, report: dict) -> None:
    plt = mpl()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout="constrained")
    for i, (seed, method) in enumerate((seed, arm) for seed in SEEDS for arm in ("A", "B")):
        rows = report["cells"][f"{seed}-{method}"]["batches"]
        for j, (key, title) in enumerate(
            (
                ("weighted_ratio", "Weighted OPSD / RL gradient norm"),
                ("rl_opsd_cosine", "RL–OPSD cosine"),
            )
        ):
            values = [row[key] for row in rows if row[key] is not None]
            box = axes[j].boxplot(
                values,
                positions=[i],
                widths=0.55,
                patch_artist=True,
                medianprops={"color": "#1c282f"},
                flierprops={"markersize": 3},
            )
            box["boxes"][0].set(facecolor=COLORS[method], alpha=0.55)
            axes[j].set(title=title, xticks=range(4), xticklabels=["A 26", "B 26", "A 27", "B 27"])
        gates = [sum(row["gate_histogram"][str(k)] for row in rows) for k in range(5)]
        if sum(gates):
            axes[2].plot(
                range(5),
                [100 * count / sum(gates) for count in gates],
                marker=MARKERS[SEEDS.index(seed)],
                color=COLORS[method],
                linestyle=("-", "--")[SEEDS.index(seed)],
                label=f"{method} {seed}",
            )
    axes[1].axhline(0, color=".55", linestyle=":")
    axes[2].set(
        title="Recorded gate histogram",
        ylabel="Tokens (%)",
        xticks=range(5),
        xticklabels=["0–.2", ".2–.4", ".4–.6", ".6–.8", ".8–1"],
    )
    axes[2].legend(frameon=False, fontsize=8)
    finish(fig, output, "07-distillation")


def failure_figure(output: Path, report: dict) -> None:
    plt = mpl()
    sensitivity = report.get("followup", {}).get("relaxed")
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), layout="constrained")
    reasons = {
        "turn_tokens": "#334e6d",
        "turn_limit": "#809fac",
        "episode_tokens": "#bf994e",
        "context_limit": "#8d6850",
    }
    for i, seed in enumerate(SEEDS):
        rows = [report["cells"][f"{seed}-{arm}"]["scores"][PRIMARY] for arm in METHODS]
        for j, (key, palette, title) in enumerate(
            (
                ("failures", CATS, "Final verdict categories"),
                ("truncation_reasons", reasons, "Terminal reasons among truncation failures"),
            )
        ):
            ax = axes[i, j]
            bottom = [0] * len(rows)
            for category, color in palette.items():
                values = [row[key].get(category, 0) for row in rows]
                if any(values):
                    ax.bar(
                        METHODS,
                        values,
                        bottom=bottom,
                        color=color,
                        label=category.replace("_", " "),
                        width=0.7,
                    )
                    bottom = [p + q for p, q in zip(bottom, values, strict=True)]
            ax.set(title=f"{seed} · {title}", ylabel="Episodes / 64", ylim=(0, 67))
            ax.tick_params(axis="x", labelrotation=27)
            ax.legend(frameon=False, fontsize=6.5, loc="upper right")
        ax = axes[i, 2]
        if sensitivity:
            cells = [sensitivity["cells"][f"{seed}-{arm}"] for arm in METHODS]
            ax.plot(
                METHODS, [cell["baseline_verified"] for cell in cells], "o-", label="4,096 tokens"
            )
            ax.plot(METHODS, [cell["verified"] for cell in cells], "^-", label="6,144 tokens")
            ax.set(
                title=f"{seed} · Per-turn cap sensitivity",
                ylabel=f"Verified repairs / {cells[0]['count']}",
            )
        else:
            for k, field in enumerate(("median_tokens_success", "median_tokens_failure")):
                ax.plot(
                    METHODS,
                    [row[field] for row in rows],
                    marker=MARKERS[k],
                    label=("Verified", "Failed")[k],
                )
            ax.set(title=f"{seed} · Generated tokens by verdict", ylabel="Median generated tokens")
        ax.tick_params(axis="x", labelrotation=27)
        ax.legend(frameon=False, fontsize=8)
    finish(fig, output, "08-failure-and-budget")


def csv_export(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tables(output: Path, report: dict) -> None:
    cells = report["cells"]
    rows = []
    for method in METHODS:
        seed_cells = [cells[f"{seed}-{method}"] for seed in SEEDS]
        primary = report["primary"][method]
        row = {"method": method}
        for seed, cell in zip(SEEDS, seed_cells, strict=True):
            for split, values in cell["scores"].items():
                row[f"{seed} {split}"] = f"{values['verified']}/{values['count']}"
            row[f"{seed} train H200h"] = round(cell["training_h200_seconds"] / 3600, 3)
            row[f"{seed} end-to-end H200h"] = round(
                (cell["training_h200_seconds"] + cell["evaluation_h200_seconds"]) / 3600, 3
            )
        rows.append(row | primary)
    csv_export(output / "table-2-main.csv", rows)
    csv_export(
        output / "table-3-contrasts.csv",
        [
            {
                "contrast": name,
                "split": split,
                "checkpoint_average_pp": row["difference_pp"],
                "conditional_95_low": row["ci95_pp"][0],
                "conditional_95_high": row["ci95_pp"][1],
                "tasks": row["tasks"],
                "families": row["families"],
                **{f"{seed}_pp": row["seed"][str(seed)]["difference_pp"] for seed in SEEDS},
            }
            for name, splits in report["contrasts"].items()
            for split, row in splits.items()
        ],
    )
    csv_export(
        output / "table-4-mechanisms.csv",
        [
            {
                "seed": seed,
                "method": method,
                "verified_online": t["verified"],
                "attempted": t["attempted"],
                "first_128_rate": t["first_128_rate"],
                "last_128_rate": t["last_128_rate"],
                "mixed_group_fraction": t["mixed_fraction"],
                "weighted_opsd_to_rl": t["median_weighted_ratio"],
                "rl_opsd_cosine": t["median_rl_opsd_cosine"],
                "primary_truncations": cell["scores"][PRIMARY]["failures"].get("truncation", 0),
                "student_tokens": t["student_tokens"],
                "proposer_tokens": t["proposer_tokens"],
                "steps": t["optimizer_steps"],
                "train_h200h": cell["training_h200_seconds"] / 3600,
                "near_zero_observed_alignment": sum(
                    b["near_zero_alignments"] for b in cell["batches"]
                ),
            }
            for seed in SEEDS
            for method in METHODS[1:]
            for cell in [cells[f"{seed}-{method}"]]
            for t in [cell["training"]]
        ],
    )


def export(output: Path, report: dict) -> None:
    framework(output)
    primary_figure(output, report)
    transfer_figure(output, report)
    development_figure(output, report)
    dynamics_figure(output, report)
    alignment_figure(output, report)
    strength_figure(output, report)
    failure_figure(output, report)
    primary_figure(output, report, compact=True)
    development_figure(output, report, compact=True)
    alignment_figure(output, report, compact=True)
    tables(output, report)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.output, read_json(args.output / "statistics.json"))
