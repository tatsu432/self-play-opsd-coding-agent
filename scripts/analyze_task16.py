"""Analyze frozen Task 15 evidence without inference or candidate execution."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

from selfplay_opsd.task16_analysis import run_analysis

COLORS = {"A": "#355C9A", "B": "#C35628"}


def figures(summary, output, destination):
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 220,
        }
    )
    with (output / "learning-curves.csv").open() as f:
        curves = list(csv.DictReader(f))

    def save(fig, name):
        for suffix in ("png", "pdf"):
            fig.savefig(destination / f"{name}.{suffix}", bbox_inches="tight")
        plt.close(fig)

    for field, label, name in (
        ("episodes", "Attempted student episodes", "development-episodes"),
        ("allocated_gpu_hours", "Allocated H200 GPU-hours", "development-compute"),
    ):
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.1), sharey=True, layout="constrained")
        for ax, seed in zip(axes, (2026, 2027, 2028), strict=True):
            for method in ("A", "B"):
                rows = [r for r in curves if int(r["seed"]) == seed and r["method"] == method]
                ax.plot(
                    [float(r[field]) for r in rows],
                    [100 * float(r["parent_macro"]) for r in rows],
                    "o-",
                    color=COLORS[method],
                    label=method,
                    markersize=4,
                )
            ax.set(title=f"Seed {seed}", xlabel=label, ylim=(0, 60))
            ax.grid(alpha=0.18)
        axes[0].set_ylabel("Development parent-macro accuracy (%)\n41 tasks / 23 parents")
        axes[-1].legend(title="Method")
        save(fig, name)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), layout="constrained")
    selections = summary["checkpoint_selection"]
    for i, r in enumerate(selections):
        axes[0].scatter(
            i, r["final_parent_macro"] * 100, color=COLORS[r["method"]], marker="o", s=55
        )
        axes[0].scatter(
            i,
            r["selected_parent_macro"] * 100,
            facecolors="none",
            edgecolors="black",
            marker="s",
            s=85,
        )
        frequency = r["stability"]["selection_frequency"].get(str(r["final_iteration"]), 0)
        axes[1].bar(i, 100 * frequency, color=COLORS[r["method"]])
    labels = [f"{r['method']}\n{r['seed']}" for r in selections]
    for ax in axes:
        ax.set_xticks(range(6), labels)
    axes[0].set(
        ylabel="Development parent-macro accuracy (%)", title="Final (circle) = selected (square)"
    )
    axes[1].set(
        ylabel="Bootstrap selections of final (%)",
        ylim=(0, 100),
        title="Family-resampled selection stability",
    )
    save(fig, "checkpoint-selection")
    categories = [
        "success",
        "public_pass_hidden_fail",
        "budget_exhausted",
        "no_edit_patch_rejected",
        "no_edit_parse_failure",
        "no_edit_other",
        "edited_semantic_failure",
        "edited_runtime_failure",
        "infrastructure",
        "unknown",
    ]
    groups = summary["evaluation_distributions"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.6), layout="constrained", sharex=True)
    cmap = plt.get_cmap("tab10")
    for ax, split in zip(axes, ("final_id", "external"), strict=True):
        rows = [r for r in groups if r["split"] == split]
        bottom = np.zeros(len(rows))
        for i, cat in enumerate(categories):
            values = np.array([100 * r["categories"].get(cat, 0) / r["tasks"] for r in rows])
            ax.bar(
                range(len(rows)), values, bottom=bottom, color=cmap(i), label=cat.replace("_", " ")
            )
            bottom += values
        ax.set(
            ylabel="Evaluated tasks (%)",
            ylim=(0, 100),
            title=("MBPP+ (115 tasks)" if split == "final_id" else "HumanEval+ (116 tasks)"),
        )
        ax.set_xticks(
            range(len(rows)),
            ["base" if r["method"] == "base" else f"{r['method']} {r['seed']}" for r in rows],
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right center", fontsize=8)
    fig.suptitle("Terminal attribution; trained action subtypes unavailable locally", fontsize=10)
    save(fig, "failure-categories")
    rows = [
        r for r in summary["paired"] if r["comparison"] == "B-minus-A" and r["split"] == "final_id"
    ]
    fig, ax = plt.subplots(figsize=(6, 3.2), layout="constrained")
    for i, r in enumerate(rows):
        mean = 100 * r["difference_parent_macro"]
        lo, hi = [100 * x for x in r["family_bootstrap_95_interval"]]
        ax.errorbar(mean, i, xerr=[[mean - lo], [hi - mean]], fmt="o", color=COLORS["B"], capsize=4)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set(
        yticks=range(3),
        yticklabels=[r["seed"] for r in rows],
        xlabel="B − A parent-macro accuracy (percentage points)",
        ylabel="Training seed",
        title="Final MBPP+: paired family-bootstrap 95% intervals",
    )
    save(fig, "final-id-paired")
    with (output / "training-health.csv").open() as f:
        health = list(csv.DictReader(f))
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), layout="constrained")
    metrics = (
        ("valid_proposals", "proposal_attempts", "Cumulative proposal validity (%)"),
        (
            "historical_replay_samples",
            "valid_proposals",
            "Cumulative historical repeats / valid proposals (%)",
        ),
        ("mixed_groups", "allocated_gpu_hours", "Cumulative mixed groups / H200-hour"),
        ("successes", "episodes", "Cumulative verified episode success (%)"),
    )
    for ax, (num, den, title) in zip(axes.flat, metrics, strict=True):
        for method in ("A", "B"):
            for seed, style in zip((2026, 2027, 2028), ("-", "--", ":"), strict=True):
                rs = [r for r in health if r["method"] == method and int(r["seed"]) == seed]
                a = (
                    np.cumsum([float(r[num]) for r in rs])
                    if den
                    else np.array([float(r[num]) for r in rs])
                )
                if den:
                    b = (
                        np.array([float(r[den]) for r in rs])
                        if den == "allocated_gpu_hours"
                        else np.cumsum([float(r[den]) for r in rs])
                    )
                    a = np.divide(a, b, out=np.zeros_like(a), where=b > 0) * (
                        1 if den == "allocated_gpu_hours" else 100
                    )
                ax.plot(
                    [int(r["episodes_end"]) for r in rs],
                    a,
                    color=COLORS[method],
                    linestyle=style,
                    linewidth=1,
                    label=f"{method}/{seed}",
                )
        ax.set(xlabel="Attempted student episodes", title=title)
        ax.grid(alpha=0.18)
    axes.flat[0].legend(fontsize=7, ncol=2)
    save(fig, "training-signal")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument(
        "--run-root", type=Path, help="Override arm directories by run ID after fetching"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figures", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument(
        "--tokenizer", type=Path, help="Local frozen tokenizer snapshot; never downloaded"
    )
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        parser.error("Use at least 100 bootstrap replicates")
    tokenizer = (
        AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        if args.tokenizer
        else None
    )
    summary = run_analysis(
        args.manifest,
        args.data,
        args.base,
        args.output,
        tokenizer=tokenizer,
        bootstrap_samples=args.bootstrap_samples,
        run_root=args.run_root,
    )
    figures(summary, args.output, args.figures or args.output / "figures")
    print(
        f"Analyzed six arms and common base; {summary['trajectory_coverage']}; GPU pilot required: {summary['gpu_pilot']['required']}"
    )


if __name__ == "__main__":
    main()
