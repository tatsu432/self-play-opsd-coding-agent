"""Plot common-family edit application and verified repair, without causal attribution."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cells = json.loads(args.summary.read_text())["cells"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True, layout="constrained")
    modes = ("strict", "normalized", "text")
    if "base-native" in cells and "trained-native" in cells:
        modes += ("native",)
    labels = {
        "strict": "Strict diff",
        "normalized": "Normalized diff",
        "text": "Exact text",
        "native": "Qwen native\nexact text",
    }
    for ax, snapshot, title in zip(
        axes, ("base", "trained"), ("Fresh base", "Frozen trained B2026"), strict=True
    ):
        rows = [cells[f"{snapshot}-{mode}"] for mode in modes]
        if any(row["tasks"] != 40 for row in rows):
            raise ValueError("Figure requires forty primary families in every cell")
        positions = np.arange(len(modes))
        edited = ax.bar(
            positions,
            [row["tasks_with_applied_edit"] for row in rows],
            width=0.65,
            facecolor="#d9e3ea",
            edgecolor="#777777",
            label="At least one edit applied",
        )
        repaired = ax.bar(
            positions,
            [row["repairs"] for row in rows],
            width=0.65,
            color="#286da8",
            label="Fully verified repair",
        )
        ax.bar_label(edited, padding=3, fontsize=9)
        ax.bar_label(repaired, padding=-14, color="white", fontsize=9)
        ax.set(
            title=title,
            xticks=positions,
            xticklabels=[labels[mode] for mode in modes],
            ylim=(0, 44),
            yticks=np.arange(0, 41, 10),
        )
    axes[0].set_ylabel("Tasks out of 40 training families")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.supxlabel(
        "Previously inspected families; different edited subsets are not causal partitions",
        fontsize=9,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output / f"interface-repairs.{suffix}", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
