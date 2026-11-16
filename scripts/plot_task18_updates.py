"""Plot actual restored-state update comparisons, separately from historical events."""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.replays.read_text())
    labels = ["Mixed group\ncalibrated 2028", "Uniform groups\ncalibrated 2026"]
    if len(rows) != 2 or any("updates" not in r for r in rows):
        raise ValueError("Expected the two completed state-restored update sources")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for offset, key, label, color in [
        (-0.18, "repeat", "Identical-operation repeat", "#777777"),
        (0.18, "rescored_diagnostic", "Diagnostic denominator substitution", "#286da8"),
    ]:
        updates = [row["updates"]["comparisons"][key]["update"] for row in rows]
        values = [100 * u["relative_l2"] for u in updates]
        angles = [math.degrees(math.acos(max(-1, min(1, u["cosine"])))) for u in updates]
        for ax, data in zip(axes, (values, angles), strict=True):
            bars = ax.bar(np.arange(2) + offset, data, width=0.36, label=label, color=color)
            ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    axes[0].set(
        ylabel="Relative L2 difference (%)",
        title="Actual Adam displacement difference",
        ylim=(0, 24),
    )
    axes[1].set(
        ylabel="Angle between updates (degrees)", title="Actual Adam update direction", ylim=(0, 14)
    )
    for ax in axes:
        ax.set(xticks=[0, 1], xticklabels=labels)
    axes[0].legend(fontsize=8)
    args.output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output / f"update-sensitivity.{suffix}", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
