"""Privacy-safe plot of historical aggregate numerical diagnostics, not new GPU evidence."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.summary.read_text())
    labels = {"A": "Nominal ungated", "B": "Gated", "C": "Calibrated ungated", "D": "GRPO"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained")
    for arm in ("D", "B", "A", "C"):
        batches = [b for r in report["runs"] if r["arm"] == arm for b in r["batches"]]
        axes[0].scatter(
            [b["iteration"] for b in batches],
            [b["pre_update_logprob_abs_max"] for b in batches],
            label=labels[arm],
            s=22,
            alpha=0.7,
        )
        axes[1].scatter(
            [b["iteration"] for b in batches],
            [100 * b["pre_update_logprob_abs_above_02_fraction"] for b in batches],
            label=labels[arm],
            s=22,
            alpha=0.7,
        )
    axes[0].set(
        ylabel="Maximum absolute log-probability difference",
        title="Largest sampled-token discrepancy",
    )
    axes[1].set(ylabel="Sampled tokens above 0.2 (%)", title="Affected fraction within each batch")
    for ax in axes:
        ax.set(xlabel="Training iteration (three seeds per method)", xticks=[1, 4, 7, 10])
    axes[0].legend(fontsize=8)
    args.output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output / f"numerical-history.{suffix}", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
