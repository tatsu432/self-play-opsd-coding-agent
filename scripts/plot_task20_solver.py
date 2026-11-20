"""Plot the completed Task 20 solver pilot from verified artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from selfplay_opsd.core import read_json
from selfplay_opsd.storage import CheckpointManager, file_hash


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    marker = read_json(args.solver / "complete-solver.json")
    checkpoint = args.solver / "checkpoints" / marker["checkpoint"]
    metadata = CheckpointManager.verify(checkpoint)
    if (
        file_hash(checkpoint / "complete.json") != marker["checkpoint_sha256"]
        or metadata["identity"] != marker["identity"]
        or metadata["snapshot"] != marker["snapshot"]
    ):
        raise ValueError("Solver checkpoint identity or checksums differ")
    iterations = []
    for path in sorted((args.solver / "iterations").glob("*.json")):
        record = read_json(path)
        if record["batch"] is None:
            continue
        batch = args.solver / "batches" / record["batch"]["path"]
        if (
            record["identity"] != marker["identity"]
            or file_hash(batch) != record["batch"]["sha256"]
        ):
            raise ValueError("Training iteration identity or checksum differs")
        iterations.append(record["metrics"])
    if len(iterations) != metadata["iteration"]:
        raise ValueError("Training iterations are incomplete")
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout="constrained")
    labels = ("Training set", "Training probe")
    for index, name in enumerate(("solver_train", "solver_probe")):
        blocks = marker["result"][name]
        for block, offset, color in (("paired-0", -0.11, "#356a80"), ("paired-1", 0.11, "#bd8058")):
            item = blocks[block]
            before, after, count = item["before"], item["after"], item["count"]
            axes[0].plot(
                [index - 0.2 + offset, index + 0.2 + offset],
                [before / count, after / count],
                marker="o",
                color=color,
                linewidth=1.7,
                label=block if index == 0 else None,
            )
            axes[0].annotate(
                f"{before}/{count}",
                (index - 0.2 + offset, before / count),
                xytext=(0, -13),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
            axes[0].annotate(
                f"{after}/{count}",
                (index + 0.2 + offset, after / count),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    axes[0].set(
        xticks=range(2),
        xticklabels=labels,
        ylim=(0, 1),
        ylabel="Verified repair rate",
        title="Repeated training-only checks",
    )
    axes[0].legend(frameon=False, fontsize=8)
    episodes = [row["attempted_episodes"] for row in iterations]
    axes[1].plot(
        episodes,
        [row["learning"]["mixed_reward_fraction"] for row in iterations],
        color="#356a80",
        marker="o",
    )
    axes[1].set(
        ylim=(0, 1),
        ylabel="Mixed-reward fraction",
        xlabel="Attempted training episodes",
        title="Solver reward variation",
    )
    axes[2].plot(
        episodes,
        [row["shared"]["update_norm"] for row in iterations],
        color="#bd8058",
        marker="o",
        label="Adapter displacement",
    )
    axes[2].plot(
        episodes,
        [row["rl"]["norm"] for row in iterations],
        color="#356a80",
        marker="o",
        label="RL gradient norm",
    )
    axes[2].set(
        xlabel="Attempted training episodes", ylabel="L2 norm", title="Eight nonzero solver updates"
    )
    axes[2].legend(frameon=False, fontsize=8)
    figure.suptitle("Task 20 solver-only pilot · seed 2026")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220)
    plt.close(figure)


if __name__ == "__main__":
    main()
