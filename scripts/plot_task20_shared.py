"""Plot checkpoint-verified shared-pilot solver and proposer signals."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sync_task20_telemetry import records

from selfplay_opsd.core import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records(args.pilot, "shared-pilot")
    rows = [
        row["metrics"]
        for path in sorted((args.pilot / "iterations").glob("*.json"))
        if (row := read_json(path))["batch"] is not None
    ]
    labels = [str(row["attempted_episodes"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    bottom = [0] * len(rows)
    for key, label, color in (
        ("all_fail_groups", "All fail", "#cf785b"),
        ("mixed_groups", "Mixed", "#447ca3"),
        ("all_pass_groups", "All pass", "#82a77b"),
    ):
        values = [row["student"][key] for row in rows]
        axes[0].bar(labels, values, bottom=bottom, label=label, color=color)
        bottom = [a + b for a, b in zip(bottom, values, strict=True)]
    axes[0].set(
        xlabel="Attempted solver episodes", ylabel="Solver groups", title="Solver reward signal"
    )
    axes[0].legend(frameon=True, facecolor="white", framealpha=1)
    axes[1].bar(
        labels,
        [row["proposer"]["groups"] for row in rows],
        color="#d8dde2",
        label="Proposal groups",
    )
    axes[1].bar(
        labels,
        [row["proposer"]["distinct_reward_groups"] for row in rows],
        color="#447ca3",
        label="Distinct valid rewards",
    )
    axes[1].set(
        xlabel="Attempted solver episodes",
        ylabel="Proposal groups",
        title="Valid-task discrimination",
    )
    axes[1].legend(frameon=True, facecolor="white", framealpha=1)
    fig.suptitle("Shared-control pilot: one seed, 64 episodes", fontsize=12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
