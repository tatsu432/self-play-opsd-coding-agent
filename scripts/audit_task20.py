"""Reconstruct Task 19 training signal from verified private batch exports."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from statistics import mean

from selfplay_opsd.core import read_json
from selfplay_opsd.storage import atomic_json, file_hash


def audit_arm(root: Path, token_limit: int) -> list[dict]:
    rows = []
    for path in sorted((root / "iterations").glob("*.json")):
        record = read_json(path)
        if record["batch"] is None:
            continue
        batch_file = root / "batches" / record["batch"]["path"]
        if file_hash(batch_file) != record["batch"]["sha256"]:
            raise ValueError(f"Historical batch checksum differs in {path.name}")
        batch = read_json(batch_file)
        metric = record["metrics"]
        histogram: Counter[int] = Counter()
        failures: Counter[str] = Counter()
        group_tokens = tokens = 0
        families = set()
        for repairs in batch["repairs"].values():
            if len(repairs) != 4:
                raise ValueError("Task 19 audit expects four rollouts per task")
            success = sum(episode["success"] for episode in repairs)
            histogram[success] += 1
            failures.update(episode["failure"] for episode in repairs)
            families.update(episode["parent_id"] for episode in repairs)
            length = sum(len(turn["ids"]) for episode in repairs for turn in episode["turns"])
            tokens += length
            if 0 < success < 4:
                group_tokens += length
        if tokens != sum(
            metric["student"][kind]
            for kind in ("reasoning_tokens", "tool_tokens", "other_sampled_tokens")
        ):
            raise ValueError(f"Historical sampled-token count differs in {path.name}")
        proposals = metric["proposer"]
        shared = metric["shared"]
        rows.append(
            {
                "iteration": record["iteration"],
                "episodes": metric["batch_episodes"],
                "parents": len(families),
                "proposal_candidates": proposals["attempted"],
                "valid_proposals": proposals["valid"],
                "proposal_groups": proposals["groups"],
                "mutation_families": proposals["families"],
                "distinct_valid_reward_groups": proposals["distinct_reward_groups"],
                "reward_histogram": [histogram[n] for n in range(5)],
                "sampled_tokens": tokens,
                "fixed_normalization_utilization": tokens / (4 * token_limit * histogram.total())
                if histogram.total()
                else None,
                "mixed_group_tokens": group_tokens,
                "failures": dict(failures),
                "student_outcomes": metric["student"],
                "rl": metric["rl"],
                "opsd": metric["opsd"],
                "shared": shared,
                "step_without_rl": shared["optimizer_step"] and metric["rl"]["norm"] == 0,
            }
        )
    if not rows:
        raise ValueError(f"No archived training iterations under {root}")
    return rows


def draw_signal(arms: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"axes.spines.right": False, "axes.spines.top": False, "font.size": 11})
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained")
    labels = ["Control", "A", "B", "C", "D"]
    values = list(arms.values())
    totals = [0] * len(values)
    for reward, color in zip(
        range(5), ("#697586", "#c4a53d", "#9a6d21", "#c6813f", "#52756b"), strict=True
    ):
        bars = [row["groups"][reward] for row in values]
        axes[0].bar(labels, bars, bottom=totals, color=color, label=f"{reward}/4")
        totals = [a + b for a, b in zip(totals, bars, strict=True)]
    axes[0].set(ylabel="Student rollout groups", ylim=(0, 30), title="Verified outcomes per group")
    axes[0].legend(title="Repairs", ncols=3, frameon=False, fontsize=8)
    solver = [row["nonzero_rl_iterations"] for row in values]
    other = [row["steps_without_rl"] for row in values]
    axes[1].bar(labels, solver, color="#31657b", label="Nonzero solver RL")
    axes[1].bar(labels, other, bottom=solver, color="#ac795a", label="Zero solver RL")
    axes[1].set(
        ylabel="Joint optimizer steps", ylim=(0, 9), title="Steps by current solver RL signal"
    )
    axes[1].legend(frameon=False, fontsize=8)
    figure.suptitle("Task 19 training signal at 96 attempted episodes")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=240)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    control = read_json(args.control)
    limit = control["episode"]["max_generated_tokens"]
    manifest = read_json(args.manifest)
    arms = {}
    for arm in ("control", "A", "B", "C", "D"):
        rows = audit_arm(Path(manifest[arm]["path"]), limit)
        episodes = sum(row["episodes"] for row in rows)
        tokens = sum(row["sampled_tokens"] for row in rows)
        arms[arm] = {
            "groups": [sum(row["reward_histogram"][n] for row in rows) for n in range(5)],
            "episodes": episodes,
            "mean_sampled_response_tokens": tokens / episodes,
            "sampled_tokens": tokens,
            "mixed_group_tokens": sum(row["mixed_group_tokens"] for row in rows),
            "optimizer_steps": sum(row["shared"]["optimizer_step"] for row in rows),
            "nonzero_rl_iterations": sum(row["rl"]["norm"] > 0 for row in rows),
            "steps_without_rl": sum(row["step_without_rl"] for row in rows),
            "mean_rl_norm": mean(row["rl"]["norm"] for row in rows),
            "mean_weighted_proposer_norm": mean(
                row["shared"]["weighted_proposer_gradient_norm"] for row in rows
            ),
            "mean_weighted_opsd_norm": mean(
                row["opsd"]["gradient_norm"] * control["sd_scale"] for row in rows
            ),
            "gradient_norms_finite": all(
                math.isfinite(row["shared"]["joint_gradient_norm"]) for row in rows
            ),
            "iterations": rows,
        }
    atomic_json(
        args.output, {"archive": "Task 19 main endpoints", "token_limit": limit, "arms": arms}
    )
    if args.figure is not None:
        draw_signal(arms, args.figure)


if __name__ == "__main__":
    main()
