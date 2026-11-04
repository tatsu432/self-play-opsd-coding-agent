"""Export privacy-safe Task 15 health curves from committed private events."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze_task14_static import allocated_hours

from selfplay_opsd.storage import active_events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    rows = []
    for arm in manifest["arms"]:
        totals = {"attempted": 0, "valid": 0, "duplicates": 0, "groups": 0, "mixed": 0}
        for event in active_events(Path(arm["run_dir"])):
            if event["kind"] != "iteration_commit":
                continue
            metrics = event["payload"]["metrics"]
            prop, student, supply = metrics["proposer"], metrics["student"], metrics["supply"]
            gpu_hours = allocated_hours(arm["allocations"], event["wall_timestamp"])
            for src, dest in (
                ("attempted_count", "attempted"),
                ("valid_count", "valid"),
                ("duplicate_count", "duplicates"),
            ):
                totals[dest] += prop.get(src, 0)
            for key, count in student.get("success_group_mix", {}).items():
                size = int(key)
                totals["groups"] += count
                totals["mixed"] += count if 0 < size < 4 else 0
            rows.append(
                {
                    "method": arm["method"],
                    "seed": arm["seed"],
                    "episodes": student["cumulative_episodes"],
                    "generated_tokens": student["cumulative_generated_tokens"],
                    "allocated_gpu_hours": gpu_hours,
                    "student_steps": metrics["train"]["student_steps"],
                    "proposer_steps": metrics["train"]["proposer_steps"],
                    "proposal_attempts": totals["attempted"],
                    "valid_proposals": totals["valid"],
                    "duplicate_proposals": totals["duplicates"],
                    "mixed_groups": totals["mixed"],
                    "groups": totals["groups"],
                    "valid_fraction_cumulative": totals["valid"] / totals["attempted"]
                    if totals["attempted"]
                    else None,
                    "mixed_fraction_cumulative": totals["mixed"] / totals["groups"]
                    if totals["groups"]
                    else None,
                    "mixed_groups_per_gpu_hour": totals["mixed"] / gpu_hours if gpu_hours else None,
                    "invalid_action_fraction": student.get("invalid_action_fraction"),
                    "logprob_mismatch_mean": student.get("pre_update_logprob_abs_mean"),
                    "logprob_mismatch_max": student.get("pre_update_logprob_abs_max"),
                    "sd_gap_mean": metrics["distill"].get("gap", {}).get("mean"),
                    "sd_gap_clip_fraction": metrics["distill"].get("gap_clip_fraction"),
                    "gate_mean": metrics["distill"].get("gate", {}).get("mean"),
                    "gradient_cosine": student.get("gradient_interaction", {}).get("cosine"),
                    "gradient_ratio": student.get("gradient_interaction", {}).get("ratio"),
                    "fallback_groups": supply.get("cumulative_fallback_groups"),
                    "online_tasks": supply.get("online_tasks"),
                    "reservoir_tasks": supply.get("reservoir_tasks"),
                    "replay_samples": supply.get("historical_replay_samples"),
                }
            )
    if not rows:
        raise ValueError("No committed training events")
    with (args.output / "health-curves.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    metrics = [
        ("valid_fraction_cumulative", "Cumulative proposal validity"),
        ("mixed_fraction_cumulative", "Cumulative mixed-group fraction"),
        ("invalid_action_fraction", "Invalid action fraction per update"),
        ("logprob_mismatch_mean", "Mean behavior/scorer log-probability mismatch"),
    ]
    for ax, (key, title) in zip(axes.flat, metrics, strict=True):
        for arm in manifest["arms"]:
            data = [
                r
                for r in rows
                if r["method"] == arm["method"] and r["seed"] == arm["seed"] and r[key] is not None
            ]
            ax.plot(
                [r["episodes"] for r in data],
                [r[key] for r in data],
                label=f"{arm['method']} {arm['seed']}",
                linewidth=0.9,
            )
        ax.set_title(title)
        ax.set_xlabel("Attempted training episodes")
        ax.grid(alpha=0.2)
    axes.flat[0].legend(fontsize="small", ncol=2)
    fig.savefig(args.output / "health-curves.png", dpi=170)
    fig.savefig(args.output / "health-curves.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
