"""Compare static student arms at common episodes and scheduler compute."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import t

from selfplay_opsd.storage import active_events, atomic_json


def allocated_hours(allocations, timestamp):
    return sum(
        max(0, min(timestamp, item["end"]) - item["start"]) * item["gpus"] / 3600
        for item in allocations
    )


def evaluations(events, allocations):
    exposure = {0: {"episodes": 0, "student_steps": 0, "tokens": 0}}
    pending = defaultdict(list)
    rows = []
    for event in events:
        payload = event["payload"]
        if event["kind"] == "iteration_commit":
            exposure[payload["iteration"]] = {
                "episodes": payload["metrics"]["student"]["cumulative_episodes"],
                "student_steps": payload["metrics"]["train"]["student_steps"],
                "tokens": payload["metrics"]["student"]["cumulative_generated_tokens"],
            }
        elif event["kind"] == "evaluation" and payload.get("split") == "dev_monitor":
            pending[(event["segment_id"], payload["iteration"])].append(payload)
        elif event["kind"] == "evaluation_complete" and payload.get("split") == "dev_monitor":
            key = (event["segment_id"], event["completed_iteration"])
            records = pending.pop(key, [])
            if len(records) != payload["completed"] or len(records) != payload["expected"]:
                raise ValueError("Incomplete development evaluation")
            if key[1] not in exposure or any(r["success"] is None for r in records):
                raise ValueError("Evaluation exposure or outcome missing")
            tasks = {
                r["task_id"]: {"parent_id": r["parent_id"], "success": int(r["success"])}
                for r in records
            }
            if len(tasks) != len(records):
                raise ValueError("Duplicate development task")
            rows.append(
                {
                    "iteration": key[1],
                    **exposure[key[1]],
                    "allocated_gpu_hours": allocated_hours(allocations, event["wall_timestamp"]),
                    "successes": sum(t["success"] for t in tasks.values()),
                    "tasks": tasks,
                }
            )
    return rows


def paired_interval(first, second):
    if first.keys() != second.keys() or any(
        first[k]["parent_id"] != second[k]["parent_id"] for k in first
    ):
        raise ValueError("Development tasks or parents differ")
    by_parent = defaultdict(list)
    for task in first:
        by_parent[first[task]["parent_id"]].append(second[task]["success"] - first[task]["success"])
    values = np.array([np.mean(by_parent[p]) for p in sorted(by_parent)])
    samples = (
        np.random.default_rng(2026)
        .choice(values, size=(2000, len(values)), replace=True)
        .mean(axis=1)
    )
    return {
        "mean_parent_macro_difference": float(values.mean()),
        "bootstrap_95_interval": np.quantile(samples, [0.025, 0.975]).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    arms = []
    for item in manifest["arms"]:
        root = Path(item["run_dir"])
        records = evaluations(active_events(root), item["allocations"])
        if {r["episodes"] for r in records} != {0, 80, 160}:
            raise ValueError("Expected complete evaluations at 0, 80 and 160 episodes")
        arms.append(
            {
                "arm": item["arm"],
                "seed": item["seed"],
                "evaluations": records,
                "total_allocated_gpu_hours": sum(
                    (a["end"] - a["start"]) * a["gpus"] / 3600 for a in item["allocations"]
                ),
            }
        )
    pairs = []
    by_key = {(item["arm"], item["seed"]): item for item in arms}
    for seed in sorted({item["seed"] for item in arms}):
        for other in ("gated", "ungated"):
            if ("grpo", seed) not in by_key or (other, seed) not in by_key:
                continue
            a = {row["episodes"]: row for row in by_key[("grpo", seed)]["evaluations"]}
            b = {row["episodes"]: row for row in by_key[(other, seed)]["evaluations"]}
            for count in (80, 160):
                pairs.append(
                    {
                        "seed": seed,
                        "comparison": other + "-minus-grpo",
                        "episodes": count,
                        **paired_interval(a[count]["tasks"], b[count]["tasks"]),
                    }
                )
    seed_summaries = []
    for comparison in sorted({row["comparison"] for row in pairs}):
        for count in (80, 160):
            rows = [r for r in pairs if r["comparison"] == comparison and r["episodes"] == count]
            values = np.array([r["mean_parent_macro_difference"] for r in rows])
            if not len(values):
                continue
            interval = None
            if len(values) > 1:
                error = t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
                interval = [float(values.mean() - error), float(values.mean() + error)]
            seed_summaries.append(
                {
                    "comparison": comparison,
                    "episodes": count,
                    "seeds": len(values),
                    "mean_difference": float(values.mean()),
                    "seed_t_95_interval": interval,
                    "seed_differences": values.tolist(),
                }
            )
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(
        args.output / "comparison.json",
        {"arms": arms, "paired": pairs, "seed_summaries": seed_summaries},
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for item in arms:
        rows = sorted(item["evaluations"], key=lambda row: row["episodes"])
        label = f"{item['arm']} seed {item['seed']}"
        axes[0].plot(
            [r["episodes"] for r in rows], [r["successes"] for r in rows], marker="o", label=label
        )
        axes[1].plot(
            [r["allocated_gpu_hours"] for r in rows],
            [r["successes"] for r in rows],
            marker="o",
            label=label,
        )
    axes[0].set(xlabel="Attempted training episodes", ylabel="Development repairs")
    axes[1].set(xlabel="Allocated GPU-hours through evaluation", ylabel="Development repairs")
    axes[1].legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(args.output / "learning-curves.png", dpi=180)
    fig.savefig(args.output / "learning-curves.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
