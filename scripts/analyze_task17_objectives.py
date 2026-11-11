"""Analyze completed Task 17 fixed-curriculum objective runs from private records."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from selfplay_opsd.storage import active_events, atomic_json, file_hash
from selfplay_opsd.task16_analysis import paired_statistics


def load_run(item, pool_info, pool_hash, control_hash, control):
    root = Path(item["run_dir"])
    manifest = json.loads((root / "manifest.json").read_text())
    complete = json.loads((root / "complete.json").read_text())
    resolved = json.loads((root / "resolved_config.json").read_text())
    expected_method = {"A": "A", "B": "B", "C": "A", "D": "GRPO"}[item["arm"]]
    expected_lambda = control["calibration_coefficient"] if item["arm"] == "C" else 0.1
    if (
        resolved["method"] != expected_method
        or resolved["objective"]["sd_lambda"] != expected_lambda
        or resolved["sd_gap_definition"] != "same_scorer_v2"
        or resolved["episodes_per_task"] != 4
        or resolved["iterations"] != 10
    ):
        raise ValueError(f"Run {root} has an incorrect objective or exposure")
    for key, value in {
        "arm": item["arm"],
        "seed": item["seed"],
        "pool_sha256": pool_hash,
        "control_sha256": control_hash,
        "interface": "replace_text",
    }.items():
        if manifest[key] != value:
            raise ValueError(f"Run {root} differs in {key}")
    if complete["episodes"] != 80 or complete["proposer_steps"] != 0:
        raise ValueError(f"Run {root} is incomplete or changed the proposer")
    events = active_events(root)
    iterations = [e for e in events if e["kind"] == "iteration_commit"]
    if [e["completed_iteration"] for e in iterations] != list(range(1, 11)):
        raise ValueError(f"Run {root} has missing or duplicated committed iterations")
    expected = {task for task, row in pool_info.items() if row["purpose"] == "development"}
    curve, records, hashes = [], {}, {}
    for event in events:
        if event["kind"] != "task17_development":
            continue
        data = event["payload"]
        episode = int(data["episodes"])
        if episode not in {0, 40, 80} or episode in records:
            raise ValueError(f"Unexpected development milestone in {root}")
        path = root / data["path"]
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Development artifact lies outside run")
        raw = json.loads(path.read_text())
        if len(raw) != len(expected) or {r["task_id"] for r in raw} != expected:
            raise ValueError("Development task set differs")
        records[episode] = [
            {
                "task_id": r["task_id"],
                "parent_id": pool_info[r["task_id"]]["task"]["parent_id"],
                "success": bool(r["reward"]),
            }
            for r in raw
        ]
        hashes[str(path)] = file_hash(path)
        curve.append(
            {
                "episodes": episode,
                "successes": sum(r["success"] for r in records[episode]),
                "tasks": len(raw),
            }
        )
    if set(records) != {0, 40, 80}:
        raise ValueError(f"Run {root} lacks required development milestones")
    metrics = [e["payload"]["metrics"]["student"] for e in iterations]
    hours = float(item["allocated_gpu_hours"])
    if not np.isfinite(hours) or hours <= 0:
        raise ValueError("Actual allocated GPU-hours must be positive")
    identity = {
        key: manifest[key]
        for key in (
            "dataset_hash",
            "episode",
            "model",
            "snapshot",
            "task_ids",
            "prompt_hash",
            "scientific_code_hash",
        )
    }
    return (
        {
            "arm": item["arm"],
            "seed": item["seed"],
            "curve": sorted(curve, key=lambda x: x["episodes"]),
            "allocated_gpu_hours": hours,
            "rollout_successes": sum(x["success_numerator"] for x in metrics),
            "rollout_episodes": sum(x["success_denominator"] for x in metrics),
            "generated_tokens": metrics[-1]["cumulative_generated_tokens"],
            "student_steps": complete["student_steps"],
            "max_scoring_mismatch": max(x["pre_update_logprob_abs_max"] for x in metrics),
        },
        records,
        hashes,
        identity,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--pool", required=True, type=Path)
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve existing analysis; select a new output")
    items = json.loads(args.manifest.read_text())["runs"]
    if (
        not items
        or any(r["arm"] not in "ABCD" or r["seed"] not in {2026, 2027, 2028} for r in items)
        or len({(r["arm"], r["seed"]) for r in items}) != len(items)
    ):
        raise ValueError("Run arm/seed identities must be unique")
    pool = json.loads(args.pool.read_text())
    info = {r["task"]["task_id"]: r for r in pool["rows"]}
    families = {r["task"]["parent_id"]: r["family_id"] for r in pool["rows"]}
    control_hash, pool_hash = file_hash(args.control), file_hash(args.pool)
    control = json.loads(args.control.read_text())
    if control["pool_sha256"] != pool_hash or control["episodes"] != 80:
        raise ValueError("Control does not match pool and endpoint")
    summaries, by_run, hashes, identities = [], {}, {}, []
    for item in items:
        summary, records, used, identity = load_run(item, info, pool_hash, control_hash, control)
        summaries.append(summary)
        identities.append(identity)
        by_run[item["arm"], item["seed"]] = records
        hashes.update(used)
        root = Path(item["run_dir"])
        for name in ("manifest.json", "complete.json", "resolved_config.json"):
            hashes[str(root / name)] = file_hash(root / name)
    if any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("Run model, curriculum, prompt or scientific identity differs")
    comparisons = []
    for first, second in (("D", "B"), ("A", "B"), ("C", "B")):
        seeds = sorted(
            {seed for arm, seed in by_run if arm == first}
            & {seed for arm, seed in by_run if arm == second}
        )
        for seed in seeds:
            for episode in (40, 80):
                comparisons.append(
                    {
                        "first": first,
                        "second": second,
                        "seed": seed,
                        "episodes": episode,
                        "development": paired_statistics(
                            by_run[first, seed][episode], by_run[second, seed][episode], families
                        ),
                    }
                )
    args.output.mkdir(parents=True)
    atomic_json(
        args.output / "summary.json",
        {
            "runs": summaries,
            "comparisons": comparisons,
            "scope": "Family bootstrap is within seed; seed variation is separate.",
            "input_sha256": hashes
            | {
                str(p): file_hash(p)
                for p in (args.manifest, args.pool, args.control, Path(__file__))
            },
        },
    )
    labels = {"A": "Ungated", "B": "Gated", "C": "Calibrated ungated", "D": "GRPO"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for arm in "DBAC":
        group = [r for r in summaries if r["arm"] == arm]
        if not group:
            continue
        curves = np.array([[p["successes"] for p in r["curve"]] for r in group])
        axes[0].plot(
            [0, 40, 80], curves.mean(axis=0), marker="o", label=f"{labels[arm]} (n={len(group)})"
        )
        for values in curves:
            axes[0].plot([0, 40, 80], values, alpha=0.25)
        axes[1].bar(labels[arm], np.mean([r["allocated_gpu_hours"] for r in group]))
    axes[0].set(
        xlabel="Attempted training episodes",
        ylabel="Independent development repairs /16",
        ylim=(0, 16),
    )
    axes[0].legend(fontsize=8)
    axes[1].set(ylabel="Allocated H200-hours per run", title="Actual GPU allocation")
    axes[1].tick_params(axis="x", rotation=20)
    for ext in ("png", "pdf"):
        fig.savefig(args.output / f"objective-learning-efficiency.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    main()
