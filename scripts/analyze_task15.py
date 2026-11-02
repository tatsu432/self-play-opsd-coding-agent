"""Local Task 15 exposure curves and paired endpoints; missing arms stay missing."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
from analyze_task14_static import evaluations
from scipy.stats import t

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from selfplay_opsd.analysis import write_csv
from selfplay_opsd.config import RunConfig
from selfplay_opsd.core import read_json
from selfplay_opsd.storage import active_events, atomic_json


def endpoint(events, split):
    pending = defaultdict(list)
    complete = None
    for event in events:
        row = event["payload"]
        key = (event["segment_id"], row.get("iteration", event["completed_iteration"]), split)
        if event["kind"] == "evaluation" and row.get("split") == split:
            pending[key].append(row)
        elif event["kind"] == "evaluation_complete" and row.get("split") == split:
            records = pending.pop(key, [])
            if (
                len(records) != row["expected"]
                or len(records) != row["completed"]
                or any(r["success"] is None for r in records)
            ):
                raise ValueError("Incomplete final evaluation; denominator cannot be reduced")
            if len({r["task_id"] for r in records}) != len(records):
                raise ValueError("Duplicate final evaluation tasks")
            complete = records
    return complete


def paired_clustered(first, second, families):
    if first is None or second is None:
        return {"status": "missing"}
    a, b = {r["task_id"]: r for r in first}, {r["task_id"]: r for r in second}
    if a.keys() != b.keys() or not a:
        raise ValueError("Final A/B task sets differ")
    parent_deltas = defaultdict(list)
    for task_id in sorted(a):
        if a[task_id]["parent_id"] != b[task_id]["parent_id"]:
            raise ValueError("Paired task parent mismatch")
        parent_deltas[a[task_id]["parent_id"]].append(
            int(b[task_id]["success"]) - int(a[task_id]["success"])
        )
    clusters = defaultdict(list)
    for parent, values in parent_deltas.items():
        clusters[families[parent]].append(float(np.mean(values)))
    keys = sorted(clusters)
    rng = np.random.default_rng(2026)
    values = []
    for _ in range(2000):
        sampled = rng.integers(0, len(keys), len(keys))
        values.append(float(np.mean([value for i in sampled for value in clusters[keys[i]]])))
    return {
        "status": "complete",
        "b_minus_a_parent_macro": float(np.mean([np.mean(v) for v in parent_deltas.values()])),
        "family_clustered_95_interval": np.quantile(values, [0.025, 0.975]).tolist(),
        "families": len(keys),
        "parents": len(parent_deltas),
        "tasks": len(a),
        "a_successes": sum(r["success"] for r in a.values()),
        "b_successes": sum(r["success"] for r in b.values()),
        "uncertainty": "evaluation families/parents, not training-seed variation",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    expected_counts = read_json(args.data / "manifest.json")["admitted_tasks"]
    families = {
        r["parent"]["parent_id"]: r["parent"]["family_id"]
        for r in read_json(args.data / "private/parents.json")
    }
    arms, curves, health, outcomes = {}, [], [], []
    common_by_seed = {}
    for item in manifest["arms"]:
        root = Path(item["run_dir"])
        if not (root / "resolved_config.json").is_file():
            continue
        resolved = read_json(root / "resolved_config.json")
        cfg = RunConfig.model_validate(
            {k: v for k, v in resolved.items() if k in RunConfig.model_fields}
        )
        common = cfg.scientific_settings(common=True)
        seed = item["seed"]
        if cfg.seed != seed or cfg.method != item["method"] or cfg.online_supply is None:
            raise ValueError("Run identity does not match the analysis manifest")
        if seed in common_by_seed and common_by_seed[seed] != common:
            raise ValueError("Paired arms differ in common scientific settings")
        common_by_seed[seed] = common
        events = active_events(root)
        arm = (seed, item["method"])
        if arm in arms:
            raise ValueError("Duplicate logical arm in analysis manifest")
        rows = evaluations(events, item["allocations"])
        curves.extend(
            {
                "seed": seed,
                "method": item["method"],
                **{k: v for k, v in row.items() if k != "tasks"},
            }
            for row in rows
        )
        finals = {split: endpoint(events, split) for split in ("final_id", "external")}
        for split, records in finals.items():
            if records and len(records) != expected_counts[split]:
                raise ValueError("Final evaluation count differs from the immutable data manifest")
            outcomes.extend(
                {"seed": seed, "method": item["method"], "split": split, **r} for r in records or []
            )
        commits = [e["payload"] for e in events if e["kind"] == "iteration_commit"]
        for row in commits:
            health.append(
                {
                    "seed": seed,
                    "method": item["method"],
                    "iteration": row["iteration"],
                    "episodes": row["metrics"]["student"]["cumulative_episodes"],
                    "metrics": row["metrics"],
                }
            )
        completion = (
            read_json(root / "complete.json") if (root / "complete.json").exists() else None
        )
        if completion and (
            completion["scientific_hash"] != cfg.scientific_hash
            or not cfg.online_supply.target_episodes
            <= completion["episodes"]
            <= cfg.online_supply.target_episodes + 12
        ):
            raise ValueError("Completion manifest does not match the frozen exposure endpoint")
        arms[arm] = {
            "finals": finals,
            "completion": completion,
            "initial": read_json(root / "initial_versions.json"),
            "last": commits[-1] if commits else None,
            "allocated_gpu_hours": sum(
                (a["end"] - a["start"]) * a["gpus"] / 3600 for a in item["allocations"]
            ),
        }
    comparisons, summaries = [], []
    for seed in (2026, 2027, 2028):
        a, b = arms.get((seed, "A")), arms.get((seed, "B"))
        if a and b and a["initial"] != b["initial"]:
            raise ValueError("Paired initialization differs")
        for method in ("A", "B"):
            arm = arms.get((seed, method))
            summaries.append(
                {
                    "seed": seed,
                    "method": method,
                    "status": "complete"
                    if arm and arm["completion"] and all(arm["finals"].values())
                    else "incomplete",
                    "completion": arm["completion"] if arm else None,
                    "allocated_gpu_hours": arm["allocated_gpu_hours"] if arm else None,
                }
            )
        for method in ("A", "B"):
            arm = arms.get((seed, method))
            for split in ("final_id", "external"):
                records = arm["finals"][split] if arm else None
                summaries[-2 if method == "A" else -1][split + "_successes"] = (
                    sum(int(r["success"]) for r in records) if records else None
                )
                summaries[-2 if method == "A" else -1][split + "_tasks"] = (
                    len(records) if records else None
                )
        for split in ("final_id", "external"):
            comparisons.append(
                {
                    "seed": seed,
                    "split": split,
                    **paired_clustered(
                        a["finals"][split] if a else None,
                        b["finals"][split] if b else None,
                        families,
                    ),
                }
            )
    seed_variation = {}
    for split in ("final_id", "external"):
        differences = [
            r["b_minus_a_parent_macro"]
            for r in comparisons
            if r["split"] == split and r["status"] == "complete"
        ]
        n = len(differences)
        mean = float(np.mean(differences)) if n else None
        half_width = (
            float(t.ppf(0.975, n - 1) * np.std(differences, ddof=1) / np.sqrt(n)) if n > 1 else None
        )
        seed_variation[split] = {
            "n": n,
            "paired_seed_differences": differences,
            "mean": mean,
            "seed_t_95_interval": [mean - half_width, mean + half_width] if n > 1 else None,
            "limitation": "three training seeds remain very uncertain",
        }
    atomic_json(
        args.output / "summary.json",
        {"arms": summaries, "paired": comparisons, "training_seed_variation": seed_variation},
    )
    write_csv(args.output / "summary.csv", summaries)
    write_csv(args.output / "learning-curves.csv", curves)
    write_csv(args.output / "health.csv", health)
    write_csv(args.output / "final-outcomes.csv", outcomes)
    if curves:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for seed, method in sorted(arms):
            rows = [r for r in curves if r["seed"] == seed and r["method"] == method]
            for axis, field in zip(
                axes, ("episodes", "tokens", "student_steps", "allocated_gpu_hours"), strict=True
            ):
                axis.plot(
                    [r[field] for r in rows],
                    [r["successes"] for r in rows],
                    marker="o",
                    label=f"{method}/{seed}",
                )
                axis.set(xlabel=field, ylabel="Development repairs (of 41)")
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(args.output / f"learning-curves.{suffix}")
        plt.close(fig)


if __name__ == "__main__":
    main()
