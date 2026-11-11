"""Analyze completed Task 17 fixed-curriculum objective runs from private records."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from selfplay_opsd.data import TaskRepository
from selfplay_opsd.storage import active_events, atomic_json, file_hash
from selfplay_opsd.task16_analysis import paired_statistics


def validate_seed_code_hashes(code_hashes, declared_code, note):
    if declared_code:
        if not note:
            raise ValueError(
                "Cross-revision analysis requires an explicit compatibility explanation"
            )
        if any(declared_code.get(str(seed)) != code for seed, code in code_hashes.items()):
            raise ValueError("Run scientific code differs from the declared seed revision")
    elif len(set(code_hashes.values())) != 1:
        raise ValueError("Different seed revisions require explicit compatibility evidence")


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
    monitor_curve, monitor_records = [], {}
    for iteration in (0, 5, 10):
        evaluations = [
            e["payload"]
            for e in events
            if e["kind"] == "evaluation"
            and e["completed_iteration"] == iteration
            and e["payload"]["split"] == "dev_monitor"
        ]
        if len(evaluations) != 41 or len({x["task_id"] for x in evaluations}) != 41:
            raise ValueError(f"Run {root} has incomplete monitoring evaluation at {iteration}")
        monitor_curve.append(
            {
                "episodes": iteration * 8,
                "successes": sum(bool(x["success"]) for x in evaluations),
                "tasks": len(evaluations),
            }
        )
        monitor_records[iteration * 8] = [
            {"task_id": x["task_id"], "parent_id": x["parent_id"], "success": bool(x["success"])}
            for x in evaluations
        ]
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
    identity["common_training"] = {
        key: resolved[key]
        for key in (
            "iterations",
            "parents_per_iteration",
            "proposals_per_parent",
            "episodes_per_task",
            "sd_gap_definition",
            "gradient_diagnostics",
            "hardware",
            "sandbox",
        )
    }
    identity["common_training"]["objective"] = {
        key: value for key, value in resolved["objective"].items() if key != "sd_lambda"
    }
    # Allocated workers bind different loopback ports; execution limits remain paired.
    identity["common_training"]["sandbox"] = {
        key: value for key, value in resolved["sandbox"].items() if key != "remote_url"
    }
    return (
        {
            "arm": item["arm"],
            "seed": item["seed"],
            "curve": sorted(curve, key=lambda x: x["episodes"]),
            "monitor_curve": monitor_curve,
            "allocated_gpu_hours": hours,
            "rollout_successes": sum(x["success_numerator"] for x in metrics),
            "rollout_episodes": sum(x["success_denominator"] for x in metrics),
            "generated_tokens": metrics[-1]["cumulative_generated_tokens"],
            "student_steps": complete["student_steps"],
            "max_scoring_mismatch": max(x["pre_update_logprob_abs_max"] for x in metrics),
        },
        records,
        monitor_records,
        hashes,
        identity,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--pool", required=True, type=Path)
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--data", type=Path, default=Path("data/protocol-v3"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve existing analysis; select a new output")
    specification = json.loads(args.manifest.read_text())
    items = specification["runs"]
    declared_code = specification.get("scientific_code_hash_by_seed")
    if (
        not items
        or any(
            r["arm"] not in {"A", "B", "C", "D"} or r["seed"] not in {2026, 2027, 2028}
            for r in items
        )
        or len({(r["arm"], r["seed"]) for r in items}) != len(items)
    ):
        raise ValueError("Run arm/seed identities must be unique")
    pool = json.loads(args.pool.read_text())
    info = {r["task"]["task_id"]: r for r in pool["rows"]}
    families = {r["task"]["parent_id"]: r["family_id"] for r in pool["rows"]}
    repository = TaskRepository(args.data)
    families.update({p.parent_id: p.family_id for p in repository.parents("dev_monitor")})
    control_hash, pool_hash = file_hash(args.control), file_hash(args.pool)
    control = json.loads(args.control.read_text())
    if control["pool_sha256"] != pool_hash or control["episodes"] != 80:
        raise ValueError("Control does not match pool and endpoint")
    summaries, by_run, monitors, hashes, identities, snapshots, code_hashes = (
        [],
        {},
        {},
        {},
        [],
        {},
        {},
    )
    for item in items:
        summary, records, monitoring, used, identity = load_run(
            item, info, pool_hash, control_hash, control
        )
        if identity["dataset_hash"] != repository.hash:
            raise ValueError("Run dataset differs from prepared data")
        summaries.append(summary)
        code_hash = identity.pop("scientific_code_hash")
        if item["seed"] in code_hashes and code_hashes[item["seed"]] != code_hash:
            raise ValueError("Paired runs use different scientific code")
        code_hashes[item["seed"]] = code_hash
        snapshot = identity.pop("snapshot")
        if item["seed"] in snapshots and snapshots[item["seed"]] != snapshot:
            raise ValueError("Paired runs do not share the same initial adapter")
        snapshots[item["seed"]] = snapshot
        identities.append(identity)
        by_run[item["arm"], item["seed"]] = records
        monitors[item["arm"], item["seed"]] = monitoring
        hashes.update(used)
        root = Path(item["run_dir"])
        for name in ("manifest.json", "complete.json", "resolved_config.json"):
            hashes[str(root / name)] = file_hash(root / name)
    if any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("Run model, curriculum, prompt or scientific identity differs")
    validate_seed_code_hashes(
        code_hashes, declared_code, specification.get("code_compatibility_note")
    )
    comparisons = []
    for first, second in (
        ("D", "B"),
        ("A", "B"),
        ("C", "B"),
        ("D", "A"),
        ("D", "C"),
        ("C", "A"),
    ):
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
                        "monitoring": paired_statistics(
                            monitors[first, seed][episode],
                            monitors[second, seed][episode],
                            families,
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
            "scientific_code_hash_by_seed": code_hashes,
            "code_compatibility_note": specification.get("code_compatibility_note"),
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
        (line,) = axes[0].plot(
            [0, 40, 80],
            curves.mean(axis=0),
            marker={"D": "o", "B": "s", "A": "^", "C": "D"}[arm],
            markersize=9 if arm == "D" else 5,
            markerfacecolor="none" if arm == "D" else None,
            markeredgewidth=1.8,
            zorder=5 if arm == "D" else 4,
            label=f"{labels[arm]} (n={len(group)})",
        )
        for values in curves:
            axes[0].plot([0, 40, 80], values, color=line.get_color(), alpha=0.25)
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
