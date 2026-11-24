"""Publish verified Task 21 training and development curves as aggregate metrics."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import wandb

from selfplay_opsd.core import digest, read_json
from selfplay_opsd.storage import atomic_json, file_hash


def collect(private: Path, artifacts: Path) -> dict[int, dict[str, float]]:
    """Validate checkpoint-backed records and return aggregate-only curves."""
    points: dict[int, dict[str, float]] = {}
    for seed in (2026, 2027):
        for arm in ("control", "A", "B"):
            name = (
                f"matched-{arm}-verified.json"
                if seed == 2026
                else f"matched-2027-{arm}-verified.json"
            )
            verified = read_json(private / name)
            root = artifacts / verified["run_id"]
            marker = read_json(root / "complete-main.json")
            if any(
                marker[key] != verified[key]
                for key in ("identity", "snapshot", "checkpoint_sha256", "result")
            ):
                raise ValueError(f"Endpoint verification mismatch: {seed}/{arm}")
            rows = [read_json(path) for path in sorted((root / "iterations").glob("*.json"))]
            snapshots = {row["episodes"]: row["snapshot"] for row in rows}
            committed = [row for row in rows if row["batch"] is not None]
            if len(committed) != int(marker["checkpoint"]):
                raise ValueError(f"Incomplete iteration history: {seed}/{arm}")
            prefix = f"seed_{seed}/{arm}"
            for row in committed:
                batch = row["batch"]
                if (
                    row["identity"] != marker["identity"]
                    or file_hash(root / "batches" / batch["path"]) != batch["sha256"]
                ):
                    raise ValueError(f"Invalid committed batch: {seed}/{arm}")
                metrics = row["metrics"]
                batch_size = metrics["batch_episodes"]
                event = points.setdefault(row["episodes"], {})
                event[f"{prefix}/training_repair_rate"] = (
                    metrics["student"]["verified"] / batch_size
                )
                event[f"{prefix}/mixed_group_fraction"] = metrics["learning"][
                    "mixed_reward_fraction"
                ]
                event[f"{prefix}/effective_rl_updates"] = metrics["learning"][
                    "effective_rl_updates"
                ]
            for path in sorted((root / "evaluation").glob("*/dev_monitor/summary.json")):
                summary = read_json(path)
                exposure = summary["episodes"]
                if summary["identity"] != marker["identity"] or summary[
                    "snapshot"
                ] != snapshots.get(exposure):
                    raise ValueError(f"Development checkpoint mismatch: {seed}/{arm}")
                tasks = summary["tasks"]
                if (
                    len(tasks) != summary["count"]
                    or sum(row["success"] for row in tasks) != summary["verified"]
                ):
                    raise ValueError(f"Development count mismatch: {seed}/{arm}")
                points.setdefault(exposure, {})[f"{prefix}/development_accuracy"] = (
                    summary["verified"] / summary["count"]
                )
    return points


def health_rows(private: Path, artifacts: Path) -> list[list]:
    """Return per-run training diagnostics; call collect first to validate inputs."""
    result = []
    for seed in (2026, 2027):
        for arm in ("control", "A", "B"):
            name = (
                f"matched-{arm}-verified.json"
                if seed == 2026
                else f"matched-2027-{arm}-verified.json"
            )
            root = artifacts / read_json(private / name)["run_id"]
            rows = [read_json(path) for path in sorted((root / "iterations").glob("*.json"))]
            metrics = [row["metrics"] for row in rows if row["batch"] is not None]
            result.append(
                [
                    seed,
                    arm,
                    sum(row["batch_episodes"] for row in metrics),
                    sum(row["student"]["verified"] for row in metrics),
                    sum(row["student"]["mixed_groups"] for row in metrics),
                    sum(sum(row["learning"]["reward_histogram"].values()) for row in metrics),
                    metrics[-1]["learning"]["effective_rl_updates"],
                    sum(row["proposer"]["valid"] for row in metrics),
                    sum(row["proposer"]["attempted"] for row in metrics),
                    sum(row["proposer"]["distinct_reward_groups"] for row in metrics),
                    sum(
                        row["student"]["failure_categories"].get("truncation", 0) for row in metrics
                    ),
                    sum(row["student"]["invalid_calls"] for row in metrics),
                    sum(row["student"]["rejected_edits"] for row in metrics),
                    sum(row["worker_retries"] for row in metrics),
                    sum(row["rl"]["support_exits"] for row in metrics),
                    sum(not math.isfinite(row["shared"]["joint_gradient_norm"]) for row in metrics),
                ]
            )
    return result


HEALTH_COLUMNS = [
    "seed",
    "method",
    "attempts",
    "verified_repairs",
    "mixed_groups",
    "total_groups",
    "nonzero_rl_updates",
    "valid_proposals",
    "proposal_attempts",
    "distinct_reward_groups",
    "limit_terminated_failures",
    "invalid_tool_calls",
    "rejected_edits",
    "worker_retries",
    "support_exits",
    "nonfinite_joint_norms",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private", type=Path, default=Path("runs/task21"))
    parser.add_argument("--artifacts", type=Path, default=Path("runs/hpc/yale-hpc-cpsc4880"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    points = collect(args.private, args.artifacts)
    if args.dry_run:
        print(f"Validated {len(points)} exposure points from six endpoint histories")
        return
    run_id = digest(["task21-matched-curves-v1"])[:24]
    receipt = args.private / "wandb-matched-curves.json"
    if receipt.exists():
        raise ValueError("Curve upload already attempted; inspect its receipt and cloud history")
    atomic_json(receipt, {"run_id": run_id, "status": "sending", "points": len(points)})
    run = wandb.init(
        entity="tatsu432s",
        project="self-play-opsd-coding-agent",
        id=run_id,
        group="repair-selfplay-2026-task21-pdb-prompt-v2",
        name="task21-matched-learning-curves",
        config={
            "source": "verified checkpoint-backed aggregates",
            "development_tasks": 24,
            "note": "Training tasks evolve; development tasks are reused.",
        },
        dir=str(args.private),
    )
    run.define_metric("episodes")
    run.define_metric("*", step_metric="episodes")
    for episode, metrics in sorted(points.items()):
        run.log({"episodes": episode, **metrics})
    milestones = (0, 64, 128, 256, 384, 512)
    labels = [f"{seed} {arm}" for seed in (2026, 2027) for arm in ("control", "A", "B")]
    keys = [
        f"seed_{seed}/{arm}/development_accuracy"
        for seed in (2026, 2027)
        for arm in ("control", "A", "B")
    ]
    run.log(
        {
            "development_comparison": wandb.plot.line_series(
                xs=list(milestones),
                ys=[[points[step][key] for step in milestones] for key in keys],
                keys=labels,
                title="Development monitor: 24 reused tasks (not held out)",
                xname="Attempted training episodes",
            )
        }
    )
    run.log(
        {
            "training_health": wandb.Table(
                columns=HEALTH_COLUMNS, data=health_rows(args.private, args.artifacts)
            )
        }
    )
    run.finish()
    atomic_json(receipt, {"run_id": run_id, "status": "sent", "points": len(points)})
    print(
        f"Uploaded aggregate curves: https://wandb.ai/tatsu432s/self-play-opsd-coding-agent/runs/{run_id}"
    )


if __name__ == "__main__":
    main()
