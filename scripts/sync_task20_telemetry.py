"""Publish checkpoint-backed Task 20 aggregate metrics to W&B idempotently."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import wandb
from wandb.errors import CommError

from selfplay_opsd.core import digest, read_json
from selfplay_opsd.storage import CheckpointManager, atomic_json, file_hash


def flatten(values: dict, prefix: str = "") -> dict:
    result = {}
    for name, value in values.items():
        key = f"{prefix}{name}"
        if isinstance(value, dict):
            result.update(flatten(value, key + "/"))
        elif isinstance(value, (bool, int, float)) and math.isfinite(value):
            result[key] = value
    return result


def records(root: Path, stage: str = "solver") -> tuple[dict, list[dict]]:
    marker = read_json(root / f"complete-{stage}.json")
    checkpoint = root / "checkpoints" / marker["checkpoint"]
    metadata = CheckpointManager.verify(checkpoint)
    if (
        file_hash(checkpoint / "complete.json") != marker["checkpoint_sha256"]
        or metadata["identity"] != marker["identity"]
        or metadata["snapshot"] != marker["snapshot"]
    ):
        raise ValueError("Completed checkpoint mismatch")
    rows = []
    for path in sorted((root / "iterations").glob("*.json")):
        row = read_json(path)
        if row["identity"] != marker["identity"]:
            raise ValueError("Iteration identity mismatch")
        if row["batch"] is None:
            continue
        if file_hash(root / "batches" / row["batch"]["path"]) != row["batch"]["sha256"]:
            raise ValueError("Iteration batch checksum mismatch")
        rows.append(flatten(row["metrics"]) | {"train/iteration": row["iteration"]})
    if len(rows) != metadata["iteration"]:
        raise ValueError("Iteration history is incomplete")
    if stage == "solver":
        result = marker["result"]
        rows.append(
            {
                "evaluation/dev_initial": result["development"]["initial"],
                "evaluation/dev_endpoint": result["development"]["endpoint"],
                "training/probe_0_before": result["solver_probe"]["paired-0"]["before"],
                "training/probe_0_after": result["solver_probe"]["paired-0"]["after"],
                "training/probe_1_before": result["solver_probe"]["paired-1"]["before"],
                "training/probe_1_after": result["solver_probe"]["paired-1"]["after"],
            }
        )
    return marker, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--solver", type=Path)
    source.add_argument("--shared-pilot", type=Path)
    parser.add_argument("--entity", default="tatsu432s")
    parser.add_argument("--workdir", type=Path, default=Path("runs/task20/wandb-sync"))
    args = parser.parse_args()
    stage = "solver" if args.solver else "shared-pilot"
    marker, rows = records(args.solver or args.shared_pilot, stage)
    run_id = digest(marker["identity"])[:24]
    location = f"{args.entity}/self-play-opsd-coding-agent/{run_id}"
    api = wandb.Api(timeout=30)
    existing = []
    try:
        cloud = api.run(location)
        existing = list(cloud.scan_history(keys=["replay/index"]))
    except CommError as exc:
        if "not found" not in str(exc).lower() and "404" not in str(exc):
            raise
    indices = [row["replay/index"] for row in existing if "replay/index" in row]
    if (
        indices != list(range(len(indices)))
        or len(indices) > len(rows)
        or len(indices) != len(existing)
    ):
        raise ValueError("Cloud history is not the expected Task 20 replay prefix")
    args.workdir.mkdir(parents=True, exist_ok=True)
    receipt = args.workdir / f"{run_id}-replay.json"
    if len(indices) < len(rows):
        if receipt.exists():
            raise RuntimeError(
                "A replay was already attempted; wait for cloud history and reconcile before retry"
            )
        atomic_json(receipt, {"run_id": run_id, "rows": len(rows), "status": "sending"})
        run = wandb.init(
            entity=args.entity,
            project="self-play-opsd-coding-agent",
            id=run_id,
            resume="allow",
            group=marker["identity"]["study"],
            name=f"task20-{stage}-control-seed2026",
            dir=str(args.workdir),
            config={"science": marker["identity"]["science"], "source": "checkpoint-backed"},
        )
        for index in range(len(indices), len(rows)):
            run.log(rows[index] | {"replay/index": index}, step=index)
        run.finish()
        atomic_json(receipt, {"run_id": run_id, "rows": len(rows), "status": "sent"})
    verified = list(api.run(location).scan_history(keys=["replay/index"]))
    if [row["replay/index"] for row in verified] != list(range(len(rows))):
        raise RuntimeError("W&B cloud history did not verify completely")
    atomic_json(receipt, {"run_id": run_id, "rows": len(rows), "status": "verified"})
    print(
        f"Verified {len(rows)} aggregate rows at "
        f"https://wandb.ai/{args.entity}/self-play-opsd-coding-agent/runs/{run_id}"
    )


if __name__ == "__main__":
    main()
