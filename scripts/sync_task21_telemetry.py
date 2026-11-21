"""Reconcile Task 21 checkpoint-backed aggregate metrics with the study's W&B run."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import wandb
from sync_task20_telemetry import flatten
from wandb.errors import CommError

from selfplay_opsd.core import digest, read_json
from selfplay_opsd.storage import atomic_json, file_hash


def records(root: Path, stage: str, verification: dict) -> tuple[dict, list[dict]]:
    marker = read_json(root / f"complete-{stage}.json")
    if any(
        marker[key] != verification[key]
        for key in ("identity", "snapshot", "checkpoint_sha256", "result")
    ):
        raise ValueError("Stage marker differs from the verified remote endpoint")
    rows = []
    for path in sorted((root / "iterations").glob("*.json")):
        row = read_json(path)
        if row["identity"] != marker["identity"]:
            raise ValueError("Iteration identity differs from verified remote endpoint")
        batch = row["batch"]
        if batch:
            if file_hash(root / "batches" / batch["path"]) != batch["sha256"]:
                raise ValueError("Committed batch data differ from their checksum")
            rows.append(flatten(row["metrics"]) | {"train/iteration": row["iteration"]})
    if len(rows) != int(marker["checkpoint"]):
        raise ValueError("Iteration aggregate history is incomplete")
    rows.append(flatten(marker["result"], f"{stage}/"))
    return marker, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "verification", "workdir"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("feasibility", "screen", "solver", "calibrate", "shared-pilot", "main"),
        required=True,
    )
    parser.add_argument("--entity", default="tatsu432s")
    args = parser.parse_args()
    os.umask(0o077)
    marker, rows = records(args.root, args.stage, read_json(args.verification))
    identity = marker["identity"]
    run_id = digest([identity, "aggregate-replay-v1"])[:24]
    location = f"{args.entity}/self-play-opsd-coding-agent/{run_id}"
    api = wandb.Api(timeout=30)
    try:
        existing = list(api.run(location).scan_history(keys=["replay/index"]))
    except CommError as exc:
        if "not found" not in str(exc).lower() and "404" not in str(exc):
            raise
        existing = []
    indices = [row["replay/index"] for row in existing]
    if indices != list(range(len(indices))) or len(indices) > len(rows):
        raise ValueError("W&B aggregate history is inconsistent")
    args.workdir.mkdir(parents=True, exist_ok=True)
    receipt = args.workdir / f"{run_id}.json"
    if len(indices) < len(rows):
        if receipt.exists():
            raise ValueError("An aggregate upload was already attempted; reconcile its receipt")
        atomic_json(receipt, {"status": "sending", "rows": len(rows), "run_id": run_id})
        run = wandb.init(
            entity=args.entity,
            project="self-play-opsd-coding-agent",
            id=run_id,
            resume="allow",
            group=identity["study"],
            name=f"task21-{args.stage}-{identity['arm']}",
            config={"science": identity["science"], "source": "verified aggregate replay"},
            dir=str(args.workdir),
        )
        for index in range(len(indices), len(rows)):
            run.log(rows[index] | {"replay/index": index}, step=index)
        run.finish()
        atomic_json(receipt, {"status": "sent", "rows": len(rows), "run_id": run_id})
    verified = list(api.run(location).scan_history(keys=["replay/index"]))
    if [row["replay/index"] for row in verified] != list(range(len(rows))):
        raise ValueError("Task 21 W&B history has not reconciled")
    atomic_json(receipt, {"status": "verified", "rows": len(rows), "run_id": run_id})
    print(f"Verified {len(rows)} aggregate rows at https://wandb.ai/{location}")


if __name__ == "__main__":
    main()
