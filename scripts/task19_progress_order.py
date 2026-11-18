"""Restore first-wave Task 19 checkpoint key order without changing data.

Run only after independently confirming the owning allocation is terminal.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
from pathlib import Path


def reordered(data: dict) -> dict:
    if data["completed_waves"] != 0:
        raise ValueError("Only the first wave has unambiguous task ordering")
    tasks, repairs = data["tasks"], data["repairs"]
    order = list(
        dict.fromkeys(
            row["task"] for group in data["groups"] for row in group if row.get("task") in tasks
        )
    )
    if len(order) != len(tasks) or set(order) != set(tasks):
        raise ValueError("Proposal records do not identify every pending task")
    repair_order = [key for key in order if key in repairs]
    if set(repair_order) != set(repairs) or repair_order != order[: len(repair_order)]:
        raise ValueError("Saved repair groups do not form the expected prefix")
    result = dict(data)
    result["tasks"] = {key: tasks[key] for key in order}
    result["repairs"] = {key: repairs[key] for key in repair_order}
    if result != data:
        raise ValueError("Task reordering changed checkpoint data")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("progress", type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    if not args.job_id.isdecimal():
        raise ValueError("Invalid Slurm job ID")
    lock_path = args.ledger.parent / ".submit.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rewrite_stopped(args)


def rewrite_stopped(args: argparse.Namespace) -> None:
    ledger = json.loads(args.ledger.read_text())
    if (
        not ledger.get("attempts")
        or ledger["attempts"][-1]["job_id"] != args.job_id
        or ledger.get("pending_intent")
        or Path(ledger["run_dir"]).resolve() != args.progress.parent.resolve()
    ):
        raise ValueError("Controller ledger does not identify this stopped checkpoint")
    status = subprocess.run(
        ["sacct", "-n", "-P", "-j", args.job_id, "--format=JobIDRaw,State"],
        check=True,
        text=True,
        capture_output=True,
    )
    rows = [line.split("|") for line in status.stdout.splitlines()]
    states = [
        row[1].split()[0].rstrip("+") for row in rows if len(row) > 1 and row[0] == args.job_id
    ]
    if len(states) != 1 or states[0] not in {"COMPLETED", "TIMEOUT"}:
        raise ValueError("Slurm has not confirmed a terminal continuable job")
    health = json.loads((args.progress.parent / "state/health.json").read_text())
    if health.get("allocation") != args.job_id or health.get("phase") not in (
        {"stopped"} if states[0] == "COMPLETED" else {"stopped", "running"}
    ):
        raise ValueError("Worker health does not match the terminal job")
    raw = args.progress.read_bytes()
    checksum = hashlib.sha256(raw).hexdigest()
    if checksum != args.expected_sha256:
        raise ValueError("Checkpoint changed since terminal-state inspection")
    data = json.loads(raw)
    if data.get("identity") != health.get("identity"):
        raise ValueError("Checkpoint identity differs from the recorded worker")
    result = reordered(data)
    archive = args.progress.with_name("qualification-progress-before-order-" + checksum + ".json")
    with archive.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    encoded = (
        json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    temporary = args.progress.with_name(".qualification-progress-reordered.tmp")
    with temporary.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, args.progress)
    print(
        json.dumps(
            {
                "original_sha256": checksum,
                "reordered_sha256": hashlib.sha256(encoded).hexdigest(),
                "archive": str(archive),
            }
        )
    )


if __name__ == "__main__":
    main()
