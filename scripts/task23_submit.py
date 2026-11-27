"""Submit bounded, idempotent follow-up jobs using the previously accepted worker."""

from __future__ import annotations

import argparse
import base64
import fcntl
import json
import re
import shutil
import subprocess
from pathlib import Path

from hpc_remote import paths, write_json

NAMESPACE = "repair-selfplay-2026-task23-eval-v1"
ACTIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED"}


def run(args: list[str]) -> str:
    return subprocess.check_output(args, text=True).strip()


def job_state(job: str) -> str:
    records = run(["sacct", "-nP", "-X", "-j", job, "-o", "JobIDRaw,State"])
    return next(
        (
            line.split("|")[1].split()[0]
            for line in records.splitlines()
            if line.split("|")[0] == job
        ),
        "",
    )


def submit(workspace: Path, directory: Path, prior_id: str, limit: int, minutes: int) -> dict:
    if not re.fullmatch(r"[a-f0-9]{20,32}", prior_id):
        raise ValueError("Expected a verified historical controller run ID")
    if not 25 <= minutes <= 240:
        raise ValueError("Schedule a bounded Task 23 job")
    check = shutil.disk_usage(workspace)
    if check.free < 20 * 1024**3:
        raise ValueError("Shared project disk is below the previously used 20 GiB floor")
    source = json.loads((workspace / "state" / f"{prior_id}.json").read_text())
    spec = source["specification"]
    source_paths = paths(spec)
    if (
        source.get("operation") != "task21-main"
        or spec["account"] != "cpsc4880"
        or spec["gpu_type"] != "h200"
        or spec["gpu_count"] != 1
        or source_paths["base"] != workspace
        or not (Path(source["run_dir"]) / "complete-main.json").is_file()
    ):
        raise ValueError("Task 23 requires an accepted completed Task 21/22 job specification")
    if directory.name == "ood-validation":
        file = directory / "candidates.json.gz"
        gpu = 0
    else:
        file = directory / "manifest.json"
        gpu = 1
        info = json.loads(file.read_text())
        actual = info["identity"]["checkpoint_identity"]
        training = json.loads((Path(source["run_dir"]) / "complete-main.json").read_text())
        if any(
            actual[key] != training["identity"][key]
            for key in ("science", "study", "initial", "worker", "data")
        ):
            raise ValueError("Source run and frozen checkpoint differ")
    if not file.is_file():
        raise ValueError("The private evaluation inputs have not been staged")
    checkout = Path(__file__).resolve().parents[1]
    if checkout.parent != workspace / "checkouts":
        raise ValueError("Use the synchronized immutable Task 23 checkout")
    comment = f"task23:{directory.name}:{limit}"
    with (directory / "submission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger_file = directory / "allocations.json"
        ledger = json.loads(ledger_file.read_text()) if ledger_file.exists() else []
        if (directory / "complete.json").exists():
            return {"complete": True, "allocations": ledger}
        for attempt in ledger:
            if (
                "job" not in attempt
                or not job_state(attempt["job"])
                or job_state(attempt["job"]) in ACTIVE
            ):
                return {"existing": attempt}
        if len(ledger) >= 3:
            raise ValueError("Task 23 cell reached its three-allocation recovery bound")
        if limit and any(item["limit"] == limit for item in ledger):
            raise ValueError("The timing tranche was already submitted")
        output = f"/runs/{NAMESPACE}/{directory.name}"
        values = {
            "SELFPLAY_PROJECT": str(checkout),
            "SELFPLAY_WORKER_SIF": spec["worker_image"],
            "SELFPLAY_WORKER_SHA256": spec["worker_sha256"],
            "SELFPLAY_WORKER_SETUP_ID": spec["worker_setup_id"],
            "SELFPLAY_WORKER_CONFIG": base64.b64encode(
                json.dumps(spec["worker_config"]).encode()
            ).decode(),
            "SELFPLAY_TRAINER_SIF": spec["image"],
            "SELFPLAY_TRAINER_SHA256": spec["image_sha256"],
            "SELFPLAY_DATA": str(source_paths["dataset"]),
            "SELFPLAY_RUNS": str(source_paths["runs"]),
            "SELFPLAY_CACHE": str(source_paths["cache"]),
            "SELFPLAY_OPERATION": "task21-followup",
            "SELFPLAY_METHOD": directory.name,
            "SELFPLAY_HARDWARE": spec["profile"],
            "SELFPLAY_GPU_COUNT": str(gpu),
            "SELFPLAY_ALLOCATION_CPUS": str(spec["cpus"]),
            "SELFPLAY_ALLOCATION_MEMORY_MIB": str(spec["memory_gib"] * 1024),
            "SELFPLAY_SETTINGS": spec["settings"],
            "SELFPLAY_STUDY": NAMESPACE,
            "SELFPLAY_MAIN_STUDY": NAMESPACE,
            "SELFPLAY_GPU_TYPE": "h200",
            "SELFPLAY_SEED": str(spec["seed"]),
            "SELFPLAY_OUTPUT": output,
            "SELFPLAY_RUN_ID": directory.name,
            "SELFPLAY_TASK22_EXPOSURE": str(limit),
        }
        if any("," in value or "\n" in value for value in values.values()):
            raise ValueError("Invalid Slurm export value")
        args = [
            "sbatch",
            "--parsable",
            "--no-requeue",
            "--account=cpsc4880",
            f"--partition={spec['partition'] if gpu else 'day'}",
            f"--cpus-per-task={spec['cpus']}",
            f"--mem={spec['memory_gib'] if gpu else 24}G",
            f"--time={minutes // 60:02d}:{minutes % 60:02d}:00",
            f"--comment={comment}",
            f"--job-name=t23-{directory.name}",
            f"--output={directory}/%j.out",
            "--export=" + ",".join(f"{key}={value}" for key, value in values.items()),
        ]
        if gpu:
            args.append("--gpus=h200:1")
        else:
            values["SELFPLAY_ALLOCATION_MEMORY_MIB"] = str(24 * 1024)
            args[-1] = "--export=" + ",".join(f"{key}={value}" for key, value in values.items())
        args.append(str(checkout / "scripts" / "yale.slurm"))
        entry = {
            "limit": limit,
            "reserved_seconds": 60 * minutes,
            "gpus": gpu,
            "comment": comment,
            "commit": checkout.name,
            "historical_run": prior_id,
        }
        ledger.append(entry)
        write_json(ledger_file, ledger)
        result = subprocess.run(args, text=True, capture_output=True, check=False)
        if result.returncode:
            ledger.pop()
            write_json(ledger_file, ledger)
            raise RuntimeError(result.stderr)
        job = result.stdout.strip().split(";", 1)[0]
        if not job.isdecimal():
            raise RuntimeError("Unknown scheduler response. Reconcile the recorded intent.")
        entry["job"] = job
        write_json(ledger_file, ledger)
        return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--prior", required=True)
    parser.add_argument("--limit", default=0, type=int, choices=(0, 8))
    parser.add_argument("--wall", required=True, type=int)
    args = parser.parse_args()
    if not re.fullmatch(
        r"(?:relaxed|ood)-(?:2026|2027)-(?:Initial|Control|A|B|C|D)|ood-validation", args.cell
    ):
        raise ValueError("Invalid evaluation cell")
    workspace = Path(__file__).resolve().parents[2].parent
    print(
        json.dumps(
            submit(
                workspace,
                workspace / "runs" / NAMESPACE / args.cell,
                args.prior,
                args.limit,
                args.wall,
            )
        )
    )


if __name__ == "__main__":
    main()
