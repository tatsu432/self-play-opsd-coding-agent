"""Fetch the four existing Task 17 cells and analyze them; never submit a job."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from selfplay_opsd.hpc import ROOT, YaleSSH
from selfplay_opsd.storage import atomic_json

CELLS = (
    ("A", "9f12c84dfddf9b37f42dd4f1", "32805621"),
    ("B", "f7c617670830bcf229a93b0e", "32811833"),
    ("C", "c10d7452835f81c14c0726df", "32808848"),
    ("D", "819fec9d88332bf03f874d1e", "32812080"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/task17/interface-analysis-v1")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve earlier analysis; choose a new output directory")
    private = ROOT / "runs/task17"
    command = [sys.executable, "-m", "selfplay_opsd.cli", "hpc"]
    reports = {}
    for arm, run_id, job_id in CELLS:
        raw = subprocess.check_output([*command, "status", run_id], text=True, cwd=ROOT)
        status = json.loads(raw)
        attempts = status["attempts"]
        if (
            len(attempts) != 1
            or str(attempts[0]["job_id"]) != job_id
            or attempts[0]["scheduler"]["state"] != "COMPLETED"
            or attempts[0]["scheduler"]["exit_code"] != "0:0"
        ):
            raise ValueError(f"Reconcile unexpected scheduler history for {arm} before analysis")
        # Parse before replacing any saved status, so authentication errors
        # cannot truncate an earlier valid record.
        atomic_json(private / f"interface-{arm}-status.json", status)
        subprocess.run([*command, "fetch", run_id], check=True, cwd=ROOT)
        reports[job_id] = (arm, run_id)

    raw = subprocess.check_output(
        [
            *YaleSSH().ssh,
            "sacct -n -P -X -j "
            + ",".join(reports)
            + " --format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES",
        ],
        text=True,
    )
    allocations = {}
    for line in raw.splitlines():
        fields = line.strip().split("|")
        if fields[0] not in reports:
            continue
        job_id, state, exit_code, seconds, tres = fields[:5]
        resources = dict(item.split("=", 1) for item in tres.split(",") if "=" in item)
        # Generic GPU count and typed GPU count describe the same allocation.
        count = int(resources.get("gres/gpu", resources.get("gres/gpu:h200", "0")))
        if state != "COMPLETED" or exit_code != "0:0" or count != 1 or int(seconds) <= 0:
            raise ValueError(f"Unexpected allocation for {job_id}: {line}")
        allocations[job_id] = int(seconds) / 3600
    if set(allocations) != set(reports):
        raise ValueError("Missing Slurm allocation records; do not substitute requested wall time")
    atomic_json(private / "interface-accounting.json", {"sacct": raw})
    manifest_path = private / "interface-manifest.json"
    atomic_json(
        manifest_path,
        {
            "cells": [
                {
                    "arm": arm,
                    "run_dir": str(ROOT / "runs/hpc/yale-hpc-cpsc4880" / run_id),
                    "status_path": str(private / f"interface-{arm}-status.json"),
                    "allocated_gpu_hours": allocations[job_id],
                }
                for arm, run_id, job_id in CELLS
            ]
        },
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/analyze_task17_interfaces.py"),
            "--manifest",
            str(manifest_path),
            "--pool",
            str(ROOT / "runs/hpc/yale-hpc-cpsc4880/86e2df49bb04a9321f7d755d/pool.json"),
            "--output",
            str(args.output),
        ],
        check=True,
        cwd=ROOT,
    )


if __name__ == "__main__":
    main()
