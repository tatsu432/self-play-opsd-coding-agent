"""Fetch only Task 23 completion evidence and reconcile physical Slurm allocations."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from task23_experiment import NAMESPACE

from selfplay_opsd.core import read_json
from selfplay_opsd.hpc import YaleSSH
from selfplay_opsd.storage import atomic_json

TERMINAL = {
    "COMPLETED",
    "FAILED",
    "TIMEOUT",
    "CANCELLED",
    "NODE_FAIL",
    "PREEMPTED",
    "OUT_OF_MEMORY",
    "BOOT_FAIL",
    "DEADLINE",
    "REVOKED",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private", type=Path, default=Path("runs/task23"))
    args = parser.parse_args()
    remote = YaleSSH()
    context = remote.call("context")
    project = context["storage"]["project"]["project_cpsc4880"]
    root = f"{project}/selfplay-opsd-{context['user']}/runs/{NAMESPACE}"
    local = args.private / "evaluations"
    local.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "rsync",
            "-a",
            "--checksum",
            "-e",
            remote.rsync_ssh,
            "--prune-empty-dirs",
            "--include=*/",
            "--include=manifest.json",
            "--include=allocations.json",
            "--include=complete.json",
            "--exclude=*",
            f"{remote.alias}:{root}/",
            str(local) + "/",
        ],
        check=True,
    )
    accepted = local / "ood-validation/complete.json"
    if accepted.is_file():
        target = args.private / "ood-validation/complete.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and target.read_bytes() != accepted.read_bytes():
            raise ValueError("The frozen independent transfer cohort changed")
        if not target.is_file():
            accepted.replace(target)
        else:
            accepted.unlink()
    expected = {}
    for path in sorted(local.glob("*/allocations.json")):
        for entry in read_json(path):
            if "job" not in entry:
                raise ValueError(f"Ambiguous scheduler submission in {path.parent.name}")
            if entry["job"] in expected:
                raise ValueError("The same job is recorded twice")
            expected[entry["job"]] = entry | {"cell": path.parent.name}
    if not expected:
        return
    response = subprocess.check_output(
        [
            *remote.ssh,
            "sacct",
            "-nP",
            "-X",
            "-j",
            ",".join(expected),
            "-o",
            "JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES",
        ],
        text=True,
    )
    allocations = []
    for line in response.splitlines():
        job, state, exit_code, elapsed, resources = line.split("|")
        if job not in expected or state.split()[0] not in TERMINAL:
            continue
        gpus = next(
            (
                int(item.split("=")[1])
                for item in resources.split(",")
                if item.startswith("gres/gpu=")
            ),
            0,
        )
        if int(elapsed) and gpus != expected[job]["gpus"]:
            raise ValueError("The scheduler's active GPU allocation differs from its reservation")
        allocations.append(
            {
                "job": job,
                "cell": expected[job]["cell"],
                "state": state,
                "exit_code": exit_code,
                "elapsed_seconds": int(elapsed),
                "gpu_seconds": int(elapsed) * gpus,
                "commit": expected[job]["commit"],
            }
        )
    atomic_json(
        args.private / "followup-accounting.json", sorted(allocations, key=lambda row: row["job"])
    )
    print(
        json.dumps(
            {
                "terminal": len(allocations),
                "submitted": len(expected),
                "h200_seconds": sum(row["gpu_seconds"] for row in allocations),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
