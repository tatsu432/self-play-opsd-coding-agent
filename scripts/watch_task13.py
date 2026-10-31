"""Reconcile and fetch known Task 13 jobs; never submit or retry them."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from selfplay_opsd.hpc import YaleSSH, remote_context


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--interval", type=int, default=45)
    args = parser.parse_args()
    if args.interval < 30 or len(set(args.runs)) != len(args.runs):
        parser.error("Use unique run IDs and an interval of at least 30 seconds")
    remote = YaleSSH("yale-hpc-cpsc4880")
    context = remote_context(remote, remote.alias)
    root = Path("runs/hpc") / remote.alias
    audit = Path("runs/task13/monitor")
    audit.mkdir(parents=True, exist_ok=True, mode=0o700)
    active = set(args.runs)
    previous = {}
    while active:
        for run_id in sorted(active.copy()):
            status = remote.call(
                "status", {"storage_root": context["storage_root"], "run_id": run_id}
            )
            (audit / f"{run_id}-status.json").write_text(json.dumps(status, indent=2) + "\n")
            attempt = status["attempts"][-1]
            state = attempt["scheduler"]["state"]
            if previous.get(run_id) != state:
                print(
                    json.dumps({"run_id": run_id, "job_id": attempt["job_id"], "state": state}),
                    flush=True,
                )
                previous[run_id] = state
            if state not in {
                "PENDING",
                "CONFIGURING",
                "RUNNING",
                "COMPLETING",
                "SUSPENDED",
                "REQUEUED",
                "UNKNOWN",
            }:
                if state == "COMPLETED" and attempt["scheduler"].get("exit_code") == "0:0":
                    remote.fetch(status["ledger"]["run_dir"], root / run_id)
                    print(
                        json.dumps(
                            {
                                "run_id": run_id,
                                "fetched": True,
                                "complete_record": (root / run_id / "complete.json").exists(),
                            }
                        ),
                        flush=True,
                    )
                else:
                    logs = remote.call(
                        "logs",
                        {"storage_root": context["storage_root"], "run_id": run_id, "lines": 120},
                    )
                    (audit / f"{run_id}-failure.json").write_text(json.dumps(logs, indent=2) + "\n")
                    print(
                        json.dumps(
                            {
                                "run_id": run_id,
                                "failure": state,
                                "logs": str(audit / f"{run_id}-failure.json"),
                            }
                        ),
                        flush=True,
                    )
                active.remove(run_id)
        if active:
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
