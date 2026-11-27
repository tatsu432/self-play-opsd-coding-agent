"""Stage frozen Task 23 manifests and submit through the Yale allocation wrapper."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from analyze_task23 import historical_manifest
from task23_experiment import NAMESPACE

from selfplay_opsd.core import digest, read_json
from selfplay_opsd.hpc import YaleSSH
from selfplay_opsd.storage import atomic_json, file_hash
from selfplay_opsd.task21 import PdbRepository, Task21Control
from selfplay_opsd.task22 import Task22Control

TRAINING_NAMES = {"Initial": "Control"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("relaxed", "ood", "ood-validation"), required=True)
    parser.add_argument("--seed", type=int, default=2026, choices=(2026, 2027))
    parser.add_argument("--method", default="B", choices=("Initial", "Control", "A", "B", "C", "D"))
    parser.add_argument("--commit", required=True)
    parser.add_argument("--limit", default=0, type=int, choices=(0, 8))
    parser.add_argument("--wall", type=int, required=True)
    args = parser.parse_args()
    remote = YaleSSH()
    context = remote.call("context")
    workspace = context["storage"]["project"]["project_cpsc4880"]
    workspace = f"{workspace}/selfplay-opsd-{context['user']}"
    history = historical_manifest(Path("runs/task21"), Path("runs/task22"))
    name = f"{args.seed}-{args.method}"
    prior = history[f"{args.seed}-{TRAINING_NAMES.get(args.method, args.method)}"]["train_ids"][0]
    root = Path("runs/task23/evaluations")
    if args.kind == "ood-validation":
        cell = args.kind
        local = Path("runs/task23/ood-validation/candidates.json")
    else:
        cell = f"{args.kind}-{name}"
        record = history[name]
        receipt = read_json(Path(record["receipts"][0]))
        control = Path(record["config"])
        cfg_class = Task22Control if args.method in {"C", "D"} else Task21Control
        cfg = cfg_class.read(control)
        parent = receipt["identity"]
        if cfg.science != parent["science"]:
            raise ValueError("Frozen control no longer matches source checkpoint")
        limits = cfg.episode
        identity = {
            "checkpoint_relative": (
                f"{parent['study']}/{parent['arm']}-seed{args.seed}"
                f"/checkpoints/{receipt['checkpoint']}"
            ),
            "checkpoint_sha256": receipt["checkpoint_sha256"],
            "checkpoint_identity": parent,
            "snapshot": receipt["snapshot"],
            "control_sha256": file_hash(control),
            "kind": args.kind,
        }
        if args.kind == "relaxed":
            limits = limits.model_copy(update={"max_turn_generated_tokens": 6144})
            dataset = PdbRepository(Path("data/task21/pdb-solver-v1"))
            identity["tasks"] = [
                task.task_id
                for task, _ in sorted(
                    dataset.tasks("primary_final"),
                    key=lambda pair: digest([pair[1].provenance["family_id"], pair[0].task_id]),
                )
            ]
        else:
            verified = Path("runs/task23/ood-validation/complete.json")
            identity["ood_sha256"] = file_hash(verified)
            identity["ood_relative"] = f"{NAMESPACE}/ood-validation/complete.json"
        identity["episode"] = limits.model_dump(mode="json")
        local = root / cell / "manifest.json"
        manifest = {
            "identity": identity,
            "identity_sha256": digest(identity),
            "seed": args.seed,
            "method": args.method,
            "control": str(control),
            "kind": args.kind,
        }
        if local.is_file() and read_json(local) != manifest:
            raise ValueError("Existing local evaluation manifest has a different identity")
        atomic_json(local, manifest)
    if not local.is_file():
        raise ValueError("Missing frozen candidate file")
    destination = f"{workspace}/runs/{NAMESPACE}/{cell}"
    subprocess.run([*remote.ssh, "mkdir", "-p", destination], check=True)
    proc = subprocess.run(
        [*remote.ssh, "sha256sum", f"{destination}/{local.name}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode:
        remote.rsync(local, f"{destination}/{local.name}")
    elif proc.stdout.split()[0] != file_hash(local):
        raise ValueError("Existing remote input differs from the frozen local manifest")
    command = [
        *remote.ssh,
        "python3",
        f"{workspace}/checkouts/{args.commit}/scripts/task23_submit.py",
        "--cell",
        cell,
        "--prior",
        prior,
        "--limit",
        str(args.limit),
        "--wall",
        str(args.wall),
    ]
    result = json.loads(subprocess.check_output(command, text=True))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
