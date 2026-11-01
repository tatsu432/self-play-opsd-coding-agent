"""Measure student reward variation on saved verified proposer outputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from task14_checkpoint import load_b200

from selfplay_opsd.cli import make_policy
from selfplay_opsd.config import load_config
from selfplay_opsd.core import keyed_seed
from selfplay_opsd.data import TaskRepository, build_task
from selfplay_opsd.environment import check_admission
from selfplay_opsd.mutations import strict_json_object, validate_mutation
from selfplay_opsd.rollout import episode_wave
from selfplay_opsd.sandbox import SandboxBackend
from selfplay_opsd.storage import atomic_json, file_hash, restore_rng
from selfplay_opsd.structured_mutations import mutation as structured_mutation
from selfplay_opsd.structured_mutations import sites


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID") or not os.getenv("SELFPLAY_WORKER_TOKEN"):
        raise RuntimeError("Probe requires an allocated isolated worker")
    if args.output.exists():
        raise ValueError("Output already exists")
    manifest = json.loads((args.source / "manifest.json").read_text())
    rows = json.loads((args.source / "proposals.json").read_text())
    complete = json.loads((args.source / "complete.json").read_text())
    if (
        manifest["study"] != "repair-diagnosis-2026-task14-probe-v1"
        or complete["count"] != len(rows)
        or len(rows) != len(manifest["parents"]) * manifest["slots"]
    ):
        raise ValueError("Proposal source is incomplete or not Task 14")
    cfg = load_config(args.settings, method="B", hardware_profile="single_gpu_80gb")
    cfg.logging.mode = "disabled"
    repo = TaskRepository(cfg.data_root)
    if manifest["manifest_hash"] != repo.hash or manifest["model"] != cfg.model.model_dump(
        mode="json"
    ):
        raise ValueError("Proposal dataset or model differs")
    if sum(row["valid"] for row in rows) != complete["valid"]:
        raise ValueError("Proposal completion count differs")
    parents = {p.parent_id: p for p in repo.parents("train")}
    source_hashes = {p["id"]: p["source_hash"] for p in manifest["parents"]}
    from selfplay_opsd.core import digest

    if any(digest(parents[key].source) != value for key, value in source_hashes.items()):
        raise ValueError("Proposal parent source differs")
    chosen, seen = [], set()
    for row in rows:
        key = (row["parent_id"], row.get("dedup_key"))
        if not row["valid"] or row.get("in_training_history", False) or key in seen:
            continue
        seen.add(key)
        chosen.append(row)
        if len(chosen) == 32:
            break
    sandbox = SandboxBackend(cfg.sandbox)
    sandbox.preflight()
    policy = make_policy(cfg)
    try:
        if manifest["checkpoint"] is not None:
            state, _, marker = load_b200(cfg, repo)
            if marker != manifest.get("checkpoint_marker_sha256"):
                raise ValueError("Proposal checkpoint marker differs")
            policy.load_state_dict(state["policy"])
            restore_rng(state["rng"])
        args.output.mkdir(parents=True, mode=0o700)
        atomic_json(
            args.output / "manifest.json",
            {
                "source_manifest_sha256": file_hash(args.source / "manifest.json"),
                "source_proposals_sha256": file_hash(args.source / "proposals.json"),
                "representation": manifest["representation"],
                "snapshot": "b200" if manifest["checkpoint"] is not None else "base",
                "student_protocol": cfg.episode.edit_protocol,
                "selected": [
                    {"parent_id": r["parent_id"], "slot": r["slot"], "dedup_key": r["dedup_key"]}
                    for r in chosen
                ],
            },
        )
        groups = []
        with policy.active("student"):
            for row in chosen:
                parent = parents[row["parent_id"]]
                value = strict_json_object(row["raw_text"])
                if manifest["representation"] == "structured":
                    change = structured_mutation(
                        parent.source,
                        sites(parent.source)[value["site"]]["path"],
                        value["operation"],
                    )
                else:
                    change = validate_mutation(parent.source, value["patch"])
                if change.dedup_key != row["dedup_key"]:
                    raise ValueError("Reconstructed mutation differs from verified proposal")
                public, hidden = repo.parent_cases(parent.parent_id)
                task, oracle = build_task(
                    parent, change, public, hidden, "train", row["seed"], sandbox
                )
                check_admission(policy, task, oracle, cfg.episode)
                jobs = [
                    (task, oracle, keyed_seed(2026, "task14-proposal-student", task.task_id, slot))
                    for slot in range(4)
                ]
                episodes = episode_wave(policy, sandbox, jobs, cfg.episode, concurrency=1)
                groups.append(
                    {
                        "parent_id": parent.parent_id,
                        "task_id": task.task_id,
                        "dedup_key": change.dedup_key,
                        "episodes": [e.model_dump(mode="json") for e in episodes],
                    }
                )
                atomic_json(args.output / "groups.json", groups)
        atomic_json(
            args.output / "complete.json",
            {
                "groups": len(groups),
                "mixed": sum(0 < sum(e["reward"] for e in g["episodes"]) < 4 for g in groups),
            },
        )
    finally:
        policy.close()


if __name__ == "__main__":
    main()
