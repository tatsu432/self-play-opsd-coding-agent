"""Factorial saved-rollout replay on the existing training-only diff probe."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from task13_static_continue import SOURCE, restore_pool
from task14_checkpoint import load_b200
from task14_numerical import select_tasks

from selfplay_opsd.cli import make_policy
from selfplay_opsd.config import load_config
from selfplay_opsd.core import EpisodeRecord
from selfplay_opsd.data import TaskRepository
from selfplay_opsd.environment import score_teacher
from selfplay_opsd.sandbox import SandboxBackend
from selfplay_opsd.storage import atomic_json, file_hash, restore_rng
from selfplay_opsd.task15_audit import audit_updates, teacher_action_discrimination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID") or not os.getenv("SELFPLAY_WORKER_TOKEN"):
        raise RuntimeError("Numerical audit requires an allocated isolated worker")
    if args.output.exists():
        raise ValueError("Audit output already exists; reconcile its ledger")
    manifest = json.loads((args.source / "manifest.json").read_text())
    records = json.loads((args.source / "episodes.json").read_text())
    complete = json.loads((args.source / "complete.json").read_text())
    if complete["episodes"] != len(records) or len(records) != len(manifest["tasks"]) * 4:
        raise ValueError("Saved source probe is incomplete")
    cfg = load_config(args.settings, method="B", hardware_profile="single_gpu_80gb")
    if cfg.online_supply or manifest["protocol"] != "unified_diff":
        raise ValueError("Replay uses diagnostic settings and the existing diff probe")
    cfg.logging.mode = "disabled"
    cfg.output_root = args.output  # Isolate adapter exports from historical study directories.
    repo = TaskRepository(cfg.data_root)
    if manifest["dataset"] != repo.hash or manifest["pool_sha256"] != file_hash(
        SOURCE / "pool.json"
    ):
        raise ValueError("Source training-pool identity differs")
    selected = set(select_tasks(manifest, records, prefer_mixed=True))
    episodes = [EpisodeRecord.model_validate(row) for row in records if row["task_id"] in selected]
    pool = {task.task_id: (task, oracle) for task, oracle, _ in restore_pool(SOURCE, repo)}
    sandbox = SandboxBackend(cfg.sandbox)
    sandbox.preflight()
    policy = make_policy(cfg)
    try:
        if manifest["snapshot"] == "b200":
            state, _, marker = load_b200(cfg, repo)
            if marker != manifest["checkpoint_marker_sha256"]:
                raise ValueError("B200 source checkpoint differs")
            policy.load_state_dict(state["policy"])
            restore_rng(state["rng"])
        elif manifest["snapshot"] != "base":
            raise ValueError("Unknown source snapshot")
        scored = [
            score_teacher(policy, episode, pool[episode.task_id][1], cfg.episode)[0]
            for episode in episodes
        ]
        report = audit_updates(policy, cfg, scored)
        report["teacher_action_discrimination"] = teacher_action_discrimination(
            policy, cfg, scored, {task_id: oracle for task_id, (_, oracle) in pool.items()}
        )
        atomic_json(
            args.output / "audit.json",
            report
            | {
                "source_sha256": file_hash(args.source / "episodes.json"),
                "source_manifest_sha256": file_hash(args.source / "manifest.json"),
                "snapshot": manifest["snapshot"],
                "audit_scientific_hash": cfg.scientific_hash,
                "audit_method": cfg.method,
                "tasks": len(selected),
                "episodes": len(episodes),
                "selection": "up to four mixed and four uniform saved training groups, then manifest order; targeted diagnostic, not a population estimate",
            },
        )
    finally:
        policy.close()


if __name__ == "__main__":
    main()
