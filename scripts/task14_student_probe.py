"""Paired training-only inference comparison of student editing protocols."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from task13_static_continue import SOURCE, restore_pool

from selfplay_opsd.cli import make_policy
from selfplay_opsd.config import load_config
from selfplay_opsd.core import digest, keyed_seed
from selfplay_opsd.data import TaskRepository
from selfplay_opsd.rollout import episode_wave
from selfplay_opsd.sandbox import SandboxBackend
from selfplay_opsd.storage import atomic_json, file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--edit-protocol", choices=("unified_diff", "replace_function_body"), required=True
    )
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID") or not os.getenv("SELFPLAY_WORKER_TOKEN"):
        raise RuntimeError("Probe requires an allocated isolated worker")
    if args.output.exists():
        raise ValueError("Output already exists")
    cfg = load_config(args.settings, method="B", hardware_profile="single_gpu_80gb")
    cfg.episode.edit_protocol = args.edit_protocol
    cfg.logging.mode = "disabled"
    repo = TaskRepository(cfg.data_root)
    pool = sorted(
        restore_pool(SOURCE, repo), key=lambda item: digest([2026, "task14-edit", item[0].task_id])
    )[:16]
    sandbox = SandboxBackend(cfg.sandbox)
    sandbox.preflight()
    policy = make_policy(cfg)
    args.output.mkdir(parents=True, mode=0o700)
    atomic_json(
        args.output / "manifest.json",
        {
            "dataset": repo.hash,
            "pool_sha256": file_hash(SOURCE / "pool.json"),
            "protocol": args.edit_protocol,
            "snapshot": "base",
            "slots": 4,
            "tasks": [
                {"task_id": t.task_id, "parent_id": t.parent_id, "source_hash": t.source_hash}
                for t, _, _ in pool
            ],
            "seed_key": [2026, "task14-edit-rollout", "task_id", "slot"],
            "episode": cfg.episode.model_dump(mode="json"),
        },
    )
    rows = []
    try:
        with policy.active("student"):
            for task, oracle, _ in pool:
                jobs = [
                    (task, oracle, keyed_seed(2026, "task14-edit-rollout", task.task_id, i))
                    for i in range(4)
                ]
                episodes = episode_wave(policy, sandbox, jobs, cfg.episode, concurrency=1)
                rows.extend(e.model_dump(mode="json") for e in episodes)
                atomic_json(args.output / "episodes.json", rows)
        atomic_json(
            args.output / "complete.json",
            {"episodes": len(rows), "successes": sum(e["reward"] for e in rows)},
        )
    finally:
        policy.close()


if __name__ == "__main__":
    main()
