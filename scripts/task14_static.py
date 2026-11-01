"""Matched-exposure fixed-task trainer for Task 14 experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from task13_static import StaticTrainer
from task13_static_continue import SOURCE, restore_pool

from selfplay_opsd.cli import make_policy
from selfplay_opsd.config import load_config
from selfplay_opsd.data import TaskRepository
from selfplay_opsd.sandbox import SandboxBackend
from selfplay_opsd.storage import atomic_json, file_hash


class ExposureTrainer(StaticTrainer):
    """Evaluate at fixed episode counts, regardless of skipped optimizer steps."""

    def run_iteration(self, k):
        # Avoid Task 13's step-count stop and evaluation rules.
        record = super(StaticTrainer, self).run_iteration(k)
        if self.total_episodes in {80, 160}:
            self.evaluate_development()
        if self.total_episodes >= 160:
            self.stop_requested = True
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("GRPO", "B", "A"), required=True)
    parser.add_argument("--seed", type=int, choices=(2026, 2027, 2028), required=True)
    parser.add_argument("--ungated-lambda", type=float)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID") or not os.getenv("SELFPLAY_WORKER_TOKEN"):
        raise RuntimeError("Static experiment requires an allocated isolated worker")
    if args.method == "A" and (args.ungated_lambda is None or args.ungated_lambda <= 0):
        raise ValueError("Ungated arm requires a positive training-calibrated coefficient")
    if args.method != "A" and args.ungated_lambda is not None:
        raise ValueError("Ungated coefficient is only valid for the ungated arm")
    if args.output.exists() and not args.resume:
        raise ValueError("Output already exists")
    if args.resume and (not args.output.is_dir() or not args.resume.is_relative_to(args.output)):
        raise ValueError("Resume checkpoint must belong to the existing static run")
    cfg = load_config(args.settings, method=args.method, hardware_profile="single_gpu_80gb")
    cfg.seed = args.seed
    cfg.study_id = "repair-diagnosis-2026-task14-static-v1"
    cfg.output_root = args.output
    cfg.logging.mode = "disabled"
    cfg.parents_per_iteration = cfg.proposals_per_parent = 2
    cfg.iterations, cfg.evaluate_every, cfg.checkpoint_every = 20, 21, 1
    if args.method == "A":
        cfg.objective.sd_lambda = args.ungated_lambda
    repo = TaskRepository(cfg.data_root)
    pool = restore_pool(SOURCE, repo)
    schedule = json.loads((SOURCE / "schedule.json").read_text())
    by_id = {item[0].task_id: item for item in pool}
    if set(schedule["order"]) != set(by_id) or len(schedule["order"]) != len(by_id):
        raise ValueError("Static schedule does not match the verified pool")
    pool = [by_id[task_id] for task_id in schedule["order"]]
    manifest = {
        "pool_sha256": file_hash(SOURCE / "pool.json"),
        "schedule_sha256": file_hash(SOURCE / "schedule.json"),
        "method": args.method,
        "seed": args.seed,
        "sd_lambda": cfg.objective.sd_lambda,
        "episode_checkpoints": [80, 160],
    }
    if args.resume:
        if json.loads((args.output / "input_manifest.json").read_text()) != manifest:
            raise ValueError("Static experiment input manifest changed")
    else:
        args.output.mkdir(parents=True, mode=0o700)
        atomic_json(args.output / "input_manifest.json", manifest)
    sandbox = SandboxBackend(cfg.sandbox)
    sandbox.preflight()
    policy = make_policy(cfg)
    try:
        trainer = ExposureTrainer(cfg, policy, repo, sandbox, pool=pool, resume=args.resume)
        trainer.train()
        if trainer.total_episodes == 160:
            atomic_json(
                args.output / "complete.json",
                {"episodes": trainer.total_episodes, "student_steps": trainer.student_steps},
            )
    finally:
        policy.close()


if __name__ == "__main__":
    main()
