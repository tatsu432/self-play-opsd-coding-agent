"""Run a Task 15 exposure arm; never confuse a clean allocation stop with completion."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from selfplay_opsd.cli import make_policy
from selfplay_opsd.config import load_config
from selfplay_opsd.data import TaskRepository
from selfplay_opsd.sandbox import SandboxBackend
from selfplay_opsd.storage import atomic_json
from selfplay_opsd.task15 import OnlineExposureTrainer
from selfplay_opsd.task15_freeze import check_freeze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("A", "B"), required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, choices=(2026, 2027, 2028), default=2026)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--stabilization", action="store_true")
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID") or not os.getenv("SELFPLAY_WORKER_TOKEN"):
        raise RuntimeError("Task 15 requires an allocated isolated worker")
    cfg = load_config(
        args.settings, method=args.method, seed=args.seed, hardware_profile="single_gpu_80gb"
    )
    if cfg.online_supply is None or cfg.checkpoint_every != 1:
        raise ValueError("Task 15 requires explicit exposure policy and per-update checkpoints")
    if args.stabilization:
        if "stabilization" not in cfg.study_id or cfg.online_supply.target_episodes > 128:
            raise ValueError("Stabilization must be a separately identified <=128-episode study")
    elif not args.freeze:
        raise ValueError("Main training requires the Task 15 scientific freeze")
    cfg.output_root = args.output
    if cfg.run_dir.exists() and not args.resume:
        raise ValueError("Logical run already exists; inspect its ledger rather than overwrite")
    if args.resume and not args.resume.resolve().is_relative_to(
        cfg.run_dir.resolve() / "checkpoints"
    ):
        raise ValueError("Resume checkpoint must belong to this exact logical run")
    repo = TaskRepository(cfg.data_root)
    if not args.stabilization:
        check_freeze(cfg, repo, args.freeze)
    sandbox = SandboxBackend(cfg.sandbox)
    sandbox.preflight()
    policy = make_policy(cfg)
    try:
        trainer = OnlineExposureTrainer(cfg, policy, repo, sandbox, resume=args.resume)
        trainer.train()
        if trainer.total_episodes >= cfg.online_supply.target_episodes:
            atomic_json(
                cfg.run_dir / "complete.json",
                {
                    "iterations": trainer.iteration,
                    "episodes": trainer.total_episodes,
                    "target_episodes": cfg.online_supply.target_episodes,
                    "student_steps": trainer.student_steps,
                    "proposer_steps": trainer.proposer_steps,
                    "scientific_hash": cfg.scientific_hash,
                },
            )
    finally:
        policy.close()


if __name__ == "__main__":
    main()
