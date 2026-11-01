"""Run one bounded Task 14 online self-play arm."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from selfplay_opsd.cli import make_policy
from selfplay_opsd.config import load_config
from selfplay_opsd.data import TaskRepository
from selfplay_opsd.sandbox import SandboxBackend
from selfplay_opsd.storage import atomic_json
from selfplay_opsd.trainer import SelfPlayTrainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=tuple("ABCD"), required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID") or not os.getenv("SELFPLAY_WORKER_TOKEN"):
        raise RuntimeError("Online training requires an allocated isolated worker")
    if args.output.exists() and not args.resume:
        raise ValueError("Output already exists")
    if args.resume and (not args.output.is_dir() or not args.resume.is_relative_to(args.output)):
        raise ValueError("Resume checkpoint must belong to the existing online run")
    cfg = load_config(args.settings, method=args.method, hardware_profile="single_gpu_80gb")
    if cfg.iterations != 20 or cfg.checkpoint_every != 1 or not cfg.gradient_diagnostics:
        raise ValueError(
            "Task 14 online run requires 20 iterations and per-iteration diagnostics/checkpoints"
        )
    if cfg.logging.mode != "online":
        raise ValueError("Task 14 online run requires W&B online logging")
    cfg.output_root = args.output
    repo = TaskRepository(cfg.data_root)
    sandbox = SandboxBackend(cfg.sandbox)
    sandbox.preflight()
    policy = make_policy(cfg)
    try:
        if not args.resume:
            args.output.mkdir(parents=True, mode=0o700)
            atomic_json(
                args.output / "initial_versions.json",
                {
                    "student": policy.version("student"),
                    "proposer": policy.version("proposer"),
                    "seed": cfg.seed,
                },
            )
        trainer = SelfPlayTrainer(cfg, policy, repo, sandbox, resume=args.resume)
        trainer.train()
        if trainer.iteration == cfg.iterations:
            atomic_json(
                args.output / "complete.json",
                {
                    "iterations": trainer.iteration,
                    "episodes": trainer.total_episodes,
                    "student_steps": trainer.student_steps,
                    "proposer_steps": trainer.proposer_steps,
                },
            )
    finally:
        policy.close()


if __name__ == "__main__":
    main()
