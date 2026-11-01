"""Matched-exposure fixed-task trainer for Task 14 experiments."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
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
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID") or not os.getenv("SELFPLAY_WORKER_TOKEN"):
        raise RuntimeError("Static experiment requires an allocated isolated worker")
    control = json.loads(args.control.read_text())
    run_id = control["calibration_run_id"]
    if not re.fullmatch(r"[0-9a-f]{24}", run_id):
        raise ValueError("Invalid calibration run ID")
    calibration_path = Path("/runs/repair-diagnosis-2026-task14-calibrate-v1") / run_id
    if file_hash(calibration_path / "calibration.json") != control["calibration_sha256"]:
        raise ValueError("Calibration checksum differs from frozen control")
    calibration = json.loads((calibration_path / "calibration.json").read_text())
    calibration_manifest = json.loads((calibration_path / "manifest.json").read_text())
    completion = json.loads((calibration_path / "complete.json").read_text())
    coefficient = calibration.get("coefficient")
    if (
        completion.get("tasks") != 8
        or coefficient is None
        or coefficient <= 0
        or completion.get("coefficient") != coefficient
    ):
        raise ValueError("Static arms require completed nondegenerate calibration")
    if calibration_manifest["protocol"] != control["edit_protocol"]:
        raise ValueError("Calibration protocol differs from frozen control")
    if args.output.exists() and not args.resume:
        raise ValueError("Output already exists")
    if args.resume and (not args.output.is_dir() or not args.resume.is_relative_to(args.output)):
        raise ValueError("Resume checkpoint must belong to the existing static run")
    cfg = load_config(args.settings, method=args.method, hardware_profile="single_gpu_80gb")
    cfg.episode.edit_protocol = calibration_manifest["protocol"]
    cfg.seed = args.seed
    cfg.study_id = "repair-diagnosis-2026-task14-static-v1"
    cfg.output_root = args.output
    cfg.logging.mode = "disabled"
    cfg.parents_per_iteration = cfg.proposals_per_parent = 2
    cfg.iterations, cfg.evaluate_every, cfg.checkpoint_every = 20, 21, 1
    if args.method == "A":
        cfg.objective.sd_lambda = coefficient
    repo = TaskRepository(cfg.data_root)
    if calibration_manifest["dataset"] != repo.hash or calibration_manifest[
        "pool_sha256"
    ] != file_hash(SOURCE / "pool.json"):
        raise ValueError("Calibration dataset or static pool differs")
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
        "edit_protocol": cfg.episode.edit_protocol,
        "calibration_sha256": control["calibration_sha256"],
        "control_sha256": file_hash(args.control),
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
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    policy = make_policy(cfg)
    try:
        if not args.resume:
            atomic_json(
                args.output / "initial_versions.json",
                {
                    "student": policy.version("student"),
                    "proposer": policy.version("proposer"),
                    "seed": args.seed,
                },
            )
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
