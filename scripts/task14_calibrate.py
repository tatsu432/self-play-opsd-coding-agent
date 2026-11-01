"""Calibrate ungated distillation strength on shared training-only rollouts."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

from task13_static_continue import SOURCE, restore_pool

from selfplay_opsd.cli import make_policy
from selfplay_opsd.config import load_config
from selfplay_opsd.core import digest, keyed_seed
from selfplay_opsd.data import TaskRepository
from selfplay_opsd.environment import score_teacher
from selfplay_opsd.rollout import episode_wave
from selfplay_opsd.sandbox import SandboxBackend
from selfplay_opsd.storage import atomic_json, file_hash
from selfplay_opsd.trainer import student_examples


def matched_coefficient(records):
    gated_sq = sum(r["gated_norm"] ** 2 for r in records)
    ungated_sq = sum(r["ungated_unit_norm"] ** 2 for r in records)
    return (
        math.sqrt(gated_sq / ungated_sq) if gated_sq > 0 and ungated_sq > 0 else None,
        gated_sq,
        ungated_sq,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--edit-protocol", choices=("unified_diff", "replace_function_body"), required=True
    )
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID") or not os.getenv("SELFPLAY_WORKER_TOKEN"):
        raise RuntimeError("Calibration requires an allocated isolated worker")
    if args.output.exists():
        raise ValueError("Output already exists")
    cfg = load_config(args.settings, method="B", hardware_profile="single_gpu_80gb")
    cfg.episode.edit_protocol = args.edit_protocol
    cfg.logging.mode = "disabled"
    repo = TaskRepository(cfg.data_root)
    pool = sorted(
        restore_pool(SOURCE, repo),
        key=lambda item: digest([2026, "task14-calibration", item[0].task_id]),
    )[:8]
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
            "tasks": [t.task_id for t, _, _ in pool],
            "snapshot": "base",
            "seed": 2026,
            "slots": 4,
        },
    )
    records = []
    try:
        with policy.active("student"):
            for task, oracle, _ in pool:
                jobs = [
                    (task, oracle, keyed_seed(2026, "task14-calibration", task.task_id, i))
                    for i in range(4)
                ]
                episodes = [
                    score_teacher(policy, e, oracle, cfg.episode)[0]
                    for e in episode_wave(policy, sandbox, jobs, cfg.episode, concurrency=1)
                ]
                gated, _ = student_examples(episodes, cfg, kind="sd")
                ungated_cfg = cfg.model_copy(deep=True)
                ungated_cfg.method = "A"
                ungated_cfg.objective.sd_lambda = 1.0
                ungated, _ = student_examples(episodes, ungated_cfg, kind="sd")
                _, gated_metrics = policy.gradients(gated)
                _, ungated_metrics = policy.gradients(ungated)
                records.append(
                    {
                        "task_id": task.task_id,
                        "gated_norm": gated_metrics["grad_norm"],
                        "ungated_unit_norm": ungated_metrics["grad_norm"],
                    }
                )
                atomic_json(args.output / "partial.json", records)
        coefficient, gated_sq, ungated_sq = matched_coefficient(records)
        atomic_json(
            args.output / "calibration.json",
            {
                "coefficient": coefficient,
                "gated_lambda": cfg.objective.sd_lambda,
                "gated_squared_norm_sum": gated_sq,
                "ungated_unit_squared_norm_sum": ungated_sq,
                "records": records,
                "selection": "training-only initial snapshot",
            },
        )
        atomic_json(
            args.output / "complete.json", {"tasks": len(records), "coefficient": coefficient}
        )
    finally:
        policy.close()


if __name__ == "__main__":
    main()
