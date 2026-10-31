"""Continue the fixed Task 13 control after its original allocation has stopped."""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path

from task13_static import StaticTrainer

from selfplay_opsd.cli import make_policy
from selfplay_opsd.config import load_config
from selfplay_opsd.core import OracleSpec, TaskSpec, digest, keyed_seed
from selfplay_opsd.data import TaskRepository
from selfplay_opsd.protocol import PROMPT_HASH
from selfplay_opsd.readiness import execution_identity
from selfplay_opsd.rollout import episode_wave
from selfplay_opsd.sandbox import SandboxBackend
from selfplay_opsd.storage import CheckpointManager, atomic_json, file_hash, restore_rng, rng_state

SOURCE = Path("/runs/repair-diagnosis-2026-static-v1/9a44124b61147aee02c685d6")


def arm_config(settings: Path, method: str, study: str, source: Path):
    cfg = load_config(settings, method=method, hardware_profile="single_gpu_80gb")
    cfg.study_id, cfg.output_root = study, source
    cfg.logging.mode = "disabled"
    cfg.parents_per_iteration = cfg.proposals_per_parent = 2
    cfg.iterations, cfg.evaluate_every, cfg.checkpoint_every = 80, 80, 1
    return cfg


def verified_checkpoint(path: Path, cfg, repo: TaskRepository) -> dict:
    state, metadata = CheckpointManager.load(path)
    runtime = execution_identity(cfg, repository=repo)
    if (
        metadata["scientific_hash"] != cfg.scientific_hash
        or metadata["manifests_hash"] != repo.hash
        or metadata["prompt_hash"] != PROMPT_HASH
        or metadata.get("scientific_code_hash") != runtime["scientific_code_hash"]
        or metadata.get("runtime_identity") != runtime
        or metadata["iteration"] != state["iteration"]
    ):
        raise ValueError("Static checkpoint identity differs")
    return state


def restore_pool(source: Path, repo: TaskRepository) -> list:
    record = json.loads((source / "pool.json").read_text())
    encoded = record["tasks"]
    expected = digest(
        {
            "dataset": repo.hash,
            "runner": file_hash(Path(__file__).with_name("task13_static.py")),
            "pool": encoded,
        }
    )
    if record["dataset"] != repo.hash or record["pool_hash"] != expected or len(encoded) != 32:
        raise ValueError("Static pool or original runner identity differs")
    pool = [
        (TaskSpec.model_validate(p["task"]), OracleSpec.model_validate(p["oracle"]), p["key"])
        for p in encoded
    ]
    if len({t.task_id for t, _, _ in pool}) != 32 or any(
        t.split != "train" or t.task_id != o.task_id for t, o, _ in pool
    ):
        raise ValueError("Invalid static training pool")
    return pool


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID") or not os.getenv("SELFPLAY_WORKER_TOKEN"):
        raise RuntimeError("Continuation requires an allocation and isolated worker")
    checkpoints = list(
        SOURCE.glob(
            "repair-diagnosis-2026-static-*/B-seed2026/checkpoints/[0-9][0-9][0-9][0-9][0-9][0-9]/complete.json"
        )
    )
    if (SOURCE / "complete.json").exists() or not (
        (SOURCE / "stopped.json").is_file() or checkpoints
    ):
        raise RuntimeError("Original control has no recoverable checkpoint or is already complete")
    if args.output.exists():
        raise RuntimeError("Continuation output already exists")
    stop = Path(os.environ["SELFPLAY_STOP_FILE"])
    args.output.mkdir(parents=True, mode=0o700)
    with (SOURCE / "continuation.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg = load_config(args.settings, method="B", hardware_profile="single_gpu_80gb")
        cfg.logging.mode = "disabled"
        repo = TaskRepository(cfg.data_root)
        pool = restore_pool(SOURCE, repo)
        pool_record = json.loads((SOURCE / "pool.json").read_text())
        pool_hash = pool_record["pool_hash"]
        sandbox = SandboxBackend(cfg.sandbox)
        sandbox.preflight()
        policy = make_policy(cfg)
        initial_state, initial_rng = copy.deepcopy(policy.state_dict()), rng_state()
        try:
            screen_path = SOURCE / "screen.json"
            screened = json.loads(screen_path.read_text()) if screen_path.exists() else []
            if len(screened) > len(pool) or any(
                row["task_id"] != pool[i][0].task_id
                or len(row["episodes"]) != 8
                or row["successes"] != sum(e["reward"] for e in row["episodes"])
                for i, row in enumerate(screened)
            ):
                raise ValueError("Partial static screen does not match the original pool")
            with policy.active("student"):
                for task, oracle, _ in pool[len(screened) :]:
                    if stop.exists():
                        atomic_json(
                            args.output / "stopped.json",
                            {"phase": "screen", "completed": len(screened)},
                        )
                        return
                    jobs = [
                        (task, oracle, keyed_seed(2026, "static-screen", task.task_id, i))
                        for i in range(8)
                    ]
                    episodes = episode_wave(policy, sandbox, jobs, cfg.episode, concurrency=1)
                    screened.append(
                        {
                            "task_id": task.task_id,
                            "successes": sum(e.reward for e in episodes),
                            "episodes": [e.model_dump(mode="json") for e in episodes],
                        }
                    )
                    atomic_json(screen_path, screened)
            mixed_ids = {r["task_id"] for r in screened if 0 < r["successes"] < 8}
            pool.sort(key=lambda p: (p[0].task_id not in mixed_ids, p[0].task_id))
            schedule = {
                "pool_hash": pool_hash,
                "mixed_tasks": len(mixed_ids),
                "order": [p[0].task_id for p in pool],
            }
            schedule_path = SOURCE / "schedule.json"
            if schedule_path.exists() and json.loads(schedule_path.read_text()) != schedule:
                raise ValueError("Saved static task schedule differs")
            atomic_json(schedule_path, schedule)
            study = "repair-diagnosis-2026-static-" + pool_hash[:12]
            initial_checkpoint = SOURCE / study / "GRPO-seed2026/checkpoints/000000"
            if initial_checkpoint.is_dir():
                initial_checkpoint_state = verified_checkpoint(
                    initial_checkpoint, arm_config(args.settings, "GRPO", study, SOURCE), repo
                )
                if initial_checkpoint_state["iteration"] != 0:
                    raise ValueError("Initial checkpoint is not iteration zero")
                initial_state, initial_rng = (
                    initial_checkpoint_state["policy"],
                    initial_checkpoint_state["rng"],
                )
            for method in ("GRPO", "B"):
                if stop.exists():
                    atomic_json(
                        args.output / "stopped.json", {"before_arm": method, "pool_hash": pool_hash}
                    )
                    return
                if method == "B" and initial_checkpoint.is_dir():
                    first_state = verified_checkpoint(
                        initial_checkpoint, arm_config(args.settings, "GRPO", study, SOURCE), repo
                    )
                    if first_state["iteration"] != 0:
                        raise ValueError("Initial checkpoint is not iteration zero")
                    initial_state, initial_rng = first_state["policy"], first_state["rng"]
                arm_cfg = arm_config(args.settings, method, study, SOURCE)
                checkpoint_root = arm_cfg.run_dir / "checkpoints"
                checkpoints = sorted(
                    p for p in checkpoint_root.glob("[0-9][0-9][0-9][0-9][0-9][0-9]") if p.is_dir()
                )
                resume = checkpoints[-1] if checkpoints else None
                if resume:
                    state = verified_checkpoint(resume, arm_cfg, repo)
                    if state["student_steps"] >= 20 or state["iteration"] >= 80:
                        continue
                policy.load_state_dict(copy.deepcopy(initial_state))
                restore_rng(initial_rng)
                policy.config = arm_cfg
                trainer = StaticTrainer(arm_cfg, policy, repo, sandbox, pool=pool, resume=resume)
                if resume:
                    trainer.reviewed_steps = {n for n in (10, 20) if n <= trainer.student_steps}
                elif initial_checkpoint.is_dir():
                    # Checkpoint zero captures the original arm's exact pre-training RNG.
                    policy.load_state_dict(copy.deepcopy(initial_state))
                    restore_rng(initial_rng)
                trainer.train()
                if stop.exists():
                    atomic_json(
                        args.output / "stopped.json",
                        {
                            "phase": method,
                            "student_steps": trainer.student_steps,
                            "pool_hash": pool_hash,
                        },
                    )
                    return
            result = {
                "pool_hash": pool_hash,
                "mixed_tasks": len(mixed_ids),
                "arms": ["GRPO", "B"],
                "continued": True,
            }
            atomic_json(SOURCE / "complete.json", result)
            atomic_json(args.output / "complete.json", result)
        finally:
            policy.close()


if __name__ == "__main__":
    main()
