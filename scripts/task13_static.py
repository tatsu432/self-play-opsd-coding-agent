"""Build a verified training-only pool and compare static GRPO student objectives."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

from selfplay_opsd.cli import make_policy
from selfplay_opsd.config import load_config
from selfplay_opsd.core import ProposalRecord, TokenTurn, digest, keyed_seed
from selfplay_opsd.data import TaskRepository, build_task
from selfplay_opsd.environment import check_admission
from selfplay_opsd.mutations import fixed_mutation
from selfplay_opsd.rollout import episode_wave
from selfplay_opsd.sandbox import SandboxBackend
from selfplay_opsd.storage import atomic_json, file_hash, restore_rng, rng_state
from selfplay_opsd.trainer import SelfPlayTrainer


class StaticTrainer(SelfPlayTrainer):
    """Reuse the trainer's student, joint-checkpoint and evaluation machinery."""

    def __init__(self, *args, pool, **kwargs):
        self.pool = pool
        self.reviewed_steps = set()
        super().__init__(*args, **kwargs)

    def proposals(self, k):
        proposals, tasks = [], {}
        for index in range(2):
            task, oracle, key = self.pool[(2 * (k - 1) + index) % len(self.pool)]
            tasks[task.task_id] = (task, oracle)
            for slot in range(2):
                turn = TokenTurn(
                    prompt_ids=[],
                    target_ids=[],
                    old_logps=[],
                    history=[],
                    raw_text="",
                    action="static_task",
                    parse_status="static",
                )
                proposals.append(
                    ProposalRecord(
                        proposal_id=digest([k, task.parent_id, slot]),
                        parent_id=task.parent_id,
                        slot=slot,
                        seed=keyed_seed(2026, "static", k, task.task_id, slot),
                        policy_version=self.policy.version("proposer"),
                        turn=turn,
                        valid=slot == 0,
                        reason="fixed_task" if slot == 0 else "placeholder",
                        task_id=task.task_id if slot == 0 else None,
                        dedup_key=key if slot == 0 else None,
                    )
                )
        return proposals, tasks

    def run_iteration(self, k):
        record = super().run_iteration(k)
        if self.student_steps in {10, 20} and self.student_steps not in self.reviewed_steps:
            self.reviewed_steps.add(self.student_steps)
            self.evaluate_development()
        if self.student_steps >= 20:
            self.stop_requested = True
        return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID") or not os.getenv("SELFPLAY_WORKER_TOKEN"):
        raise RuntimeError("Static diagnostic requires allocated Slurm and isolated worker")
    if args.output.exists():
        raise RuntimeError("Static diagnostic output already exists")
    args.output.mkdir(parents=True, mode=0o700)
    stop = Path(os.environ.get("SELFPLAY_STOP_FILE", str(args.output / "stop_requested")))
    cfg = load_config(args.settings, method="B", hardware_profile="single_gpu_80gb")
    cfg.logging.mode = "disabled"
    repo = TaskRepository(cfg.data_root)
    sandbox = SandboxBackend(cfg.sandbox)
    sandbox.preflight()
    screening_policy = make_policy(cfg)
    initial_state = copy.deepcopy(screening_policy.state_dict())
    initial_rng = rng_state()
    initial_versions = {role: screening_policy.version(role) for role in ("student", "proposer")}
    try:
        parents = sorted(repo.parents("train"), key=lambda p: digest([2026, p.parent_id]))
        pool = []
        for parent in parents[:64]:
            if stop.exists():
                atomic_json(args.output / "stopped.json", {"phase": "pool"})
                screening_policy.close()
                return
            for attempt in range(4):
                seed = keyed_seed(2026, "static-pool", parent.parent_id, attempt)
                try:
                    change = fixed_mutation(parent.source, seed)
                    public, hidden = repo.parent_cases(parent.parent_id)
                    task, oracle = build_task(
                        parent, change, public, hidden, "train", seed, sandbox
                    )
                    check_admission(screening_policy, task, oracle, cfg.episode)
                except (ValueError, SyntaxError):
                    continue
                pool.append((task, oracle, change.dedup_key))
                break
            if len(pool) == 32:
                break
        if len(pool) != 32:
            raise ValueError(f"Only {len(pool)} verified training parents available")
        encoded = [
            {"task": t.model_dump(mode="json"), "oracle": o.model_dump(mode="json"), "key": k}
            for t, o, k in pool
        ]
        pool_hash = digest(
            {"dataset": repo.hash, "runner": file_hash(Path(__file__)), "pool": encoded}
        )
        atomic_json(
            args.output / "pool.json",
            {"pool_hash": pool_hash, "dataset": repo.hash, "tasks": encoded},
        )
        screened = []
        with screening_policy.active("student"):
            for task, oracle, _key in pool:
                if stop.exists():
                    atomic_json(
                        args.output / "stopped.json",
                        {"phase": "screen", "completed": len(screened)},
                    )
                    screening_policy.close()
                    return
                jobs = [
                    (task, oracle, keyed_seed(2026, "static-screen", task.task_id, i))
                    for i in range(8)
                ]
                episodes = episode_wave(screening_policy, sandbox, jobs, cfg.episode, concurrency=1)
                screened.append(
                    {
                        "task_id": task.task_id,
                        "successes": sum(e.reward for e in episodes),
                        "episodes": [e.model_dump(mode="json") for e in episodes],
                    }
                )
                atomic_json(args.output / "screen.json", screened)
    except BaseException:
        screening_policy.close()
        raise
    mixed_ids = {r["task_id"] for r in screened if 0 < r["successes"] < 8}
    pool.sort(key=lambda p: (p[0].task_id not in mixed_ids, p[0].task_id))
    atomic_json(
        args.output / "schedule.json",
        {
            "pool_hash": pool_hash,
            "mixed_tasks": len(mixed_ids),
            "order": [p[0].task_id for p in pool],
        },
    )
    try:
        for method in ("GRPO", "B"):
            if stop.exists():
                atomic_json(
                    args.output / "stopped.json", {"before_arm": method, "pool_hash": pool_hash}
                )
                return
            cfg = load_config(args.settings, method=method, hardware_profile="single_gpu_80gb")
            cfg.study_id = "repair-diagnosis-2026-static-" + pool_hash[:12]
            cfg.output_root = args.output
            cfg.logging.mode = "disabled"
            cfg.parents_per_iteration = 2
            cfg.proposals_per_parent = 2
            cfg.iterations = 80
            cfg.evaluate_every = 80
            cfg.checkpoint_every = 1
            screening_policy.load_state_dict(copy.deepcopy(initial_state))
            restore_rng(initial_rng)
            if any(
                screening_policy.version(role) != version
                for role, version in initial_versions.items()
            ):
                raise RuntimeError("Static control adapter initialization differs")
            screening_policy.config = cfg
            trainer = StaticTrainer(cfg, screening_policy, repo, sandbox, pool=pool)
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
    finally:
        screening_policy.close()
    atomic_json(
        args.output / "complete.json",
        {"pool_hash": pool_hash, "mixed_tasks": len(mixed_ids), "arms": ["GRPO", "B"]},
    )


if __name__ == "__main__":
    main()
