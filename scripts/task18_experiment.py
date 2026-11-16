"""Allocated Task 18 diagnostics and interface cells using the existing isolated runner."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from task17_experiment import Task17Trainer, load_historical_weights

from selfplay_opsd.cli import make_policy
from selfplay_opsd.config import ObjectiveConfig, load_config
from selfplay_opsd.core import EpisodeRecord, OracleSpec, TaskSpec, digest, keyed_seed
from selfplay_opsd.data import TaskRepository
from selfplay_opsd.environment import check_admission, run_episode, score_teacher
from selfplay_opsd.evaluation import verify_base_adapter
from selfplay_opsd.protocol import PROMPT_HASH
from selfplay_opsd.sandbox import SandboxBackend
from selfplay_opsd.storage import (
    CheckpointManager,
    atomic_json,
    file_hash,
    provenance,
    restore_rng,
    rng_state,
    scientific_code_hash,
)
from selfplay_opsd.task15_audit import gradient_mode_probe
from selfplay_opsd.task17 import load_pool
from selfplay_opsd.task18 import OPERATIONS, configure_operation, load_native_probe
from selfplay_opsd.task18_numerical import (
    audit_updates,
    cached_scoring_probe,
    frozen_rollout_replay,
    saved_token_report,
)
from selfplay_opsd.task18_protocol import NativePolicy
from selfplay_opsd.trainer import SelfPlayTrainer


class Task18Trainer(Task17Trainer):
    def run_iteration(self, k):
        record = SelfPlayTrainer.run_iteration(self, k)
        if self.total_episodes in {32, 40, 80}:
            self.evaluate_development()
        return record


def replay_numerical(policy, cfg, repo, pool, control, output):
    reports = []
    oracles = {
        row["task"]["task_id"]: OracleSpec.model_validate(row["oracle"])
        for row in pool["rows"]
        if row["purpose"] == "curriculum"
    }
    for source in control["numerical_sources"]:
        path, checkpoint = Path(source["episodes"]), Path(source["checkpoint"])
        if (
            file_hash(path) != source["episodes_sha256"]
            or file_hash(checkpoint / "complete.json") != source["checkpoint_complete_sha256"]
        ):
            raise ValueError("Numerical source checksum differs")
        state, metadata = CheckpointManager.load(checkpoint)
        if metadata["manifests_hash"] != repo.hash or metadata["resolved_config"][
            "model"
        ] != cfg.model.model_dump(mode="json"):
            raise ValueError("Numerical source model or dataset differs")
        episodes = [EpisodeRecord.model_validate(row) for row in json.loads(path.read_text())]
        if not episodes or any(e.task_id not in oracles for e in episodes):
            raise ValueError("Replay must use frozen training curriculum tasks")
        cfg.method = metadata["resolved_config"]["method"]
        if metadata["resolved_config"]["sd_gap_definition"] != "same_scorer_v2":
            raise ValueError("Task 18 replay requires the original same-scorer SD gap")
        cfg.objective = ObjectiveConfig.model_validate(metadata["resolved_config"]["objective"])
        policy.load_state_dict(state["policy"])
        restore_rng(state["rng"])
        if any(e.policy_version != policy.version("student") for e in episodes):
            raise ValueError("Replay checkpoint is not the exact generating snapshot")
        if control.get("numerical_stage") == "cached-scoring":
            reports.append(
                {
                    "source": source,
                    "cached_scoring": cached_scoring_probe(policy, episodes),
                    "rollout_replay": frozen_rollout_replay(
                        policy, episodes, control.get("replay_rollout_turns", 4)
                    ),
                }
            )
            atomic_json(output / "numerical-replays.json", reports)
            continue
        original = saved_token_report(episodes, cfg.objective.clip)
        scored = [score_teacher(policy, e, oracles[e.task_id], cfg.episode)[0] for e in episodes]
        reports.append(
            {
                "source": source,
                "original_saved_tokens": original,
                "fresh_saved_tokens": saved_token_report(scored, cfg.objective.clip),
                "gradient_mode": gradient_mode_probe(policy, cfg, scored),
                "updates": audit_updates(
                    policy,
                    cfg,
                    scored,
                    control["update_relative_l2_screen"],
                    control["update_cosine_screen"],
                ),
            }
        )
        atomic_json(output / "numerical-replays.json", reports)
    atomic_json(
        output / "complete.json",
        {
            "replayed_sources": len(reports),
            "generated_episodes": 0,
            "generated_tokens": sum(
                r.get("rollout_replay", {}).get("new_generated_tokens", 0) for r in reports
            ),
            "training_ready": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--operation", choices=OPERATIONS, required=True)
    parser.add_argument("--arm", choices=("A", "B", "D"), default="A")
    parser.add_argument("--seed", type=int, choices=(2026, 2027), default=2026)
    parser.add_argument(
        "--control", type=Path, default=Path("/workspace/configs/task18-control.json")
    )
    parser.add_argument(
        "--pool",
        type=Path,
        default=Path("/runs/repair-selfplay-2026-task17-interface-v1/prepared/pool.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/runs/repair-selfplay-2026-task15-ab-v2/B-seed2026/checkpoints/000245"),
    )
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID") or not os.getenv("SELFPLAY_WORKER_TOKEN"):
        raise RuntimeError("Task 18 requires allocated Slurm and the accepted isolated worker")
    if args.resume and args.operation != "student":
        raise ValueError("Only student training supports continuation")
    if args.output.exists() and not args.resume:
        raise ValueError("Output exists; reconcile instead of duplicating")
    control = json.loads(args.control.read_text())
    cfg = configure_operation(
        load_config(args.settings, seed=args.seed, hardware_profile="single_gpu_80gb"),
        args.operation,
        args.arm,
        args.control,
    )
    repo = TaskRepository(cfg.data_root)
    pool, pool_hash = load_pool(args.pool, repo)
    if pool_hash != control["pool_sha256"]:
        raise ValueError("Frozen pool differs")
    purpose = (
        "curriculum"
        if args.operation == "student"
        else "calibration"
        if args.operation == "numerical"
        else "interface"
    )
    rows = [
        row
        for row in pool["rows"]
        if row["purpose"] == purpose and row["presentation"] == "original"
    ]
    if args.operation == "native":
        if control["native_families"] not in {8, 40}:
            raise ValueError("Native probe must use eight or forty frozen families")
        rows = sorted(rows, key=lambda row: digest(["task18-native", row["task"]["task_id"]]))[
            : control["native_families"]
        ]
    if args.operation == "numerical":
        if (
            type(control["numerical_groups"]) is not int
            or not 1 <= control["numerical_groups"] <= 4
        ):
            raise ValueError("Numerical generation is bounded to four groups")
        rows = rows[: control["numerical_groups"]]
    items = [
        (TaskSpec.model_validate(row["task"]), OracleSpec.model_validate(row["oracle"]))
        for row in rows
    ]
    sandbox = SandboxBackend(cfg.sandbox)
    sandbox.preflight()
    args.output.mkdir(parents=True, exist_ok=bool(args.resume), mode=0o700)
    if args.operation == "student":
        cfg.output_root = args.output.parents[1]
        if cfg.run_dir != args.output:
            raise ValueError("Student output must equal its effective run directory")
        if args.resume and not args.resume.resolve().is_relative_to(
            args.output.resolve() / "checkpoints"
        ):
            raise ValueError("Continuation must belong to this exact arm")
    else:
        cfg.output_root = args.output / "policy-runtime"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    policy = make_policy(cfg)
    if cfg.episode.tool_serialization == "qwen_native":
        policy = NativePolicy(policy)
    try:
        if args.operation == "numerical" and control.get("numerical_sources"):
            atomic_json(
                args.output / "manifest.json",
                {
                    "study_id": cfg.study_id,
                    "operation": "numerical-replay",
                    "numerical_stage": control.get("numerical_stage", "updates"),
                    "control_sha256": file_hash(args.control),
                    "provenance": provenance(),
                    "sources": control["numerical_sources"],
                },
            )
            replay_numerical(policy, cfg, repo, pool, control, args.output)
            return
        imported = (
            load_historical_weights(policy, cfg, repo, args.checkpoint)
            if args.operation in {"native", "normalized"} and args.arm == "B"
            else None
        )
        if imported is None:
            verify_base_adapter(policy)
        for task, oracle in items:
            check_admission(policy, task, oracle, cfg.episode)
        manifest = {
            "study_id": cfg.study_id,
            "operation": args.operation,
            "arm": args.arm,
            "seed": args.seed,
            "pool_sha256": pool_hash,
            "dataset_hash": repo.hash,
            "control_sha256": file_hash(args.control),
            "scientific_hash": cfg.scientific_hash,
            "scientific_code_hash": scientific_code_hash(),
            "prompt_hash": PROMPT_HASH,
            "snapshot": policy.version("student"),
            "imported_checkpoint": imported,
            "episode": cfg.episode.model_dump(mode="json"),
            "model": cfg.model.model_dump(mode="json"),
            "task_ids": [t.task_id for t, _ in items],
            "task_features": {
                t.task_id: {
                    "source_tokens": len(
                        policy.tokenizer.encode(t.initial_source, add_special_tokens=False)
                    ),
                    "spec_tokens": len(
                        policy.tokenizer.encode(t.specification, add_special_tokens=False)
                    ),
                    "public_cases": len(o.public_cases),
                }
                for t, o in items
            },
            "provenance": provenance(),
        }
        reused = []
        if args.operation == "native" and control["native_families"] == 40:
            reference = control.get("native_probe_reuse", {}).get(args.arm)
            if not reference:
                raise ValueError("Full native evaluation must reuse its completed eight-task probe")
            reused = load_native_probe(reference, manifest, {t.task_id for t, _ in items[:8]})
            manifest["reused_probe"] = reference
            manifest["new_task_ids"] = [
                t.task_id for t, _ in items if t.task_id not in {e.task_id for e in reused}
            ]
        if args.resume:
            previous = json.loads((args.output / "manifest.json").read_text())
            if any(
                manifest[key] != previous[key]
                for key in manifest
                if key not in {"provenance", "snapshot"}
            ):
                raise ValueError("Continuation science differs")
        else:
            atomic_json(args.output / "manifest.json", manifest)
        if args.operation == "student":
            development = [
                (TaskSpec.model_validate(r["task"]), OracleSpec.model_validate(r["oracle"]))
                for r in pool["rows"]
                if r["purpose"] == "development"
            ]
            trainer = Task18Trainer(
                cfg,
                policy,
                repo,
                sandbox,
                pool=[(t, o, digest([t.task_id, "static"])) for t, o in items],
                development=development,
                endpoint=80,
                resume=args.resume,
            )
            trainer.train()
            if trainer.total_episodes == 80:
                atomic_json(
                    args.output / "complete.json",
                    {
                        "episodes": 80,
                        "student_steps": trainer.student_steps,
                        "proposer_steps": trainer.proposer_steps,
                    },
                )
            return
        if args.operation == "numerical":
            CheckpointManager(args.output).save(
                0,
                {"policy": copy.deepcopy(policy.state_dict()), "rng": rng_state()},
                {
                    "iteration": 0,
                    "artifact_kind": "task18_diagnostic_only",
                    "scientific_hash": cfg.scientific_hash,
                    "snapshot": policy.version("student"),
                },
            )
        episodes = list(reused)
        reused_ids = {e.task_id for e in reused}
        with policy.active("student"):
            for task, oracle in items:
                if task.task_id in reused_ids:
                    continue
                for slot in range(4 if args.operation == "numerical" else 1):
                    episode = run_episode(
                        policy,
                        sandbox,
                        task,
                        oracle,
                        cfg.episode,
                        keyed_seed(args.seed, "task17", task.task_id, slot),
                        greedy=args.operation != "numerical",
                    )
                    if args.operation == "numerical":
                        episode = score_teacher(policy, episode, oracle, cfg.episode)[0]
                    episodes.append(episode)
                    atomic_json(
                        args.output / "episodes.json", [e.model_dump(mode="json") for e in episodes]
                    )
        if args.operation == "numerical":
            report = {
                "saved_tokens": saved_token_report(episodes, cfg.objective.clip),
                "gradient_mode": gradient_mode_probe(policy, cfg, episodes),
                "updates": audit_updates(
                    policy,
                    cfg,
                    episodes,
                    control["update_relative_l2_screen"],
                    control["update_cosine_screen"],
                ),
                "sampling": {
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "min_p": 0.0,
                    "repetition_penalty": 1.0,
                },
                "source": "fresh bounded calibration-family diagnostic; not a historical mismatch reproduction",
            }
            atomic_json(args.output / "numerical.json", report)
        atomic_json(
            args.output / "complete.json",
            {
                "episodes": len(episodes),
                "successes": sum(e.reward for e in episodes),
                "pool_sha256": pool_hash,
                "episodes_sha256": file_hash(args.output / "episodes.json"),
                "reused_episodes": len(reused),
                "generated_episodes": len(episodes) - len(reused),
            },
        )
    finally:
        policy.close()


if __name__ == "__main__":
    main()
