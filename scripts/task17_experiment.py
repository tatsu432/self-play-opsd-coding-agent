"""Bounded Task 17 operations, using the existing policy/trainer/isolated worker."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import torch
from task13_static import StaticTrainer
from task14_calibrate import matched_coefficient

from selfplay_opsd.cli import make_policy
from selfplay_opsd.config import load_config
from selfplay_opsd.core import OracleSpec, TaskSpec, digest, keyed_seed
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
    scientific_code_hash,
)
from selfplay_opsd.task17 import STUDY, configure_operation, load_pool, prepare_pool
from selfplay_opsd.task17_recovery import admit_repair
from selfplay_opsd.trainer import SelfPlayTrainer, student_examples

ROOT = Path("/runs") / STUDY
HISTORICAL_B2026 = {
    "iteration": 245,
    "artifact_kind": "fully_resumable",
    "scientific_hash": "3fb8aaa585826f21695e89758e1a937fa371e1e83ed341b6ba3d1454d03ab709",
    "manifests_hash": "fd47b9117ffa627531a24b938999d9e063f1e6eee47e4b0b91b1bbf46c4ed714",
    "prompt_hash": "680ba2be6503a764e64dce27318397f16bc37afb2d508dc6aac088d8c2497ecb",
    "scientific_code_hash": "4f0ec3e0269d90ec45e6ba1d1dd43ef5b377acde69fc8cd6239834adbca432ab",
    "student_version": "791c821e07b6d312d2cf55e39e6d847ef48430f0e02c536085302a8c5eaf71cf",
}


def load_historical_weights(policy, cfg, repo, checkpoint):
    """Verify old identity, import weights with fresh optimizer; never claim resume."""
    state, metadata = CheckpointManager.load(checkpoint)
    old = load_config(
        Path("configs/task15-v2.yaml"), method="B", seed=2026, hardware_profile="single_gpu_80gb"
    )
    if (
        metadata["scientific_hash"] != old.scientific_hash
        or metadata["manifests_hash"] != repo.hash
        or metadata["iteration"] != 245
        or metadata["resolved_config"]["model"] != cfg.model.model_dump(mode="json")
    ):
        raise ValueError("Historical Task 15 B2026 checkpoint identity differs")
    if any(metadata.get(key) != value for key, value in HISTORICAL_B2026.items()):
        raise ValueError("Historical checkpoint provenance differs from recovered Task 15 evidence")
    original = state["policy"]
    fresh = policy.state_dict()
    policy.load_state_dict(original | {"optimizers": fresh["optimizers"]})
    if policy.version("student") != metadata["student_version"]:
        raise ValueError("Imported adapter differs from historical checkpoint")
    return {
        "path": str(checkpoint),
        "complete_sha256": file_hash(checkpoint / "complete.json"),
        "historical_prompt_hash": metadata["prompt_hash"],
        "new_prompt_hash": PROMPT_HASH,
        "import": "weights_only_fresh_optimizer",
    }


def validate_control(control, pool_hash):
    if control.get("pool_sha256") != pool_hash or not control.get("phase1_informative"):
        raise ValueError("Objective study requires an informative recorded Phase 1 decision")
    if control.get("interface") not in {"unified_diff", "replace_text"} or control.get(
        "episodes"
    ) not in {80, 128}:
        raise ValueError("Invalid frozen interface or episode endpoint")
    evidence = control.get("evidence", [])
    if not evidence:
        raise ValueError("Phase 1 decision requires actual checksummed result evidence")
    for item in evidence:
        if file_hash(Path(item["path"])) != item["sha256"]:
            raise ValueError("Decision evidence checksum differs")


class Task17Trainer(StaticTrainer):
    def __init__(self, *args, pool, development, endpoint, **kwargs):
        self.task17_development, self.endpoint = development, endpoint
        super().__init__(*args, pool=pool, **kwargs)

    def run_iteration(self, k):
        # Skip the historical step-count stopping rule: exposure is the endpoint.
        record = SelfPlayTrainer.run_iteration(self, k)
        if self.total_episodes in {self.endpoint // 2, self.endpoint}:
            self.evaluate_development()
        return record

    def evaluate_development(self):
        super().evaluate_development()
        rows = []
        with self.policy.active("student"):
            for task, oracle in self.task17_development:
                episode = run_episode(
                    self.policy,
                    self.sandbox,
                    task,
                    oracle,
                    self.config.episode,
                    keyed_seed(self.config.seed, "task17-dev", task.task_id),
                    greedy=True,
                )
                rows.append(episode.model_dump(mode="json"))
        if self.store:
            path = f"task17-development-{self.iteration:06d}.json"
            atomic_json(self.root / path, rows)
            self.store.emit(
                "task17_development",
                {
                    "episodes": self.total_episodes,
                    "path": path,
                    "successes": sum(r["reward"] for r in rows),
                    "tasks": len(rows),
                },
                committed=True,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--operation",
        choices=("prepare", "interface", "calibrate", "student", "replay"),
        required=True,
    )
    parser.add_argument("--arm", choices=("A", "B", "C", "D"), default="B")
    parser.add_argument("--seed", type=int, choices=(2026, 2027, 2028), default=2026)
    parser.add_argument("--pool", type=Path, default=ROOT / "prepared/pool.json")
    parser.add_argument(
        "--control", type=Path, default=Path("configs/task17-objective-control.json")
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/runs/repair-selfplay-2026-task15-ab-v2/B-seed2026/checkpoints/000245"),
    )
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.resume and args.operation != "student":
        raise ValueError("Only student training supports checkpoint resume")
    if not os.getenv("SLURM_JOB_ID") or not os.getenv("SELFPLAY_WORKER_TOKEN"):
        raise RuntimeError("Task 17 execution requires the allocated accepted isolated worker")
    cfg = load_config(args.settings, method="B", seed=args.seed, hardware_profile="single_gpu_80gb")
    cfg = configure_operation(cfg, args.operation, args.arm, args.control)
    repo, sandbox = TaskRepository(cfg.data_root), SandboxBackend(cfg.sandbox)
    sandbox.preflight()
    if args.output.exists() and not args.resume:
        raise ValueError("Output exists; reconcile the existing operation instead of duplicating")
    args.output.mkdir(parents=True, exist_ok=bool(args.resume), mode=0o700)
    if args.operation == "prepare":
        # This CPU operation produces current runtime evidence and admits the
        # actual new tasks; avoid a separate redundant orchestration stage.
        from selfplay_opsd.readiness import validate_runtime

        validate_runtime(cfg)
        cache = None
        invalidated = []
        descriptor_path = ROOT / "admission-recovery.json"
        if descriptor_path.is_file():
            descriptor = json.loads(descriptor_path.read_text())
            cache_path = Path(descriptor["path"])
            if (
                not cache_path.resolve().is_relative_to(ROOT.resolve())
                or file_hash(cache_path) != descriptor["sha256"]
            ):
                raise ValueError("Admission recovery source/checksum differs")
            for evidence in descriptor.get("invalidation_evidence", []):
                evidence_path = Path(evidence["path"])
                if (
                    not evidence_path.resolve().is_relative_to(ROOT.resolve())
                    or file_hash(evidence_path) != evidence["sha256"]
                ):
                    raise ValueError("Invalidation evidence path/checksum differs")
                invalidated.append(evidence["parent_id"])
            cache = json.loads(cache_path.read_text())
            atomic_json(args.output / "admission-reused-from.json", descriptor)
        prepare_pool(
            repo, sandbox, args.output / "pool.json", admission=cache, invalidated=invalidated
        )
        atomic_json(
            args.output / "complete.json", {"pool_sha256": file_hash(args.output / "pool.json")}
        )
        return
    if args.operation == "replay":
        if not args.candidates:
            raise ValueError("Replay requires the checksummed private candidate file")
        candidates = json.loads(args.candidates.read_text())
        tasks = {
            t.task_id: (t, o) for split in ("final_id", "external") for t, o in repo.tasks(split)
        }
        rows = []
        verified_sources = {}
        for candidate in candidates:
            task, oracle = tasks[candidate["task_id"]]
            if digest(candidate["source"]) != candidate["source_hash"]:
                raise ValueError("Replay source hash differs")
            admit_repair(candidate["source"], task.initial_source)
            key = (task.task_id, candidate["source_hash"])
            reused = key in verified_sources
            if not reused:
                verified_sources[key] = sandbox.verify(
                    candidate["source"],
                    task.entry_point,
                    oracle.public_cases + oracle.hidden_cases,
                    task.verifier_seconds,
                )
            result = verified_sources[key]
            rows.append(
                {k: candidate[k] for k in ("episode_id", "task_id", "turn", "source_hash")}
                | {
                    "passed": result.passed,
                    "reused_identical_source": reused,
                    "status": result.status.value,
                    "semantic_failures": result.semantic_failures,
                    "runtime_failures": result.runtime_failures,
                }
            )
            atomic_json(args.output / "replays.json", rows)
        atomic_json(
            args.output / "complete.json",
            {"candidates_sha256": file_hash(args.candidates), "verified": len(rows)},
        )
        return
    pool, pool_hash = load_pool(args.pool, repo)
    control = None
    if args.operation in {"calibrate", "student"}:
        control = json.loads(args.control.read_text())
        validate_control(control, pool_hash)
        cfg.episode.edit_protocol = control["interface"]
    else:
        cfg.episode.edit_protocol = "unified_diff" if args.arm in {"A", "C"} else "replace_text"
    purpose = {"interface": "interface", "calibrate": "calibration", "student": "curriculum"}[
        args.operation
    ]
    items = [
        (TaskSpec.model_validate(r["task"]), OracleSpec.model_validate(r["oracle"]))
        for r in pool["rows"]
        if r["purpose"] == purpose
    ]
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    import random

    import numpy as np

    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.operation == "student":
        cfg.output_root = args.output.parents[1]
        if cfg.run_dir != args.output:
            raise ValueError("Student output must equal the effective configuration run directory")
        if args.arm == "C":
            calibration_path = ROOT / "calibration/calibration.json"
            calibration = json.loads(calibration_path.read_text())
            assert control is not None
            if (
                calibration["pool_sha256"] != pool_hash
                or calibration["interface"] != cfg.episode.edit_protocol
                or calibration["coefficient"] != cfg.objective.sd_lambda
                or file_hash(calibration_path) != control.get("calibration_sha256")
            ):
                raise ValueError("Frozen calibration differs")
    if args.operation != "student":
        cfg.output_root = args.output / "policy-runtime"
    policy = make_policy(cfg)
    telemetry = None
    try:
        for task, oracle in items:
            check_admission(policy, task, oracle, cfg.episode)
        imported = None
        if args.operation == "interface" and args.arm in {"C", "D"}:
            imported = load_historical_weights(policy, cfg, repo, args.checkpoint)
        else:
            verify_base_adapter(policy)
        manifest = {
            "study_id": cfg.study_id,
            "pool_sha256": pool_hash,
            "dataset_hash": repo.hash,
            "arm": args.arm,
            "seed": args.seed,
            "interface": cfg.episode.edit_protocol,
            "episode": cfg.episode.model_dump(mode="json"),
            "model": cfg.model.model_dump(mode="json"),
            "snapshot": policy.version("student"),
            "imported_checkpoint": imported,
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
            "control_sha256": file_hash(args.control) if control is not None else None,
            "scientific_code_hash": scientific_code_hash(),
            "prompt_hash": PROMPT_HASH,
            "provenance": provenance(),
        }
        if args.resume:
            old_manifest = json.loads((args.output / "manifest.json").read_text())
            for field in (
                "study_id",
                "pool_sha256",
                "dataset_hash",
                "arm",
                "seed",
                "interface",
                "episode",
                "model",
                "task_ids",
                "prompt_hash",
                "control_sha256",
            ):
                if manifest[field] != old_manifest[field]:
                    raise ValueError("Resumed operation input differs")
        else:
            atomic_json(args.output / "manifest.json", manifest)
        if args.operation == "student":
            assert control is not None
            if args.resume and not args.resume.resolve().is_relative_to(
                cfg.run_dir.resolve() / "checkpoints"
            ):
                raise ValueError("Resume checkpoint must belong to this exact objective arm")
            development = [
                (TaskSpec.model_validate(r["task"]), OracleSpec.model_validate(r["oracle"]))
                for r in pool["rows"]
                if r["purpose"] == "development"
            ]
            trainer = Task17Trainer(
                cfg,
                policy,
                repo,
                sandbox,
                pool=[(t, o, digest([t.task_id, "static"])) for t, o in items],
                development=development,
                endpoint=control["episodes"],
                resume=args.resume,
            )
            trainer.train()
            if trainer.total_episodes == control["episodes"]:
                atomic_json(
                    args.output / "complete.json",
                    {
                        "episodes": trainer.total_episodes,
                        "student_steps": trainer.student_steps,
                        "proposer_steps": trainer.proposer_steps,
                    },
                )
            return
        if cfg.logging.mode != "disabled":
            try:
                import wandb

                telemetry = wandb.init(
                    project=cfg.logging.project,
                    entity=cfg.logging.entity,
                    group=cfg.study_id,
                    name=f"{args.operation}-{args.arm}-{args.seed}",
                    id=digest(
                        [
                            cfg.study_id,
                            args.operation,
                            args.arm,
                            args.seed,
                            pool_hash,
                            manifest["snapshot"],
                            PROMPT_HASH,
                            manifest["scientific_code_hash"],
                        ]
                    )[:24],
                    mode=cfg.logging.mode,
                    dir=str(args.output),
                    config={
                        k: manifest[k]
                        for k in (
                            "study_id",
                            "arm",
                            "seed",
                            "interface",
                            "snapshot",
                            "pool_sha256",
                            "prompt_hash",
                        )
                    },
                    settings=wandb.Settings(init_timeout=15, disable_code=True),
                )
            except Exception as exc:
                atomic_json(
                    args.output / "logging-warning.json",
                    {"error_type": type(exc).__name__, "local_data_preserved": True},
                )
        rows, calibration_rows = [], []
        with policy.active("student"):
            for task, oracle in items:
                count = 1 if args.operation == "interface" else 4
                episodes = [
                    run_episode(
                        policy,
                        sandbox,
                        task,
                        oracle,
                        cfg.episode,
                        keyed_seed(args.seed, "task17", task.task_id, slot),
                        greedy=args.operation == "interface",
                    )
                    for slot in range(count)
                ]
                rows.extend(e.model_dump(mode="json") for e in episodes)
                atomic_json(args.output / "episodes.json", rows)
                if telemetry is not None:
                    try:
                        telemetry.log(
                            {
                                "episodes": len(rows),
                                "verified_repairs": sum(e["reward"] for e in rows),
                                "generated_tokens": sum(
                                    len(t["target_ids"]) for e in rows for t in e["turns"]
                                ),
                            }
                        )
                    except Exception as exc:
                        atomic_json(
                            args.output / "logging-warning.json",
                            {"error_type": type(exc).__name__, "local_data_preserved": True},
                        )
                if args.operation == "calibrate":
                    scored = [score_teacher(policy, e, oracle, cfg.episode)[0] for e in episodes]
                    gated, _ = student_examples(scored, cfg, kind="sd")
                    ungated_cfg = cfg.model_copy(deep=True)
                    ungated_cfg.method = "A"
                    ungated_cfg.objective.sd_lambda = 1.0
                    ungated, _ = student_examples(scored, ungated_cfg, kind="sd")
                    before = copy.deepcopy(policy.state_dict())
                    _, gm = policy.gradients(gated, step=True)
                    policy.load_state_dict(before)
                    _, um = policy.gradients(ungated, step=True)
                    policy.load_state_dict(before)
                    calibration_rows.append(
                        {
                            "task_id": task.task_id,
                            "gated_norm": gm["grad_norm"],
                            "ungated_unit_norm": um["grad_norm"],
                            "gated_auxiliary_optimizer_displacement": gm["update_norm"],
                            "ungated_unit_auxiliary_optimizer_displacement": um["update_norm"],
                        }
                    )
        if args.operation == "calibrate":
            coefficient, gs, us = matched_coefficient(calibration_rows)
            atomic_json(
                args.output / "calibration.json",
                {
                    "coefficient": coefficient,
                    "pool_sha256": pool_hash,
                    "interface": cfg.episode.edit_protocol,
                    "gated_squared_norm_sum": gs,
                    "ungated_unit_squared_norm_sum": us,
                    "records": calibration_rows,
                },
            )
        atomic_json(
            args.output / "complete.json",
            {
                "episodes": len(rows),
                "successes": sum(e["reward"] for e in rows),
                "pool_sha256": pool_hash,
            },
        )
    finally:
        if telemetry is not None:
            try:
                telemetry.finish()
            except Exception as exc:
                atomic_json(
                    args.output / "logging-warning.json",
                    {"error_type": type(exc).__name__, "local_data_preserved": True},
                )
        policy.close()


if __name__ == "__main__":
    main()
