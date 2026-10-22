"""Full-workload acceptance of each non-pilot method, using training parents only."""

from __future__ import annotations

import os
from pathlib import Path

from selfplay_opsd import readiness
from selfplay_opsd.config import RunConfig
from selfplay_opsd.core import ExperimentError
from selfplay_opsd.readiness import evidence_root
from selfplay_opsd.storage import (
    CheckpointManager,
    active_events,
    atomic_json,
    file_hash,
    read_json,
)

SMOKE_CHECKS = {
    "committed_one_iteration",
    "student_update",
    "proposer_update",
    "generated_episodes",
    "method_alignment_path",
    "method_distillation_gate",
    "joint_checkpoint",
}


def smoke_config(config: RunConfig, method: str) -> RunConfig:
    return config.model_copy(
        update={"method": method, "study_id": config.study_id + "-method-smoke", "iterations": 1},
        deep=True,
    )


def smoke_path(config: RunConfig, method: str) -> Path:
    return evidence_root(config) / "method-smokes" / f"{method}.json"


def verify_smoke(config: RunConfig, method: str) -> dict:
    report = read_json(smoke_path(config, method))
    expected = smoke_config(config, method)
    if (
        not report.get("ready")
        or report.get("method") != method
        or report.get("runtime_identity") != readiness.execution_identity(config)
        or report.get("common_settings")
        != config.scientific_settings(common=True, calibration=True)
        or report.get("hardware") != config.hardware.model_dump(mode="json")
        or report.get("gpu_type") != os.getenv("SELFPLAY_GPU_TYPE", "h100")
        or set(report.get("checks", {})) != SMOKE_CHECKS
        or not all(report["checks"].values())
        or report.get("smoke_runtime_identity") != readiness.execution_identity(expected)
    ):
        raise ExperimentError(f"Missing or incompatible method {method} smoke evidence")
    checkpoint = expected.run_dir / "checkpoints/000001"
    if file_hash(checkpoint / "complete.json") != report["checkpoint_marker_sha256"]:
        raise ExperimentError("Smoke checkpoint marker changed")
    metadata = CheckpointManager.verify(checkpoint)
    if (
        metadata.get("iteration") != 1
        or metadata.get("scientific_hash") != expected.scientific_hash
        or metadata.get("runtime_identity") != report["smoke_runtime_identity"]
        or metadata.get("manifests_hash") != report["runtime_identity"]["manifest_hash"]
    ):
        raise ExperimentError("Smoke checkpoint identity mismatch")
    return report


def run_smoke(config: RunConfig, policy, repository, sandbox) -> None:
    from selfplay_opsd.pilot import training_reference_proxies
    from selfplay_opsd.trainer import SelfPlayTrainer

    if config.method not in "ABC" or len(config.method) != 1:
        raise ExperimentError("Method smoke accepts A, B or C; D uses the continuation pilot")
    original = config
    config = smoke_config(config, config.method)
    # Fresh initialization; previous incomplete smoke state is never silently
    # resumed or overwritten. The trainer enforces the existing-output guard.
    trainer = SelfPlayTrainer(config, policy, repository, sandbox, development_enabled=False)
    if config.candidate_objective:
        with trainer.phase("smoke_reference_preparation"):
            trainer.reference_anchors = training_reference_proxies(trainer)
    trainer.train()
    if policy.is_coordinator:
        checkpoint = config.run_dir / "checkpoints/000001"
        state, metadata = CheckpointManager.load(checkpoint)
        commits = [
            e["payload"] for e in active_events(config.run_dir) if e["kind"] == "iteration_commit"
        ]
        alignment = sum(e["metrics"]["alignment"]["scored_count"] for e in commits)
        checks = {
            "committed_one_iteration": trainer.iteration == 1 and len(commits) == 1,
            "student_update": trainer.student_steps > 0,
            "proposer_update": trainer.proposer_steps > 0,
            "generated_episodes": trainer.total_episodes > 0,
            "method_alignment_path": alignment > 0 if config.method == "C" else alignment == 0,
            "method_distillation_gate": all(
                e["metrics"]["distill"]["gating_enabled"] == (config.method != "A") for e in commits
            ),
            "joint_checkpoint": metadata["artifact_kind"] == "fully_resumable"
            and state["iteration"] == 1
            and "rng" in state
            and bool(state["policy"]["adapters"])
            and set(state["policy"]["optimizers"]) == {"student", "proposer"},
        }
        report = {
            "schema_version": 1,
            "ready": all(checks.values()),
            "checks": checks,
            "method": config.method,
            "runtime_identity": readiness.execution_identity(original),
            "smoke_runtime_identity": trainer.runtime_identity,
            "common_settings": original.scientific_settings(common=True, calibration=True),
            "hardware": original.hardware.model_dump(mode="json"),
            "gpu_type": os.getenv("SELFPLAY_GPU_TYPE", "h100"),
            "checkpoint_marker_sha256": file_hash(checkpoint / "complete.json"),
        }
        atomic_json(smoke_path(original, config.method), report)
        if not report["ready"]:
            raise ExperimentError("Method smoke did not exercise its required update path")
    policy.barrier()
