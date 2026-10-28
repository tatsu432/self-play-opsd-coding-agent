"""Full-workload acceptance of each non-pilot method, using training parents only."""

from __future__ import annotations

import os
import re
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

ACCEPTANCE_ITERATIONS = 3

SMOKE_CHECKS = {
    "committed_acceptance_window",
    "student_update",
    "proposer_update",
    "generated_episodes",
    "method_alignment_path",
    "method_distillation_gate",
    "joint_checkpoint",
}


def smoke_config(config: RunConfig, method: str) -> RunConfig:
    return config.model_copy(
        update={
            "method": method,
            "study_id": config.study_id + "-method-smoke",
            "iterations": ACCEPTANCE_ITERATIONS,
        },
        deep=True,
    )


def smoke_path(config: RunConfig, method: str) -> Path:
    return evidence_root(config) / "method-smokes" / f"{method}.json"


def verify_smoke(config: RunConfig, method: str) -> dict:
    report = read_json(smoke_path(config, method))
    source_config = config
    reuse = report.get("reuse_provenance")
    if reuse:
        source_study = reuse["source_study"]
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", source_study):
            raise ExperimentError("Unsafe reused smoke study")
        source_config = config.model_copy(update={"study_id": source_study}, deep=True)
        source = smoke_path(source_config, method)
        original = read_json(source)
        if (
            file_hash(source) != reuse["sha256"]
            or {k: v for k, v in report.items() if k != "reuse_provenance"} != original
        ):
            raise ExperimentError("Reused smoke provenance differs from source")
    expected = smoke_config(source_config, method)
    if (
        not report.get("ready")
        or report.get("method") != method
        or report.get("runtime_identity") != readiness.execution_identity(config)
        or report.get("common_settings")
        != config.scientific_settings(common=True, calibration=True)
        or readiness.hardware_settings(report.get("hardware"))
        != readiness.hardware_settings(config.hardware.model_dump(mode="json"))
        or report.get("gpu_type") != os.getenv("SELFPLAY_GPU_TYPE", "h100")
        or set(report.get("checks", {})) != SMOKE_CHECKS
        or not all(report["checks"].values())
        or report.get("smoke_runtime_identity") != readiness.execution_identity(expected)
    ):
        raise ExperimentError(f"Missing or incompatible method {method} smoke evidence")
    checkpoint = expected.run_dir / f"checkpoints/{ACCEPTANCE_ITERATIONS:06d}"
    if file_hash(checkpoint / "complete.json") != report["checkpoint_marker_sha256"]:
        raise ExperimentError("Smoke checkpoint marker changed")
    metadata = CheckpointManager.verify(checkpoint)
    if (
        metadata.get("iteration") != ACCEPTANCE_ITERATIONS
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
    # State collection and gradients can be collective, even for fixture checks.
    mechanics = optimizer_mechanics(policy)
    if policy.is_coordinator:
        checkpoint = config.run_dir / f"checkpoints/{ACCEPTANCE_ITERATIONS:06d}"
        state, metadata = CheckpointManager.load(checkpoint)
        commits = [
            e["payload"] for e in active_events(config.run_dir) if e["kind"] == "iteration_commit"
        ]
        alignment = sum(e["metrics"]["alignment"]["scored_count"] for e in commits)
        checks = {
            "committed_acceptance_window": trainer.iteration == ACCEPTANCE_ITERATIONS
            and len(commits) == ACCEPTANCE_ITERATIONS,
            "student_update": update_path_accepted(
                commits, "student", trainer.student_steps, mechanics["ready"]
            ),
            "proposer_update": update_path_accepted(
                commits, "proposer", trainer.proposer_steps, mechanics["ready"]
            ),
            "generated_episodes": trainer.total_episodes > 0,
            "method_alignment_path": alignment > 0 if config.method == "C" else alignment == 0,
            "method_distillation_gate": all(
                e["metrics"]["distill"]["gating_enabled"] == (config.method != "A") for e in commits
            ),
            "joint_checkpoint": metadata["artifact_kind"] == "fully_resumable"
            and state["iteration"] == ACCEPTANCE_ITERATIONS
            and "rng" in state
            and bool(state["policy"]["adapters"])
            and set(state["policy"]["optimizers"]) == {"student", "proposer"},
        }
        report = {
            "schema_version": 2,
            "acceptance_iterations": ACCEPTANCE_ITERATIONS,
            "optimizer_mechanics_fixture": mechanics,
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


def update_path_accepted(commits: list[dict], role: str, steps: int, mechanics_ready: bool) -> bool:
    import math

    rows = [e["metrics"][role] for e in commits]
    if not rows or any(not math.isfinite(row["grad_norm"]) for row in rows):
        return False
    correct_steps = all(
        row["stepped"] > 0 if row["grad_norm"] > 0 else row["stepped"] == 0 for row in rows
    )
    zero_path = mechanics_ready and (
        role == "student" or all(row["nonzero_advantage_count"] == 0 for row in rows)
    )
    return correct_steps and (steps > 0 or zero_path)


def optimizer_mechanics(policy) -> dict:
    """Labelled deterministic mechanics only; restore state before returning."""
    import copy
    import time

    started = time.monotonic()
    started_at = time.time()
    from selfplay_opsd.diagnostics import _optimizer_digest
    from selfplay_opsd.policy import LossExample

    state = copy.deepcopy(policy.state_dict())
    before = _optimizer_digest(state)
    checks = {}
    try:
        prompt = policy.encode_history([{"role": "user", "content": "Synthetic optimizer fixture"}])
        target = policy.tokenizer.encode('{"action":"finish"}', add_special_tokens=False)
        for role in ("student", "proposer"):
            with policy.active(role):
                old_version = policy.version(role)
                old = policy.score(prompt, target).tolist()
                _, metrics = policy.gradients(
                    [LossExample(prompt, target, old, 1.0, [0.0] * len(target), 1 / len(target))],
                    step=True,
                )
                checks[role] = (
                    metrics["grad_norm"] > 0
                    and metrics["stepped"] == 1
                    and policy.version(role) != old_version
                )
    finally:
        policy.load_state_dict(state)
    checks["restored"] = _optimizer_digest(policy.state_dict()) == before
    return {
        "fixture_only": True,
        "ready": all(checks.values()),
        "checks": checks,
        "seconds": time.monotonic() - started,
        "started_at": started_at,
        "ended_at": time.time(),
    }
