"""Evidence shared by runtime checks, study freezing and main launch gates."""

from __future__ import annotations

import os
from pathlib import Path

from selfplay_opsd.config import RunConfig
from selfplay_opsd.core import ExperimentError, digest
from selfplay_opsd.data import TaskRepository
from selfplay_opsd.protocol import PROMPT_HASH
from selfplay_opsd.sandbox import SandboxBackend
from selfplay_opsd.storage import (
    active_events,
    atomic_json,
    code_hash,
    read_json,
    scientific_code_hash,
)


def evidence_root(config: RunConfig) -> Path:
    return (
        config.output_root
        / "readiness"
        / os.getenv("SELFPLAY_MAIN_STUDY", config.study_id.removesuffix("-pilot"))
    )


def validation_path(config: RunConfig) -> Path:
    return Path(
        os.getenv("SELFPLAY_VALIDATION_REPORT", str(config.data_root / "validation_report.json"))
    )


def execution_identity(config: RunConfig, *, repository: TaskRepository | None = None) -> dict:
    repository = repository if repository is not None else TaskRepository(config.data_root)
    if repository.root.resolve() != config.data_root.resolve():
        raise ExperimentError("Repository root differs from configured data_root")
    return {
        "code_hash": code_hash(),
        "scientific_code_hash": scientific_code_hash(),
        "common_hash": digest(config.scientific_settings(common=True, calibration=True)),
        "manifest_hash": repository.hash,
        "prompt_hash": PROMPT_HASH,
        "model_revision": config.model.revision,
        "tokenizer_revision": config.model.tokenizer_revision,
        "trainer_sha256": os.getenv("SELFPLAY_IMAGE_DIGEST"),
        "worker_sha256": os.getenv("SELFPLAY_WORKER_SHA256"),
    }


def validate_runtime(config: RunConfig) -> dict:
    from selfplay_opsd.evaluation import validate_data

    root = evidence_root(config)
    sandbox = SandboxBackend(config.sandbox)
    isolation = sandbox.preflight()
    validation = validate_data(
        TaskRepository(config.data_root), sandbox, output=root / "validation.json"
    )
    report = {
        "ready": bool(isolation["ready"] and validation["ready"]),
        "isolation": isolation,
        "validation": validation,
        "runtime_identity": execution_identity(config),
    }
    atomic_json(root / "runtime.json", report)
    return report


def readiness_report(config: RunConfig, *, require_freeze: bool = True) -> dict:
    """Read evidence without allocating GPUs; runtime preflight still runs per job."""
    root = evidence_root(config)
    pilot_suffix = os.getenv("SELFPLAY_PILOT_SUFFIX", "-pilot")
    gpu_type = os.getenv("SELFPLAY_GPU_TYPE", "h100")
    identity = execution_identity(config)
    gates: dict[str, dict] = {}

    def gate(name: str, action):
        try:
            value = action()
            if value is not True:
                raise ValueError(str(value))
            gates[name] = {"ready": True}
        except (OSError, KeyError, ValueError, ExperimentError) as exc:
            gates[name] = {"ready": False, "reason": str(exc)}

    def report_matches(path: Path) -> dict:
        report = read_json(path)
        if report.get("runtime_identity") != identity:
            raise ValueError(f"Stale or incompatible evidence: {path.name}")
        return report

    def runtime():
        report = report_matches(root / "runtime.json")
        validation = read_json(root / "validation.json")
        return bool(
            report.get("ready")
            and report["isolation"].get("ready")
            and all(report["isolation"].get("checks", {}).values())
            and validation.get("executed")
            and validation.get("ready")
            and validation.get("manifest_hash") == identity["manifest_hash"]
            and validation.get("sandbox_image_digest") == "sha256:" + str(identity["worker_sha256"])
            and validation.get("execution_limits") == config.scientific_settings()["sandbox"]
        )

    def backend(profile: str):
        report = report_matches(config.output_root / "compatibility" / f"{profile}.json")
        count = 1 if profile == "single_gpu_80gb" else config.hardware.gpu_count
        return bool(
            report.get("ready")
            and report.get("gpu_count") == count
            and report.get("hardware_profile") == profile
            and (count == 1 or report.get("gpu_type") == gpu_type)
            and report.get("objective_probes") == ["A", "B", "C", "D"]
            and (count == 1 or report.get("single_process_reference_compared"))
        )

    def base():
        base_root = config.output_root / config.study_id / "base"
        if read_json(base_root / "runtime_identity.json") != identity:
            raise ValueError("Base evaluation runtime identity differs")
        return any(
            e["kind"] == "evaluation_complete" and e["payload"]["split"] == "dev_monitor"
            for e in active_events(base_root)
        )

    def pilot():
        report = report_matches(
            config.output_root
            / f"{config.study_id}{pilot_suffix}/D-seed{config.seed}/pilot_report.json"
        )
        pilot_config = RunConfig.model_validate(report["pilot_config"])
        return bool(
            report.get("iterations", 0) >= 10
            and report.get("includes_alignment")
            and report.get("target_hardware_verified")
            and report.get("scored_candidates", 0) > 0
            and report.get("student_steps", 0) > 0
            and report.get("proposer_steps", 0) > 0
            and 2 in report.get("resume_history", [report.get("resumed_from_iteration")])
            and pilot_config.scientific_settings(common=True, calibration=True)
            == config.scientific_settings(common=True, calibration=True)
            and report["hardware"]["profile"] == config.hardware.profile
        )

    gate(
        "images",
        lambda: all(
            isinstance(identity[k], str) and len(identity[k]) == 64
            for k in ("trainer_sha256", "worker_sha256")
        ),
    )
    gate("runtime_and_data", runtime)
    gate("single_gpu_reference", lambda: backend("single_gpu_80gb"))
    gate("target_backend", lambda: backend(config.hardware.profile))
    gate("base_evaluation", base)
    gate("pilot_and_continuation", pilot)
    if require_freeze:

        def frozen():
            report = read_json(config.output_root / config.study_id / "study_frozen.json")
            return (
                report.get("common_config") == config.scientific_settings(common=True)
                and report.get("runtime_identity") == identity
            )

        gate("frozen_study", frozen)
    files = [
        root / "runtime.json",
        root / "validation.json",
        config.output_root / "compatibility/single_gpu_80gb.json",
        config.output_root / "compatibility" / f"{config.hardware.profile}.json",
        config.output_root / config.study_id / "base/evaluation_complete.json",
        config.output_root / config.study_id / "base/runtime_identity.json",
        config.output_root
        / f"{config.study_id}{pilot_suffix}/D-seed{config.seed}/pilot_report.json",
    ]
    if require_freeze:
        files.append(config.output_root / config.study_id / "study_frozen.json")
    from selfplay_opsd.storage import file_hash

    evidence_files = {
        str(path.relative_to(config.output_root)): file_hash(path)
        for path in files
        if path.is_file()
    }
    return {
        "gpu_type": gpu_type,
        "pilot_suffix": pilot_suffix,
        "common_config": config.scientific_settings(common=True),
        "evidence_files": evidence_files,
        "study_id": config.study_id,
        "runtime_identity": identity,
        "gates": gates,
        "blockers": [name + ": " + g["reason"] for name, g in gates.items() if not g["ready"]],
        "eligible_to_submit_main": all(g["ready"] for g in gates.values()),
        "runtime_preflight_required_per_job": True,
    }
