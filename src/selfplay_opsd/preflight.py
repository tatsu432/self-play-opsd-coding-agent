"""Readiness reports describe observed capabilities, not assumed GPU support."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
from pathlib import Path
from typing import Any

import psutil
import torch

from selfplay_opsd.config import RunConfig
from selfplay_opsd.core import ExperimentError, digest
from selfplay_opsd.data import TaskRepository
from selfplay_opsd.protocol import PROMPT_HASH
from selfplay_opsd.sandbox import SandboxBackend
from selfplay_opsd.storage import atomic_json, provenance, read_json


def doctor(config: RunConfig, *, isolation: bool = False) -> dict:
    checks: dict[str, Any] = {}
    blockers: list[str] = []
    cfg = config
    checks["architecture"] = platform.machine()
    checks["python"] = platform.python_version()
    checks["backend"] = cfg.model.backend
    checks["gpu_available"] = torch.cuda.is_available()
    checks["visible_gpu_count"] = torch.cuda.device_count()
    checks["cpu_allocation"] = int(os.getenv("SLURM_CPUS_PER_TASK", str(psutil.cpu_count() or 1)))
    checks["host_available_bytes"] = psutil.virtual_memory().available
    checks["disk_available_bytes"] = shutil.disk_usage(Path.cwd()).free
    checks["devices"] = [
        {
            "index": i,
            "name": torch.cuda.get_device_name(i),
            "total_vram_bytes": torch.cuda.get_device_properties(i).total_memory,
            "free_vram_bytes": torch.cuda.mem_get_info(i)[0],
        }
        for i in range(torch.cuda.device_count())
    ]
    if cfg.hardware.gpu_count:
        if not checks["gpu_available"] or checks["visible_gpu_count"] < cfg.hardware.gpu_count:
            blockers.append("Allocated CUDA devices unavailable")
        if any(
            d["total_vram_bytes"] < cfg.hardware.min_vram_gib * 1024**3
            for d in checks["devices"][: cfg.hardware.gpu_count]
        ):
            blockers.append("Device VRAM below selected profile")
        if cfg.model.backend != "verl":
            blockers.append("GPU production profiles require verl; HF is the debug backend")
        for package, expected in (("verl", "0.7.1"), ("vllm", "0.11.0"), ("torch", "2.8.0")):
            try:
                version = importlib.metadata.version(package)
                if version.split("+")[0] != expected:
                    blockers.append(f"{package} version {version} differs from pinned {expected}")
            except importlib.metadata.PackageNotFoundError:
                blockers.append(f"{package} not installed; uv sync --group gpu")
    if cfg.hardware.cpu_workers > checks["cpu_allocation"]:
        blockers.append("CPU workers exceed scheduler allocation")
    if not cfg.model.tiny and any(
        x == "UNRESOLVED" or len(x) != 40
        for x in (cfg.model.revision, cfg.model.tokenizer_revision)
    ):
        blockers.append("Exact model/tokenizer commit revisions required")
    if cfg.logging.mode == "online" and not os.getenv("WANDB_API_KEY"):
        checks["wandb_auth"] = "no key in environment; existing SDK authentication not inspected"
    else:
        checks["wandb_auth"] = "offline/disabled or environment key present"
    checks["sandbox"] = {"tested": False}
    if isolation:
        try:
            checks["sandbox"] = {"tested": True, **SandboxBackend(cfg.sandbox).preflight()}
        except (ExperimentError, OSError) as exc:
            checks["sandbox"] = {"tested": True, "ready": False, "error": str(exc)}
            blockers.append("Sandbox isolation unavailable")
    else:
        blockers.append("Sandbox isolation not tested in this preflight")
    try:
        repo = TaskRepository(cfg.data_root)
        checks["data"] = {"manifest_hash": repo.hash, "ready": repo.manifest.get("ready", False)}
        if not repo.manifest.get("ready"):
            blockers.append("Data/anchors not finalized")
        if repo.manifest.get("prompt_hash") != PROMPT_HASH:
            blockers.append("Prepared prompt hash mismatch")
        if repo.manifest.get("episode_limits_hash") != digest(cfg.episode.model_dump()):
            blockers.append("Prepared episode budget mismatch")
        if cfg.hardware.gpu_count:
            from selfplay_opsd.readiness import validation_path as target_validation

            validation_path = target_validation(cfg)
            if not validation_path.exists():
                blockers.append("Target sandbox data validation missing")
            else:
                validated = read_json(validation_path)
                if (
                    not validated.get("ready")
                    or not validated.get("executed")
                    or validated.get("manifest_hash") != repo.hash
                    or not cfg.sandbox.matches_execution_limits(
                        validated.get("execution_limits", {})
                    )
                ):
                    blockers.append("Data validation does not match the actual execution contract")
                elif isolation and validated.get("sandbox_image_digest") != checks["sandbox"].get(
                    "image_digest"
                ):
                    blockers.append("Revalidate data against the selected sandbox image")
    except (OSError, ValueError, KeyError) as exc:
        checks["data"] = {"ready": False, "error": str(exc)}
        blockers.append("Frozen dataset unavailable")
    if cfg.hardware.gpu_count:
        from selfplay_opsd.readiness import (
            BACKEND_EVIDENCE_DIRECTORY,
            backend_environment,
            backend_identity,
        )

        proof_path = cfg.output_root / BACKEND_EVIDENCE_DIRECTORY / f"{cfg.hardware.profile}.json"
        if not proof_path.exists():
            blockers.append("Target backend acceptance probe missing; run backend-check")
        else:
            proof = read_json(proof_path)
            valid = (
                proof.get("backend_identity") == backend_identity(cfg)
                and proof.get("objective_probes") == ["A", "B", "C", "D"]
                and proof.get("ready")
                and proof.get("model_revision") == cfg.model.revision
                and proof.get("gpu_count") == cfg.hardware.gpu_count
            )
            if cfg.hardware.gpu_count > 1:
                valid = valid and proof.get("single_process_reference_compared")
            try:
                valid = valid and proof.get("environment") == backend_environment(gpu=True)
            except OSError:
                valid = False
            if os.getenv("SELFPLAY_GPU_TYPE"):
                valid = valid and proof.get("gpu_type") == os.environ["SELFPLAY_GPU_TYPE"]
            checks["backend_acceptance"] = {"valid": bool(valid), "proof": str(proof_path)}
            if not valid:
                blockers.append(
                    "Backend acceptance proof stale or missing single-process gradient comparison"
                )
    checks["expected_memory"] = {
        "profile_host_ram_gib": cfg.hardware.host_ram_gib,
        "measured_training_peak": None,
        "feasibility": "unmeasured until pilot",
    }
    from selfplay_opsd.readiness import readiness_report

    full_ready = not blockers and readiness_report(cfg)["eligible_to_submit_main"]
    return {
        "checks": checks,
        "blockers": blockers,
        "ready_for_smoke": not any("Sandbox" in b for b in blockers),
        "ready_for_pilot": not blockers,
        "ready_for_full_runs": full_ready,
        "full_run_gate": "requires measured target-hardware pilot and frozen study config",
        "provenance": provenance(),
    }


def freeze_study(config: RunConfig, pilot: Path) -> dict:
    import fcntl

    from selfplay_opsd.readiness import execution_identity, readiness_report

    expected_pilot = (
        config.output_root
        / f"{config.study_id}{os.getenv('SELFPLAY_PILOT_SUFFIX', '-pilot')}/D-seed{config.seed}/pilot_report.json"
    )
    if pilot.resolve() != expected_pilot.resolve():
        raise ExperimentError("Freeze must use this study's measured D pilot")
    readiness = readiness_report(config, require_freeze=False)
    if not readiness["eligible_to_submit_main"]:
        raise ExperimentError("Study readiness gates failed: " + "; ".join(readiness["blockers"]))
    report = read_json(pilot)
    root = config.output_root / config.study_id
    root.mkdir(parents=True, exist_ok=True)
    from selfplay_opsd.telemetry import telemetry_identity

    payload = {
        "schema_version": 2,
        "telemetry_identity": telemetry_identity(),
        "study_id": config.study_id,
        "gpu_type": os.getenv("SELFPLAY_GPU_TYPE", "h100"),
        "pilot_suffix": os.getenv("SELFPLAY_PILOT_SUFFIX", "-pilot"),
        "common_config": config.scientific_settings(common=True),
        "manifest_hash": TaskRepository(config.data_root).hash,
        "prompt_hash": PROMPT_HASH,
        "runtime_identity": execution_identity(config),
        "pilot_hash": digest(report),
        "pilot": report,
        "readiness": readiness,
        "execution_commit": provenance()["git_commit"],
        "requires_fresh_initialization": True,
    }
    plan_path = root / "resource_plan.json"
    if plan_path.exists():
        from selfplay_opsd.readiness import evidence_root, hardware_settings

        plan = read_json(plan_path)
        if (
            plan.get("runtime_identity") != payload["runtime_identity"]
            or plan.get("common_config") != payload["common_config"]
            or plan.get("pilot_hash") != payload["pilot_hash"]
            or plan.get("checkpoint_every") != config.checkpoint_every
            or plan.get("hardware") != hardware_settings(config.hardware.model_dump(mode="json"))
            or plan.get("gpu_type") != payload["gpu_type"]
            or plan.get("baseline_hash")
            != digest(read_json(root / "base/evaluation_complete.json"))
            or plan.get("method_hashes")
            != {
                method: digest(
                    read_json(evidence_root(config) / "method-smokes" / f"{method}.json")
                )
                for method in "ABC"
            }
        ):
            raise ExperimentError("Resource plan does not match measured common-study evidence")
        for method in "ABCD":
            entry = plan["per_method"][method]
            if not (
                0 < entry["planning_projection_200_hours"] < float("inf")
                and type(entry["signal_seconds"]) is int
                and type(entry["max_continuations"]) is int
                and 300 <= entry["signal_seconds"] <= 65535
                and 1 <= entry["max_continuations"] <= 32
            ):
                raise ExperimentError("Resource plan has an invalid continuation envelope")
        payload["resource_plan"] = plan
        payload["resource_plan_hash"] = digest(plan)
    path = root / "study_frozen.json"
    with (root / ".freeze.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if path.exists():
            if read_json(path) != payload:
                raise ExperimentError("Conflicting study freeze; use a new study ID")
        else:
            atomic_json(path, payload)
    return payload


def check_frozen(config: RunConfig, repository: TaskRepository) -> None:
    from selfplay_opsd.readiness import execution_identity, readiness_report

    path = config.output_root / config.study_id / "study_frozen.json"
    if not path.exists():
        raise ExperimentError(
            "Study not frozen: complete runtime, backend, base evaluation and D pilot"
        )
    frozen = read_json(path)
    if (
        frozen.get("common_config") != config.scientific_settings(common=True)
        or frozen.get("manifest_hash") != repository.hash
        or frozen.get("prompt_hash") != PROMPT_HASH
        or frozen.get("runtime_identity") != execution_identity(config)
    ):
        raise ExperimentError("Main study differs from frozen science/data/prompts/runtime")
    report = readiness_report(config)
    if not report["eligible_to_submit_main"]:
        raise ExperimentError("Frozen study evidence is stale: " + "; ".join(report["blockers"]))
