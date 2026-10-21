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
from selfplay_opsd.storage import active_events, atomic_json, code_hash, provenance, read_json


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
            validation_path = cfg.data_root / "validation_report.json"
            if not validation_path.exists():
                blockers.append("Target sandbox data validation missing")
            else:
                validated = read_json(validation_path)
                if (
                    not validated.get("executed")
                    or validated.get("manifest_hash") != repo.hash
                    or validated.get("execution_limits")
                    != cfg.sandbox.model_dump(
                        mode="json",
                        exclude={"backend", "image", "runtime", "remote_url", "remote_token_env"},
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
        proof_path = cfg.output_root / "compatibility" / f"{cfg.hardware.profile}.json"
        if not proof_path.exists():
            blockers.append("Target backend acceptance probe missing; run backend-check")
        else:
            proof = read_json(proof_path)
            valid = (
                proof.get("ready")
                and proof.get("code_hash") == code_hash()
                and proof.get("model_revision") == cfg.model.revision
                and proof.get("gpu_count") == cfg.hardware.gpu_count
            )
            if cfg.hardware.gpu_count > 1:
                valid = valid and proof.get("single_process_reference_compared")
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
    return {
        "checks": checks,
        "blockers": blockers,
        "ready_for_smoke": not any("Sandbox" in b for b in blockers),
        "ready_for_pilot": not blockers,
        "ready_for_full_runs": not blockers
        and (cfg.output_root / cfg.study_id / "study_frozen.json").exists(),
        "full_run_gate": "requires measured target-hardware pilot and frozen study config",
        "provenance": provenance(),
    }


def freeze_study(config: RunConfig, pilot: Path) -> dict:
    report = read_json(pilot)
    if "pilot_config" not in report:
        raise ExperimentError("Pilot report lacks resolved scientific settings")
    excluded = {
        "study_id",
        "method",
        "iterations",
        "hardware",
        "logging",
        "output_root",
        "data_root",
        "max_active_hours",
        "max_gpu_hours",
    }
    pilot_common = {k: v for k, v in report["pilot_config"].items() if k not in excluded}
    common = {k: v for k, v in config.model_dump(mode="json").items() if k not in excluded}
    if pilot_common != common:
        raise ExperimentError(
            "Pilot common settings differ from the main study; reset/recalibrate explicitly"
        )
    if (
        report.get("iterations", 0) < 10
        or not report.get("target_hardware_verified")
        or not report.get("includes_alignment")
    ):
        raise ExperimentError(
            "Freeze requires >=10 measured target-hardware iterations including C/D alignment"
        )
    root = config.output_root / config.study_id
    base_events = active_events(root / "base") if (root / "base/events").exists() else []
    if not any(
        e["kind"] == "evaluation_complete" and e["payload"]["split"] == "dev_monitor"
        for e in base_events
    ):
        raise ExperimentError(
            "Evaluate the shared untrained base on monitoring development tasks before freezing the main study"
        )
    path = root / "study_frozen.json"
    if path.exists():
        raise ExperimentError(
            "Study config already frozen; create a new study ID for scientific changes"
        )
    repository = TaskRepository(config.data_root)
    if report.get("manifest_hash") != repository.hash:
        raise ExperimentError("Pilot dataset differs from the main study")
    payload = {
        "study_id": config.study_id,
        "common_config": config.model_dump(
            mode="json",
            exclude={
                "method",
                "hardware",
                "logging",
                "output_root",
                "data_root",
                "max_active_hours",
                "max_gpu_hours",
            },
        ),
        "manifest_hash": repository.hash,
        "prompt_hash": PROMPT_HASH,
        "pilot_hash": digest(report),
        "pilot": report,
        "requires_fresh_initialization": True,
    }
    atomic_json(path, payload)
    return payload


def check_frozen(config: RunConfig, repository: TaskRepository) -> None:
    path = config.output_root / config.study_id / "study_frozen.json"
    if not path.exists():
        raise ExperimentError(
            "Study not frozen: complete backend acceptance, shared base evaluation and target-hardware pilot, then freeze-study"
        )
    frozen = read_json(path)
    common = config.model_dump(
        mode="json",
        exclude={
            "method",
            "hardware",
            "logging",
            "output_root",
            "data_root",
            "max_active_hours",
            "max_gpu_hours",
        },
    )
    if (
        frozen["common_config"] != common
        or frozen["manifest_hash"] != repository.hash
        or frozen["prompt_hash"] != PROMPT_HASH
    ):
        raise ExperimentError("Main study differs from frozen common config/data/prompts")
