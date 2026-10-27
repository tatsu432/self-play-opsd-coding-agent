"""Evidence shared by runtime checks, study freezing and main launch gates."""

from __future__ import annotations

import ast
import os
import platform
from pathlib import Path

from selfplay_opsd.config import RunConfig
from selfplay_opsd.core import ExperimentError, digest
from selfplay_opsd.data import TaskRepository
from selfplay_opsd.protocol import PROMPT_HASH
from selfplay_opsd.sandbox import SandboxBackend
from selfplay_opsd.storage import (
    active_events,
    atomic_json,
    file_hash,
    read_json,
    scientific_code_hash,
)
from selfplay_opsd.worker_runtime import allocation_fingerprint, supervisor_hash

BACKEND_EVIDENCE_DIRECTORY = "compatibility/backend-v2"


def backend_environment(*, gpu: bool) -> dict:
    """Actual allocated-host ABI; checked again before generated GPU work."""
    import torch

    return {
        "kernel": platform.release(),
        "nvidia_driver": Path("/proc/driver/nvidia/version").read_text().strip() if gpu else None,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


def allocation_code_hash() -> str:
    return allocation_fingerprint(Path(__file__).resolve().parents[2] / "scripts/hpc_job.py")


def implementation_hash(*, backend: bool = False) -> str:
    """Dependency-scoped compatibility; complete source provenance stays in reports."""
    package = Path(__file__).parent
    excluded = {"hpc.py", "hpc_images.py", "hpc_readiness.py"}
    if backend:
        included = {
            "backend_check.py",
            "core.py",
            "objectives.py",
            "policy.py",
            "verl_backend.py",
            "vllm_worker.py",
            "protocol.py",
            "storage.py",
            "codec.py",
        }
    else:
        included = {p.name for p in package.glob("*.py")} - excluded
    project = package.parents[1]
    selected = {}
    if backend:
        for filename, function in (
            ("cli.py", "make_policy"),
            ("diagnostics.py", "_optimizer_digest"),
        ):
            tree = ast.parse((package / filename).read_text())
            selected[filename] = ast.dump(
                next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function),
                include_attributes=False,
            )
    return digest(
        {
            **{name: file_hash(package / name) for name in sorted(included)},
            **selected,
            **{name: file_hash(project / name) for name in ("uv.lock", "pyproject.toml")},
        }
    )


def backend_identity(config: RunConfig) -> dict:
    """Trusted synthetic probes need neither a dataset nor a generated-code worker.

    Hardware count is recorded separately so one-GPU gradients can be compared
    with distributed gradients from the same model/objective implementation.
    """
    return {
        "schema_version": 2,
        "dependency_schema": 1,
        "code_hash": implementation_hash(backend=True),
        "allocation_code_hash": allocation_code_hash(),
        "scientific_code_hash": implementation_hash(backend=True),
        "model": config.model.model_dump(mode="json"),
        "objective": config.objective.model_dump(mode="json"),
        "prompt_hash": PROMPT_HASH,
        "trainer_sha256": os.getenv("SELFPLAY_IMAGE_DIGEST"),
    }


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
        "code_hash": scientific_code_hash(),
        "allocation_code_hash": allocation_code_hash(),
        "scientific_code_hash": scientific_code_hash(),
        "common_hash": digest(config.scientific_settings(common=True, calibration=True)),
        "manifest_hash": repository.hash,
        "prompt_hash": PROMPT_HASH,
        "model_revision": config.model.revision,
        "tokenizer_revision": config.model.tokenizer_revision,
        "trainer_sha256": os.getenv("SELFPLAY_IMAGE_DIGEST"),
        "worker_sha256": os.getenv("SELFPLAY_WORKER_SHA256"),
        "worker_setup_id": os.getenv("SELFPLAY_WORKER_SETUP_ID", ""),
    }


def validate_runtime(config: RunConfig) -> dict:
    from selfplay_opsd.evaluation import validate_data
    from selfplay_opsd.storage import provenance

    root = evidence_root(config)
    sandbox = SandboxBackend(config.sandbox)
    isolation = sandbox.preflight()
    repository = TaskRepository(config.data_root)
    validation = validate_data(repository, sandbox, output=root / "validation.json")
    report = {
        "ready": bool(isolation["ready"] and validation["ready"]),
        "isolation": isolation,
        "validation": validation,
        "runtime_identity": execution_identity(config, repository=repository),
        "provenance": provenance(),
    }
    atomic_json(root / "runtime.json", report)
    return report


def readiness_report(config: RunConfig, *, require_freeze: bool = True) -> dict:
    """Read evidence without allocating GPUs; runtime preflight still runs per job."""
    root = evidence_root(config)
    pilot_suffix = os.getenv("SELFPLAY_PILOT_SUFFIX", "-pilot")
    gpu_type = os.getenv("SELFPLAY_GPU_TYPE", "h100")
    identity = execution_identity(config)
    backend_expected = backend_identity(config)
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
        if config.sandbox.resource_backend == "slurm":
            worker = read_json(root / "worker.json")
            if not (
                worker.get("ready")
                and worker.get("supervisor_hash") == supervisor_hash()
                and worker.get("checks", {}).get("allocation_termination_cleanup")
                and worker.get("checks", {}).get("oom_candidate_cleanup")
                and worker.get("termination_proof", {}).get("ready")
            ):
                return False
        return bool(
            report.get("ready")
            and report["isolation"].get("ready")
            and all(report["isolation"].get("checks", {}).values())
            and validation.get("executed")
            and validation.get("ready")
            and validation.get("manifest_hash") == identity["manifest_hash"]
            and validation.get("sandbox_image_digest") == "sha256:" + str(identity["worker_sha256"])
            and config.sandbox.matches_execution_limits(validation.get("execution_limits", {}))
        )

    def backend(profile: str):
        report = read_json(config.output_root / BACKEND_EVIDENCE_DIRECTORY / f"{profile}.json")
        count = 1 if profile == "single_gpu_80gb" else config.hardware.gpu_count
        return bool(
            report.get("ready")
            and report.get("backend_identity") == backend_expected
            and report.get("gpu_count") == count
            and report.get("hardware_profile") == profile
            and (profile != config.hardware.profile or report.get("gpu_type") == gpu_type)
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
    from selfplay_opsd.method_smoke import verify_smoke

    for method in "ABC":
        gate(f"method_{method}", lambda method=method: bool(verify_smoke(config, method)))
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
        config.output_root / BACKEND_EVIDENCE_DIRECTORY / "single_gpu_80gb.json",
        config.output_root / BACKEND_EVIDENCE_DIRECTORY / f"{config.hardware.profile}.json",
        config.output_root / config.study_id / "base/evaluation_complete.json",
        config.output_root / config.study_id / "base/runtime_identity.json",
        config.output_root
        / f"{config.study_id}{pilot_suffix}/D-seed{config.seed}/pilot_report.json",
    ]
    if require_freeze:
        files.append(config.output_root / config.study_id / "study_frozen.json")
    files.extend(root / "method-smokes" / f"{method}.json" for method in "ABC")
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
