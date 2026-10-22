"""Local, explicit HPC operations; the scheduler adapter is replaceable for AWS.

No GPU allocation is made by inspect, doctor or sync. SSH uses the named course
aliases and a bounded, non-interactive connection; MFA must first be established
through the user's normal Yale login flow when required.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Protocol

import typer

from selfplay_opsd.config import PROFILES, load_config
from selfplay_opsd.core import digest
from selfplay_opsd.worker_runtime import supervisor_hash

app = typer.Typer(no_args_is_help=True, help="Inspect and operate approved Yale Slurm jobs")
ROOT = Path(__file__).resolve().parents[2]
REMOTE_SCRIPT = ROOT / "scripts/hpc_remote.py"
ALIASES = ("yale-hpc-cpsc4880", "yale-hpc-cpsc4710", "yale-hpc-sds6190")
GPU_PARTITIONS = {"h100": "gpu_h100", "h200": "gpu_h200", "rtx_pro_6000_blackwell": "gpu_rtx6000"}
GPU_VRAM = {"h100": 80, "h200": 141, "rtx_pro_6000_blackwell": 96}
ACTIVE = {"PENDING", "CONFIGURING", "RUNNING", "COMPLETING", "SUSPENDED", "REQUEUED"}
TRANSIENT = {"PREEMPTED", "NODE_FAIL", "BOOT_FAIL"}


class HpcError(RuntimeError):
    pass


class Scheduler(Protocol):
    def call(self, command: str, payload: dict | None = None) -> dict: ...


@dataclass
class YaleSSH:
    alias: str = ALIASES[0]

    def __post_init__(self) -> None:
        if self.alias not in ALIASES:
            raise HpcError(f"Unknown course SSH alias: {self.alias}")

    @property
    def control_path(self) -> Path:
        return Path.home() / ".ssh/selfplay-control" / self.alias

    @property
    def ssh(self) -> list[str]:
        return [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ControlMaster=auto",
            "-o",
            f"ControlPath={self.control_path}",
            self.alias,
        ]

    @property
    def rsync_ssh(self) -> str:
        return f"ssh -o BatchMode=yes -o ConnectTimeout=10 -o ControlMaster=auto -o ControlPath={self.control_path}"

    def login(self) -> dict:
        directory = self.control_path.parent
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(directory, 0o700)
        existing = subprocess.run(
            ["ssh", "-S", str(self.control_path), "-O", "check", self.alias],
            capture_output=True,
            text=True,
            check=False,
        )
        if existing.returncode == 0:
            return {"alias": self.alias, "authenticated_master": True, "expires_after_idle": "2h"}
        result = subprocess.run(
            [
                "ssh",
                "-M",
                "-S",
                str(self.control_path),
                "-o",
                "BatchMode=no",
                "-o",
                "ControlPersist=2h",
                "-fN",
                self.alias,
            ],
            check=False,
        )
        if result.returncode:
            raise HpcError("Yale SSH login failed; complete the normal key/MFA flow")
        return {"alias": self.alias, "authenticated_master": True, "expires_after_idle": "2h"}

    def call(self, command: str, payload: dict | None = None) -> dict:
        if command not in {
            "inspect",
            "context",
            "study",
            "preview",
            "prepare_image",
            "prepare_source_image",
            "resources",
            "prepare",
            "finalize",
            "status",
            "submit",
            "recover",
            "logs",
        }:
            raise HpcError("Unsupported remote operation")
        encoded = base64.b64encode(json.dumps(payload or {}, sort_keys=True).encode()).decode()
        result = subprocess.run(
            [*self.ssh, "python3", "-", command, encoded],
            input=REMOTE_SCRIPT.read_text(),
            capture_output=True,
            text=True,
            timeout=45 if command == "inspect" else 180,
            check=False,
        )
        try:
            report = json.loads(result.stdout.strip().splitlines()[-1])
        except (IndexError, ValueError) as exc:
            detail = result.stderr.strip()[-1200:]
            raise HpcError(
                f"Yale SSH failed ({result.returncode}): {detail or 'no JSON response'}"
            ) from exc
        if result.returncode or "error" in report:
            raise HpcError(report.get("error", result.stderr.strip()[-1200:]))
        return report

    def rsync(self, source: Path, destination: str) -> None:
        if not re.fullmatch(r"/[A-Za-z0-9_./-]+", destination):
            raise HpcError("Unsafe remote transfer path")
        source_arg = str(source) + ("/" if source.is_dir() else "")
        result = subprocess.run(
            [
                "rsync",
                "-a",
                "--checksum",
                "--chmod=Du=rwx,Dgo=,Fu=rw,Fgo=",
                "-e",
                self.rsync_ssh,
                source_arg,
                f"{self.alias}:{destination}",
            ],
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
        if result.returncode:
            raise HpcError(f"Rsync failed ({result.returncode}): {result.stderr.strip()[-1200:]}")

    def fetch(self, source: str, destination: Path, *, full: bool = False) -> None:
        if not re.fullmatch(r"/[A-Za-z0-9_./-]+", source):
            raise HpcError("Unsafe remote artifact path")
        destination.mkdir(parents=True, exist_ok=True, mode=0o700)
        excludes = (
            []
            if full
            else [
                "--exclude=checkpoints/",
                "--exclude=trajectories/",
                "--exclude=wandb/",
                "--exclude=*.sif",
                "--exclude=*.tar",
            ]
        )
        result = subprocess.run(
            [
                "rsync",
                "-a",
                "--no-perms",
                "--no-owner",
                "--no-group",
                "--checksum",
                *excludes,
                "-e",
                self.rsync_ssh,
                f"{self.alias}:{source}/",
                str(destination) + "/",
            ],
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
        if result.returncode:
            raise HpcError(
                f"Artifact retrieval failed ({result.returncode}): {result.stderr.strip()[-1200:]}"
            )


def output(value: dict) -> None:
    typer.echo(json.dumps(value, indent=2, sort_keys=True))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def snapshot() -> dict:
    if git("status", "--porcelain", "--untracked-files=all"):
        raise HpcError("Git checkout has uncommitted files; make an immutable run commit first")
    manifest = ROOT / "data/prepared/manifest.json"
    validation = ROOT / "data/prepared/validation_report.json"
    if not manifest.is_file() or not validation.is_file():
        raise HpcError("Prepared and validated data are required for Yale synchronization")
    manifest_data = json.loads(manifest.read_text())
    validation_data = json.loads(validation.read_text())
    if (
        not manifest_data.get("ready")
        or not validation_data.get("ready")
        or not validation_data.get("executed")
        or validation_data.get("manifest_hash") != digest(manifest_data)
    ):
        raise HpcError("Prepared data or validation is not ready")
    return {"commit": git("rev-parse", "HEAD"), "manifest_sha256": sha256(manifest)}


def choose_storage(report: dict, alias: str) -> str:
    course = alias.removeprefix("yale-hpc-")
    candidate = report["storage"]["project"].get(f"project_{course}")
    if not candidate:
        raise HpcError(f"Cannot uniquely identify project_{course} storage on Yale")
    return candidate


def choose_scratch(report: dict, alias: str) -> str:
    course = alias.removeprefix("yale-hpc-")
    candidates = [
        path for name, path in report["storage"]["scratch"].items() if name.endswith(course)
    ]
    if len(candidates) != 1:
        raise HpcError(f"Cannot uniquely identify scratch storage for {course}")
    return candidates[0]


def remote_context(remote: Scheduler, alias: str) -> dict:
    report = remote.call("context")
    return {"storage_root": choose_storage(report, alias), "inspection": report}


def sync(remote: YaleSSH) -> dict:
    identity = snapshot()
    context = remote_context(remote, remote.alias)
    scratch = choose_scratch(context["inspection"], remote.alias)
    payload = {**identity, "storage_root": context["storage_root"], "scratch_root": scratch}
    paths = remote.call("prepare", payload)
    with tempfile.TemporaryDirectory(prefix="selfplay-hpc-") as temp:
        bundle = Path(temp) / f"{identity['commit']}.bundle"
        subprocess.run(
            ["git", "-C", str(ROOT), "bundle", "create", str(bundle), "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        remote.rsync(bundle, paths["bundle"])
    remote.rsync(ROOT / "data/prepared/", paths["dataset"] + "/")
    return {
        **remote.call("finalize", payload),
        **identity,
        "storage_root": context["storage_root"],
        "scratch_root": scratch,
    }


def choose_account(report: dict, partition: str) -> str:
    association = report["slurm"]["associations"]
    if association["exit_code"] != 0:
        raise HpcError("Cannot verify Slurm account associations")
    accounts = {
        parts[0]
        for line in association["stdout"].splitlines()
        if (parts := line.split("|"))
        and len(parts) >= 2
        and parts[0]
        and parts[1] in ("", partition)
    }
    if len(accounts) != 1:
        raise HpcError(f"Expected one permitted account for {partition}; found {sorted(accounts)}")
    return accounts.pop()


def run_payload(
    remote: Scheduler,
    alias: str,
    *,
    operation: str,
    method: str,
    profile: str,
    gpu_type: str | None,
    account: str | None,
    image: str,
    image_sha256: str,
    settings: Path,
    iterations: int = 10,
    resume: str | None = None,
    base: bool = False,
    split: str = "dev_monitor",
    checkpoint_iteration: int | None = None,
    endpoint: str = "primary",
    secrets_file: str | None = None,
    worker_setup_id: str = "",
) -> dict:
    if operation not in {
        "backend-check",
        "allocation-check",
        "worker-diagnose",
        "pilot",
        "train",
        "evaluate",
        "diagnose",
        "image-convert",
        "image-build",
        "method-smoke",
        "worker-check",
        "runtime-check",
        "freeze-study",
        "readiness-check",
    }:
        raise HpcError("Unsupported submission operation")
    if method not in "ABCD" or len(method) != 1:
        raise HpcError("Method must be A, B, C, or D")
    if profile not in PROFILES or profile == "cpu_debug":
        raise HpcError("Choose a production hardware profile")
    if operation == "pilot" and method != "D":
        raise HpcError("The calibration pilot must use method D")
    if image and not re.fullmatch(r"/[A-Za-z0-9_./-]+", image):
        raise HpcError("Trainer SIF must be an absolute remote path")
    if image_sha256 and not re.fullmatch(r"[0-9a-f]{64}", image_sha256):
        raise HpcError("Trainer SIF requires its SHA-256 digest")
    if iterations < 1 or iterations > 200:
        raise HpcError("Pilot iteration count must be between 1 and 200")
    if split not in {"dev_monitor", "final_id", "external"} or endpoint not in {
        "primary",
        "secondary",
    }:
        raise HpcError("Unsupported frozen evaluation split or endpoint")
    if operation == "diagnose" and checkpoint_iteration is None:
        raise HpcError("Diagnostics require a checkpoint iteration")
    if operation == "evaluate" and not base and checkpoint_iteration is None:
        raise HpcError("Evaluation requires --base or --checkpoint-iteration")
    if base and (operation != "evaluate" or checkpoint_iteration is not None):
        raise HpcError("--base is only valid for evaluation without a checkpoint")
    if secrets_file and not re.fullmatch(r"/[A-Za-z0-9_./-]+", secrets_file):
        raise HpcError("Secrets file must be an absolute remote path")
    snap = snapshot()
    inspection = remote.call("context")
    storage = choose_storage(inspection, alias)
    scratch = choose_scratch(inspection, alias)
    chosen_gpu = gpu_type or ("h200" if profile == "dual_gpu_141gb" else "h100")
    if chosen_gpu not in GPU_PARTITIONS:
        raise HpcError("Unsupported GPU type")
    partition = GPU_PARTITIONS[chosen_gpu]
    cfg = load_config(settings, method=method, hardware_profile=profile)
    state = remote.call("study", {"storage_root": storage, "study_id": cfg.study_id})
    if operation == "train" or (operation == "evaluate" and split in {"final_id", "external"}):
        pinned_commit = state["reports"]["freeze"].get("execution_commit")
        if pinned_commit:
            snap["commit"] = pinned_commit
    trainer = state["images"]["trainer"]
    worker = state["images"]["worker"]
    worker_setup_id = worker_setup_id or state["reports"]["worker"].get("worker_setup_id", "")
    image = image or trainer.get("path", "")
    image_sha256 = image_sha256 or trainer.get("sha256", "")
    if operation not in {"image-convert", "image-build", "worker-check", "worker-diagnose"} and (
        not image or not image_sha256
    ):
        raise HpcError("Build and convert the trainer image first: hpc images build --kind trainer")
    from selfplay_opsd.readiness import backend_identity, execution_identity

    backend_expected = backend_identity(cfg) | {"trainer_sha256": image_sha256 or None}
    runtime_identity = (
        None
        if operation in {"backend-check", "allocation-check", "worker-diagnose"}
        else execution_identity(cfg)
        | {
            "trainer_sha256": image_sha256 or None,
            "worker_sha256": worker.get("sha256"),
            "worker_setup_id": worker_setup_id,
        }
    )
    pilot_suffix = "-pilot" if chosen_gpu == "h100" else "-pilot-" + chosen_gpu
    checkpoint_cfg = cfg.model_copy(deep=True)
    if operation == "pilot":
        checkpoint_cfg.study_id += pilot_suffix
        checkpoint_cfg.iterations = iterations
    if checkpoint_iteration is not None and not 0 <= checkpoint_iteration <= cfg.iterations:
        raise HpcError("Checkpoint iteration is outside the study budget")
    if cfg.hardware.min_vram_gib > GPU_VRAM[chosen_gpu]:
        raise HpcError("GPU VRAM is below the selected hardware profile")
    selected_account = account or choose_account(inspection, partition)
    relative_settings = settings.resolve().relative_to(ROOT)
    if not re.fullmatch(r"[A-Za-z0-9_./-]+", str(relative_settings)):
        raise HpcError("Settings path has unsafe characters")
    study = cfg.study_id + (pilot_suffix if operation == "pilot" else "")
    logical = {
        "commit": snap["commit"],
        "manifest_sha256": snap["manifest_sha256"],
        "settings_sha256": sha256(settings),
        "scientific_hash": checkpoint_cfg.scientific_hash,
        "image_sha256": image_sha256,
        "worker_sha256": worker.get("sha256") if runtime_identity else None,
        "worker_setup_id": worker_setup_id if runtime_identity else None,
        "backend_identity": backend_expected if operation == "backend-check" else None,
        "operation": operation,
        "method": method,
        "profile": profile,
        "gpu_type": chosen_gpu,
        "study": study,
        "seed": cfg.seed,
        "iterations": iterations if operation == "pilot" else cfg.iterations,
        "base": base,
        "split": split if operation == "evaluate" else None,
        "checkpoint_iteration": checkpoint_iteration,
        "endpoint": endpoint if operation == "evaluate" else None,
    }
    identity = hashlib.sha256(json.dumps(logical, sort_keys=True).encode()).hexdigest()
    gpu_operation = operation in {
        "backend-check",
        "allocation-check",
        "pilot",
        "method-smoke",
        "train",
        "evaluate",
        "diagnose",
    }
    count = cfg.hardware.gpu_count if gpu_operation else 0
    if not gpu_operation:
        partition = "day"
        selected_account = account or choose_account(inspection, partition)
    payload = {
        **snap,
        "storage_root": storage,
        "scratch_root": scratch,
        "identity": identity,
        "runtime_identity": runtime_identity,
        "backend_identity": backend_expected,
        "common_config": cfg.scientific_settings(common=True),
        "signal_seconds": max(
            300,
            min(
                3600,
                int(state["reports"]["freeze"].get("pilot", {}).get("p90_iteration_seconds", 0))
                + 120,
            ),
        )
        if operation == "train"
        else 30
        if operation in {"allocation-check", "worker-diagnose"}
        else 300,
        "worker_image": worker.get("path", "") if runtime_identity else "",
        "worker_sha256": worker.get("sha256", "") if runtime_identity else "",
        "worker_config": cfg.sandbox.model_dump(mode="json"),
        "worker_setup_id": worker_setup_id,
        "supervisor_hash": supervisor_hash(),
        "pilot_suffix": pilot_suffix,
        "target_iterations": iterations if operation == "pilot" else cfg.iterations,
        "max_continuations": 1
        if operation == "pilot"
        else min(
            32,
            max(
                1,
                math.ceil(
                    state["reports"]["freeze"].get("pilot", {}).get("projection_200_hours", 0) / 22
                ),
            ),
        )
        if operation == "train"
        else 0,
        "estimated_main_hours_from_D_pilot": state["reports"]["freeze"]
        .get("pilot", {})
        .get("projection_200_hours"),
        "retry_setup": not gpu_operation or operation in {"allocation-check", "backend-check"},
        "readiness_budget": 24
        if operation in {"backend-check", "allocation-check", "pilot", "method-smoke"}
        or (operation == "evaluate" and base)
        else None,
        "run_id": identity[:24],
        "scientific_hash": checkpoint_cfg.scientific_hash,
        "manifests_hash": runtime_identity["manifest_hash"] if runtime_identity else None,
        "operation": operation,
        "method": method,
        "profile": profile,
        "profile_gpu_count": cfg.hardware.gpu_count,
        "gpu_count": count,
        "min_vram_gib": cfg.hardware.min_vram_gib,
        "gpu_type": chosen_gpu,
        "partition": partition,
        "account": selected_account,
        "cpus": 2
        if operation == "allocation-check"
        else cfg.hardware.cpu_workers
        if gpu_operation
        else 8
        if operation in {"image-convert", "runtime-check"}
        else 2,
        "memory_gib": 8
        if operation == "allocation-check"
        else cfg.hardware.host_ram_gib
        if gpu_operation
        else 64
        if operation == "image-convert"
        else 16
        if operation == "runtime-check"
        else 4
        if operation == "worker-check"
        else 8,
        "time": "00:05:00"
        if operation in {"allocation-check", "worker-diagnose"}
        else "01:00:00"
        if operation == "backend-check"
        else "06:00:00"
        if operation == "pilot"
        else "02:00:00"
        if (operation == "evaluate" and base) or operation == "method-smoke"
        else "00:15:00"
        if operation in {"worker-check", "freeze-study", "readiness-check"}
        else "01:00:00"
        if not gpu_operation
        else "23:00:00",
        "image": image,
        "image_sha256": image_sha256,
        "settings": str(relative_settings),
        "study_id": cfg.study_id,
        "seed": cfg.seed,
        "iterations": iterations,
        "resume": resume,
        "output": f"/runs/compatibility/backend-v2/{profile}"
        if operation == "backend-check"
        else f"/runs/compatibility/{profile}",
        "reference": "/runs/compatibility/backend-v2/single_gpu_80gb/reference_gradients.pt"
        if count > 1
        else "",
        "base": base,
        "split": split,
        "endpoint": endpoint,
        "checkpoint": f"/runs/{cfg.study_id}/{method}-seed{cfg.seed}/checkpoints/{checkpoint_iteration:06d}"
        if checkpoint_iteration is not None
        else "",
        "secrets_file": secrets_file or "",
    }
    if operation == "diagnose":
        payload["output"] = f"/runs/{cfg.study_id}/diagnostics/{checkpoint_iteration:06d}"
    return payload


@app.command("allocation-check")
def allocation_check(
    profile: str = "single_gpu_80gb",
    gpu_type: str = "h100",
    settings: Path = Path("configs/yale.yaml"),
    dry_run: bool = False,
    alias: str = ALIASES[0],
):
    """Prove physical GPU ownership and container UUIDs without loading a model."""
    if (
        alias != ALIASES[0]
        or gpu_type != "h100"
        or profile not in {"single_gpu_80gb", "dual_gpu_80gb"}
    ):
        raise HpcError("Allocation diagnostics require cpsc4880 and one or two H100s")
    remote = YaleSSH(alias)
    payload = run_payload(
        remote,
        alias,
        operation="allocation-check",
        method="D",
        profile=profile,
        gpu_type=gpu_type,
        settings=settings,
        account=None,
        image="",
        image_sha256="",
    )
    output(remote.call("preview" if dry_run else "submit", payload))


@app.command("worker-diagnose")
def worker_diagnose(
    settings: Path = Path("configs/yale.yaml"),
    dry_run: bool = False,
    alias: str = ALIASES[0],
):
    """Inspect compute-node worker prerequisites; execute no candidate code."""
    if alias != ALIASES[0]:
        raise HpcError("Worker diagnostics are authorized only on cpsc4880")
    remote = YaleSSH(alias)
    payload = run_payload(
        remote,
        alias,
        operation="worker-diagnose",
        method="D",
        profile="dual_gpu_80gb",
        gpu_type="h100",
        settings=settings,
        account=None,
        image="",
        image_sha256="",
    )
    output(remote.call("preview" if dry_run else "submit", payload))


@app.command("inspect")
def inspect(alias: Annotated[str, typer.Option()] = ALIASES[0]):
    """Read current access, associations, partitions, jobs, storage and software."""
    output(YaleSSH(alias).call("inspect"))


@app.command("login")
def login_command(alias: Annotated[str, typer.Option()] = ALIASES[0]):
    """Establish a normal MFA authenticated SSH master connection."""
    output(YaleSSH(alias).login())


@app.command("doctor")
def doctor(
    alias: Annotated[str, typer.Option()] = ALIASES[0], settings: Path = Path("configs/yale.yaml")
):
    """Report local prerequisites and lightweight remote state without allocation."""
    manifest_path = ROOT / "data/prepared/manifest.json"
    validation_path = ROOT / "data/prepared/validation_report.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    validation = json.loads(validation_path.read_text()) if validation_path.is_file() else {}
    local = {
        "git_clean": not bool(git("status", "--porcelain", "--untracked-files=all")),
        "prepared_data": bool(manifest.get("ready")),
        "data_validation": bool(
            validation.get("ready")
            and validation.get("executed")
            and validation.get("manifest_hash") == digest(manifest)
        ),
        "manifest_sha256": sha256(manifest_path) if manifest else None,
    }
    try:
        remote = YaleSSH(alias).call(
            "inspect",
            {"commit": git("rev-parse", "HEAD"), "manifest_sha256": local["manifest_sha256"]},
        )
        result: dict[str, Any] = {"local": local, "remote": remote, "blockers": []}
        if not local["git_clean"]:
            result["blockers"].append("Local Git commit is not clean for immutable synchronization")
        if not local["prepared_data"] or not local["data_validation"]:
            result["blockers"].append("Prepared and executed dataset validation missing")
        if not remote["storage"]["project"]:
            result["blockers"].append("Project storage not discovered")
        if not remote["storage"]["scratch"]:
            result["blockers"].append("Scratch storage not discovered")
        if remote["slurm"]["associations"]["exit_code"] != 0:
            result["blockers"].append("Slurm associations unavailable")
        if not remote["workspace"].get("current_snapshot"):
            result["blockers"].append("Current immutable Yale checkout and data not synchronized")
        from selfplay_opsd.readiness import backend_identity, execution_identity

        cfg = load_config(settings)
        target = YaleSSH(alias)
        context = {
            "storage_root": choose_storage(remote, alias),
            "study_id": cfg.study_id,
            "profile": cfg.hardware.profile,
            "profile_gpu_count": cfg.hardware.gpu_count,
            "supervisor_hash": supervisor_hash(),
            "seed": cfg.seed,
        }
        state = target.call("study", context)
        context["gpu_type"] = state["reports"]["freeze"].get("gpu_type", "h100")
        context["pilot_suffix"] = state["reports"]["freeze"].get("pilot_suffix", "-pilot")
        identity = execution_identity(cfg) | {
            "trainer_sha256": state["images"]["trainer"].get("sha256"),
            "worker_sha256": state["images"]["worker"].get("sha256"),
            "worker_setup_id": state["reports"]["worker"].get("worker_setup_id", ""),
        }
        state = target.call(
            "study",
            context
            | {
                "runtime_identity": identity,
                "backend_identity": backend_identity(cfg)
                | {"trainer_sha256": state["images"]["trainer"].get("sha256")},
            },
        )
        result["study"] = state
        result["eligible_to_submit_main"] = (
            state["eligible_to_submit_main"] and not result["blockers"]
        )
        result["blockers"] += [
            "Readiness gate missing or stale: " + name
            for name, ready in state["gates"].items()
            if not ready
        ]
        if not result["eligible_to_submit_main"] and not result["blockers"]:
            result["blockers"].append("Frozen readiness evidence missing or stale")
        result["next_action"] = (
            "Run hpc prepare-study; inspect its next action or failed-stage evidence"
        )
    except HpcError as exc:
        result = {
            "local": local,
            "remote": None,
            "blockers": [f"SSH inspection unavailable: {exc}"],
            "next_action": "Run hpc login in a terminal and complete Yale MFA",
        }
    output(result)


@app.command("resources")
def resources_command(
    profile: str = "dual_gpu_80gb",
    gpu_type: str | None = None,
    account: str | None = None,
    settings: Path = Path("configs/yale.yaml"),
    alias: str = ALIASES[0],
):
    """Check an intended GPU request against live Slurm metadata; allocate none."""
    if profile not in PROFILES or profile == "cpu_debug":
        raise HpcError("Choose a production hardware profile")
    cfg = load_config(settings, hardware_profile=profile)
    chosen_gpu = gpu_type or ("h200" if profile == "dual_gpu_141gb" else "h100")
    if chosen_gpu not in GPU_PARTITIONS or GPU_VRAM[chosen_gpu] < cfg.hardware.min_vram_gib:
        raise HpcError("GPU type cannot satisfy the profile")
    partition = GPU_PARTITIONS[chosen_gpu]
    remote = YaleSSH(alias)
    inspection = remote.call("inspect")
    selected_account = account or choose_account(inspection, partition)
    payload = {
        "account": selected_account,
        "partition": partition,
        "gpu_type": chosen_gpu,
        "gpu_count": cfg.hardware.gpu_count,
        "cpus": cfg.hardware.cpu_workers,
        "memory_gib": cfg.hardware.host_ram_gib,
        "time": "01:00:00",
    }
    output(remote.call("resources", payload))


@app.command("sync")
def sync_command(alias: Annotated[str, typer.Option()] = ALIASES[0]):
    """Transfer a clean Git commit and independently hashed prepared dataset."""
    output(sync(YaleSSH(alias)))


@app.command("submit")
def submit_command(
    operation: Annotated[
        str, typer.Option(help="GPU operation or CPU runtime-check/freeze-study/worker-check")
    ],
    image: Annotated[str, typer.Option(help="Absolute trainer SIF path on Yale")] = "",
    image_sha256: Annotated[str, typer.Option(help="SHA-256 of the Yale trainer SIF")] = "",
    method: str = "D",
    profile: str = "dual_gpu_80gb",
    gpu_type: str | None = None,
    account: str | None = None,
    settings: Path = Path("configs/yale.yaml"),
    iterations: int = 10,
    base: bool = False,
    split: str = "dev_monitor",
    checkpoint_iteration: int | None = None,
    endpoint: str = "primary",
    secrets_file: str | None = None,
    dry_run: bool = False,
    alias: str = ALIASES[0],
):
    """Submit one immutable run after live Slurm authorization checks."""
    remote = YaleSSH(alias)
    payload = run_payload(
        remote,
        alias,
        operation=operation,
        method=method,
        profile=profile,
        gpu_type=gpu_type,
        account=account,
        image=image,
        image_sha256=image_sha256,
        settings=settings,
        iterations=iterations,
        base=base,
        split=split,
        checkpoint_iteration=checkpoint_iteration,
        endpoint=endpoint,
        secrets_file=secrets_file,
    )
    output(remote.call("preview" if dry_run else "submit", payload))


@app.command("status")
def status_command(run_id: str, alias: str = ALIASES[0]):
    """Reconcile a logical run with live Slurm state."""
    remote = YaleSSH(alias)
    context = remote_context(remote, alias)
    output(remote.call("status", {"storage_root": context["storage_root"], "run_id": run_id}))


@app.command("logs")
def logs_command(run_id: str, lines: int = 100, alias: str = ALIASES[0]):
    """Read the bounded tail of the latest attempt's Slurm output."""
    remote = YaleSSH(alias)
    context = remote_context(remote, alias)
    output(
        remote.call(
            "logs", {"storage_root": context["storage_root"], "run_id": run_id, "lines": lines}
        )
    )


@app.command("fetch")
def fetch_command(run_id: str, full: bool = False, alias: str = ALIASES[0]):
    """Copy run records locally; include checkpoints only with --full."""
    remote = YaleSSH(alias)
    context = remote_context(remote, alias)
    report = remote.call("status", {"storage_root": context["storage_root"], "run_id": run_id})
    run_dir = Path(report["ledger"].get("run_dir", ""))
    allowed_root = (
        Path(context["storage_root"]) / f"selfplay-opsd-{context['inspection']['user']}/runs"
    )
    if not run_dir.is_relative_to(allowed_root) and not run_dir.is_relative_to(
        allowed_root.parent / "images"
    ):
        raise HpcError("Ledger run directory lies outside managed runs")
    target = ROOT / "runs/hpc" / alias / run_id
    remote.fetch(str(run_dir), target, full=full)
    output({"run_id": run_id, "local": str(target), "includes_checkpoints": full})


@app.command("resume")
def resume_command(run_id: str, continue_stopped: bool = False, alias: str = ALIASES[0]):
    """Use the persisted immutable specification; opt in to bounded clean continuation."""
    remote = YaleSSH(alias)
    context = remote_context(remote, alias)
    output(
        remote.call(
            "recover",
            {
                "storage_root": context["storage_root"],
                "run_id": run_id,
                "explicit_continue": continue_stopped,
            },
        )
    )


@app.command("watch")
def watch_command(
    run_id: str, interval: int = 60, continue_stopped: bool = False, alias: str = ALIASES[0]
):
    """Monitor persisted state; only authorized transient retries/continuations run."""
    if interval < 30:
        raise HpcError("Poll interval must be at least 30 seconds")
    remote = YaleSSH(alias)
    context = remote_context(remote, alias)
    payload = {
        "storage_root": context["storage_root"],
        "run_id": run_id,
        "explicit_continue": continue_stopped,
    }
    while True:
        result = remote.call("recover", payload)
        output(result)
        if result.get("state") not in ACTIVE and not result.get("submitted"):
            break
        time.sleep(interval)


def _register_readiness():
    from selfplay_opsd.hpc_readiness import register

    register(app)


_register_readiness()
