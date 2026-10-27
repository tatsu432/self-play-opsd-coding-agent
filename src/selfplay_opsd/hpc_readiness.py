"""Local image distribution and resumable preparation; main runs are never started here."""

from __future__ import annotations

import json
import shutil
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Annotated

import typer

from selfplay_opsd.config import load_config
from selfplay_opsd.core import digest
from selfplay_opsd.worker_runtime import supervisor_hash


def build_archive(
    kind: str, source_root: Path, directory: Path, commit: str, builder: str | None = None
) -> tuple[Path, dict]:
    """Build/export locally; remote synchronization and submission belong to the caller."""
    from selfplay_opsd.hpc import HpcError, sha256
    from selfplay_opsd.storage import atomic_json

    directory.mkdir(parents=True, exist_ok=True)
    dockerfile_hash = sha256(source_root / f"docker/{kind}.Dockerfile")
    lock_hash = sha256(source_root / "uv.lock")
    expected_metadata = {
        "kind": kind,
        "platform": "linux/amd64",
        "source_commit": commit,
        "lock_sha256": lock_hash,
        "dockerfile_sha256": dockerfile_hash,
    }
    key = digest(
        {
            "commit": commit,
            "dockerfile": dockerfile_hash,
            "lock": lock_hash,
            "platform": "linux/amd64",
        }
    )[:24]
    archive = directory / f"{key}.tar"
    metadata_path = directory / f"{key}.json"
    if archive.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if sha256(archive) != metadata["archive_sha256"]:
            raise HpcError("Cached Docker archive checksum mismatch")
        if any(metadata.get(key) != value for key, value in expected_metadata.items()):
            raise HpcError("Cached Docker archive provenance mismatch")
    else:
        required = (60 if kind == "trainer" else 2) * 1024**3
        free = shutil.disk_usage(directory).free
        if free < required:
            raise HpcError(
                f"Insufficient local capacity for {kind}: need {required // 1024**3} GiB free, have {free / 1024**3:.1f} GiB"
            )
        pending = archive.with_suffix(".pending.tar")
        tag = f"selfplay-{kind}:readiness-{key}"
        log_path = directory / f"{key}.build.log"
        typer.echo(f"Building {kind} for linux/amd64; log: {log_path}")
        try:
            with log_path.open("w") as log:
                subprocess.run(
                    [
                        "docker",
                        "buildx",
                        "build",
                        *(["--builder", builder] if builder else []),
                        "--platform",
                        "linux/amd64",
                        "--tag",
                        tag,
                        "--file",
                        str(source_root / f"docker/{kind}.Dockerfile"),
                        "--output",
                        f"type=docker,dest={pending}",
                        str(source_root),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=14400,
                )
            with tarfile.open(pending) as tar:
                manifest_file = tar.extractfile("manifest.json")
                assert manifest_file is not None
                manifest = json.load(manifest_file)
                config_file = tar.extractfile(manifest[0]["Config"])
                assert config_file is not None
                config_bytes = config_file.read()
                image_config = json.loads(config_bytes)
            if image_config.get("architecture") != "amd64" or image_config.get("os") != "linux":
                raise HpcError("Docker archive is not Linux AMD64")
            import hashlib

            metadata = {
                **expected_metadata,
                "docker_image_id": "sha256:" + hashlib.sha256(config_bytes).hexdigest(),
                "archive_sha256": sha256(pending),
            }
            pending.replace(archive)
            atomic_json(metadata_path, metadata)
        finally:
            pending.unlink(missing_ok=True)
    return archive, metadata


def register(app: typer.Typer):
    # Imported after the controller's primitives are defined, avoiding transport duplication.
    from selfplay_opsd.hpc import (
        ACTIVE,
        ALIASES,
        ROOT,
        HpcError,
        YaleSSH,
        output,
        remote_context,
        run_payload,
        snapshot,
        sync,
    )

    images = typer.Typer(help="Build pinned images on Yale CPU nodes or import Docker archives")
    app.add_typer(images, name="images")

    def payload_for(remote, operation, settings, **options):
        return run_payload(
            remote,
            remote.alias,
            operation=operation,
            method=options.pop("method", "D"),
            profile=options.pop("profile", load_config(settings).hardware.profile),
            gpu_type=options.pop("gpu_type", "h100"),
            account=None,
            image="",
            image_sha256="",
            settings=settings,
            **options,
        )

    def study(remote, settings, gpu_type="h100", *, backend_only=False):
        cfg = load_config(settings)
        context = remote_context(remote, remote.alias)
        request = {
            "storage_root": context["storage_root"],
            "study_id": cfg.study_id,
            "profile": cfg.hardware.profile,
            "profile_gpu_count": cfg.hardware.gpu_count,
            "gpu_type": gpu_type,
            "pilot_suffix": "-pilot" if gpu_type == "h100" else "-pilot-h200",
            "supervisor_hash": supervisor_hash(),
            "seed": cfg.seed,
        }
        result = remote.call("study", request)
        from selfplay_opsd.readiness import backend_identity, execution_identity

        trainer_digest = result["images"]["trainer"].get("sha256")
        request["backend_identity"] = backend_identity(cfg) | {
            "trainer_sha256": trainer_digest,
        }
        if not backend_only:
            request["runtime_identity"] = execution_identity(cfg) | {
                "trainer_sha256": trainer_digest,
                "worker_sha256": result["images"]["worker"].get("sha256"),
                "worker_setup_id": result["reports"]["worker"].get("worker_setup_id", ""),
            }
        return remote.call("study", request), request

    @images.command("build")
    def build(
        kind: str = "worker",
        settings: Path = Path("configs/yale.yaml"),
        alias: str = ALIASES[0],
        build_root: Annotated[Path | None, typer.Option(envvar="SELFPLAY_IMAGE_BUILD_ROOT")] = None,
        builder: Annotated[str | None, typer.Option(help="Existing Buildx builder name")] = None,
        build_mode: str = "docker",
        check_only: bool = False,
    ):
        """Build worker first; --kind all also attempts the much larger trainer."""
        if kind not in {"worker", "trainer", "all"}:
            raise HpcError("Image kind must be worker, trainer or all")
        if build_mode not in {"docker", "yale"}:
            raise HpcError("Build mode must be docker or yale")
        if (check_only or build_mode == "yale") and kind != "trainer":
            raise HpcError("Yale source builds/checks apply to the trainer only")
        if check_only and build_mode != "yale":
            raise HpcError("--check-only requires --build-mode yale")
        if build_mode == "yale" and (build_root or builder or alias != ALIASES[0]):
            raise HpcError("Yale builds use cpsc4880 scratch; local build options do not apply")
        remote = YaleSSH(alias)
        snap = sync(remote)
        if build_mode == "yale":
            from selfplay_opsd.hpc_images import (
                BUILD_CHECK_DOCKERFILE,
                base_environment,
                source_build,
            )

            with tempfile.TemporaryDirectory() as temporary:
                dockerfile = ROOT / "docker/trainer.Dockerfile"
                if check_only:
                    dockerfile = Path(temporary) / "check.Dockerfile"
                    dockerfile.write_text(BUILD_CHECK_DOCKERFILE)
                inputs = source_build(
                    ROOT,
                    dockerfile,
                    snap["commit"],
                    base_env=base_environment(dockerfile),
                    check=check_only,
                )
            payload = payload_for(remote, "image-build", settings)
            payload.update(inputs)
            payload.update(
                {
                    "cpus": 2 if check_only else 8,
                    "memory_gib": 4 if check_only else 32,
                    "time": "00:15:00" if check_only else "04:00:00",
                }
            )
            payload.update(remote.call("prepare_source_image", payload))
            payload["identity"] = digest([payload["identity"], inputs["build_key"]])
            payload["run_id"] = payload["identity"][:24]
            output(remote.call("submit", payload))
            return
        for current in ["worker", "trainer"] if kind == "all" else [kind]:
            directory = (build_root or ROOT / "runs/hpc/images").expanduser().resolve() / current
            if directory.is_relative_to(ROOT) and not directory.is_relative_to(ROOT / "runs"):
                raise HpcError("Build root inside the checkout must be under ignored runs/")
            if "," in str(directory):
                raise HpcError("Build root cannot contain a comma (Buildx output syntax)")
            archive, metadata = build_archive(current, ROOT, directory, snap["commit"], builder)
            payload = payload_for(remote, "image-convert", settings)
            location = remote.call(
                "prepare_image",
                payload
                | {
                    "kind": current,
                    "archive_sha256": metadata["archive_sha256"],
                    "archive_bytes": archive.stat().st_size,
                },
            )
            remote.rsync(archive, location["archive"])
            payload.update(
                location
                | {
                    "archive_sha256": metadata["archive_sha256"],
                    "image_metadata": metadata,
                    "time": "04:00:00",
                }
            )
            payload["identity"] = digest([payload["identity"], metadata])
            payload["run_id"] = payload["identity"][:24]
            output(remote.call("submit", payload))

    @app.command("prepare-study")
    def prepare_study(
        settings: Path = Path("configs/yale.yaml"),
        gpu_hours_limit: float = 24,
        watch: bool = False,
        alias: str = ALIASES[0],
        gpu_type: str = "h100",
        stage: str = "all",
        worker_setup_id: str = "",
        diagnostic_partition: str | None = None,
        exclude_node: list[str] | None = None,
    ):
        """Advance readiness one allocation at a time; stop before main training."""
        if gpu_type not in {"h100", "h200"}:
            raise HpcError("Readiness supports authorized H100 or H200 resources")
        if stage not in {"all", "backend"}:
            raise HpcError("Stage must be all or backend")
        if diagnostic_partition not in {None, "scavenge_gpu"}:
            raise HpcError("Diagnostic partition override must be scavenge_gpu")
        if alias != ALIASES[0]:
            raise HpcError("Readiness authorization is restricted to cpsc4880")
        if not 0 < gpu_hours_limit <= 24:
            raise HpcError("Readiness cap must be positive and at most 24 GPU-hours")
        snapshot()
        remote = YaleSSH(alias)
        while True:
            state, request = study(remote, settings, gpu_type, backend_only=stage == "backend")
            if stage == "all" and not state["images"]["worker"].get("ready"):
                output(
                    {
                        "eligible_to_submit_main": False,
                        "next_action": "hpc images build --kind worker",
                        "state": state,
                    }
                )
                return
            stages: list[tuple[str, str, dict]] = [
                ("worker", "worker-check", {}),
                ("runtime", "runtime-check", {}),
                ("single", "backend-check", {"profile": "single_gpu_80gb"}),
                ("target", "backend-check", {}),
                ("base", "evaluate", {"base": True}),
                ("pilot", "pilot", {}),
                ("methods", "method-smoke", {}),
                ("freeze", "freeze-study", {}),
            ]
            cfg = load_config(settings)
            if cfg.hardware.gpu_count == 1:
                # The target backend proof is already a single-process reference.
                stages = [entry for entry in stages if entry[0] != "single"]
            if stage == "backend":
                stages = [entry for entry in stages if entry[0] in {"single", "target"}]
            if "allocation_checks" in state:
                stages = [
                    ("allocation_single", "allocation-check", {"profile": "single_gpu_80gb"}),
                    ("allocation_target", "allocation-check", {}),
                ] + stages
                if cfg.hardware.gpu_count == 1:
                    stages = [entry for entry in stages if entry[0] != "allocation_single"]
                for key, ready in state["allocation_checks"].items():
                    state["gates"]["allocation_" + key] = ready
            next_stage = next((entry for entry in stages if not state["gates"][entry[0]]), None)
            if stage == "backend" and next_stage is None:
                output(
                    {
                        "backend_ready": True,
                        "eligible_to_submit_main": state["eligible_to_submit_main"],
                        "gates": state["gates"],
                        "budget": state["budget"],
                    }
                )
                return
            if next_stage is None and not state["eligible_to_submit_main"]:
                next_stage = ("readiness", "readiness-check", {})
            if next_stage is None:
                plans = []
                for method in "ABCD":
                    payload = payload_for(
                        remote, "train", settings, method=method, gpu_type=gpu_type
                    )
                    plans.append(remote.call("preview", payload))
                output(
                    {"eligible_to_submit_main": True, "main_runs": plans, "budget": state["budget"]}
                )
                return
            name, operation, options = next_stage
            if operation in {"allocation-check", "backend-check"} and diagnostic_partition:
                options["partition"] = diagnostic_partition
            if operation in {"allocation-check", "backend-check"} and exclude_node:
                options["exclude_nodes"] = exclude_node
            if operation == "worker-check":
                options["worker_setup_id"] = worker_setup_id
                prior = state["reports"].get("worker", {})
                detail = json.dumps(prior.get("details", {})).lower()
                if (
                    not prior.get("ready")
                    and ("dbus_session_bus_address" in detail or "delegat" in detail)
                    and worker_setup_id == prior.get("worker_setup_id", "")
                    and prior.get("supervisor_hash") == supervisor_hash()
                ):
                    output(
                        {
                            "stage": name,
                            "state": "failed",
                            "failure_category": "site_setup",
                            "eligible_to_submit_main": False,
                            "next_action": "Run hpc worker-diagnose; obtain a supported setup in YCRC ticket 79403, then use --worker-setup-id with that documented change",
                            "evidence": prior,
                        }
                    )
                    return
            if operation != "worker-check" and not state["images"]["trainer"].get("ready"):
                output(
                    {
                        "eligible_to_submit_main": False,
                        "next_action": "hpc images build --kind trainer --build-mode yale",
                        "state": state,
                    }
                )
                return
            payload = payload_for(
                remote,
                operation,
                settings,
                gpu_type=gpu_type,
                **options,
            )
            payload["readiness_budget"] = gpu_hours_limit
            if operation == "pilot":
                payload.update(
                    {"stop_after_iteration": 2, "time": "02:00:00", "continuation_time": "04:00:00"}
                )
            report = remote.call("status", {**request, "run_id": payload["run_id"]})
            if report["ledger"].get("pending_intent"):
                output({"stage": name, "blocked": "Unreconciled submission", "status": report})
                return
            if report["attempts"]:
                latest = report["attempts"][-1]["scheduler"]["state"]
                if latest in ACTIVE:
                    output({"stage": name, "state": latest, "run_id": payload["run_id"]})
                    if not watch:
                        return
                    time.sleep(30)
                    continue
                state = remote.call("study", request)
                if (
                    (
                        payload["gpu_count"] == 0
                        or operation in {"allocation-check", "backend-check"}
                    )
                    and latest
                    in {
                        "PREEMPTED",
                        "NODE_FAIL",
                        "BOOT_FAIL",
                        "TIMEOUT",
                    }
                    and (payload["gpu_count"] == 0 or latest != "TIMEOUT")
                ):
                    result = remote.call("submit", payload)
                    output(result)
                    if result.get("submitted") and watch:
                        time.sleep(30)
                        continue
                    return
                if operation == "pilot" and latest in {
                    "PREEMPTED",
                    "NODE_FAIL",
                    "BOOT_FAIL",
                    "COMPLETED",
                    "TIMEOUT",
                }:
                    result = remote.call(
                        "recover",
                        {**request, "run_id": payload["run_id"], "explicit_continue": True},
                    )
                    output(result)
                    if result.get("submitted") and watch:
                        time.sleep(30)
                        continue
                    if result.get("submitted"):
                        return
                if gpu_type == "h100" and state["h100_memory_fit_failure"]:
                    return prepare_study(
                        settings,
                        gpu_hours_limit,
                        watch,
                        alias,
                        "h200",
                        stage,
                        worker_setup_id,
                        diagnostic_partition,
                        exclude_node,
                    )
                output(
                    {
                        "stage": name,
                        "blocked": "Allocation ended without valid stage evidence; inspect logs",
                        "status": report,
                        "evidence": state["reports"].get(name),
                    }
                )
                return
            result = remote.call("submit", payload)
            output({"stage": name, **result})
            if watch and result.get("reason") == "slot_busy" and result.get("state") in ACTIVE:
                time.sleep(30)
                continue
            if not watch or not result.get("submitted"):
                return
            time.sleep(30)

    @app.command("fetch-study")
    def fetch_study(study_id: str, full: bool = False, alias: str = ALIASES[0]):
        """Fetch study-shaped outputs, pilot, compatibility and readiness reports."""
        remote = YaleSSH(alias)
        context = remote_context(remote, alias)
        state = remote.call(
            "study", {"storage_root": context["storage_root"], "study_id": study_id}
        )
        target = ROOT / "runs/hpc" / alias / "studies" / study_id
        for name, source in state["artifact_directories"].items():
            destination = target / name
            remote.fetch(source, destination, full=full)
        output(
            {"local": str(target), "study": str(target / study_id), "includes_checkpoints": full}
        )
