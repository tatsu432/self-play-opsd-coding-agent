"""Local image distribution and resumable preparation; main runs are never started here."""

from __future__ import annotations

import json
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

import typer

from selfplay_opsd.config import load_config
from selfplay_opsd.core import digest
from selfplay_opsd.worker_runtime import supervisor_hash


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
        sha256,
        snapshot,
        sync,
    )

    images = typer.Typer(help="Build AMD64 Docker archives and convert on Yale CPU nodes")
    app.add_typer(images, name="images")

    def payload_for(remote, operation, settings, **options):
        return run_payload(
            remote,
            remote.alias,
            operation=operation,
            method=options.pop("method", "D"),
            profile=options.pop("profile", "dual_gpu_80gb"),
            gpu_type=options.pop("gpu_type", "h100"),
            account=None,
            image="",
            image_sha256="",
            settings=settings,
            **options,
        )

    def study(remote, settings, gpu_type="h100"):
        cfg = load_config(settings)
        context = remote_context(remote, remote.alias)
        request = {
            "storage_root": context["storage_root"],
            "study_id": cfg.study_id,
            "profile": cfg.hardware.profile,
            "gpu_type": gpu_type,
            "pilot_suffix": "-pilot" if gpu_type == "h100" else "-pilot-h200",
            "supervisor_hash": supervisor_hash(),
            "seed": cfg.seed,
        }
        result = remote.call("study", request)
        from selfplay_opsd.readiness import execution_identity

        request["runtime_identity"] = execution_identity(cfg) | {
            "trainer_sha256": result["images"]["trainer"].get("sha256"),
            "worker_sha256": result["images"]["worker"].get("sha256"),
        }
        return remote.call("study", request), request

    @images.command("build")
    def build(
        kind: str = "worker", settings: Path = Path("configs/yale.yaml"), alias: str = ALIASES[0]
    ):
        """Build worker first; --kind all also attempts the much larger trainer."""
        if kind not in {"worker", "trainer", "all"}:
            raise HpcError("Image kind must be worker, trainer or all")
        remote = YaleSSH(alias)
        snap = snapshot()
        sync(remote)
        for current in ["worker", "trainer"] if kind == "all" else [kind]:
            directory = ROOT / "runs/hpc/images" / current
            directory.mkdir(parents=True, exist_ok=True)
            key = digest(
                {
                    "commit": snap["commit"],
                    "dockerfile": sha256(ROOT / f"docker/{current}.Dockerfile"),
                    "lock": sha256(ROOT / "uv.lock"),
                    "platform": "linux/amd64",
                }
            )[:24]
            archive = directory / f"{key}.tar"
            metadata_path = directory / f"{key}.json"
            if archive.exists() and metadata_path.exists():
                metadata = json.loads(metadata_path.read_text())
                if sha256(archive) != metadata["archive_sha256"]:
                    raise HpcError("Cached Docker archive checksum mismatch")
            else:
                required = (60 if current == "trainer" else 2) * 1024**3
                free = shutil.disk_usage(directory).free
                if free < required:
                    raise HpcError(
                        f"Insufficient local capacity for {current}: need {required // 1024**3} GiB free, have {free / 1024**3:.1f} GiB"
                    )
                pending = archive.with_suffix(".pending.tar")
                tag = f"selfplay-{current}:readiness-{key}"
                log_path = directory / f"{key}.build.log"
                typer.echo(f"Building {current} for linux/amd64; log: {log_path}")
                try:
                    with log_path.open("w") as log:
                        subprocess.run(
                            [
                                "docker",
                                "buildx",
                                "build",
                                "--platform",
                                "linux/amd64",
                                "--tag",
                                tag,
                                "--file",
                                str(ROOT / f"docker/{current}.Dockerfile"),
                                "--output",
                                f"type=docker,dest={pending}",
                                str(ROOT),
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
                    if (
                        image_config.get("architecture") != "amd64"
                        or image_config.get("os") != "linux"
                    ):
                        raise HpcError("Docker archive is not Linux AMD64")
                    import hashlib

                    metadata = {
                        "kind": current,
                        "platform": "linux/amd64",
                        "source_commit": snap["commit"],
                        "lock_sha256": sha256(ROOT / "uv.lock"),
                        "docker_image_id": "sha256:" + hashlib.sha256(config_bytes).hexdigest(),
                        "dockerfile_sha256": sha256(ROOT / f"docker/{current}.Dockerfile"),
                        "archive_sha256": sha256(pending),
                    }
                    pending.replace(archive)
                    metadata_path.write_text(json.dumps(metadata, sort_keys=True) + "\n")
                finally:
                    pending.unlink(missing_ok=True)
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
    ):
        """Advance readiness one allocation at a time; stop before main training."""
        if gpu_type not in {"h100", "h200"}:
            raise HpcError("Readiness supports H100, or H200 after verified H100 memory failure")
        if alias != ALIASES[0]:
            raise HpcError("Readiness authorization is restricted to cpsc4880")
        if not 0 < gpu_hours_limit <= 24:
            raise HpcError("Readiness cap must be positive and at most 24 GPU-hours")
        snapshot()
        remote = YaleSSH(alias)
        while True:
            state, request = study(remote, settings, gpu_type)
            if gpu_type == "h200" and not state["h100_memory_fit_failure"]:
                raise HpcError(
                    "H200 fallback requires a recorded managed H100 CUDA out-of-memory failure"
                )
            if not state["images"]["worker"].get("ready"):
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
                ("freeze", "freeze-study", {}),
            ]
            next_stage = next((entry for entry in stages if not state["gates"][entry[0]]), None)
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
            if operation != "worker-check" and not state["images"]["trainer"].get("ready"):
                output(
                    {
                        "eligible_to_submit_main": False,
                        "next_action": "hpc images build --kind trainer",
                        "state": state,
                    }
                )
                return
            payload = payload_for(
                remote,
                operation,
                settings,
                gpu_type=gpu_type,
                **{k: v for k, v in options.items() if k != "profile"},
            )
            if "profile" in options:
                payload = run_payload(
                    remote,
                    alias,
                    operation=operation,
                    method="D",
                    profile=options["profile"],
                    gpu_type="h100",
                    account=None,
                    image="",
                    image_sha256="",
                    settings=settings,
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
                if payload["gpu_count"] == 0 and latest in {
                    "PREEMPTED",
                    "NODE_FAIL",
                    "BOOT_FAIL",
                    "TIMEOUT",
                }:
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
                    return prepare_study(settings, gpu_hours_limit, watch, alias, "h200")
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
