#!/usr/bin/env python3
"""Standard-library allocation entrypoint; no host training environment required."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import platform
import secrets
import signal
import subprocess
import sys
import time
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, sort_keys=True) + "\n")
    os.replace(tmp, path)


def checked_image(path: str, expected: str):
    if sha256(Path(path)) != expected:
        raise ValueError("Pinned image SHA-256 mismatch")


def convert(env: dict):
    archive = Path(env["SELFPLAY_ARCHIVE"])
    target = Path(env["SELFPLAY_IMAGE_TARGET"])
    expected = env["SELFPLAY_ARCHIVE_SHA256"]
    checked_image(str(archive), expected)
    manifest = Path(env["SELFPLAY_IMAGE_MANIFEST"])
    if manifest.exists():
        report = json.loads(manifest.read_text())
        if report.get("archive_sha256") == expected:
            checked_image(report["path"], report["sha256"])
            return
    target.parent.mkdir(parents=True, exist_ok=True)
    # Space for extraction plus SIF compression; never use the home filesystem.
    import shutil

    cache = Path(env["SELFPLAY_CACHE"])
    cache.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(cache).free < archive.stat().st_size * 3:
        raise ValueError("Insufficient scratch space for archive extraction and SIF")
    staged = target.with_suffix(".pending.sif")
    try:
        subprocess.run(
            ["apptainer", "build", "--force", str(staged), f"docker-archive:{archive}"],
            check=True,
            env=env
            | {"APPTAINER_CACHEDIR": str(cache / "apptainer"), "APPTAINER_TMPDIR": str(cache)},
        )
        os.replace(staged, target)
    finally:
        staged.unlink(missing_ok=True)
    report = json.loads(base64.b64decode(env["SELFPLAY_IMAGE_METADATA"])) | {
        "path": str(target),
        "sha256": sha256(target),
        "archive_sha256": expected,
        "architecture": platform.machine(),
        "ready": True,
    }
    write(manifest, report)
    write(target.with_suffix(".json"), report)


def broker_args(env: dict, ready: Path, report: Path) -> list[str]:
    checked_image(env["SELFPLAY_WORKER_SIF"], env["SELFPLAY_WORKER_SHA256"])
    return [
        sys.executable,
        "-m",
        "selfplay_opsd.remote_worker",
        "--backend",
        "apptainer",
        "--runtime",
        "apptainer",
        "--image",
        env["SELFPLAY_WORKER_SIF"],
        "--config-json",
        base64.b64decode(env["SELFPLAY_WORKER_CONFIG"]).decode(),
        "--port",
        "0",
        "--ready-file",
        str(ready),
        "--report",
        str(report),
    ]


def execute(env: dict) -> int:
    operation = env["SELFPLAY_OPERATION"]
    runs = Path(env["SELFPLAY_RUNS"])
    study = env["SELFPLAY_STUDY"]
    study_root = env.get("SELFPLAY_MAIN_STUDY", study.removesuffix("-pilot"))
    pilot_suffix = env.get("SELFPLAY_PILOT_SUFFIX", "-pilot")
    readiness = runs / "readiness" / study_root
    readiness.mkdir(parents=True, exist_ok=True)
    attempt = env["SLURM_JOB_ID"]
    ready = readiness / f"worker-ready-{attempt}.json"
    broker_report = readiness / f"worker-{attempt}.json"
    ready.unlink(missing_ok=True)
    process_env = env | {
        "APPTAINER_CACHEDIR": str(Path(env["SELFPLAY_CACHE"]) / "apptainer"),
        "PYTHONPATH": str(Path.cwd() / "src"),
        "SELFPLAY_WORKER_TOKEN": secrets.token_urlsafe(32),
    }
    if operation == "worker-check":
        return subprocess.run(
            broker_args(process_env, ready, readiness / "worker.json") + ["--check-only"],
            env=process_env,
            check=False,
        ).returncode
    checked_image(env["SELFPLAY_TRAINER_SIF"], env["SELFPLAY_TRAINER_SHA256"])
    broker = None
    try:
        if env.get("SELFPLAY_WORKER_SIF"):
            broker = subprocess.Popen(
                broker_args(process_env, ready, broker_report), env=process_env
            )
            for _ in range(180):
                if ready.exists():
                    process_env["SELFPLAY_WORKER_URL"] = json.loads(ready.read_text())["url"]
                    break
                if broker.poll() is not None:
                    raise RuntimeError(
                        "Allocation worker isolation/startup failed; inspect worker report"
                    )
                time.sleep(0.5)
            else:
                raise RuntimeError("Allocation worker startup timed out")
        stop = (
            runs
            / study
            / f"{env['SELFPLAY_METHOD']}-seed{env['SELFPLAY_SEED']}"
            / "state"
            / f"stop_requested.{attempt}"
        )

        def request_stop(*_):
            stop.parent.mkdir(parents=True, exist_ok=True)
            stop.touch()

        for sig in (signal.SIGUSR1, signal.SIGTERM):
            signal.signal(sig, request_stop)
        overrides = {
            "SELFPLAY_MAIN_STUDY": study_root,
            "SELFPLAY_PILOT_SUFFIX": pilot_suffix,
            "SELFPLAY_GPU_TYPE": env.get("SELFPLAY_GPU_TYPE", "h100"),
            "SELFPLAY_DATA_ROOT": "/data/prepared",
            "SELFPLAY_OUTPUT_ROOT": "/runs",
            "PYTHONPATH": "/workspace/src",
            "HF_HOME": "/cache/huggingface",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "1",
            "SELFPLAY_IMAGE_DIGEST": env["SELFPLAY_TRAINER_SHA256"],
            "SELFPLAY_STOP_FILE": "/runs/" + str(stop.relative_to(runs)),
            "SELFPLAY_VALIDATION_REPORT": f"/runs/readiness/{study_root}/validation.json",
            "SELFPLAY_ALLOCATED_GPU_COUNT": env["SELFPLAY_GPU_COUNT"],
            "SELFPLAY_STOP_AFTER_ITERATION": env.get("SELFPLAY_STOP_AFTER_ITERATION", "0"),
        }
        for key in (
            "SELFPLAY_WORKER_TOKEN",
            "SELFPLAY_WORKER_URL",
            "SELFPLAY_WORKER_SHA256",
            "WANDB_API_KEY",
            "CUDA_VISIBLE_DEVICES",
            "SLURM_JOB_ID",
            "SLURM_CPUS_PER_TASK",
            "SLURM_JOB_GPUS",
            "SLURM_JOB_ACCOUNT",
        ):
            if key in process_env:
                overrides[key] = process_env[key]
        process_env.update({"APPTAINERENV_" + k: v for k, v in overrides.items()})
        gpu_count = int(env["SELFPLAY_GPU_COUNT"])
        args = (
            ["torchrun", "--standalone", "--nnodes=1", f"--nproc-per-node={gpu_count}", "-m"]
            if gpu_count
            else ["python", "-m"]
        )
        args += [
            "selfplay_opsd.cli",
            operation,
            "--settings",
            "/workspace/" + env["SELFPLAY_SETTINGS"],
        ]
        if operation in {"train", "pilot", "evaluate", "diagnose"}:
            args += ["--config", env["SELFPLAY_METHOD"], "--hardware", env["SELFPLAY_HARDWARE"]]
        if operation in {"train", "pilot"} and env.get("SELFPLAY_RESUME"):
            args += ["--resume", env["SELFPLAY_RESUME"]]
        if operation == "pilot":
            args += ["--iterations", env["SELFPLAY_ITERATIONS"]]
        if operation in {"backend-check", "diagnose"}:
            args += ["--output", env["SELFPLAY_OUTPUT"]]
        if operation in {"backend-check", "readiness-check"}:
            args += ["--hardware", env["SELFPLAY_HARDWARE"]]
        if operation == "backend-check" and env.get("SELFPLAY_REFERENCE"):
            args += ["--reference", env["SELFPLAY_REFERENCE"]]
        if operation == "evaluate":
            args += [
                "--split",
                env["SELFPLAY_SPLIT"],
                "--endpoint",
                env["SELFPLAY_ENDPOINT"],
                "--protocol",
                "frozen",
            ]
            args += (
                ["--base"]
                if env.get("SELFPLAY_BASE") == "1"
                else ["--checkpoint", env["SELFPLAY_CHECKPOINT"]]
            )
        if operation == "diagnose":
            args += ["--checkpoint", env["SELFPLAY_CHECKPOINT"]]
        if operation == "freeze-study":
            args += [
                "--pilot-report",
                f"/runs/{study_root}{pilot_suffix}/D-seed{env['SELFPLAY_SEED']}/pilot_report.json",
            ]
        cache = env["SELFPLAY_CACHE"]
        Path(cache).mkdir(parents=True, exist_ok=True)
        command = ["apptainer", "exec"] + (["--nv"] if gpu_count else [])
        command += [
            "--containall",
            "--cleanenv",
            "--bind",
            env["SELFPLAY_PROJECT"] + ":/workspace:ro",
            "--bind",
            env["SELFPLAY_DATA"] + ":/data/prepared:ro",
            "--bind",
            str(runs) + ":/runs",
            "--bind",
            cache + ":/cache/huggingface",
            "--pwd",
            "/workspace",
            env["SELFPLAY_TRAINER_SIF"],
        ]
        status = subprocess.run(command + args, env=process_env, check=False).returncode
        if status == 0 and operation == "freeze-study":
            command_report = [
                "python",
                "-m",
                "selfplay_opsd.cli",
                "readiness-check",
                "--settings",
                "/workspace/" + env["SELFPLAY_SETTINGS"],
                "--hardware",
                env["SELFPLAY_HARDWARE"],
            ]
            status = subprocess.run(
                command + command_report, env=process_env, check=False
            ).returncode
        return status
    finally:
        if broker:
            broker.terminate()
            try:
                broker.wait(timeout=15)
            except subprocess.TimeoutExpired:
                broker.kill()
                broker.wait()
        ready.unlink(missing_ok=True)


def main():
    env = dict(os.environ)
    if env["SELFPLAY_OPERATION"] == "image-convert":
        convert(env)
    else:
        raise SystemExit(execute(env))


if __name__ == "__main__":
    main()
