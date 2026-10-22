#!/usr/bin/env python3
"""Standard-library allocation entrypoint; no host training environment required."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import platform
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
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
    with tmp.open("w") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def checked_image(path: str, expected: str):
    if sha256(Path(path)) != expected:
        raise ValueError("Pinned image SHA-256 mismatch")


@contextmanager
def sampled_scratch_usage(path: Path):
    """Sample this build's logical file sizes, excluding symlinks and other jobs."""
    usage = {"sampled_peak_logical_bytes": 0}
    stopped = threading.Event()

    def sample():
        total = 0
        for item in path.rglob("*"):
            try:
                if not item.is_symlink() and item.is_file():
                    total += item.stat().st_size
            except FileNotFoundError:
                continue
        usage["sampled_peak_logical_bytes"] = max(usage["sampled_peak_logical_bytes"], total)

    def monitor():
        while not stopped.wait(10):
            sample()

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        yield usage
    finally:
        stopped.set()
        thread.join()
        sample()


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


def source_image(env: dict):
    """Build trusted committed sources, independently of candidate worker capabilities."""
    metadata = json.loads(base64.b64decode(env["SELFPLAY_IMAGE_METADATA"]))
    key = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    if key != env["SELFPLAY_BUILD_KEY"] or platform.machine() != "x86_64":
        raise ValueError("Source build identity or architecture mismatch")
    root = Path(env["SELFPLAY_PROJECT"])
    if (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        != metadata["source_commit"]
    ):
        raise ValueError("Source build checkout mismatch")
    checked_image(str(root / "uv.lock"), metadata["lock_sha256"])
    if metadata["kind"] == "trainer":
        checked_image(str(root / "docker/trainer.Dockerfile"), metadata["dockerfile_sha256"])
    definition = Path(env["SELFPLAY_BUILD_DEFINITION"])
    checked_image(str(definition), metadata["definition_sha256"])
    target, manifest = Path(env["SELFPLAY_IMAGE_TARGET"]), Path(env["SELFPLAY_IMAGE_MANIFEST"])
    if manifest.exists():
        existing = json.loads(manifest.read_text())
        if existing.get("build_key") == key:
            checked_image(existing["path"], existing["sha256"])
            return
    check = metadata["kind"] == "build-check"
    cache = Path(env["SELFPLAY_CACHE"]).parent / "image-builds"
    cache.mkdir(parents=True, exist_ok=True)
    minimum = (2 if check else 150) * 1024**3
    if shutil.disk_usage(cache).free < minimum:
        raise ValueError(f"Insufficient build scratch capacity: need {minimum // 1024**3} GiB")
    if shutil.disk_usage(target.parent).free < (1 if check else 30) * 1024**3:
        raise ValueError("Insufficient project capacity for completed SIF")
    started = time.monotonic()
    staged = target.with_name(f".{env['SLURM_JOB_ID']}.pending.sif")
    version = subprocess.check_output(["apptainer", "version"], text=True).strip()
    initial_free = shutil.disk_usage(cache).free
    try:
        with tempfile.TemporaryDirectory(prefix=f"{key[:12]}-", dir=cache) as temporary:
            build = Path(temporary)
            context = build / "context"
            context.mkdir()
            for relative, expected in metadata["context_files"].items():
                path = root / relative
                if (
                    Path(relative).is_absolute()
                    or ".." in Path(relative).parts
                    or path.is_symlink()
                    or not path.resolve().is_relative_to(root.resolve())
                ):
                    raise ValueError("Source context path escapes immutable checkout")
                checked_image(str(path), expected)
                destination = context / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
            recipe = build / "image.def"
            shutil.copyfile(definition, recipe)
            process_env = dict(env)
            # Build setup receives no trainer/worker/W&B credentials or implicit
            # host binds. No host privilege changes are made here.
            for name in list(process_env):
                if name.startswith(("APPTAINERENV_", "SINGULARITYENV_")) or name in {
                    "SELFPLAY_WORKER_TOKEN",
                    "WANDB_API_KEY",
                    "APPTAINER_BIND",
                    "APPTAINER_BINDPATH",
                }:
                    process_env.pop(name)
            process_env.update(
                {"APPTAINER_CACHEDIR": str(build / "cache"), "APPTAINER_TMPDIR": str(build / "tmp")}
            )
            (build / "tmp").mkdir()
            with sampled_scratch_usage(build) as usage:
                subprocess.run(
                    ["apptainer", "build", "--force", str(staged), str(recipe)],
                    cwd=context,
                    env=process_env,
                    check=True,
                    timeout=780 if check else 13800,
                )
            command = ["apptainer", "exec", "--containall", "--cleanenv", "--no-home", str(staged)]
            if check:
                subprocess.run(
                    ["apptainer", "run", "--containall", "--cleanenv", "--no-home", str(staged)],
                    env=process_env,
                    check=True,
                    timeout=60,
                )
            else:
                validation = (
                    "import importlib.metadata as m, torch, transformers, peft, verl, vllm; "
                    + "; ".join(
                        f"assert m.version({package!r}).split('+')[0]=={expected!r}"
                        for package, expected in (
                            ("torch", "2.8.0"),
                            ("transformers", "4.57.1"),
                            ("peft", "0.17.1"),
                            ("verl", "0.7.1"),
                            ("vllm", "0.11.0"),
                        )
                    )
                )
                subprocess.run(
                    command + ["/opt/selfplay/.venv/bin/python", "-c", validation],
                    env=process_env,
                    check=True,
                    timeout=180,
                )
                if (
                    subprocess.check_output(
                        command + ["uv", "--version"], env=process_env, text=True, timeout=30
                    ).strip()
                    != "uv 0.8.15"
                ):
                    raise ValueError("Trainer uv version mismatch")
                subprocess.run(
                    command + ["/opt/selfplay/.venv/bin/selfplay", "--help"],
                    env=process_env,
                    check=True,
                    timeout=180,
                )
            # Measure retained scratch use before deleting this build's context,
            # extracted image and temporary cache. Filesystem changes by other
            # jobs may affect the free-space delta; it is not a peak guarantee.
            scratch_bytes = sum(
                p.stat().st_size for p in build.rglob("*") if p.is_file() and not p.is_symlink()
            )
            free_delta = max(0, initial_free - shutil.disk_usage(cache).free)
            os.replace(staged, target)
    finally:
        staged.unlink(missing_ok=True)
    report = metadata | {
        "path": str(target),
        "sha256": sha256(target),
        "build_key": key,
        "architecture": platform.machine(),
        "apptainer_version": version,
        "elapsed_seconds": time.monotonic() - started,
        "scratch_retained_bytes": scratch_bytes,
        "scratch_free_delta_bytes": free_delta,
        **usage,
        "ready": True,
    }
    write(target.with_suffix(".json"), report)
    write(manifest, report)


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


def user_bus_environment(env: dict, directory: Path | None = None) -> dict:
    """Use an existing user-owned systemd bus; never create privileged services."""
    result = dict(env)
    directory = directory or Path(f"/run/user/{os.getuid()}")
    bus = directory / "bus"
    if (
        directory.is_dir()
        and directory.stat().st_uid == os.getuid()
        and bus.is_socket()
        and bus.stat().st_uid == os.getuid()
    ):
        result.setdefault("XDG_RUNTIME_DIR", str(directory))
        result.setdefault("DBUS_SESSION_BUS_ADDRESS", "unix:path=" + str(bus))
    return result


def execute(env: dict) -> int:
    operation = env["SELFPLAY_OPERATION"]
    if operation not in {
        "backend-check",
        "worker-check",
        "runtime-check",
        "train",
        "pilot",
        "method-smoke",
        "evaluate",
        "diagnose",
        "freeze-study",
        "readiness-check",
    }:
        raise ValueError("Unsupported allocation operation")
    uses_worker = operation in {
        "worker-check",
        "runtime-check",
        "train",
        "pilot",
        "method-smoke",
        "evaluate",
        "diagnose",
    }
    if uses_worker and not env.get("SELFPLAY_WORKER_SIF"):
        raise ValueError("Generated-code operations require the allocation worker")
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
    }
    # Do not inherit stale worker endpoints or credentials into trusted-only jobs.
    for key in ("SELFPLAY_WORKER_URL", "SELFPLAY_WORKER_TOKEN"):
        process_env.pop(key, None)
    if uses_worker:
        process_env["SELFPLAY_WORKER_TOKEN"] = secrets.token_urlsafe(32)
        process_env = user_bus_environment(process_env)
    runtime_directory = Path(f"/run/user/{os.getuid()}")
    write(
        readiness / f"node-environment-{attempt}.json",
        {
            "job_id": attempt,
            "host": platform.node(),
            "uid": os.getuid(),
            "user_runtime_directory_exists": runtime_directory.is_dir(),
            "user_bus_socket_exists": (runtime_directory / "bus").is_socket(),
            "dbus_session_bus_address_set": bool(process_env.get("DBUS_SESSION_BUS_ADDRESS")),
            "cgroups_v2": Path("/sys/fs/cgroup/cgroup.controllers").exists(),
        },
    )
    if operation == "worker-check":
        return subprocess.run(
            broker_args(process_env, ready, readiness / "worker.json") + ["--check-only"],
            env=process_env,
            check=False,
        ).returncode
    checked_image(env["SELFPLAY_TRAINER_SIF"], env["SELFPLAY_TRAINER_SHA256"])
    broker = None
    try:
        if uses_worker:
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
        if operation in {"train", "pilot", "method-smoke", "evaluate", "diagnose"}:
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
            str(runs) + ":/runs",
            "--bind",
            cache + ":/cache/huggingface",
            "--pwd",
            "/workspace",
            env["SELFPLAY_TRAINER_SIF"],
        ]
        if operation != "backend-check":
            command[2:2] = ["--bind", env["SELFPLAY_DATA"] + ":/data/prepared:ro"]
        if operation == "method-smoke":
            # Separate processes provide fresh model, RNG and optimizer initialization.
            status = 0
            method_index = args.index("--config") + 1
            for method in "ABC":
                if stop.exists():
                    return 75
                method_args = list(args)
                method_args[method_index] = method
                status = subprocess.run(
                    command + method_args, env=process_env, check=False
                ).returncode
                if status:
                    break
        else:
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
    if env["SELFPLAY_OPERATION"] == "image-build":
        source_image(env)
    elif env["SELFPLAY_OPERATION"] == "image-convert":
        convert(env)
    else:
        raise SystemExit(execute(env))


if __name__ == "__main__":
    main()
