#!/usr/bin/env python3
"""One resource-aware launcher for local/AWS Docker and Yale Apptainer.

No provisioning, shell interpolation, credentials in arguments, or implicit runs.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
from pathlib import Path


def command(args):
    from selfplay_opsd.config import PROFILES, load_config

    count = PROFILES[args.hardware]["gpu_count"]
    root = Path.cwd().resolve()
    settings = Path(args.settings).resolve()
    try:
        relative = settings.relative_to(root)
    except ValueError:
        raise ValueError("Settings must be inside the mounted project directory") from None
    cfg = load_config(settings, method=args.method, hardware_profile=args.hardware)
    mounts = [(root, Path("/workspace"), False)]
    for path, readonly in ((cfg.data_root, True), (cfg.output_root, False)):
        if path.is_absolute():
            mounts.append((path, path, readonly))
        elif ".." in path.parts:
            raise ValueError(
                "Use an absolute mounted path instead of parent traversal in data/output settings"
            )
    cache = (
        Path(
            getattr(args, "cache", None)
            or os.getenv("SELFPLAY_CACHE_DIR", str(root / ".cache/huggingface"))
        )
        .expanduser()
        .resolve()
    )
    mounts.append((cache, Path("/cache/huggingface"), False))
    if not getattr(args, "dry_run", False):
        cache.mkdir(parents=True, exist_ok=True)
        cfg.output_root.mkdir(parents=True, exist_ok=True)
    visible = [
        x
        for x in os.environ.get("CUDA_VISIBLE_DEVICES", ",".join(map(str, range(count)))).split(",")
        if x
    ]
    if len(visible) < count:
        raise ValueError("Requested GPU count exceeds CUDA_VISIBLE_DEVICES")
    devices = ",".join(visible[:count])
    operation = args.operation
    inner = [
        "torchrun",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={count}",
        "-m",
        "selfplay_opsd.cli",
        operation,
        "--settings",
        str(Path("/workspace") / relative),
    ]
    inner += ["--config", args.method, "--hardware", args.hardware]
    if args.resume:
        resume = Path(args.resume).resolve().relative_to(root)
        inner += ["--resume", str(Path("/workspace") / resume)]
    if operation == "pilot":
        inner += ["--iterations", str(args.iterations)]
    environment = [
        "WANDB_API_KEY",
        "WANDB_ENTITY",
        "HF_TOKEN",
        "SELFPLAY_WORKER_TOKEN",
        "SELFPLAY_IMAGE_DIGEST",
        "SELFPLAY_STOP_FILE",
        "HF_HOME",
        "CUDA_VISIBLE_DEVICES",
        "SLURM_CPUS_PER_TASK",
        "SLURM_JOB_ID",
        "SELFPLAY_ALLOCATED_GPU_COUNT",
        "PYTHONPATH",
    ]
    if args.environment == "yale":
        if not os.getenv("SLURM_JOB_ID"):
            raise ValueError("Yale GPU execution requires a Slurm allocation")
        allocated = os.getenv("CUDA_VISIBLE_DEVICES", "").split(",")
        if len([x for x in allocated if x]) < count:
            raise ValueError("Requested profile exceeds CUDA_VISIBLE_DEVICES allocation")
        for key in environment:
            if key in os.environ:
                os.environ[f"APPTAINERENV_{key}"] = os.environ[key]
        return [
            "apptainer",
            "exec",
            "--nv",
            "--containall",
            "--cleanenv",
            *sum(
                (
                    ["--bind", f"{source}:{destination}" + (":ro" if ro else "")]
                    for source, destination, ro in mounts
                ),
                [],
            ),
            "--pwd",
            "/workspace",
            args.image,
            *inner,
        ]
    return [
        "docker",
        "run",
        "--rm",
        "--init",
        "--network=host",
        "--gpus",
        json.dumps(f"device={devices}"),
        "--shm-size=16g",
        *sum(
            (
                [
                    "--mount",
                    f"type=bind,src={source},dst={destination}" + (",readonly" if ro else ""),
                ]
                for source, destination, ro in mounts
            ),
            [],
        ),
        "--workdir",
        "/workspace",
        *sum((["--env", key] for key in environment if key in os.environ), []),
        "--entrypoint",
        "torchrun",
        args.image,
        *inner[1:],
    ]


def main():
    from selfplay_opsd.config import PROFILES

    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", choices=["local", "aws", "yale"], required=True)
    parser.add_argument(
        "--hardware", choices=[p for p in PROFILES if p != "cpu_debug"], default="dual_gpu_80gb"
    )
    parser.add_argument("--settings", default="configs/main.yaml")
    parser.add_argument("--method", choices=list("ABCD"), default="B")
    parser.add_argument("--image", required=True)
    parser.add_argument("--operation", choices=["train", "pilot"], default="train")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--resume")
    parser.add_argument("--cache")
    parser.add_argument("--allocated-gpus", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    from selfplay_opsd.config import load_config

    cfg = load_config(Path(args.settings), method=args.method, hardware_profile=args.hardware)
    if args.operation == "pilot":
        cfg.study_id += "-pilot"
    stop_file = cfg.run_dir / "state/stop_requested"
    os.environ["SELFPLAY_STOP_FILE"] = str(Path("/workspace") / stop_file)
    os.environ["HF_HOME"] = "/cache/huggingface"
    os.environ["PYTHONPATH"] = "/workspace/src"
    allocated = (
        args.allocated_gpus
        if args.allocated_gpus is not None
        else len(os.getenv("CUDA_VISIBLE_DEVICES", "").split(","))
        if os.getenv("SLURM_JOB_ID")
        else cfg.hardware.gpu_count
    )
    os.environ["SELFPLAY_ALLOCATED_GPU_COUNT"] = str(allocated)
    os.environ["SELFPLAY_IMAGE_DIGEST"] = args.image
    result = command(args)
    # Redact actual credential values in Apptainer arguments. Execution still
    # inherits secrets via environment, never writes them into a plan/log.
    if args.dry_run:
        print(
            json.dumps(
                [
                    part.split("=", 1)[0] + "=<redacted>"
                    if any(
                        part.startswith(k + "=")
                        for k in ("WANDB_API_KEY", "HF_TOKEN", "SELFPLAY_WORKER_TOKEN")
                    )
                    else part
                    for part in result
                ],
                indent=2,
            )
        )
    else:
        if stop_file.exists():
            stop_file.unlink()  # A new explicit launch clears the preceding stop request.

        def stop(signum, frame):
            stop_file.parent.mkdir(parents=True, exist_ok=True)
            stop_file.touch()

        for signum in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, stop)
        process = subprocess.Popen(result)
        status = process.wait()
        if status:
            raise subprocess.CalledProcessError(status, result)


if __name__ == "__main__":
    main()
