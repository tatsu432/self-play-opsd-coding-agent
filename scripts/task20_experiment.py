"""Run a bounded Task 20 stage under the existing Yale Slurm controller."""

from __future__ import annotations

import argparse
import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
from task19_experiment import execution_identity, telemetry

from selfplay_opsd.core import digest, read_json
from selfplay_opsd.data import TaskRepository
from selfplay_opsd.sandbox import SandboxBackend
from selfplay_opsd.storage import atomic_json, file_hash
from selfplay_opsd.task19 import ARMS
from selfplay_opsd.task19_policy import SharedPolicy
from selfplay_opsd.task19_train import SegmentStop
from selfplay_opsd.task20 import Mode, Task20Control, Task20Study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--operation",
        required=True,
        choices=("screen", "solver", "calibrate", "shared-pilot", "main", "evaluate"),
    )
    parser.add_argument("--arm", default="control", choices=ARMS)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint-iteration", type=int)
    args = parser.parse_args()
    required_env = {
        "SELFPLAY_GPU_TYPE": "h200",
        "SLURM_JOB_ACCOUNT": "cpsc4880",
        "SELFPLAY_ALLOCATED_GPU_COUNT": "1",
    }
    if (
        any(os.environ.get(key) != value for key, value in required_env.items())
        or not os.getenv("SELFPLAY_WORKER_TOKEN")
        or not os.getenv("SLURM_JOB_ID")
    ):
        raise RuntimeError("Task 20 requires a controller-managed H200 and an isolated Yale worker")
    os.umask(0o077)
    cfg = Task20Control.read(args.settings.with_suffix(".json"))
    if cfg.study != args.output.parent.name:
        raise ValueError("Output does not belong to the Task 20 control")
    if args.operation in {"main", "evaluate"} and cfg.endpoint is None:
        raise ValueError("Main and final evaluation require a frozen Task 20 control")
    args.output.mkdir(parents=True, exist_ok=True)

    stopped = False

    def request_stop(*_) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGUSR1, request_stop)
    stop_file = Path(os.environ["SELFPLAY_STOP_FILE"])
    end = float(os.getenv("SLURM_JOB_END_TIME", "inf"))

    def should_stop() -> bool:
        return stopped or stop_file.exists() or time.time() + 300 >= end

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    worker = SandboxBackend(
        cfg.sandbox.model_copy(update={"remote_url": os.environ["SELFPLAY_WORKER_URL"]})
    )
    worker.preflight()
    repository = TaskRepository(Path(os.environ["SELFPLAY_DATA_ROOT"]))
    if repository.manifest.get("benchmark") != "task20-evalplus-harder-v1":
        raise ValueError("Expected the verified Task 20 fallback manifest")
    canaries = repository.tasks("screen")[:2]
    if len(canaries) != 2:
        raise ValueError("Missing training-only verifier canaries")
    for task, oracle in canaries:
        cases = oracle.public_cases + oracle.hidden_cases
        if (
            not worker.verify(
                oracle.correct_source, task.entry_point, cases, task.verifier_seconds
            ).passed
            or not worker.verify(
                task.initial_source, task.entry_point, cases, task.verifier_seconds
            ).semantic_failures
        ):
            raise ValueError("Yale worker disagrees with frozen Task 20 task verification")

    execution = execution_identity()
    files = (
        Path(__file__),
        Path(__file__).parents[1] / "src/selfplay_opsd/task20.py",
    )
    execution["task20_code"] = digest({path.name: file_hash(path) for path in files})
    mode: Mode = "fixed" if args.operation in {"screen", "solver"} else "shared"
    if args.operation == "shared-pilot":
        mode = cfg.shared_pilot_mode
    study = Task20Study(
        cfg,
        args.arm,
        SharedPolicy(cfg),
        worker,
        repository,
        args.output,
        execution,
        mode=mode,
    )
    if args.resume is not None:
        pointer = read_json(study.checkpoints.root / "latest.json")
        if args.resume.resolve() != (study.checkpoints.root / pointer["path"]).resolve():
            raise ValueError("Controller resume differs from the latest committed checkpoint")

    allocation = os.environ["SLURM_JOB_ID"]
    health = args.output / "state/health.json"
    segment = {"operation": args.operation, "allocation": allocation, "identity": study.identity}
    atomic_json(args.output / "segments" / f"{allocation}.json", segment)
    atomic_json(health, {"phase": "running", **segment})
    try:
        with telemetry(study) as log:
            started = time.monotonic()

            def publish(metrics: dict, iteration: int) -> None:
                log(metrics | {"allocation_elapsed_seconds": time.monotonic() - started}, iteration)

            study.run(args.operation, publish, should_stop, checkpoint=args.checkpoint_iteration)
    except SegmentStop:
        atomic_json(health, {"phase": "stopped", "episodes": study.episodes, **segment})
    except BaseException as exc:
        atomic_json(health, {"phase": "failed", "error_type": type(exc).__name__, **segment})
        raise
    else:
        atomic_json(health, {"phase": "completed", "episodes": study.episodes, **segment})


if __name__ == "__main__":
    main()
