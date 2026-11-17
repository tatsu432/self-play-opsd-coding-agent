"""Run the Task 19 study inside the controller's accepted Yale allocation."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import random
import signal
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from selfplay_opsd.core import digest, read_json
from selfplay_opsd.data import TaskRepository
from selfplay_opsd.sandbox import SandboxBackend
from selfplay_opsd.storage import atomic_json, file_hash, scientific_code_hash
from selfplay_opsd.task19 import ARMS, MODEL, PILOT, REVISION, Task19Control
from selfplay_opsd.task19_policy import SharedPolicy
from selfplay_opsd.task19_qualification import qualify, verify_evidence
from selfplay_opsd.task19_train import SegmentStop, Study

ROOT = Path(__file__).resolve().parents[1]


def execution_identity() -> dict:
    package = ROOT / "src/selfplay_opsd"
    files = [
        package / name
        for name in (
            "task19.py",
            "task19_policy.py",
            "task19_protocol.py",
            "task19_qualification.py",
            "task19_rollout.py",
            "task19_train.py",
        )
    ] + [Path(__file__)]
    files += [ROOT / "docker/task19/uv.lock", ROOT / "docker/task19.Dockerfile"]
    return {
        "code": digest({str(path.relative_to(ROOT)): file_hash(path) for path in files}),
        "legacy_dependencies": scientific_code_hash(),
        "image": os.environ["SELFPLAY_IMAGE_DIGEST"],
        "worker_image": os.environ["SELFPLAY_WORKER_SHA256"],
        "packages": {
            name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft")
        },
        "model": MODEL,
        "model_revision": REVISION,
        "tokenizer_revision": REVISION,
        "device": torch.cuda.get_device_name(0),
    }


@contextmanager
def telemetry(study: Study):
    record = {"mode": "online", "sdk_calls": 0, "status": "not_started", "run_url": None}
    status = study.output / "telemetry.json"
    name = f"qwen35-shared-drgrpo-{study.arm}-s{study.cfg.seed}"
    run = None
    try:
        import wandb

        if os.getenv("WANDB_API_KEY"):
            run = wandb.init(
                project="self-play-opsd-coding-agent",
                entity=os.getenv("WANDB_ENTITY"),
                group=study.cfg.study,
                name=name,
                id=digest(study.identity)[:24],
                resume="allow",
                dir=str(study.output),
                settings=wandb.Settings(init_timeout=30),
                config={"method": study.arm, "seed": study.cfg.seed, "science": study.cfg.science},
            )
            record.update(status="initialized", run_url=run.url)
        else:
            record.update(status="missing_credentials", mode="local")
    except Exception as exc:
        record.update(status=type(exc).__name__, mode="local")
    atomic_json(status, record)

    def publish(metrics, iteration):
        if run is None:
            return

        def flatten(values, prefix=""):
            out = {}
            for name, value in values.items():
                key = prefix + str(name)
                if isinstance(value, dict):
                    out.update(flatten(value, key + "/"))
                elif isinstance(value, (int, float, bool)):
                    out[key] = value
            return out

        try:
            run.log(flatten(metrics) | {"train/iteration": iteration})
            record["sdk_calls"] += 1
        except Exception as exc:
            record["status"] = type(exc).__name__
        atomic_json(status, record)

    try:
        yield publish
    finally:
        if run is not None:
            try:
                run.finish()
            except Exception as exc:
                record["status"] = type(exc).__name__
            atomic_json(status, record)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--operation",
        choices=("qualify", "qualify-uncached", "base", "pilot", "train"),
        required=True,
    )
    p.add_argument("--arm", choices=ARMS, default="control")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--settings", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    args = p.parse_args()
    if (
        not os.getenv("SLURM_JOB_ID")
        or not os.getenv("SELFPLAY_WORKER_TOKEN")
        or os.getenv("SELFPLAY_GPU_TYPE") != "h200"
        or os.getenv("SLURM_JOB_ACCOUNT") != "cpsc4880"
    ):
        raise RuntimeError(
            "Task 19 requires a controller-managed Yale cpsc4880 H200 and isolated worker"
        )
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=True)
    initial = Task19Control.read(args.settings.parent / "task19-control.json")
    root = args.output.parent
    qualify_dir = root / "qualification"
    uncached = root / "qualification-uncached"
    if args.operation == "qualify":
        cfg = initial
    elif args.operation == "qualify-uncached":
        cfg = Task19Control.read(qualify_dir / "uncached-control.json")
        if cfg.cached_rollout:
            raise ValueError("Fallback qualification must use uncached execution")
    else:
        qualified = [
            path for path in (qualify_dir, uncached) if (path / "calibrated-control.json").is_file()
        ]
        if len(qualified) != 1:
            raise ValueError("Expected one successfully qualified execution")
        qualify_dir = qualified[0]
        control = (
            root / "main-control.json"
            if args.operation == "train"
            else qualify_dir / "calibrated-control.json"
        )
        cfg = Task19Control.read(control)
    if cfg.study != root.name:
        raise ValueError("Unexpected Task 19 study path")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    worker_url = os.environ["SELFPLAY_WORKER_URL"]
    sandbox = SandboxBackend(cfg.sandbox.model_copy(update={"remote_url": worker_url}))
    result = sandbox.preflight()
    if not result["ready"]:
        raise RuntimeError("Allocated worker did not pass isolation preflight")
    repository = TaskRepository(Path(os.environ["SELFPLAY_DATA_ROOT"]))
    policy = SharedPolicy(cfg)
    study = Study(cfg, args.arm, policy, sandbox, repository, args.output, execution_identity())
    if args.operation not in {"qualify", "qualify-uncached"}:
        verify_evidence(cfg, qualify_dir, study.execution, repository.hash, study.initial)
    if args.operation == "train":
        pilot = read_json(args.output / f"complete-{PILOT}.json")
        checkpoint = args.output / "checkpoints" / pilot["checkpoint"] / "complete.json"
        if (
            pilot["identity"] != study.identity
            or file_hash(checkpoint) != pilot["checkpoint_sha256"]
        ):
            raise ValueError("Main requires the matching completed pilot checkpoint")
    if args.resume is not None:
        latest = read_json(study.checkpoints.root / "latest.json")
        if args.resume.resolve() != (study.checkpoints.root / latest["path"]).resolve():
            raise ValueError("Controller resume differs from the latest committed checkpoint")
    segment = {
        "operation": args.operation,
        "allocation": os.environ["SLURM_JOB_ID"],
        "controller_run": os.environ.get("SELFPLAY_TASK19_RUN_ID"),
        "commit": os.environ.get("SELFPLAY_TASK19_COMMIT"),
        "identity": study.identity,
    }
    atomic_json(args.output / "segments" / f"{segment['allocation']}.json", segment)
    health = args.output / "state/health.json"
    atomic_json(health, {"phase": "running", **segment})
    stopped = False

    def signal_stop(*_):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, signal_stop)
    signal.signal(signal.SIGUSR1, signal_stop)
    stop_path = Path(os.environ["SELFPLAY_STOP_FILE"])
    end_time = float(os.getenv("SLURM_JOB_END_TIME", "inf"))

    def should_stop():
        return stopped or stop_path.exists() or time.time() + 300 >= end_time

    try:
        if args.operation.startswith("qualify"):
            qualify(study)
        elif args.operation == "base":
            pointer = study.checkpoints.root / "latest.json"
            if pointer.is_file():
                study.restore(study.checkpoints.root / read_json(pointer)["path"])
            else:
                study.commit({}, None)
            for split in ("dev_monitor", "final_id", "external"):
                study.evaluate(split, should_stop)
            atomic_json(
                args.output / "complete-base.json",
                {"identity": study.identity, "snapshot": study.snapshot},
            )
        else:
            with telemetry(study) as publish:
                study.train(
                    pilot=args.operation == "pilot", publish=publish, should_stop=should_stop
                )
    except SegmentStop:
        atomic_json(health, {"phase": "stopped", "episodes": study.episodes, **segment})
    except BaseException as exc:
        atomic_json(health, {"phase": "failed", "error_type": type(exc).__name__, **segment})
        raise
    else:
        atomic_json(health, {"phase": "completed", "episodes": study.episodes, **segment})


if __name__ == "__main__":
    main()
