"""Run Task 21 using the accepted Yale H200, worker, and checkpoint controller."""

from __future__ import annotations

import argparse
import fcntl
import os
import random
import signal
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from task19_experiment import execution_identity, telemetry

from selfplay_opsd.core import digest, read_json
from selfplay_opsd.pdb_verifier import PdbOracle, verify_pdb
from selfplay_opsd.sandbox import SandboxBackend
from selfplay_opsd.storage import atomic_json, file_hash
from selfplay_opsd.task19 import ARMS
from selfplay_opsd.task19_policy import SharedPolicy
from selfplay_opsd.task19_train import SegmentStop
from selfplay_opsd.task21 import PdbRepository, Task21Control, Task21Study
from selfplay_opsd.task22 import Task22Control, Task22Study, qualify
from selfplay_opsd.task22_contract import (
    B_FALLBACK,
    BASE_STUDY,
    FINAL_STAGES,
    MATCHED_DATA,
    PREFIX,
    PRIMARY,
    QUALIFY,
    code_identity,
)


def run_sample_stage(study: Task21Study, stage: str, publish, should_stop) -> None:
    with (study.output / ".run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pointer = study.checkpoints.root / "latest.json"
        if pointer.is_file():
            study.restore(study.checkpoints.root / read_json(pointer)["path"])
        else:
            study.commit({}, None)
        marker = study.output / f"complete-{stage}.json"
        if marker.exists():
            if read_json(marker)["identity"] != study.identity:
                raise ValueError("Completed Task 21 stage identity differs")
            return
        result = study.sample(
            stage, 1 if stage == "feasibility" else 8, should_stop, block="difficulty"
        )
        by_task: dict[str, list[dict]] = defaultdict(list)
        for row in result["rows"]:
            by_task[row["task_id"]].append(row)
        summary: dict = {"count": result["count"], "verified": result["verified"]}
        if stage == "screen" and study.repository.tasks("feasibility_multi"):
            multi = study.sample("feasibility_multi", 1, should_stop, block="difficulty")
            summary["multi"] = {key: multi[key] for key in ("count", "verified")}
        for size in (4, 8) if stage == "screen" else ():
            groups = [sorted(rows, key=lambda row: row["slot"])[:size] for rows in by_task.values()]
            histogram = Counter(sum(row["success"] for row in group) for group in groups)
            seconds = sum(sum(row["timings"].values()) for group in groups for row in group)
            mixed = len(groups) - histogram[0] - histogram[size]
            summary[f"group_{size}"] = {
                "groups": len(groups),
                "mixed": mixed,
                "histogram": [histogram[n] for n in range(size + 1)],
                "episode_seconds": seconds,
                "mixed_groups_per_episode_hour": mixed * 3600 / seconds,
            }
        study.finish(marker, summary)
        publish(summary, study.iteration)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--operation",
        required=True,
        choices=(
            "feasibility",
            "screen",
            "solver",
            "calibrate",
            "shared-pilot",
            "main",
            "evaluate",
            PRIMARY,
            QUALIFY,
        ),
    )
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, default="control")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint-iteration", type=int)
    parser.add_argument("--evaluation-exposure", type=int, default=512)
    args = parser.parse_args()
    required = {
        "SELFPLAY_GPU_TYPE": "h200",
        "SLURM_JOB_ACCOUNT": "cpsc4880",
        "SELFPLAY_ALLOCATED_GPU_COUNT": "1",
    }
    if (
        any(os.environ.get(key) != value for key, value in required.items())
        or not os.getenv("SLURM_JOB_ID")
        or not os.getenv("SELFPLAY_WORKER_TOKEN")
    ):
        raise RuntimeError("Task 21 requires the Yale controller and isolated worker")
    os.umask(0o077)
    control = read_json(args.settings.with_suffix(".json"))
    study_class: type[Task21Study] = Task21Study
    cfg: Task21Control
    if control["study"].startswith(PREFIX):
        cfg = Task22Control.model_validate(control)
        study_class = Task22Study
        extension = (
            cfg.parent_marker_sha256
            and args.arm in {"control", "A", "B"}
            and (
                args.operation == "main"
                or args.operation in FINAL_STAGES
                and args.evaluation_exposure == cfg.endpoint
            )
        )
        baseline_fallback = (
            args.arm == "B"
            and args.operation in FINAL_STAGES
            and B_FALLBACK.get(args.evaluation_exposure) == args.checkpoint_iteration
            and args.evaluation_exposure in B_FALLBACK
            and cfg.alignment_gate_hash
        )
        if not (
            extension
            or baseline_fallback
            or (
                args.operation == QUALIFY
                and args.arm == "control"
                and not cfg.alignment_gate_hash
                and not cfg.parent_marker_sha256
            )
            or (
                args.operation in FINAL_STAGES | {"main"}
                and args.arm in {"C", "D"}
                and cfg.alignment_gate_hash
            )
        ):
            raise ValueError("Task 22 runner requires the qualified operation and arm")
    else:
        cfg = Task21Control.model_validate(control)
        if args.operation in {PRIMARY, QUALIFY}:
            raise ValueError("Task 22 operations require the new study identity")
    if cfg.study != args.output.parent.name:
        raise ValueError("Task 21 output is outside its study")
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
    proof = worker.preflight()
    repository = PdbRepository(Path(os.environ["SELFPLAY_DATA_ROOT"]))
    if study_class is Task22Study and repository.hash != MATCHED_DATA:
        raise ValueError("Task 22 PDB bundle does not match Task 21")
    if args.operation in {"feasibility", "screen"} and not repository.tasks(args.operation):
        raise ValueError("Task 21 stage lacks a verified partition")
    canaries = repository.tasks("feasibility")
    if not canaries:
        raise ValueError("Task 21 bundle lacks verifier canaries")
    handlers = set()
    for task, oracle in canaries:
        if not isinstance(oracle, PdbOracle):
            raise ValueError("Task 21 verifier oracle is missing")
        if oracle.handler in handlers:
            continue
        if (
            not verify_pdb(worker, oracle.correct_source, task, oracle).passed
            or not verify_pdb(worker, task.initial_source, task, oracle).semantic_failures
        ):
            raise ValueError("Yale worker disagrees with local PDB canary")
        handlers.add(oracle.handler)

    execution = execution_identity()
    root = Path(__file__).resolve().parents[1]
    files = [
        Path(__file__),
        *(
            root / "src/selfplay_opsd" / name
            for name in (
                "task21.py",
                "task20.py",
                "pdb_proposer.py",
                "pdb_verifier.py",
                "pdb_data.py",
                "environment.py",
                "sandbox.py",
            )
        ),
    ]
    execution["task21_code"] = digest({path.name: file_hash(path) for path in files})
    mode = (
        "fixed"
        if args.operation in {"feasibility", "screen", "solver"}
        or (args.operation in {"calibrate", "main", "evaluate"} and cfg.main_mode == "fixed")
        else "shared"
    )
    if study_class is Task22Study:
        execution["task22_code"] = code_identity(root)
    study = study_class(
        cfg, args.arm, SharedPolicy(cfg), worker, repository, args.output, execution, mode=mode
    )
    if isinstance(study, Task22Study):
        study.source_commit = os.environ["SELFPLAY_TASK19_COMMIT"]
        study.run_id = os.environ["SELFPLAY_TASK19_RUN_ID"]
    if args.resume is not None and not (
        isinstance(study, Task22Study) and args.operation in FINAL_STAGES
    ):
        pointer = read_json(study.checkpoints.root / "latest.json")
        if args.resume.resolve() != (study.checkpoints.root / pointer["path"]).resolve():
            raise ValueError("Task 21 resume is outside the verified latest checkpoint")

    allocation = os.environ["SLURM_JOB_ID"]
    health = args.output / "state/health.json"
    segment = {"operation": args.operation, "allocation": allocation, "identity": study.identity}
    atomic_json(args.output / "worker.json", proof)
    atomic_json(args.output / "segments" / f"{allocation}.json", segment)
    atomic_json(health, {"phase": "running", **segment})
    try:
        with telemetry(study) as publish:
            if args.operation in {"feasibility", "screen"}:
                run_sample_stage(study, args.operation, publish, should_stop)
            elif isinstance(study, Task22Study) and args.operation == QUALIFY:
                qualify(study, should_stop, publish)
            elif (
                isinstance(study, Task22Study)
                and args.operation == "main"
                and study.cfg.parent_marker_sha256
            ):
                if not (study.checkpoints.root / "latest.json").is_file():
                    with (study.output / ".run.lock").open("a+") as lock:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        marker = (
                            study.output.parent.parent
                            / BASE_STUDY
                            / f"{study.arm}-seed2026"
                            / "complete-main.json"
                        )
                        study.import_checkpoint(int(read_json(marker)["checkpoint"]), 512)
                study.run("main", publish, should_stop)
            elif isinstance(study, Task22Study) and args.operation in FINAL_STAGES:
                if args.checkpoint_iteration is None:
                    raise ValueError("Task 22 evaluation requires a checkpoint")
                with (study.output / ".run.lock").open("a+") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    study.evaluation(
                        args.operation,
                        args.checkpoint_iteration,
                        args.evaluation_exposure,
                        should_stop,
                    )
            else:
                study.run(
                    args.operation, publish, should_stop, checkpoint=args.checkpoint_iteration
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
