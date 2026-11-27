"""Evaluate original Task 21/22 weights with a separately frozen Task 23 protocol."""

from __future__ import annotations

import argparse
import fcntl
import os
import signal
import time
from pathlib import Path
from typing import cast

from task19_experiment import execution_identity

from selfplay_opsd.core import TaskSpec, digest, keyed_seed, read_json
from selfplay_opsd.pdb_verifier import PdbEnvironment, PdbOracle, verify_pdb
from selfplay_opsd.sandbox import SandboxBackend
from selfplay_opsd.storage import CheckpointManager, atomic_json, file_hash
from selfplay_opsd.task19_policy import SharedPolicy
from selfplay_opsd.task19_rollout import repair
from selfplay_opsd.task21 import PdbRepository, Task21Control
from selfplay_opsd.task22 import Task22Control

NAMESPACE = "repair-selfplay-2026-task23-eval-v1"


def select_tasks(
    repository: PdbRepository, task_ids: list[str]
) -> list[tuple[TaskSpec, PdbOracle]]:
    tasks = sorted(
        ((task, cast(PdbOracle, oracle)) for task, oracle in repository.tasks("primary_final")),
        key=lambda row: digest([row[1].provenance["family_id"], row[0].task_id]),
    )
    index = {task.task_id: (task, oracle) for task, oracle in tasks}
    if task_ids != [task.task_id for task, _ in tasks] or len(tasks) != 64:
        raise ValueError("Primary cohort is not the frozen matched set")
    return [index[task_id] for task_id in task_ids]


def stopped() -> bool:
    return Path(os.environ["SELFPLAY_STOP_FILE"]).exists() or time.time() + 300 >= float(
        os.getenv("SLURM_JOB_END_TIME", "inf")
    )


def run(output: Path, limit: int) -> None:
    manifest = read_json(output / "manifest.json")
    identity = manifest["identity"]
    kind = manifest["kind"]
    if digest(identity) != manifest["identity_sha256"]:
        raise ValueError("Evaluation identity verification failed")
    control_file = Path("/workspace") / manifest["control"]
    if file_hash(control_file) != identity["control_sha256"]:
        raise ValueError("Frozen source control differs")
    control = read_json(control_file)
    cfg_class = Task22Control if manifest["method"] in {"C", "D"} else Task21Control
    config = cfg_class.model_validate(control)
    original = identity["checkpoint_identity"]
    if config.seed != manifest["seed"] or config.science != original["science"]:
        raise ValueError("Source scientific control differs from the original checkpoint")
    worker = SandboxBackend(
        config.sandbox.model_copy(update={"remote_url": os.environ["SELFPLAY_WORKER_URL"]})
    )
    worker.preflight()
    if worker.identity() != original["worker"]:
        raise ValueError("The isolated verifier worker differs from the original")
    runtime = execution_identity()
    for key in (
        "image",
        "worker_image",
        "packages",
        "model",
        "model_revision",
        "tokenizer_revision",
        "device",
    ):
        if runtime[key] != original["execution"][key]:
            raise ValueError(f"Original runtime setting differs: {key}")
    repository = PdbRepository(Path(os.environ["SELFPLAY_DATA_ROOT"]))
    if repository.hash != original["data"]:
        raise ValueError("Original frozen PDB bundle differs")
    path = Path("/runs") / identity["checkpoint_relative"]
    if file_hash(path / "complete.json") != identity["checkpoint_sha256"]:
        raise ValueError("Verified checkpoint marker checksum differs")
    state, metadata = CheckpointManager.load(path)
    if (
        metadata["identity"] != original
        or metadata["snapshot"] != identity["snapshot"]
        or metadata["episodes"] != (0 if manifest["method"] == "Initial" else 512)
    ):
        raise ValueError("Verified checkpoint identity or exposure differs")
    if kind == "relaxed":
        cases = [("primary_final", select_tasks(repository, identity["tasks"]))]
        limits = config.episode.model_copy(update={"max_turn_generated_tokens": 6144})
    elif kind == "ood":
        candidate_path = Path("/runs") / identity["ood_relative"]
        if file_hash(candidate_path) != identity["ood_sha256"]:
            raise ValueError("Independent transfer manifest differs")
        accepted = read_json(candidate_path)
        cases = [
            (
                split,
                [
                    (TaskSpec.model_validate(row["task"]), PdbOracle.model_validate(row["oracle"]))
                    for row in accepted["cohorts"][split]
                ],
            )
            for split in ("ood_source", "ood_complexity")
            if accepted["cohorts"][split]
        ]
        limits = config.episode
    else:
        raise ValueError("Unsupported Task 23 evaluation")
    if limits.model_dump(mode="json") != identity["episode"]:
        raise ValueError("Evaluation episode budget differs")
    if not cases:
        raise ValueError("Evaluation has no verified tasks")
    for task, raw_oracle in repository.tasks("feasibility")[:1]:
        oracle = cast(PdbOracle, raw_oracle)
        if (
            not verify_pdb(worker, oracle.correct_source, task, oracle).passed
            or not verify_pdb(worker, task.initial_source, task, oracle).semantic_failures
        ):
            raise ValueError("Isolated worker failed the existing PDB canary")
    policy = SharedPolicy(config)
    policy.restore(state["policy"])
    if policy.version() != identity["snapshot"]:
        raise ValueError("Restored LoRA weights differ from the original snapshot")
    del state
    eval_config = config.model_copy(update={"episode": limits})
    all_rows = {}
    seen = 0
    for split, tasks in cases:
        collected = []
        for task, oracle in tasks:
            seed = keyed_seed(config.seed, "evaluation", split, task.task_id)
            task_identity = {"task": task.task_id, "seed": seed, "evaluation": digest(identity)}
            location = output / split / "tasks" / f"{task.task_id}.json"
            if location.is_file():
                row = read_json(location)
                if row["identity"] != task_identity:
                    raise ValueError("Cached result has a different frozen identity")
            else:
                if stopped() or (limit and seen >= limit):
                    return
                rollout = repair(
                    policy, worker, eval_config, task, oracle, seed, environment=PdbEnvironment
                )
                row = {
                    "identity": task_identity,
                    "task_id": task.task_id,
                    "parent_id": task.parent_id,
                    "family": oracle.provenance["family_id"],
                    **rollout.metrics(),
                    "verifier_status": rollout.verifier_status,
                }
                atomic_json(location, row)
            collected.append(row)
            seen += 1
        all_rows[split] = sorted(collected, key=lambda row: row["task_id"])
    atomic_json(output / "complete.json", {"identity": identity, "splits": all_rows})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", default=0, type=int)
    args = parser.parse_args()
    if (
        args.output.parent != Path("/runs") / NAMESPACE
        or os.environ.get("SLURM_JOB_ACCOUNT") != "cpsc4880"
        or not os.getenv("SLURM_JOB_ID")
        or not os.getenv("SELFPLAY_WORKER_TOKEN")
        or not args.output.is_dir()
    ):
        raise RuntimeError("Task 23 requires a guarded Yale allocation and isolated worker")
    os.umask(0o077)
    if args.output.name == "ood-validation":
        from prepare_task23_ood import validate

        validate(args.output, stopped)
        return
    if os.environ.get("SELFPLAY_ALLOCATED_GPU_COUNT") != "1":
        raise RuntimeError("Task 23 evaluation requires the same single H200 runtime")
    signal.signal(signal.SIGTERM, lambda *_: None)
    with (args.output / ".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not (args.output / "complete.json").is_file():
            run(args.output, args.limit)


if __name__ == "__main__":
    main()
