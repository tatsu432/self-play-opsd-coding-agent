"""Frozen greedy repair evaluation and token-budget-admitted canonical anchors."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from selfplay_opsd.config import EpisodeConfig
from selfplay_opsd.core import EvaluationRecord, ExperimentError, Status, digest, keyed_seed
from selfplay_opsd.data import TaskRepository
from selfplay_opsd.environment import canonical_trajectory, check_admission, run_episode
from selfplay_opsd.policy import HFPolicy
from selfplay_opsd.protocol import PROMPT_HASH, VERSION
from selfplay_opsd.sandbox import SandboxBackend
from selfplay_opsd.storage import EventStore, atomic_json, file_hash, read_json


def verify_base_adapter(policy: HFPolicy) -> None:
    """Require zero LoRA B weights, including coordinates on other FSDP ranks."""
    import torch

    nonzero = torch.zeros((), dtype=torch.int64, device=policy.device)
    for name, parameter in policy.named_parameters("student"):
        if ".lora_B." in name:
            local = parameter.to_local() if hasattr(parameter, "to_local") else parameter
            nonzero += torch.count_nonzero(local.detach())
    if policy.distributed:
        torch.distributed.all_reduce(nonzero, op=torch.distributed.ReduceOp.SUM)
    if nonzero.item():
        raise ExperimentError("Base evaluation adapter is not a no-op")


def finalize_data(
    repository: TaskRepository, policy: HFPolicy, sandbox: SandboxBackend, limits: EpisodeConfig
) -> dict:
    if repository.manifest.get("ready"):
        raise ExperimentError("Dataset already finalized; immutable manifests cannot be edited")
    anchors = {}
    for split in ("reference", "dev_diagnostic"):
        rows = []
        for task, oracle in repository.tasks(split):
            try:
                rows.append(canonical_trajectory(policy, sandbox, task, oracle, limits))
            except (ValueError, ExperimentError) as exc:
                raise ExperimentError(
                    f"Canonical admission failed {split}/{task.task_id}: {exc}", Status.ORACLE
                ) from exc
        if not rows:
            raise ExperimentError(f"No canonical anchors in {split}", Status.ORACLE)
        path = repository.root / "private" / f"anchors-{split}.json"
        atomic_json(path, rows)
        anchors[split] = file_hash(path)
    admitted: dict[str, int] = defaultdict(int)
    for split in ("reference", "dev_monitor", "dev_diagnostic", "final_id", "external"):
        for task, oracle in repository.tasks(split):
            try:
                check_admission(policy, task, oracle, limits)
            except ValueError as exc:
                raise ExperimentError(
                    f"Admission failed {split}/{task.task_id}: {exc}", Status.ORACLE
                ) from exc
            admitted[split] += 1
    manifest = repository.manifest | {
        "ready": True,
        "anchor_hashes": anchors,
        "tokenizer_name": policy.config.model.name,
        "tokenizer_revision": policy.config.model.tokenizer_revision,
        "episode_limits_hash": digest(limits.model_dump()),
        "prompt_hash": PROMPT_HASH,
        "admitted_tasks": dict(admitted),
    }
    atomic_json(repository.root / "manifest.json", manifest)
    return manifest


def validate_data(
    repository: TaskRepository,
    sandbox: SandboxBackend,
    *,
    execute: bool = True,
    output: Path | None = None,
) -> dict:
    manifest, root = repository.manifest, repository.root
    if file_hash(root / "private/parents.json") != manifest["parents_hash"]:
        raise ExperimentError("Parent manifest modified", Status.ORACLE)
    for name, expected in manifest["task_hashes"].items():
        if file_hash(root / "private" / name) != expected:
            raise ExperimentError(f"Frozen task manifest modified: {name}", Status.ORACLE)
    for split, expected in manifest.get("anchor_hashes", {}).items():
        if file_hash(root / "private" / f"anchors-{split}.json") != expected:
            raise ExperimentError("Canonical trajectories modified", Status.ORACLE)
    roles: dict[str, str] = {}
    families: dict[str, str] = {}
    for item in read_json(root / "private/parents.json"):
        parent, split = item["parent"], item["split"]
        if parent["parent_id"] in roles:
            raise ExperimentError("Parent duplicated across records", Status.ORACLE)
        roles[parent["parent_id"]] = split
        family = parent["family_id"]
        if (
            family in families
            and families[family] != split
            and not split.startswith("excluded")
            and not families[family].startswith("excluded")
        ):
            raise ExperimentError("Family crosses partitions", Status.ORACLE)
        families[family] = split
    checked: dict[str, int] = defaultdict(int)
    for split in ("reference", "dev_monitor", "dev_diagnostic", "final_id", "external"):
        for task, oracle in repository.tasks(split):
            if task.split != split or roles[task.parent_id] != split:
                raise ExperimentError("Task parent/split mismatch", Status.ORACLE)
            # This is structural boundary verification, not unreliable substring
            # matching when public specs naturally share text with correct code.
            if set(task.student_files()) != {
                "README.md",
                "src/solution.py",
                "tests/test_public.py",
            }:
                raise ExperimentError("Student workspace export not allowlisted")
            if not oracle.hidden_cases:
                raise ExperimentError("Task lacks additional verifier coverage", Status.ORACLE)
            if execute:
                restored = sandbox.verify(
                    oracle.correct_source,
                    task.entry_point,
                    oracle.public_cases + oracle.hidden_cases,
                    task.verifier_seconds,
                )
                mutant = sandbox.verify(
                    task.initial_source,
                    task.entry_point,
                    oracle.public_cases + oracle.hidden_cases,
                    task.verifier_seconds,
                )
                if not restored.passed or mutant.semantic_failures < 1:
                    raise ExperimentError("Oracle/mutation verification changed", Status.ORACLE)
            checked[split] += 1
    report = {
        "ready": bool(manifest.get("ready")),
        "executed": execute,
        "checked_tasks": dict(checked),
        "manifest_hash": repository.hash,
        "split_audit": "pass",
        "family_audit": "pass",
        "workspace_boundary": "explicit_allowlist",
        "sandbox_image_digest": sandbox.identity() if execute else None,
        "execution_limits": sandbox.config.execution_limits(),
    }
    atomic_json(output or root / "validation_report.json", report)
    return report


def evaluate_tasks(
    policy: HFPolicy,
    sandbox: SandboxBackend,
    repository: TaskRepository,
    split: str,
    limits: EpisodeConfig,
    iteration: int,
    seed: int,
    store: EventStore | None,
    *,
    checkpoint: str | None = None,
    endpoint: str = "primary",
) -> list[EvaluationRecord]:
    if split not in {"dev_monitor", "dev_diagnostic", "final_id", "external"}:
        raise ValueError("Evaluation split is not permitted")
    tasks = repository.tasks(split)
    if not tasks:
        raise ExperimentError(f"No frozen evaluation tasks: {split}", Status.ORACLE)
    identity = checkpoint or policy.version("student")
    if store:
        store.emit(
            "evaluation_start",
            {
                "split": split,
                "checkpoint": identity,
                "endpoint": endpoint,
                "expected_task_ids": [t.task_id for t, _ in tasks],
                "manifest_hash": repository.hash,
            },
            committed=True,
            phase="evaluation",
        )
    records: list[EvaluationRecord] = []
    for task, oracle in tasks:
        episode_seed = keyed_seed(seed, "evaluation", split, task.task_id)
        try:
            episode = run_episode(policy, sandbox, task, oracle, limits, episode_seed, greedy=True)
        except ExperimentError as exc:
            record = EvaluationRecord(
                checkpoint=identity,
                endpoint=endpoint,
                iteration=iteration,
                split=split,
                task_id=task.task_id,
                parent_id=task.parent_id,
                success=None,
                status=exc.status,
                episode_path=None,
                metrics={"incomplete": True, "error": str(exc)},
            )
            if store:
                store.emit(
                    "evaluation", record.model_dump(mode="json"), committed=True, phase="evaluation"
                )
                store.emit(
                    "evaluation_incomplete",
                    {
                        "endpoint": endpoint,
                        "expected": len(tasks),
                        "completed": len(records),
                        "split": split,
                        "checkpoint": identity,
                    },
                    committed=True,
                    phase="evaluation",
                )
            raise
        path = (
            store.shard(
                iteration, f"eval-{split}-{task.task_id[:12]}", [episode.model_dump(mode="json")]
            )
            if store
            else None
        )
        record = EvaluationRecord(
            checkpoint=identity,
            endpoint=endpoint,
            iteration=iteration,
            split=split,
            task_id=task.task_id,
            parent_id=task.parent_id,
            success=bool(episode.reward),
            status=episode.status,
            episode_path=path,
            metrics={
                "tokens": sum(len(t.target_ids) for t in episode.turns),
                "turns": len(episode.turns),
                "test_calls": episode.test_calls,
                "seconds": episode.seconds,
                "public_pass": episode.public_pass,
                "hidden_pass": episode.hidden_pass,
                "final_source_hash": episode.final_source_hash,
                "termination": episode.termination,
                "verifier_status": episode.verifier_status,
                "runtime_failures": episode.runtime_failures,
                "semantic_failures": episode.semantic_failures,
                "protocol": VERSION,
                "task_creation_protocol": task.protocol_version,
                "seed": episode_seed,
                "manifest_hash": repository.hash,
            },
        )
        records.append(record)
        if store:
            store.emit(
                "evaluation", record.model_dump(mode="json"), committed=True, phase="evaluation"
            )
    if store:
        store.emit(
            "evaluation_complete",
            {
                "split": split,
                "checkpoint": identity,
                "endpoint": endpoint,
                "expected": len(tasks),
                "completed": len(records),
            },
            committed=True,
            phase="evaluation",
        )
    return records
