"""Prepare and freeze private harder EvalPlus tasks through the isolated worker."""

from __future__ import annotations

import argparse
import ast
import os
import random
from collections import Counter
from pathlib import Path

from selfplay_opsd.config import load_config
from selfplay_opsd.core import OracleSpec, ParentProgram, TaskSpec, digest, keyed_seed, read_json
from selfplay_opsd.data import TaskRepository, build_task, render_public_tests
from selfplay_opsd.mutations import Mutation, fixed_mutation, make_patch, validate_mutation
from selfplay_opsd.sandbox import SandboxBackend
from selfplay_opsd.storage import atomic_json, file_hash

PARTITIONS = {
    "screen": ("train", 32),
    "solver_probe": ("dev_diagnostic", 8),
    "dev_monitor": ("dev_monitor", 50),
    "final_id": ("final_id", 100),
    "external": ("external", 24),
}


def compound(source: str, seed: int) -> Mutation:
    """Preserve the original source except for two independent literal/boundary edits."""
    lines = source.encode().splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    candidates = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Constant) or type(node.value) is not int:
            continue
        assert node.end_lineno is not None and node.end_col_offset is not None
        if node.lineno == node.end_lineno:
            start = offsets[node.lineno - 1] + node.col_offset
            end = offsets[node.lineno - 1] + node.end_col_offset
            candidates.append((node.lineno, start, end, node.value))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    if not candidates:
        raise ValueError("No integer mutation sites")
    first = candidates[0]
    second = next((item for item in candidates if item[0] != first[0]), None)
    if second is None:
        raise ValueError("No second line with an independent mutation site")
    changed = source.encode()
    for _, start, end, value in sorted((first, second), reverse=True):
        changed = changed[:start] + str(value + rng.choice((-1, 1))).encode() + changed[end:]
    result = validate_mutation(source, make_patch(source, changed.decode()))
    if result.changed_nodes < 2 or result.changed_lines < 4:
        raise ValueError("Mutant does not contain two admitted edits")
    return result


def freeze(
    output: Path, source: TaskRepository, partitions: dict[str, list], metadata: dict
) -> dict:
    if (output / "manifest.json").exists():
        raise ValueError("Task 20 prepared data is immutable")
    parent_records = []
    roles = {"dev_diagnostic": "solver_probe"}
    seen_roles: dict[str, str] = {}
    source_records = read_json(source.root / "private" / "parents.json")
    for record in source_records:
        role = roles.get(record["split"], record["split"])
        if role not in {
            "train",
            "solver_probe",
            "dev_monitor",
            "final_id",
            "external",
            "reference",
        }:
            continue
        parent = ParentProgram.model_validate(record["parent"])
        prior = seen_roles.setdefault(parent.family_id, role)
        if prior != role:
            raise ValueError("A parent family crosses a frozen partition")
        parent_records.append(record | {"split": role})
    task_roles = {"screen": "train", "solver_train": "train"}
    parent_roles = {row["parent"]["parent_id"]: row["split"] for row in parent_records}
    for split, tasks in partitions.items():
        if not tasks:
            raise ValueError(f"No tasks prepared for {split}")
        ids = set()
        for row in tasks:
            task = TaskSpec.model_validate(row["task"])
            oracle = OracleSpec.model_validate(row["oracle"])
            if (
                task.split != split
                or oracle.task_id != task.task_id
                or parent_roles[task.parent_id] != task_roles.get(split, split)
                or task.task_id in ids
            ):
                raise ValueError(f"Task or oracle crosses a frozen partition: {split}")
            ids.add(task.task_id)
    private = output / "private"
    private.mkdir(mode=0o700, parents=True, exist_ok=True)
    atomic_json(private / "parents.json", parent_records)
    task_hashes = {}
    for split, tasks in sorted(partitions.items()):
        path = private / f"tasks-{split}.json"
        atomic_json(path, tasks)
        task_hashes[path.name] = file_hash(path)
    anchor = source.root / "private" / "anchors-reference.json"
    anchor_hashes = {}
    if anchor.is_file():
        source.anchors()
        atomic_json(private / anchor.name, read_json(anchor))
        anchor_hashes[anchor.name] = file_hash(private / anchor.name)
    result = {
        "schema_version": 1,
        "benchmark": "task20-evalplus-harder-v1",
        "source_data": source.manifest.get("source_data", source.hash),
        "parents_hash": file_hash(private / "parents.json"),
        "task_hashes": task_hashes,
        "anchor_hashes": anchor_hashes,
        "protocol": source.manifest["protocol"],
        "ready": True,
        **metadata,
    }
    atomic_json(
        output / "validation_report.json",
        {
            "ready": True,
            "executed": True,
            "manifest_hash": digest(result),
            "checked_tasks": {split: len(tasks) for split, tasks in partitions.items()},
            "sandbox_image_digest": metadata["preparation_worker"],
            "split_audit": "pass",
            "family_audit": "pass",
        },
    )
    atomic_json(output / "manifest.json", result)
    return result


def validate_prepared(repository: TaskRepository, sandbox: SandboxBackend) -> dict:
    """Recheck the frozen split and workspace, and replay one canary per partition."""
    image = sandbox.preflight()["image_digest"]
    if image != repository.manifest["preparation_worker"]:
        raise ValueError("Use the verified preparation worker for local validation")
    seen: dict[str, str] = {}
    role_by_parent = {}
    for row in read_json(repository.root / "private/parents.json"):
        parent = ParentProgram.model_validate(row["parent"])
        role = row["split"]
        if parent.parent_id in role_by_parent or seen.setdefault(parent.family_id, role) != role:
            raise ValueError("Duplicate parent or cross-partition parent family")
        role_by_parent[parent.parent_id] = role
    checked = {}
    for name in repository.manifest["task_hashes"]:
        split = name.removeprefix("tasks-").removesuffix(".json")
        tasks = repository.tasks(split)
        if not tasks:
            raise ValueError(f"Empty frozen partition {split}")
        role = "train" if split in {"screen", "solver_train"} else split
        for task, oracle in tasks:
            if (
                task.split != split
                or role_by_parent[task.parent_id] != role
                or oracle.task_id != task.task_id
                or not oracle.hidden_cases
                or task.public_tests != render_public_tests(task.entry_point, oracle.public_cases)
            ):
                raise ValueError(f"Frozen partition or public export differs: {split}")
        task, oracle = tasks[0]
        cases = oracle.public_cases + oracle.hidden_cases
        if (
            not sandbox.verify(
                oracle.correct_source, task.entry_point, cases, task.verifier_seconds
            ).passed
            or not sandbox.verify(
                task.initial_source, task.entry_point, cases, task.verifier_seconds
            ).semantic_failures
        ):
            raise ValueError(f"Verifier canary disagrees in {split}")
        checked[split] = len(tasks)
    report = {
        "ready": True,
        "executed": True,
        "manifest_hash": repository.hash,
        "checked_tasks": checked,
        "sandbox_image_digest": image,
        "split_audit": "pass",
        "family_audit": "pass",
        "workspace_boundary": "explicit public-case export",
        "replayed_canaries": len(checked),
    }
    atomic_json(repository.root / "validation_report.json", report)
    return report


def prepare(source: TaskRepository, output: Path, sandbox: SandboxBackend, seed: int) -> dict:
    attestation = sandbox.preflight()
    partitions: dict[str, list] = {}
    counts = {}
    for split, (parent_split, target) in PARTITIONS.items():
        parents = sorted(
            source.parents(parent_split), key=lambda p: keyed_seed(seed, split, p.parent_id)
        )
        found = []
        families: Counter[str] = Counter()
        used = set()
        # First take one per family, then fill remaining capacity with different
        # parents and task variants within the same partition.
        families_seen: set[str] = set()
        unique, remainder = [], []
        for parent in parents:
            (remainder if parent.family_id in families_seen else unique).append(parent)
            families_seen.add(parent.family_id)
        for rotation in range(2):
            for parent in unique + remainder:
                if len(found) >= target:
                    break
                public, hidden = source.parent_cases(parent.parent_id)
                for attempt in range(8):
                    trial = keyed_seed(seed, split, parent.parent_id, rotation, attempt)
                    try:
                        generator = compound if attempt < 4 else fixed_mutation
                        if generator is fixed_mutation and split == "external":
                            break
                        mutant = generator(parent.source, trial)
                        key = (parent.parent_id, mutant.dedup_key)
                        if key in used:
                            continue
                        task, oracle = build_task(
                            parent, mutant, public, hidden, split, trial, sandbox
                        )
                    except ValueError:
                        continue
                    used.add(key)
                    label = "two_line_literal" if generator is compound else mutant.family
                    provenance = oracle.provenance | {
                        "mutation_family": label,
                        "changed_ast_nodes": mutant.changed_nodes,
                        "changed_lines": mutant.changed_lines,
                    }
                    found.append(
                        {
                            "task": task.model_dump(mode="json"),
                            "oracle": oracle.model_copy(
                                update={"provenance": provenance}
                            ).model_dump(mode="json"),
                        }
                    )
                    families[label] += 1
                    break
        partitions[split] = found
        counts[split] = {"actual": len(found), "target": target, "mutations": dict(families)}
        atomic_json(output / "preparation-progress.json", counts)
    return freeze(
        output,
        source,
        partitions,
        {"seed": seed, "coverage": counts, "preparation_worker": attestation["image_digest"]},
    )


def select_curriculum(source: TaskRepository, output: Path, summary: Path, count: int) -> dict:
    record = read_json(summary)
    if record["data"] != source.hash:
        raise ValueError("Difficulty screen belongs to a different data manifest")
    samples = record["rows"]
    passed: Counter[str] = Counter()
    seen: Counter[str] = Counter()
    for row in samples:
        if row["slot"] < 4:
            passed[row["task_id"]] += row["success"]
            seen[row["task_id"]] += 1
    partitions = {
        split: [
            {"task": task.model_dump(mode="json"), "oracle": oracle.model_dump(mode="json")}
            for task, oracle in source.tasks(split)
        ]
        for split in PARTITIONS
    }
    if set(seen) != {row["task"]["task_id"] for row in partitions["screen"]} or any(
        amount != 4 for amount in seen.values()
    ):
        raise ValueError("A complete four-rollout screen is required")
    candidates = [row for row in partitions["screen"] if 0 < passed[row["task"]["task_id"]] < 4]
    candidates.sort(
        key=lambda row: (
            abs(passed[row["task"]["task_id"]] - 2),
            keyed_seed(source.manifest["seed"], "curriculum", row["task"]["task_id"]),
        )
    )
    chosen, families = [], set()
    parent_families = {p.parent_id: p.family_id for p in source.parents("train")}
    for row in candidates:
        family = parent_families[row["task"]["parent_id"]]
        if family in families:
            continue
        families.add(family)
        chosen.append(row | {"task": row["task"] | {"split": "solver_train"}})
        if len(chosen) == count:
            break
    if len(chosen) < min(4, count):
        raise ValueError("Fewer than four parent-distinct mixed tasks; revise the training screen")
    partitions["solver_train"] = chosen
    return freeze(
        output,
        source,
        partitions,
        {
            "seed": source.manifest["seed"],
            "coverage": source.manifest["coverage"] | {"solver_train": {"actual": len(chosen)}},
            "preparation_worker": source.manifest["preparation_worker"],
            "difficulty_screen_sha256": file_hash(summary),
            "screened_data": source.hash,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--settings", type=Path, default=Path("configs/local.yaml"))
    parser.add_argument("--screen-summary", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--curriculum-size", type=int, default=8, choices=range(4, 13))
    args = parser.parse_args()
    os.umask(0o077)
    source = TaskRepository(args.source)
    if args.validate_only:
        if args.output or args.screen_summary:
            parser.error("--validate-only uses only --source and --settings")
        print(validate_prepared(source, SandboxBackend(load_config(args.settings).sandbox)))
        return
    if args.output is None:
        parser.error("--output is required for preparation or curriculum selection")
    if (args.output / "manifest.json").exists():
        raise ValueError("Task 20 prepared data is immutable")
    if args.screen_summary:
        report = select_curriculum(source, args.output, args.screen_summary, args.curriculum_size)
    else:
        report = prepare(
            source, args.output, SandboxBackend(load_config(args.settings).sandbox), 2026
        )
    print(report["coverage"])


if __name__ == "__main__":
    main()
