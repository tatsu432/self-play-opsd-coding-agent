"""Task 17 frozen training-only candidates, isolation validation and provenance."""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

from selfplay_opsd.core import OracleSpec, TaskSpec, digest, keyed_seed
from selfplay_opsd.data import TaskRepository, build_task
from selfplay_opsd.mutations import fixed_mutation, make_patch, validate_source
from selfplay_opsd.storage import atomic_json, file_hash
from selfplay_opsd.task15 import eligible_training_parents

STUDY = "repair-selfplay-2026-task17-interface-v1"
COUNTS = {"interface": 40, "curriculum": 40, "calibration": 8, "development": 16}


def partition_parents(repository: TaskRepository) -> dict:
    """Freeze one parent per family, balance length/structure without outcomes."""
    families = defaultdict(list)
    for parent in eligible_training_parents(repository.parents("train"), ["Mbpp/590"]):
        families[parent.family_id].append(parent)
    representatives = [
        min(group, key=lambda p: digest([2026, "task17-parent", p.parent_id]))
        for group in families.values()
    ]
    representatives.sort(key=lambda p: (len(p.source), digest([2026, "task17-order", p.parent_id])))
    strata = defaultdict(list)
    for index, parent in enumerate(representatives):
        tree = validate_source(parent.source)
        complexity = any(
            isinstance(n, (ast.If, ast.For, ast.While, ast.comprehension)) for n in ast.walk(tree)
        )
        strata[(min(3, 4 * index // len(representatives)), complexity)].append(parent)
    ordered = []
    while any(strata.values()):
        for key in sorted(strata):
            if strata[key]:
                ordered.append(strata[key].pop(0))
    if len(ordered) < sum(COUNTS.values()):
        raise ValueError("Insufficient disjoint eligible families; freeze a revised smaller design")
    result: dict[str, list[str]] = {key: [] for key in COUNTS}
    # Weighted interleaving prevents assigning all long sources to one purpose.
    for parent in ordered:
        available = [key for key in COUNTS if len(result[key]) < COUNTS[key]]
        if not available:
            break
        purpose_key = min(
            available, key=lambda k: (len(result[k]) / COUNTS[k], list(COUNTS).index(k))
        )
        result[purpose_key].append(parent.parent_id)
    assigned = {p for values in result.values() for p in values}
    result["reserve"] = [p.parent_id for p in ordered if p.parent_id not in assigned]
    return result


def prepare_pool(repository, sandbox, output: Path, *, admission: dict | None = None) -> dict:
    if output.exists():
        raise ValueError("Prepared pool already exists; inspect instead of replacing")
    requested = partition_parents(repository)
    partitions = {key: list(requested[key]) for key in COUNTS}
    reserve = list(requested["reserve"])
    by_id = {p.parent_id: p for p in repository.parents("train")}
    rows, exclusions, replacements = [], [], []
    if admission is not None:
        if (
            admission["dataset_hash"] != repository.hash
            or admission["requested_partitions"] != requested
            or admission["admitted_by_purpose"] != COUNTS
        ):
            raise ValueError("Cached admission dataset/partition/coverage differs")
        partitions, rows = admission["partitions"], admission["rows"]
        exclusions, replacements = admission["exclusions"], admission["replacements"]
        if len(rows) != sum(COUNTS.values()):
            raise ValueError("Cached admission is incomplete")
        for row in rows:
            task, oracle = (
                TaskSpec.model_validate(row["task"]),
                OracleSpec.model_validate(row["oracle"]),
            )
            parent = by_id[task.parent_id]
            expected = fixed_mutation(parent.source, task.mutation_seed)
            public, hidden = repository.parent_cases(task.parent_id)
            if (
                task.task_id
                != digest([parent.parent_id, expected.dedup_key, "train", task.mutation_seed])
                or task.initial_source != expected.source
                or task.source_hash != digest(expected.source)
                or oracle.correct_source != parent.source
                or oracle.public_cases != public
                or oracle.hidden_cases != hidden
                or row["presentation"] != "original"
                or task.parent_id not in partitions[row["purpose"]]
            ):
                raise ValueError("Cached verified task differs from deterministic source/cases")
    else:
        for purpose in COUNTS:
            pending = list(partitions[purpose])
            for parent_id in pending:
                parent = by_id[parent_id]
                public, hidden = repository.parent_cases(parent_id)
                admitted = False
                seen_mutations: set[str] = set()
                for attempt in range(32):
                    seed = keyed_seed(2026, "task17-mutation", parent_id, attempt)
                    try:
                        mutation = fixed_mutation(parent.source, seed)
                        if mutation.dedup_key in seen_mutations:
                            raise ValueError("Repeated deterministic mutation already rejected")
                        seen_mutations.add(mutation.dedup_key)
                        task, oracle = build_task(
                            parent, mutation, public, hidden, "train", seed, sandbox
                        )
                    except ValueError as exc:
                        exclusions.append(
                            {"parent_id": parent_id, "attempt": attempt, "reason": str(exc)}
                        )
                        continue
                    rows.append(
                        {
                            "purpose": purpose,
                            "family_id": parent.family_id,
                            "presentation": "original",
                            "task": task.model_dump(mode="json"),
                            "oracle": oracle.model_dump(mode="json"),
                        }
                    )
                    admitted = True
                    break
                if not admitted and reserve:
                    replacement = reserve.pop(0)
                    pending.append(replacement)
                    partitions[purpose].remove(parent_id)
                    partitions[purpose].append(replacement)
                    replacements.append(
                        {
                            "purpose": purpose,
                            "excluded": parent_id,
                            "replacement": replacement,
                            "reason": "32 oracle-only mutation attempts failed",
                        }
                    )
            atomic_json(
                output.with_name("admission-progress.json"),
                {
                    "dataset_hash": repository.hash,
                    "requested_partitions": requested,
                    "partitions": partitions,
                    "exclusions": exclusions,
                    "replacements": replacements,
                    "rows": rows,
                    "admitted_by_purpose": {
                        key: sum(r["purpose"] == key for r in rows) for key in COUNTS
                    },
                },
            )
    original = [r for r in rows if r["purpose"] == "interface"]
    if len(original) < COUNTS["interface"]:
        raise ValueError(
            "Interface admission deficit: preserve evidence and revise before inference"
        )
    variant_checks = []
    for index, row in enumerate(
        sorted(original, key=lambda r: digest([2026, "task17-variant", r["task"]["task_id"]]))[:16]
    ):
        task = TaskSpec.model_validate(row["task"])
        oracle = OracleSpec.model_validate(row["oracle"])
        if index < 8:
            prefix = "".join("# " + line + "\n" for line in task.specification.splitlines()) * 4
            source = prefix + task.initial_source
            correct = prefix + oracle.correct_source
            specification, presentation = task.specification, "source_comments"
        else:
            source, correct = task.initial_source, oracle.correct_source
            specification, presentation = (
                "\n\n".join([task.specification] * 4),
                "specification_repeat",
            )
        identifier = digest([task.task_id, "task17", presentation])
        variant = task.model_copy(
            update={
                "task_id": identifier,
                "initial_source": source,
                "source_hash": digest(source),
                "specification": specification,
            }
        )
        variant_oracle = oracle.model_copy(
            update={
                "task_id": identifier,
                "correct_source": correct,
                "inverse_patch": make_patch(source, correct),
            }
        )
        if ast.dump(ast.parse(correct)) != ast.dump(ast.parse(oracle.correct_source)) or ast.dump(
            ast.parse(source)
        ) != ast.dump(ast.parse(task.initial_source)):
            raise ValueError("Presentation variant changed executable AST")
        # Isolated verification also checks comment-sensitive source edge cases.
        for kind, candidate in (("mutant", source), ("correct", correct)):
            result = sandbox.verify(
                candidate,
                task.entry_point,
                oracle.public_cases + oracle.hidden_cases,
                task.verifier_seconds,
            )
            from dataclasses import asdict

            check = {
                "parent_id": task.parent_id,
                "task_id": task.task_id,
                "presentation": presentation,
                "kind": kind,
                "verification": asdict(result),
            }
            variant_checks.append(check)
            atomic_json(output.with_name("variant-verification.json"), variant_checks)
            if (kind == "correct" and not result.passed) or (
                kind == "mutant" and result.semantic_failures < 1
            ):
                before = task.initial_source if kind == "mutant" else oracle.correct_source
                check["original_recheck"] = asdict(
                    sandbox.verify(
                        before,
                        task.entry_point,
                        oracle.public_cases + oracle.hidden_cases,
                        task.verifier_seconds,
                    )
                )
                check["variant_repeat"] = asdict(
                    sandbox.verify(
                        candidate,
                        task.entry_point,
                        oracle.public_cases + oracle.hidden_cases,
                        task.verifier_seconds,
                    )
                )
                atomic_json(output.with_name("variant-verification.json"), variant_checks)
                raise ValueError(
                    f"Presentation admission failed for {task.parent_id}/{presentation}/{kind}; see variant-verification.json"
                )
        rows.append(
            row
            | {
                "presentation": presentation,
                "task": variant.model_dump(mode="json"),
                "oracle": variant_oracle.model_dump(mode="json"),
            }
        )
    counts = {
        purpose: sum(r["purpose"] == purpose and r["presentation"] == "original" for r in rows)
        for purpose in COUNTS
    }
    if counts != COUNTS:
        raise ValueError("Pool admission deficit; do not silently shrink frozen training design")
    result = {
        "study_id": STUDY,
        "dataset_hash": repository.hash,
        "requested_partitions": requested,
        "partitions": partitions,
        "replacements": replacements,
        "mutation_attempts_per_parent": 32,
        "counts": counts,
        "rows": rows,
        "exclusions": exclusions,
        "isolated_verified": True,
    }
    atomic_json(output, result)
    return result


def load_pool(path: Path, repository: TaskRepository) -> tuple[dict, str]:
    import json

    pool_hash = file_hash(path)
    marker = json.loads(path.with_name("complete.json").read_text())
    if marker.get("pool_sha256") != pool_hash:
        raise ValueError("Prepared pool checksum differs from completion marker")
    pool = json.loads(path.read_text())
    if (
        pool["study_id"] != STUDY
        or pool["dataset_hash"] != repository.hash
        or not pool["isolated_verified"]
    ):
        raise ValueError("Task 17 pool identity or verification differs")
    requested = partition_parents(repository)
    if pool["requested_partitions"] != requested:
        raise ValueError("Task 17 partition assignment differs")
    for purpose in COUNTS:
        actual = pool["partitions"][purpose]
        if len(actual) != COUNTS[purpose] or not set(actual) <= set(
            requested[purpose] + requested["reserve"]
        ):
            raise ValueError("Task 17 replacement crossed reserved purpose boundaries")
    from collections import Counter

    counts = Counter(r["purpose"] for r in pool["rows"])
    if dict(counts) != (COUNTS | {"interface": 56}) or len(
        {r["task"]["task_id"] for r in pool["rows"]}
    ) != len(pool["rows"]):
        raise ValueError("Prepared pool coverage or unique task identity differs")
    heldout_families = {
        r["parent"]["family_id"] for r in repository._parents if r["split"] != "train"
    }
    seen: dict[str, str] = {}
    for row in pool["rows"]:
        task, oracle = (
            TaskSpec.model_validate(row["task"]),
            OracleSpec.model_validate(row["oracle"]),
        )
        if (
            task.parent_id not in pool["partitions"][row["purpose"]]
            or task.source_hash != digest(task.initial_source)
            or oracle.task_id != task.task_id
        ):
            raise ValueError("Pool task identity differs")
        family = row["family_id"]
        if family in heldout_families or family in seen and seen[family] != row["purpose"]:
            raise ValueError("Pool family leakage")
        seen[family] = row["purpose"]
    return pool, pool_hash


def configure_operation(cfg, operation: str, arm: str, control_path: Path):
    """Resolve the same effective science in controller and allocated runner."""
    import json

    cfg.method = "B"
    if operation == "interface":
        cfg.episode.edit_protocol = "unified_diff" if arm in {"A", "C"} else "replace_text"
    elif operation in {"calibrate", "student"}:
        if not control_path.is_file():
            raise ValueError(
                "Freeze configs/task17-objective-control.json after Phase 1 before Phase 2"
            )
        control = json.loads(control_path.read_text())
        if control.get("interface") not in {"unified_diff", "replace_text"} or control.get(
            "episodes"
        ) not in {80, 128}:
            raise ValueError("Invalid objective-study interface or endpoint")
        cfg.episode.edit_protocol = control["interface"]
        if operation == "student":
            cfg.method = {"A": "A", "B": "B", "C": "A", "D": "GRPO"}[arm]
            cfg.study_id = f"repair-selfplay-2026-task17-objectives-{arm}-v1"
            cfg.iterations = control["episodes"] // 8
            cfg.parents_per_iteration = cfg.proposals_per_parent = 2
            cfg.evaluate_every = cfg.iterations + 1
            cfg.gradient_diagnostics = True
            if arm == "C":
                coefficient = control.get("calibration_coefficient")
                if not isinstance(coefficient, (int, float)) or coefficient <= 0:
                    raise ValueError("Calibrated arm needs a finite positive frozen coefficient")
                cfg.objective.sd_lambda = coefficient
    return cfg
