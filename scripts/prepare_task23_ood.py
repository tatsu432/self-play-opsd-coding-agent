"""Prepare and verify new parent-disjoint PDB repair transfer cohorts."""

from __future__ import annotations

import argparse
import gzip
import json
import os
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import cast

from capture_task21_bcb import available, recorder
from prepare_task21 import BCB_SHA, collect_lcb

from selfplay_opsd.codec import decode, encode
from selfplay_opsd.config import SandboxConfig
from selfplay_opsd.core import Case, Status, TaskSpec, digest, read_json
from selfplay_opsd.pdb_data import (
    BCB_TRANSFER,
    bcb_cases,
    build_task,
    eligible,
    families,
    parent_key,
    release_rows,
    variant_order,
)
from selfplay_opsd.pdb_verifier import PdbOracle, verify_pdb
from selfplay_opsd.sandbox import SandboxBackend
from selfplay_opsd.storage import atomic_json, file_hash
from selfplay_opsd.structured_mutations import mutate_source, sites
from selfplay_opsd.task21 import PdbRepository, Task21Control

SPLITS = ("ood_source", "ood_complexity")
TARGET = 32


def serialized(task: TaskSpec, oracle: PdbOracle, split: str) -> dict:
    return {
        "task": task.model_copy(update={"split": split}).model_dump(mode="json"),
        "oracle": oracle.model_dump(mode="json"),
    }


def prepare(pdb: Path, upstream: Path, historical: Path, output: Path) -> None:
    if (output / "candidates.json.gz").exists():
        raise ValueError("Use a new output directory for a new candidate freeze")
    releases = release_rows(pdb)
    bcb_path = upstream / "bigcodebench.json"
    if file_hash(bcb_path) != BCB_SHA:
        raise ValueError("Pinned BigCodeBench fixture checksum differs")
    bcb = read_json(bcb_path)
    lcb, _ = collect_lcb(upstream, releases, False)
    observations = read_json(upstream / "bcb-observations.json")
    known = []
    for manifest in sorted(historical.glob("*/manifest.json")):
        metadata = read_json(manifest)
        if "bundle_sha256" in metadata:
            repo = PdbRepository(manifest.parent)
            for split in repo.partition_names():
                known.extend((task, cast(PdbOracle, oracle)) for task, oracle in repo.tasks(split))
        else:
            payload = read_json(manifest.with_name("pdb.json"))
            checksum = payload.pop("hash")
            if checksum != metadata["hash"] or checksum != digest(payload):
                raise ValueError("Historical feasibility bundle checksum differs")
            known.extend(
                (TaskSpec.model_validate(row["task"]), PdbOracle.model_validate(row["oracle"]))
                for group in payload["partitions"].values()
                for row in group
            )
    used_parents = {task.parent_id for task, _ in known}
    releases_for_audit = {name: list(rows) for name, rows in releases.items()}
    audit_rows = releases_for_audit["PDB-Single-Full"]
    audit_rows.extend(
        {
            "source_dataset": task.parent_id.split(":", 1)[0],
            "task_id": task.parent_id.split(":", 1)[1] + "_0",
            "gt_solution": oracle.correct_source,
        }
        for task, oracle in known
    )
    candidates = []
    pdb_parents = {parent_key(row) for rows in releases.values() for row in rows}
    for key, fixture in sorted(bcb.items()):
        if f"bigcodebench:{key}" in pdb_parents:
            continue
        try:
            if not available(fixture):
                continue
        except (ArithmeticError, ValueError, TypeError, SyntaxError):
            continue
        row = {
            "task_id": f"{key}_0",
            "source_dataset": "bigcodebench",
            "gt_solution": fixture["code_prompt"] + "\n" + fixture["canonical_solution"],
            "task_prompt": fixture["instruct_prompt"],
            "is_buggy": True,
            "bug_count": 1,
            "bug_type": "semantic_mutation",
        }
        candidates.append((row, fixture))
        audit_rows.append(row)
    family_map = families(releases_for_audit)
    excluded = {family_map[parent] for parent in used_parents}
    counts = Counter()
    source: dict[str, list] = {}
    for row, fixture in sorted(candidates, key=lambda item: variant_order(item[0])):
        family = family_map[parent_key(row)]
        if family in excluded or family in source:
            continue
        edits = [
            (site["path"], operation)
            for site in sites(row["gt_solution"])
            for operation in site["operations"]
        ]
        variants = []
        try:
            cases = bcb_cases(fixture)
        except (ArithmeticError, ValueError, TypeError, SyntaxError):
            cases = None
        for path, operation in sorted(edits, key=lambda item: digest([row["task_id"], item]))[:4]:
            try:
                variant = row | {
                    "buggy_code": mutate_source(row["gt_solution"], path, operation),
                    "bug_subtype": operation,
                }
                if cases:
                    variants.append(
                        serialized(
                            *build_task(variant, cases, family, BCB_TRANSFER, "ood_source"),
                            "ood_source",
                        )
                    )
                else:
                    variants.append(
                        {
                            "variant": variant,
                            "family": family,
                            "entry": fixture["entry_point"],
                            "test": fixture["test"],
                        }
                    )
            except (ArithmeticError, SyntaxError, ValueError, TypeError):
                counts["source_build_rejections"] += 1
        if variants:
            source[family] = variants

    prepared, failures = eligible(releases, lcb, bcb, observations.get("parents"))
    complexity: dict[str, list] = {}
    for task, oracle in prepared["PDB-Multi"]:
        family = family_map[task.parent_id]
        if family in excluded:
            continue
        oracle = oracle.model_copy(update={"provenance": oracle.provenance | {"family_id": family}})
        complexity.setdefault(family, []).append(serialized(task, oracle, "ood_complexity"))

    native_multi = len(complexity)
    for task, oracle in prepared["PDB-Single-Full"]:
        family = family_map[task.parent_id]
        if (
            family in excluded
            or family in complexity
            or oracle.provenance["source_dataset"] != "livecodebench"
        ):
            continue
        variants = double_mutants(task, oracle, family)
        if variants:
            complexity[family] = variants
    # Prioritize genuine PDB-Multi records before double-mutation supplements.
    cohorts = {
        split: [
            variants
            for _, variants in sorted(
                groups.items(),
                key=lambda item: (
                    split == "ood_complexity" and "components" in item[1][0],
                    digest([23, split, item[0]]),
                ),
            )[: 4 * TARGET]
        ]
        for split, groups in (("ood_source", source), ("ood_complexity", complexity))
    }
    output.mkdir(parents=True, exist_ok=True)
    pending = output / "candidates.pending.gz"
    with (
        pending.open("wb") as stream,
        gzip.GzipFile(fileobj=stream, mode="wb", mtime=0, filename="") as compressed,
    ):
        payload = {"cohorts": cohorts, "target": TARGET, "source_fixture_sha256": BCB_SHA}
        compressed.write(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    pending.replace(output / "candidates.json.gz")
    atomic_json(
        output / "inventory.json",
        {
            "historical_parents": len(used_parents),
            "historical_families": len(excluded),
            "candidate_families": {"ood_source": len(source), "ood_complexity": len(complexity)},
            "build_rejections": dict(counts),
            "eligible_native_multi_families": native_multi,
            "upstream_preparation_rejections": sum(failures.values()),
        },
    )


def double_mutants(task: TaskSpec, oracle: PdbOracle, family: str) -> list[dict]:
    canonical = oracle.correct_source
    operations = [(item["path"], op) for item in sites(canonical) for op in item["operations"]]
    edits = sorted(operations, key=lambda item: digest([23, family, item]))[:24]
    row = {
        "task_id": oracle.provenance["pdb_id"],
        "source_dataset": "livecodebench",
        "gt_solution": canonical,
        "task_prompt": task.specification,
        "is_buggy": True,
        "bug_count": 2,
        "bug_type": "semantic_mutation",
        "bug_subtype": "two_disjoint_ast_sites",
    }
    cases = (oracle.handler, task.entry_point, oracle.public_cases, oracle.hidden_cases)
    variants = []
    for (left, op1), (right, op2) in combinations(edits, 2):
        if left[: len(right)] == right or right[: len(left)] == left:
            continue
        try:
            one = mutate_source(canonical, left, op1)
            two = mutate_source(canonical, right, op2)
            both = mutate_source(one, right, op2)
            variant = row | {"buggy_code": both}
            variants.append(
                serialized(
                    *build_task(variant, cases, family, "PDB-Single-Full", "ood_complexity"),
                    "ood_complexity",
                )
                | {"components": [one, two]}
            )
        except (ArithmeticError, SyntaxError, TypeError, ValueError):
            continue
        if len(variants) >= 4:
            break
    return variants


def capture_cases(worker: SandboxBackend, item: dict) -> list[dict]:
    source = recorder(item["variant"]["gt_solution"], item["test"], item["entry"])
    samples = []
    for _ in range(2):
        result = worker.execute(source, "collect", [encode([])], 20, isolated_harness=True)
        if result.status in {Status.INFRA, Status.DEPENDENCY}:
            raise RuntimeError("BigCodeBench fixture-capture worker failed")
        if result.status != Status.OK or not result.results or not result.results[0]["ok"]:
            return []
        calls = decode(result.results[0]["value"])
        if not calls:
            return []
        samples.append(calls)
    if samples[0] != samples[1]:
        return []
    unique: dict[str, dict] = {}
    for inputs, (comparison, expected) in samples[0]:
        args = encode(inputs)
        case = {"args": args, "expected": encode(expected), "comparison": comparison}
        if unique.setdefault(digest(args), case) != case:
            return []
    return list(unique.values()) if len(unique) >= 2 else []


def validate(output: Path, stopped) -> None:
    source = output / "candidates.json.gz"
    with gzip.open(source, "rt") as stream:
        payload = json.load(stream)
    config = Task21Control.read(Path("/workspace/configs/task21-main.json"))
    sandbox = SandboxBackend(
        SandboxConfig.model_validate(
            config.sandbox.model_dump() | {"remote_url": os.environ["SELFPLAY_WORKER_URL"]}
        )
    )
    proof = sandbox.preflight()
    if not proof["ready"]:
        raise ValueError("OOD validation requires the isolated PDB worker")
    accepted: dict[str, list] = {split: [] for split in SPLITS}
    selected = set()
    counts = Counter()
    checksum = file_hash(source)
    for split in SPLITS:
        for group in payload["cohorts"][split]:
            if len(accepted[split]) >= payload["target"]:
                break
            for item in group:
                if stopped():
                    return
                row = item
                if "variant" in item:
                    capture = output / "fixtures" / f"{digest([checksum, item['family']])}.json"
                    calls = (
                        read_json(capture) if capture.is_file() else capture_cases(sandbox, item)
                    )
                    if not capture.is_file():
                        atomic_json(capture, calls)
                    if not calls:
                        counts["ood_source_fixture_rejections"] += 1
                        break
                    suite = [Case.model_validate(case) for case in calls]
                    try:
                        task, oracle = build_task(
                            item["variant"],
                            ("bcb_call", item["entry"], suite[:1], suite[1:]),
                            item["family"],
                            BCB_TRANSFER,
                            split,
                        )
                    except (ArithmeticError, SyntaxError, TypeError, ValueError):
                        continue
                    provenance = oracle.provenance | {
                        "verification": "stable-canonical-outputs-on-upstream-inputs"
                    }
                    row = serialized(
                        task, oracle.model_copy(update={"provenance": provenance}), split
                    )
                task = TaskSpec.model_validate(row["task"])
                oracle = PdbOracle.model_validate(row["oracle"])
                family = oracle.provenance["family_id"]
                if family in selected:
                    break
                if stopped():
                    return
                identity = digest([checksum, split, task.task_id, task.initial_source])
                cache = output / "verdicts" / f"{identity}.json"
                if cache.exists():
                    verdict = read_json(cache)
                else:
                    correct = verify_pdb(sandbox, oracle.correct_source, task, oracle)
                    buggy = verify_pdb(sandbox, task.initial_source, task, oracle)
                    verdict = {
                        "correct_pass": correct.passed,
                        "buggy_semantic_fail": buggy.semantic_failures > 0
                        and all(
                            verify_pdb(sandbox, component, task, oracle).semantic_failures > 0
                            for component in item.get("components", [])
                        ),
                    }
                    atomic_json(cache, verdict)
                counts[f"{split}_validated"] += 1
                if verdict["correct_pass"] and verdict["buggy_semantic_fail"]:
                    accepted[split].append({key: row[key] for key in ("task", "oracle")})
                    selected.add(family)
                    break
                counts[f"{split}_rejected"] += 1
                if not verdict["correct_pass"]:
                    break
    atomic_json(
        output / "complete.json",
        {
            "cohorts": accepted,
            "candidate_sha256": file_hash(source),
            "verifier_counts": dict(counts),
            "worker": proof["image_digest"],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", type=Path, default=Path("runs/task20/pdb"))
    parser.add_argument("--upstream", type=Path, default=Path("runs/task21/upstream"))
    parser.add_argument("--historical", type=Path, default=Path("data/task21"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    prepare(args.pdb, args.upstream, args.historical, args.output)


if __name__ == "__main__":
    main()
