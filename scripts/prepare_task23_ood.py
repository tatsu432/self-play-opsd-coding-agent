"""Prepare and verify new parent-disjoint PDB repair transfer cohorts."""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path
from typing import cast

from prepare_task21 import BCB_SHA, collect_lcb

from selfplay_opsd.config import SandboxConfig
from selfplay_opsd.core import TaskSpec, digest, read_json
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
    if (output / "candidates.json").exists():
        raise ValueError("Use a new output directory for a new candidate freeze")
    releases = release_rows(pdb)
    bcb_path = upstream / "bigcodebench.json"
    if file_hash(bcb_path) != BCB_SHA:
        raise ValueError("Pinned BigCodeBench fixture checksum differs")
    bcb = read_json(bcb_path)
    lcb, _ = collect_lcb(upstream, releases, False)
    observations = read_json(upstream / "bcb_observations.json")
    known = [
        (task, cast(PdbOracle, oracle))
        for manifest in historical.glob("*/manifest.json")
        for repo in [PdbRepository(manifest.parent)]
        for split in repo.partition_names()
        for task, oracle in repo.tasks(split)
    ]
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
            cases = bcb_cases(fixture)
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
        candidates.append((row, cases))
        audit_rows.append(row)
    family_map = families(releases_for_audit)
    excluded = {family_map[parent] for parent in used_parents}
    counts = Counter()
    source: dict[str, list] = {}
    for row, cases in sorted(candidates, key=lambda item: variant_order(item[0])):
        family = family_map[parent_key(row)]
        if family in excluded or family in source:
            continue
        edits = [
            (site["path"], operation)
            for site in sites(row["gt_solution"])
            for operation in site["operations"]
        ]
        variants = []
        for path, operation in sorted(edits, key=lambda item: digest([row["task_id"], item]))[:4]:
            try:
                variant = row | {
                    "buggy_code": mutate_source(row["gt_solution"], path, operation),
                    "bug_subtype": operation,
                }
                variants.append(
                    serialized(
                        *build_task(variant, cases, family, BCB_TRANSFER, "ood_source"),
                        "ood_source",
                    )
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

    # Reserve four candidates per desired family to cover verifier rejections.
    cohorts = {
        split: [
            variants
            for _, variants in sorted(
                groups.items(), key=lambda item: digest([23, split, item[0]])
            )[: 4 * TARGET]
        ]
        for split, groups in (("ood_source", source), ("ood_complexity", complexity))
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(
        output / "candidates.json",
        {"cohorts": cohorts, "target": TARGET, "source_fixture_sha256": BCB_SHA},
    )
    atomic_json(
        output / "inventory.json",
        {
            "historical_parents": len(used_parents),
            "historical_families": len(excluded),
            "candidate_families": {"ood_source": len(source), "ood_complexity": len(complexity)},
            "build_rejections": dict(counts),
            "upstream_preparation_rejections": sum(failures.values()),
        },
    )


def validate(output: Path, stopped) -> None:
    source = output / "candidates.json"
    payload = read_json(source)
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
    for split in SPLITS:
        for group in payload["cohorts"][split]:
            if len(accepted[split]) >= payload["target"]:
                break
            for row in group:
                task = TaskSpec.model_validate(row["task"])
                oracle = PdbOracle.model_validate(row["oracle"])
                family = oracle.provenance["family_id"]
                if family in selected:
                    break
                if stopped():
                    return
                identity = digest([file_hash(source), split, task.task_id, task.initial_source])
                cache = output / "verdicts" / f"{identity}.json"
                if cache.exists():
                    verdict = read_json(cache)
                else:
                    correct = verify_pdb(sandbox, oracle.correct_source, task, oracle)
                    buggy = verify_pdb(sandbox, task.initial_source, task, oracle)
                    verdict = {
                        "correct_pass": correct.passed,
                        "buggy_semantic_fail": buggy.semantic_failures > 0,
                    }
                    atomic_json(cache, verdict)
                counts[f"{split}_validated"] += 1
                if verdict["correct_pass"] and verdict["buggy_semantic_fail"]:
                    accepted[split].append(row)
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
