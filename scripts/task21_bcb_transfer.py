"""Build a separate BigCodeBench transfer set from parents outside the PDB releases."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from selfplay_opsd.config import SandboxConfig
from selfplay_opsd.core import digest, read_json
from selfplay_opsd.pdb_data import (
    BCB_TRANSFER,
    bcb_cases,
    build_task,
    families,
    parent_key,
    release_rows,
    write_bundle,
)
from selfplay_opsd.pdb_verifier import PdbOracle, verify_pdb
from selfplay_opsd.sandbox import SandboxBackend
from selfplay_opsd.storage import atomic_json, file_hash
from selfplay_opsd.structured_mutations import mutate_source, sites
from selfplay_opsd.task21 import PdbRepository

UPSTREAM_SHA = "6079ff7a9f41fb928b3b0750c9e1e160ac36c8cc7c58124efbf86138549131d6"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("pdb", "bcb", "data", "output", "report"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    if (args.output / "manifest.json").exists() or file_hash(args.bcb) != UPSTREAM_SHA:
        raise ValueError("Use a new directory and pinned BigCodeBench fixtures")
    releases = release_rows(args.pdb)
    pdb_parents = {parent_key(row) for rows in releases.values() for row in rows}
    repo = PdbRepository(args.data)
    upstream = read_json(args.bcb)
    candidates = []
    for key, row in sorted(upstream.items()):
        if f"bigcodebench:{key}" in pdb_parents:
            continue
        try:
            cases = bcb_cases(row)
        except (ValueError, TypeError, SyntaxError):
            continue
        record = {
            "task_id": f"{key}_0",
            "source_dataset": "bigcodebench",
            "gt_solution": row["code_prompt"] + "\n" + row["canonical_solution"],
            "task_prompt": row["instruct_prompt"],
            "is_buggy": True,
            "bug_count": 1,
            "bug_type": "semantic_mutation",
            "bug_subtype": "deterministic_ast",
        }
        candidates.append((record, cases))
    enriched = releases | {
        "PDB-Single-Full": [*releases["PDB-Single-Full"], *(row for row, _ in candidates)]
    }
    family_map = families(enriched)
    held_out = {family_map[p] for p in pdb_parents}
    worker = SandboxBackend(SandboxConfig())
    proof = worker.preflight()
    accepted, checked = [], []
    for row, cases in candidates:
        family = family_map[parent_key(row)]
        if family in held_out:
            continue
        choices = [
            (site["path"], variant)
            for site in sites(row["gt_solution"])
            for variant in site["operations"]
        ]
        for path, op in sorted(choices, key=lambda item: digest([row["task_id"], item]))[:4]:
            variant = row | {
                "buggy_code": mutate_source(row["gt_solution"], path, op),
                "bug_subtype": op,
            }
            try:
                task, oracle = build_task(
                    variant, cases, family, BCB_TRANSFER, "cross_source_bcb_mutants"
                )
            except (SyntaxError, ValueError):
                continue
            canonical = verify_pdb(worker, oracle.correct_source, task, oracle)
            mutant = verify_pdb(worker, task.initial_source, task, oracle)
            qualifies = canonical.passed and mutant.semantic_failures > 0
            checked.append(
                {
                    "task": task.task_id,
                    "parent": task.parent_id,
                    "canonical": canonical.passed,
                    "mutant_semantic_failures": mutant.semantic_failures,
                }
            )
            if qualifies:
                accepted.append((task, oracle))
                held_out.add(family)
                break
            if not canonical.passed:
                break
    partitions = {
        split: [
            (task, oracle) for task, oracle in repo.tasks(split) if isinstance(oracle, PdbOracle)
        ]
        for split in repo.partition_names()
    }
    if accepted:
        partitions["cross_source_bcb_mutants"] = accepted
    write_bundle(
        args.output,
        partitions,
        repo.provenance
        | {
            "cross_source_bcb_mutants": {
                "source": BCB_TRANSFER,
                "worker": proof["image_digest"],
                "parents": len(accepted),
            },
            "pilot_manifest": repo.hash,
        },
    )
    atomic_json(
        args.report,
        {"eligible_parents": len(candidates), "verified_parents": len(accepted), "checks": checked},
    )
    print(f"Verified {len(accepted)} of {len(candidates)} eligible BigCodeBench transfer parents")


if __name__ == "__main__":
    main()
