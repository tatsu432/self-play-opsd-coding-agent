"""Build a small, physically verified Task 21 feasibility and screening bundle."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from itertools import chain
from pathlib import Path
from urllib.request import urlopen

from selfplay_opsd.config import SandboxConfig
from selfplay_opsd.core import ExperimentError, TaskSpec, digest
from selfplay_opsd.pdb_data import (
    LCB_REVISION,
    RELEASES,
    SCORER,
    eligible,
    families,
    lcb_rows,
    parent_key,
    release_rows,
    write_bundle,
)
from selfplay_opsd.pdb_verifier import PdbOracle, verify_pdb
from selfplay_opsd.sandbox import SandboxBackend
from selfplay_opsd.storage import atomic_json, file_hash
from selfplay_opsd.task21 import PdbRepository

LCB_FILES = ("test.jsonl", *(f"test{i}.jsonl" for i in range(2, 7)))
BCB_SHA = "6079ff7a9f41fb928b3b0750c9e1e160ac36c8cc7c58124efbf86138549131d6"


def collect_lcb(root: Path, releases: dict, fetch: bool) -> tuple[dict, list[str]]:
    ids = {
        parent_key(row).split(":", 1)[1]
        for row in chain.from_iterable(releases.values())
        if row["source_dataset"] == "livecodebench"
    }
    data: dict = {}
    used = []
    for name in LCB_FILES:
        slim = root / f"matched-{name}.json"
        origin = root / name
        if slim.exists():
            rows = json.loads(slim.read_text())
        elif fetch:
            url = (
                "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/"
                f"{LCB_REVISION}/{name}"
            )
            with urlopen(url, timeout=45) as stream:
                rows = list(lcb_rows(stream, ids - data.keys()))
            atomic_json(slim, rows)
        elif origin.exists():
            with origin.open("rb") as stream:
                rows = list(lcb_rows(stream, ids - data.keys()))
            atomic_json(slim, rows)
        else:
            continue
        data.update({row["question_id"]: row for row in rows})
        used.append(name)
        if ids <= data.keys():
            break
    return data, used


def prepare(args) -> None:
    if (args.output / "manifest.json").exists():
        raise ValueError("PDB bundles are immutable. Use a new output directory.")
    releases = release_rows(args.pdb)
    lcb, used = collect_lcb(args.upstream, releases, args.fetch_lcb)
    bcb_path = args.upstream / "bigcodebench.json"
    if file_hash(bcb_path) != BCB_SHA:
        raise ValueError("BigCodeBench data checksum differs from the pinned PDB scorer")
    bcb = json.loads(bcb_path.read_text())
    observations = json.loads(args.bcb_observations.read_text()) if args.bcb_observations else {}
    if observations and observations.get("bcb_sha256") != BCB_SHA:
        raise ValueError("Adapted BigCodeBench observations use another fixture identity")
    prepared, excluded = eligible(releases, lcb, bcb, observations.get("parents"))
    sandbox = SandboxBackend(SandboxConfig(image=args.worker))
    proof = sandbox.preflight()
    log: list[dict] = []
    groups: dict[str, list[tuple[TaskSpec, PdbOracle]]] = {}
    used_families: set[str] = set()
    used_parents: set[str] = set()
    family_map = families(releases)
    multi_families = {family_map[parent_key(row)] for row in releases["PDB-Multi"]}
    pilot = PdbRepository(args.from_pilot) if args.from_pilot else None
    feasibility_ref = (
        PdbRepository(args.feasibility_reference) if args.feasibility_reference else None
    )
    if pilot:
        for split in pilot.partition_names():
            for task, oracle in pilot.tasks(split):
                if not isinstance(oracle, PdbOracle):
                    raise ValueError("Pilot contains an incompatible oracle")
                family = family_map[task.parent_id]
                if family not in used_families:
                    groups.setdefault(split, []).append(
                        (
                            task,
                            oracle.model_copy(
                                update={"provenance": oracle.provenance | {"family_id": family}}
                            ),
                        )
                    )
                    used_parents.add(task.parent_id)
                    used_families.add(family)

    def pick(
        split: str,
        candidates: list[tuple[TaskSpec, PdbOracle]],
        target: int,
        *,
        source: str | None = None,
        held: bool = False,
        multi: bool = False,
    ) -> None:
        bucket = groups.setdefault(split, [])
        initial = len(bucket)
        for task, oracle in candidates:
            if len(bucket) - initial >= target:
                break
            family = oracle.provenance["family_id"]
            if (
                family in used_families
                or task.parent_id in used_parents
                or (not multi and family in multi_families)
                or (source and oracle.provenance["source_dataset"] != source)
                or (held and not multi and oracle.provenance["release"] != "PDB-Single")
            ):
                continue
            try:
                canonical = verify_pdb(sandbox, oracle.correct_source, task, oracle)
                buggy = verify_pdb(sandbox, task.initial_source, task, oracle)
            except ExperimentError as exc:
                raise ExperimentError(
                    f"CPU PDB qualification interrupted ({exc.status})", exc.status
                ) from exc
            accepted = canonical.passed and not buggy.passed and buggy.semantic_failures > 0
            print(
                f"{split} {oracle.handler} canonical={canonical.passed} semantic_bug={bool(buggy.semantic_failures)}",
                flush=True,
            )
            log.append(
                {
                    "task_id": task.task_id,
                    "parent": task.parent_id,
                    "handler": oracle.handler,
                    "canonical_pass": canonical.passed,
                    "buggy_pass": buggy.passed,
                    "buggy_semantic_failures": buggy.semantic_failures,
                    "buggy_runtime_failures": buggy.runtime_failures,
                    "worker_seconds": canonical.seconds + buggy.seconds,
                    "accepted": accepted,
                }
            )
            if accepted:
                bucket.append((task, oracle))
                used_families.add(family)
                used_parents.add(task.parent_id)

    singles = sorted(
        prepared.get("PDB-Single-Full", []),
        key=lambda item: digest([21, item[0].task_id]),
    )
    multis = sorted(
        prepared.get("PDB-Multi", []),
        key=lambda item: digest([21, item[0].task_id]),
    )
    if pilot:
        if not groups.get("feasibility_multi") and not any(
            o.provenance["release"] == "PDB-Multi" for _, o in groups["feasibility"]
        ):
            pick("feasibility_multi", multis, 2, multi=True)
    else:
        pick("feasibility", multis, 2, multi=True)
        pick("feasibility", singles, 4, source="bigcodebench")
        pick("feasibility", singles, 12 - len(groups["feasibility"]), source="livecodebench")
        pick("screen", singles, args.screen, source="livecodebench")
    if args.extended:
        pick(
            "solver_probe", singles, 8 - len(groups.get("solver_probe", [])), source="livecodebench"
        )
        pick(
            "dev_monitor",
            singles,
            24 - len(groups.get("dev_monitor", [])),
            source="livecodebench",
            held=True,
        )
        pick(
            "primary_final",
            singles,
            64 - len(groups.get("primary_final", [])),
            source="livecodebench",
            held=True,
        )
        pick(
            "cross_source",
            singles,
            64 - len(groups.get("cross_source", [])),
            source="bigcodebench",
            held=True,
        )
        pick(
            "harder_transfer",
            multis,
            24 - len(groups.get("harder_transfer", [])),
            held=True,
            multi=True,
        )

    counts = {split: len(rows) for split, rows in groups.items()}
    if not groups["feasibility"]:
        raise RuntimeError("No verified PDB feasibility tasks")
    if feasibility_ref and {task.task_id for task, _ in feasibility_ref.tasks("feasibility")} != {
        task.task_id for task, _ in groups["feasibility"]
    }:
        raise ValueError("Benchmark feasibility tasks differ from the original measured pilot")
    write_bundle(
        args.output,
        {split: rows for split, rows in groups.items() if rows},
        {
            "scorer": SCORER,
            "pdb_releases": RELEASES,
            "lcb_revision": LCB_REVISION,
            "lcb_files": used,
            "bcb_sha256": BCB_SHA,
            "worker": proof["image_digest"],
            "pilot_manifest": pilot.hash if pilot else None,
            "feasibility_manifest": (
                feasibility_ref.hash
                if feasibility_ref
                else (
                    pilot.provenance.get("feasibility_manifest")
                    or pilot.provenance.get("pilot_manifest")
                    or pilot.hash
                    if pilot
                    else None
                )
            ),
            "bcb_observations": file_hash(args.bcb_observations) if args.bcb_observations else None,
        },
    )
    report = {
        "local_worker": proof,
        "eligible": {key: len(val) for key, val in prepared.items()},
        "matched_lcb_parents": len(lcb),
        "partitions": counts,
        "verification": log,
        "exclusions": dict(excluded),
        "handler_counts": dict(
            Counter(oracle.handler for rows in groups.values() for _, oracle in rows)
        ),
    }
    atomic_json(args.report, report)
    print(
        json.dumps({k: report[k] for k in ("eligible", "partitions", "handler_counts")}, indent=2)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--worker", default="selfplay-worker:local")
    parser.add_argument("--screen", type=int, default=0, choices=(0, 24))
    parser.add_argument("--extended", action="store_true")
    parser.add_argument("--from-pilot", type=Path)
    parser.add_argument("--feasibility-reference", type=Path)
    parser.add_argument("--bcb-observations", type=Path)
    parser.add_argument("--fetch-lcb", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    prepare(args)


if __name__ == "__main__":
    main()
