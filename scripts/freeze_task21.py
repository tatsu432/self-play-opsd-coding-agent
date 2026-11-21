"""Freeze a Task 21 curriculum and matched controls using measured stage evidence."""

from __future__ import annotations

import argparse
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

from selfplay_opsd.core import digest, read_json
from selfplay_opsd.pdb_data import write_bundle
from selfplay_opsd.pdb_verifier import PdbOracle
from selfplay_opsd.storage import atomic_json, file_hash
from selfplay_opsd.task21 import PdbRepository, Task21Control


def sample_groups(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["task_id"]].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: row["slot"])
        if [row["slot"] for row in rows] != list(range(8)):
            raise ValueError("Screen requires eight matched episodes per task")
    return dict(groups)


def group_efficiency(groups: dict[str, list[dict]], seconds: float) -> dict[int, dict]:
    if not groups or seconds <= 0:
        raise ValueError("Screen requires measured groups and actual H200 allocation")
    durations = {
        n: math.fsum(sum(row["timings"].values()) for rows in groups.values() for row in rows[:n])
        for n in (4, 8)
    }
    overhead = max(0.0, seconds - durations[8])
    results = {}
    for size in (4, 8):
        successes = Counter(sum(row["success"] for row in rows[:size]) for rows in groups.values())
        mixed = len(groups) - successes[0] - successes[size]
        cost = durations[size] + overhead
        results[size] = {
            "histogram": [successes[n] for n in range(size + 1)],
            "mixed": mixed,
            "estimated_h200_seconds": cost,
            "mixed_per_h200_hour": 3600 * mixed / cost,
        }
    return results


def freeze_curriculum(args) -> None:
    if (args.output / "manifest.json").exists():
        raise ValueError("A frozen curriculum is immutable")
    repo = PdbRepository(args.data)
    sample = read_json(args.samples)
    marker = read_json(args.marker)
    if (
        sample["data"] != marker["identity"]["data"]
        or sample["snapshot"] != marker["snapshot"]
        or sample["count"] != marker["result"]["count"]
        or sample["count"] != len(sample["rows"])
    ):
        raise ValueError("Screen samples do not match the completed stage marker")
    bundle = read_json(args.data / "pdb.json")
    pilot_manifest = bundle["provenance"].get("pilot_manifest")
    if sample["data"] not in {repo.hash, pilot_manifest}:
        raise ValueError("Screen was produced with a different task manifest")
    groups = sample_groups(sample["rows"])
    efficiency = group_efficiency(groups, args.h200_seconds)
    size = max((4, 8), key=lambda n: (efficiency[n]["mixed_per_h200_hour"], -n))
    candidates = []
    for task, oracle in repo.tasks("screen"):
        if not isinstance(oracle, PdbOracle) or task.task_id not in groups:
            raise ValueError("Screened task does not match frozen training partition")
        successes = sum(row["success"] for row in groups[task.task_id][:size])
        candidates.append((task, oracle, successes))
    mixed = [row for row in candidates if 0 < row[2] < size]
    if not mixed:
        raise ValueError("Screen produced no mixed groups for a solver curriculum")
    classes: Counter[str] = Counter()
    curriculum = []
    while mixed and len(curriculum) < 8:
        candidate = min(
            mixed,
            key=lambda row: (
                classes[str(row[1].provenance["bug_type"])],
                abs(row[2] / size - 0.5),
                digest([21, row[0].task_id]),
            ),
        )
        mixed.remove(candidate)
        task, oracle, _ = candidate
        classes[str(oracle.provenance["bug_type"])] += 1
        curriculum.append((task, oracle))
    unanimous = sorted(
        [row for row in candidates if row[2] in {0, size}],
        key=lambda row: (
            abs(sum(r["success"] for r in groups[row[0].task_id]) / 8 - 0.5),
            digest([21, row[0].task_id]),
        ),
    )
    for outcome in (0, size):
        for task, oracle, _ in [row for row in unanimous if row[2] == outcome][:2]:
            curriculum.append((task, oracle))
    partitions = {}
    for name in repo.partition_names():
        if name != "screen":
            partitions[name] = [
                (task, oracle) for task, oracle in repo.tasks(name) if isinstance(oracle, PdbOracle)
            ]
    partitions["solver_train"] = curriculum
    curriculum_ids = {task.task_id for task, _ in curriculum}
    partitions["proposer_train"] = [
        (task, oracle) for task, oracle, _ in candidates if task.task_id not in curriculum_ids
    ]
    if not partitions.get("solver_probe") or not partitions.get("dev_monitor"):
        raise ValueError("Reserve independent probe and development parents before solver training")
    selection = {
        "size": size,
        "efficiency": efficiency,
        "allocated_h200_seconds": args.h200_seconds,
        "samples_sha256": file_hash(args.samples),
        "screen_marker_sha256": file_hash(args.marker),
        "screen_manifest": sample["data"],
        "selection": [task.task_id for task, _ in curriculum],
    }
    write_bundle(args.output, partitions, bundle["provenance"] | {"difficulty_screen": selection})
    control = Task21Control.model_validate_json(args.control.read_text())
    control = Task21Control.model_validate(control.model_dump() | {"group_size": size})
    atomic_json(args.output_control, control.model_dump(mode="json"))
    atomic_json(args.report, selection)


def freeze_endpoint(args) -> None:
    control = Task21Control.model_validate_json(args.control.read_text())
    calibration = read_json(args.calibration)
    solver = read_json(args.solver)
    pilot = read_json(args.pilot)
    if not solver["result"]["learnable"] or not calibration["result"]["ready"]:
        raise ValueError("Task 21 has no positive solver and usable calibration evidence")
    if args.mode == "shared" and not pilot["result"]["viable"]:
        raise ValueError("Shared endpoint requires verified proposer and solver signal")
    if args.mode == "fixed" and pilot["result"]["viable"]:
        raise ValueError("A viable shared pilot takes precedence")
    if min(args.sec_per_episode, args.eval_seconds, args.available_gpu_seconds) <= 0:
        raise ValueError("Supply positive measured costs and reconciled capacity")
    if not 0 < args.effective_updates_per_episode <= 1:
        raise ValueError("Supply the observed effective-update rate")
    capacity = args.available_gpu_seconds - 1.25 * args.eval_seconds
    granularity = 4 * control.group_size
    maximum = min(1024, math.floor(capacity / (3 * args.sec_per_episode)))
    endpoint = maximum // granularity * granularity
    if endpoint < 512 or endpoint * args.effective_updates_per_episode < 12:
        raise ValueError("Measured capacity does not cover a full informative matched triad")
    result = calibration["result"]
    if result["mode"] != args.mode:
        raise ValueError("Calibration does not match the comparison's policy mode")
    updates = {
        "main_mode": args.mode,
        "endpoint": endpoint,
        "dev_milestones": tuple(endpoint * n // 4 for n in (1, 2, 3)),
        "sd_scale": result["sd_scale"],
        "proposer_scale": result["proposer_scale"],
        "calibration_hash": file_hash(args.calibration),
        "solver_gate_hash": file_hash(args.solver),
        "shared_gate_hash": file_hash(args.pilot),
    }
    final = Task21Control.model_validate(control.model_dump() | updates)
    atomic_json(args.output_control, final.model_dump(mode="json"))
    atomic_json(
        args.report,
        {
            "science": final.science,
            "mode": args.mode,
            "endpoint": endpoint,
            "estimated_train_gpu_seconds": 3 * endpoint * args.sec_per_episode,
            "reserved_evaluation_gpu_seconds": 1.25 * args.eval_seconds,
            "available_gpu_seconds": args.available_gpu_seconds,
            "expected_effective_updates": endpoint * args.effective_updates_per_episode,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    curriculum = stages.add_parser("curriculum")
    for name in ("data", "samples", "marker", "output"):
        curriculum.add_argument(f"--{name}", type=Path, required=True)
    curriculum.add_argument("--h200-seconds", type=float, required=True)
    endpoint = stages.add_parser("endpoint")
    endpoint.add_argument("--mode", choices=("shared", "fixed"), required=True)
    for name in ("calibration", "solver", "pilot"):
        endpoint.add_argument(f"--{name}", type=Path, required=True)
    for name in (
        "sec-per-episode",
        "eval-seconds",
        "available-gpu-seconds",
        "effective-updates-per-episode",
    ):
        endpoint.add_argument(f"--{name}", type=float, required=True)
    for stage in (curriculum, endpoint):
        for name in ("control", "output-control", "report"):
            stage.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    (freeze_curriculum if args.stage == "curriculum" else freeze_endpoint)(args)


if __name__ == "__main__":
    main()
