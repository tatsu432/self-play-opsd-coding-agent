"""Freeze the common Task 20 endpoint from authorized remaining allocation and pilot evidence."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from hpc_remote import task20_solver_evidence

from selfplay_opsd.core import read_json
from selfplay_opsd.storage import CheckpointManager, atomic_json, file_hash
from selfplay_opsd.task20 import Task20Control


def evidence(root: Path, stage: str) -> tuple[Path, dict]:
    path = root / f"complete-{stage}.json"
    marker = read_json(path)
    state = root / "checkpoints" / marker["checkpoint"]
    if file_hash(state / "complete.json") != marker["checkpoint_sha256"]:
        raise ValueError(f"{stage} completion checkpoint checksum differs")
    metadata = CheckpointManager.verify(state)
    if metadata["identity"] != marker["identity"] or metadata["snapshot"] != marker["snapshot"]:
        raise ValueError(f"{stage} completion checkpoint identity differs")
    return path, marker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, default=Path("configs/task20-yale.json"))
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--shared-pilot", type=Path, required=True)
    parser.add_argument("--remaining-hours", type=float, required=True)
    parser.add_argument("--evaluation-hours", type=float, required=True)
    parser.add_argument("--opsd-seconds-per-episode", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = Task20Control.read(args.settings)
    if args.output.exists():
        raise ValueError("Frozen main control is immutable; choose a separate settings path")
    solver, solver_record = evidence(args.solver, "solver")
    shared, shared_record = evidence(args.shared_pilot, "shared-pilot")
    calibration, calibration_record = evidence(args.calibration, "calibrate")
    records = (solver_record, shared_record, calibration_record)
    identity = solver_record["identity"]
    if not all(
        record["identity"][key] == identity[key]
        for record in records
        for key in ("study", "initial", "data", "worker", "execution")
    ):
        raise ValueError("Task 20 pilot identities differ")
    if cfg.study != identity["study"] or solver_record["identity"]["curriculum"] != "fixed":
        raise ValueError("Expected the configured solver-only run")
    if not task20_solver_evidence(solver_record, args.solver):
        raise ValueError("Solver-only learnability gate failed")
    scales = calibration_record["result"]
    if not scales["ready"] or shared_record["identity"]["curriculum"] != "shared":
        raise ValueError("Calibration or shared-policy mode failed")
    rows = [
        row["metrics"]
        for path in sorted((args.shared_pilot / "iterations").glob("*.json"))
        if (row := read_json(path))["batch"] is not None
    ]
    episodes = shared_record["episodes"]
    updates = shared_record["effective_rl_updates"]
    if (
        episodes != cfg.shared_pilot_episodes
        or updates < 2
        or not sum(row["proposer"]["distinct_reward_groups"] for row in rows)
        or not all(math.isfinite(row["shared"]["update_norm"]) for row in rows)
    ):
        raise ValueError("Shared pilot did not establish enough stable and discriminative updates")
    if not all(
        math.isfinite(value) and value > 0
        for value in (
            args.remaining_hours,
            args.evaluation_hours,
            args.opsd_seconds_per_episode,
        )
    ):
        raise ValueError(
            "Remaining allocation, evaluation reserve, and OPSD timing must be positive"
        )
    seconds_per_episode = sum(row["timing"]["total"] for row in rows) / episodes
    # Main consists of one control and two OPSD arms. Apply a 20% reserve to
    # measured training and evaluation before choosing a common group boundary.
    seconds_per_triad_episode = 3 * seconds_per_episode + 2 * args.opsd_seconds_per_episode
    raw = (args.remaining_hours / 1.2 - args.evaluation_hours) * 3600 / seconds_per_triad_episode
    endpoint = math.floor(raw / (4 * cfg.group_size)) * (4 * cfg.group_size)
    if endpoint < episodes or endpoint * updates / episodes < 12:
        raise ValueError(
            "Remaining allocation cannot support the minimum informative shared exposure"
        )
    if not cfg.qualification_hash:
        raise ValueError("Bind the retained or targeted execution qualification")
    frozen = Task20Control.model_validate(
        cfg.model_dump()
        | {
            "endpoint": endpoint,
            "dev_milestones": tuple(endpoint * frac // 4 for frac in (1, 2, 3)),
            "sd_scale": scales["sd_scale"],
            "proposer_scale": scales["proposer_scale"],
            "calibration_hash": file_hash(calibration),
            "solver_gate_hash": file_hash(solver),
            "shared_gate_hash": file_hash(shared),
        }
    )
    atomic_json(args.output, frozen.model_dump(mode="json"))
    print(
        f"Endpoint {endpoint}, projected effective updates {endpoint * updates / episodes:.1f}, "
        f"evaluation reserve {args.evaluation_hours:.2f} H200-hours"
    )


if __name__ == "__main__":
    main()
