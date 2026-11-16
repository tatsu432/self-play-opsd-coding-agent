"""Recover hidden-training reward bounds from saved trusted verifier counts only."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from selfplay_opsd.storage import atomic_json, file_hash


def hidden_fraction(episode, public, hidden, initial_hash):
    if public < 1 or hidden < 1:
        raise ValueError("Nonempty public and hidden training suites required")
    failed = episode["semantic_failures"] + episode["runtime_failures"]
    if not 0 <= failed <= public + hidden or episode["reward"] not in {0, 1}:
        raise ValueError("Invalid saved verifier counts")
    if bool(episode["reward"]) != (episode["public_pass"] and episode["hidden_pass"]):
        raise ValueError("Saved binary reward contradicts complete repair")
    public_low, public_high = (0, 0) if episode["public_pass"] else (1, public)
    current_hash = initial_hash
    for turn in episode["turns"]:
        observation = turn.get("observation") or {}
        if turn["parse_status"] != "valid":
            continue
        if turn["action"] in {"replace_text", "apply_patch", "replace_function_body"}:
            current_hash = observation.get("source_hash")
        elif (
            turn["action"] == "run_tests"
            and current_hash == episode["final_source_hash"]
            and observation.get("total") == public
            and episode["runtime_failures"] == 0
        ):
            passed = observation["passed"]
            if not 0 <= passed <= public or bool(passed == public) != episode["public_pass"]:
                raise ValueError("Same-source public outcomes disagree")
            public_low = public_high = public - passed
    low = max(0 if episode["hidden_pass"] else 1, failed - public_high)
    high = min(0 if episode["hidden_pass"] else hidden, failed - public_low)
    if not 0 <= low <= high <= hidden:
        raise ValueError("Saved public/hidden outcomes contradict failure counts")
    return {"lower": 1 - high / hidden, "upper": 1 - low / hidden, "exact": low == high}


def analyze(pool_path, source):
    pool = json.loads(pool_path.read_text())
    info = {r["task"]["task_id"]: r for r in pool["rows"] if r["purpose"] == "curriculum"}
    records = []
    hashes = {str(pool_path): file_hash(pool_path)}
    for path in sorted(source.glob("trajectories/*/*-episodes.json")):
        hashes[str(path)] = file_hash(path)
        for episode in json.loads(path.read_text()):
            row = info[episode["task_id"]]
            if episode.get("verifier_status") in {
                "sandbox_infrastructure_error",
                "missing_dependency",
                "data_oracle_failure",
            }:
                raise ValueError("Infrastructure outcomes are not partial rewards")
            bounds = hidden_fraction(
                episode,
                len(row["oracle"]["public_cases"]),
                len(row["oracle"]["hidden_cases"]),
                row["task"]["source_hash"],
            )
            records.append(
                {
                    "task_id": episode["task_id"],
                    "snapshot": episode["policy_version"],
                    "binary": episode["reward"],
                    "initial_source": episode["final_source_hash"] == row["task"]["source_hash"],
                    **bounds,
                }
            )
    if not records:
        raise ValueError("No saved training trajectories")
    groups = defaultdict(list)
    for row in records:
        groups[(row["task_id"], row["snapshot"])].append(row)
    metrics = []
    for group in groups.values():
        if len(group) != 4:
            raise ValueError("Expected complete four-rollout task groups")
        binary_variance = float(np.var([r["binary"] for r in group]))
        guaranteed = max(r["lower"] for r in group) > min(r["upper"] for r in group) + 1e-12
        exact = all(r["exact"] for r in group)
        metrics.append(
            {
                "binary_variance": binary_variance,
                "guaranteed_partial_variation": guaranteed,
                "possibly_partial_variation": max(r["upper"] for r in group)
                > min(r["lower"] for r in group) + 1e-12,
                "exact_partial_variance": float(np.var([r["lower"] for r in group]))
                if exact
                else None,
            }
        )
    baselines = {r["task_id"]: r for r in records if r["initial_source"]}
    improvements = [
        {
            "lower": r["lower"] - baselines[r["task_id"]]["upper"],
            "upper": r["upper"] - baselines[r["task_id"]]["lower"],
        }
        for r in records
        if r["task_id"] in baselines
    ]
    return {
        "kind": "retrospective hidden-training reward bounds; no program execution or new reward feedback",
        "episodes": len(records),
        "groups": len(groups),
        "exact_hidden_fractions": sum(r["exact"] for r in records),
        "binary_informative_groups": sum(r["binary_variance"] > 0 for r in metrics),
        "guaranteed_partial_informative_groups": sum(
            r["guaranteed_partial_variation"] for r in metrics
        ),
        "possibly_partial_informative_groups": sum(
            r["possibly_partial_variation"] for r in metrics
        ),
        "additional_guaranteed_informative_groups": sum(
            r["binary_variance"] == 0 and r["guaranteed_partial_variation"] for r in metrics
        ),
        "observed_initial_mutant_baselines": len(baselines),
        "episodes_with_initial_baseline": len(improvements),
        "guaranteed_improvements_over_initial": sum(r["lower"] > 1e-12 for r in improvements),
        "guaranteed_regressions_from_initial": sum(r["upper"] < -1e-12 for r in improvements),
        "group_metrics": metrics,
        "source_hashes": hashes,
        "new_allocated_gpu_hours": 0,
        "limitations": "Bounds retain unknown public-case allocation of failures. Concurrent repair association is not eventual generalization or causal learning benefit. Missing initial-source baselines remain unknown.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve prior analysis")
    atomic_json(args.output / "summary.json", analyze(args.pool, args.source))


if __name__ == "__main__":
    main()
