"""Summarize isolated mechanical-recovery evidence, never agent counterfactual accuracy."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from selfplay_opsd.analysis import write_csv
from selfplay_opsd.core import digest
from selfplay_opsd.storage import atomic_json, file_hash


def summarize(candidates, replays, evaluations, actions):
    inputs = {(r["episode_id"], r["turn"]): r for r in candidates}
    outputs = {(r["episode_id"], r["turn"]): r for r in replays}
    if (
        len(inputs) != len(candidates)
        or len(outputs) != len(replays)
        or inputs.keys() != outputs.keys()
    ):
        raise ValueError("Replay keys incomplete or duplicated")
    original = {(r["method"], str(r["seed"]), r["task_id"]): r for r in evaluations}
    seen, joined = {}, []
    for key, result in outputs.items():
        candidate = inputs[key]
        if (
            result["source_hash"] != candidate["source_hash"]
            or digest(candidate["source"]) != result["source_hash"]
        ):
            raise ValueError("Replayed source identity differs")
        if result["status"] in {
            "sandbox_infrastructure_error",
            "missing_dependency",
            "data_oracle_failure",
        }:
            raise ValueError("Infrastructure/oracle failure is not a repair outcome")
        logical = (candidate["method"], str(candidate["seed"]), candidate["task_id"])
        if logical not in original:
            raise ValueError("Missing original evaluation")
        source_key = (candidate["task_id"], candidate["source_hash"])
        outcome = tuple(
            result[k] for k in ("passed", "status", "semantic_failures", "runtime_failures")
        )
        if source_key in seen and seen[source_key] != outcome:
            raise ValueError("Cached source verification outcomes disagree")
        if result["reused_identical_source"] != (source_key in seen):
            raise ValueError("Verifier-call provenance differs")
        seen[source_key] = outcome
        joined.append(
            candidate | result | {"original_success": original[logical]["success"] == "True"}
        )
    first_edit = {}
    for action in actions:
        if action["action"] == "apply_patch":
            identifier = action["episode_id"]
            first_edit[identifier] = min(first_edit.get(identifier, 99), int(action["turn"]))
    groups = []
    for kind in ("base", "trained"):
        for split in ("final_id", "external"):
            selected = [
                r
                for r in joined
                if (r["method"] == "base") == (kind == "base") and r["split"] == split
            ]
            baseline = [
                r
                for r in evaluations
                if (r["method"] == "base") == (kind == "base") and r["split"] == split
            ]
            rejected = [
                r
                for r in actions
                if (r["method"] == "base") == (kind == "base")
                and r["split"] == split
                and r["action"] == "apply_patch"
                and r["parse_status"] == "tool_error"
            ]
            passed = [r for r in selected if r["passed"]]
            failed_passing = [r for r in passed if not r["original_success"]]
            earliest = {}
            for row in failed_passing:
                identifier = row["episode_id"]
                earliest[identifier] = min(earliest.get(identifier, 99), int(row["turn"]) + 1)
            groups.append(
                {
                    "kind": kind,
                    "failed_episode_earliest_correct_reconstruction_turn": {
                        "one_based_counts": dict(sorted(Counter(earliest.values()).items())),
                        "median": median(earliest.values()) if earliest else None,
                    },
                    "failed_episodes_correct_on_first_schema_valid_edit": len(
                        {
                            r["episode_id"]
                            for r in failed_passing
                            if int(r["turn"]) == first_edit[r["episode_id"]]
                        }
                    ),
                    "split": split,
                    "original_episodes": len(baseline),
                    "original_successes": sum(r["success"] == "True" for r in baseline),
                    "rejected_actions": len(rejected),
                    "admitted_reconstructions": len(selected),
                    "passing_reconstructed_actions": len(passed),
                    "unique_reconstructed_programs": len(
                        {(r["task_id"], r["source_hash"]) for r in selected}
                    ),
                    "passing_unique_programs": len(
                        {(r["task_id"], r["source_hash"]) for r in passed}
                    ),
                    "episodes_with_passing_state": len({r["episode_id"] for r in passed}),
                    "originally_failed_episodes_with_passing_state": len(
                        {r["episode_id"] for r in passed if not r["original_success"]}
                    ),
                    "parents_with_passing_state": len({r["parent_id"] for r in passed}),
                    "runtime_failure_actions": sum(bool(r["runtime_failures"]) for r in selected),
                }
            )
    per_model = defaultdict(set)
    for row in joined:
        if row["passed"]:
            per_model[(row["method"], row["seed"], row["split"])].add(row["task_id"])
    return {
        "actions": len(joined),
        "unique_verifier_calls": len(seen),
        "passing_unique_programs": sum(bool(value[0]) for value in seen.values()),
        "groups": groups,
        "per_model_correct_state_tasks": [
            {"method": k[0], "seed": k[1], "split": k[2], "tasks": len(v)}
            for k, v in sorted(per_model.items())
        ],
        "interpretation": "Correct states at logged prefixes; not measured end-to-end agent accuracy",
    }, joined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figures", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve prior analysis; output exists")
    candidate_path = args.audit / "replay-candidates.json"
    complete = json.loads((args.replay / "complete.json").read_text())
    if complete["candidates_sha256"] != file_hash(candidate_path):
        raise ValueError("Replay input checksum differs")
    candidates = json.loads(candidate_path.read_text())
    replays = json.loads((args.replay / "replays.json").read_text())
    if complete["verified"] != len(candidates):
        raise ValueError("Replay incomplete")
    with (args.audit / "episodes.csv").open() as stream:
        evaluations = list(csv.DictReader(stream))
    with (args.audit / "actions.csv").open() as stream:
        actions = list(csv.DictReader(stream))
    summary, joined = summarize(candidates, replays, evaluations, actions)
    atomic_json(args.output / "summary.json", summary)
    write_csv(
        args.output / "verified-reconstructions.csv",
        [{k: v for k, v in r.items() if k != "source"} for r in joined],
    )
    paths = [
        candidate_path,
        args.audit / "episodes.csv",
        args.audit / "actions.csv",
        args.replay / "complete.json",
        args.replay / "replays.json",
        Path(__file__),
    ]
    atomic_json(
        args.output / "input-manifest.json", {str(p.resolve()): file_hash(p) for p in paths}
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = args.figures or args.output / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    for ax, split in zip(axes, ("final_id", "external"), strict=True):
        group = next(r for r in summary["groups"] if r["kind"] == "trained" and r["split"] == split)
        values = [
            group[k]
            for k in (
                "rejected_actions",
                "admitted_reconstructions",
                "passing_reconstructed_actions",
            )
        ]
        bars = ax.bar(
            ["Rejected", "Unique-span\nreconstruction", "Pass full\nverification"],
            values,
            color=["#8B929B", "#355C9A", "#34835C"],
        )
        ax.bar_label(bars)
        ax.set(
            title="MBPP+-derived" if split == "final_id" else "HumanEval+-derived",
            ylabel="Actions across six trained snapshots",
            ylim=(0, max(values) * 1.13),
        )
    fig.suptitle("Mechanical recovery at recorded states; not new agent accuracy")
    for suffix in ("png", "pdf"):
        fig.savefig(destination / f"mechanical-recovery-verification.{suffix}", bbox_inches="tight")
    plt.close(fig)
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k not in {"groups", "per_model_correct_state_tasks"}
            }
        )
    )


if __name__ == "__main__":
    main()
