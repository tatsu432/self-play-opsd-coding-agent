"""Compare paired training-only student editing probes."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from selfplay_opsd.storage import atomic_json, file_hash


def load(path: Path):
    manifest = json.loads((path / "manifest.json").read_text())
    episodes = json.loads((path / "episodes.json").read_text())
    complete = json.loads((path / "complete.json").read_text())
    if (
        len(episodes) != complete["episodes"]
        or len(episodes) != len(manifest["tasks"]) * manifest["slots"]
    ):
        raise ValueError("Incomplete student probe")
    if sum(e["reward"] for e in episodes) != complete["successes"]:
        raise ValueError("Student probe reward count differs")
    keys = [(e["task_id"], e["seed"]) for e in episodes]
    if len(set(keys)) != len(keys):
        raise ValueError("Repeated student rollout")
    return manifest, {key: e for key, e in zip(keys, episodes, strict=True)}


def summarize(episodes):
    rows = list(episodes.values())
    edits = [
        t
        for e in rows
        for t in e["turns"]
        if t["action"] in {"apply_patch", "replace_function_body"}
    ]
    return {
        "episodes": len(rows),
        "successes": sum(e["reward"] for e in rows),
        "edit_attempts": len(edits),
        "valid_edits": sum(t["parse_status"] == "valid" for t in edits),
        "parse_status": dict(Counter(t["parse_status"] for e in rows for t in e["turns"])),
        "generated_tokens": sum(len(t["target_ids"]) for e in rows for t in e["turns"]),
    }


def compare(first, second):
    if set(first) != set(second):
        raise ValueError("Student probes do not have matching tasks and rollout seeds")
    by_task = defaultdict(list)
    for key in first:
        by_task[key[0]].append(second[key]["reward"] - first[key]["reward"])
    deltas = np.array([np.mean(by_task[k]) for k in sorted(by_task)])
    rng = np.random.default_rng(2026)
    bootstrap = rng.choice(deltas, size=(2000, len(deltas)), replace=True).mean(axis=1)
    return {
        "mean_success_difference": float(deltas.mean()),
        "paired_parent_bootstrap_95_interval": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diff", type=Path, required=True)
    parser.add_argument("--body", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    first_manifest, first = load(args.diff)
    second_manifest, second = load(args.body)
    for field in ("dataset", "pool_sha256", "tasks", "snapshot", "slots", "seed_key"):
        if first_manifest[field] != second_manifest[field]:
            raise ValueError(f"Paired probe {field} differs")
    if (
        first_manifest["protocol"] != "unified_diff"
        or second_manifest["protocol"] != "replace_function_body"
    ):
        raise ValueError("Unexpected student protocols")
    atomic_json(
        args.output,
        {
            "diff": summarize(first),
            "body": summarize(second),
            "paired": compare(first, second),
            "inputs": {str(p): file_hash(p / "episodes.json") for p in (args.diff, args.body)},
        },
    )


if __name__ == "__main__":
    main()
