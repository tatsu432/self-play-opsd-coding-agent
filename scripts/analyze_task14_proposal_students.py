"""Summarize student mixed-reward groups from verified proposal probes."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from selfplay_opsd.storage import atomic_json, file_hash


def load(path):
    manifest = json.loads((path / "manifest.json").read_text())
    groups = json.loads((path / "groups.json").read_text())
    complete = json.loads((path / "complete.json").read_text())
    if len(groups) != complete["groups"] or len(groups) != len(manifest["selected"]):
        raise ValueError("Incomplete proposal-student probe")
    keys = [(g["parent_id"], g["dedup_key"]) for g in groups]
    selected = [(g["parent_id"], g["dedup_key"]) for g in manifest["selected"]]
    if (
        keys != selected
        or len(set(keys)) != len(keys)
        or any(len(g["episodes"]) != 4 for g in groups)
    ):
        raise ValueError("Proposal-student task selection differs")
    mixed = [0 < sum(e["reward"] for e in g["episodes"]) < 4 for g in groups]
    if sum(mixed) != complete["mixed"]:
        raise ValueError("Mixed-group count differs")
    return manifest, groups


def summarize(groups):
    mixed = sum(0 < sum(e["reward"] for e in g["episodes"]) < 4 for g in groups)
    return {
        "tasks": len(groups),
        "parents": len({g["parent_id"] for g in groups}),
        "mixed_groups": mixed,
        "mixed_fraction": mixed / len(groups) if groups else None,
        "episodes": sum(len(g["episodes"]) for g in groups),
        "successful_episodes": sum(e["reward"] for g in groups for e in g["episodes"]),
    }


def compare(first, second):
    by_parent = []
    for groups in (first, second):
        values = defaultdict(list)
        for group in groups:
            values[group["parent_id"]].append(
                int(0 < sum(e["reward"] for e in group["episodes"]) < 4)
            )
        by_parent.append(values)
    common = sorted(by_parent[0].keys() & by_parent[1].keys())
    if not common:
        return {"shared_parents": 0, "mean_difference": None, "bootstrap_95_interval": None}
    delta = np.array([np.mean(by_parent[1][p]) - np.mean(by_parent[0][p]) for p in common])
    bootstrap = (
        np.random.default_rng(2026)
        .choice(delta, size=(2000, len(delta)), replace=True)
        .mean(axis=1)
    )
    return {
        "shared_parents": len(common),
        "mean_difference": float(delta.mean()),
        "bootstrap_95_interval": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diff", type=Path, required=True)
    parser.add_argument("--structured", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    first_manifest, first = load(args.diff)
    second_manifest, second = load(args.structured)
    if (
        first_manifest["representation"] != "diff"
        or second_manifest["representation"] != "structured"
    ):
        raise ValueError("Unexpected representation")
    if any(first_manifest[key] != second_manifest[key] for key in ("snapshot", "student_protocol")):
        raise ValueError("Student snapshot or protocol differs")
    atomic_json(
        args.output,
        {
            "diff": summarize(first),
            "structured": summarize(second),
            "paired_shared_parents": compare(first, second),
            "inputs": {
                str(path): file_hash(path / "groups.json") for path in (args.diff, args.structured)
            },
        },
    )


if __name__ == "__main__":
    main()
