"""Compare saved Task 13 probes by parent without executing candidate code."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from selfplay_opsd.storage import atomic_json, file_hash


def load_probe(path: Path) -> dict:
    manifest = json.loads((path / "manifest.json").read_text())
    rows = json.loads((path / "proposals.json").read_text())
    complete = json.loads((path / "complete.json").read_text())
    parents = {p["id"]: p["source_hash"] for p in manifest["parents"]}
    if len(parents) != len(manifest["parents"]) or len(rows) != complete["count"]:
        raise ValueError("Incomplete probe or duplicate parent")
    if len(rows) != len(parents) * manifest["slots"]:
        raise ValueError("Probe did not complete all planned samples")
    if sum(r["valid"] for r in rows) != complete["valid"]:
        raise ValueError("Completion record differs from proposals")
    by_parent = defaultdict(list)
    for row in rows:
        if row["parent_id"] not in parents:
            raise ValueError("Proposal parent missing from manifest")
        by_parent[row["parent_id"]].append(row)
    if any(
        sorted(r["slot"] for r in values) != list(range(manifest["slots"]))
        for values in by_parent.values()
    ):
        raise ValueError("Probe slots are missing or repeated")
    return {
        "manifest": manifest,
        "rows": rows,
        "parents": parents,
        "by_parent": dict(by_parent),
        "input_sha256": {
            f: file_hash(path / f) for f in ("manifest.json", "proposals.json", "complete.json")
        },
    }


def summary(probe: dict) -> dict:
    rows = probe["rows"]
    return {
        "attempts": len(rows),
        "verified": sum(r["valid"] for r in rows),
        "unique_verified_parent_mutations": len(
            {(r["parent_id"], r["dedup_key"]) for r in rows if r["valid"]}
        ),
        "verified_in_checkpoint_history": sum(
            r["valid"] and r.get("in_training_history", False) for r in rows
        ),
        "distinct_output_sequences": len({tuple(r["target_ids"]) for r in rows}),
        "reasons": dict(Counter(r["reason"] for r in rows)),
    }


def compare(first: dict, second: dict, *, resamples: int = 10000) -> dict:
    if first["parents"] != second["parents"]:
        raise ValueError("Probe parent identities or source hashes do not match")
    if first["manifest"]["model"] != second["manifest"]["model"]:
        raise ValueError("Model identity differs; use a separately labelled comparison")
    if first["manifest"]["slots"] != second["manifest"]["slots"]:
        raise ValueError("Probe sampling budgets differ")
    if first["manifest"]["manifest_hash"] != second["manifest"]["manifest_hash"]:
        raise ValueError("Probe datasets differ")
    for parent in first["parents"]:
        first_seeds = {r["slot"]: r["seed"] for r in first["by_parent"][parent]}
        second_seeds = {r["slot"]: r["seed"] for r in second["by_parent"][parent]}
        if first_seeds != second_seeds:
            raise ValueError("Probe sampling seeds differ")
    ids = sorted(first["parents"])
    count = first["manifest"]["slots"]
    result = {"parents": len(ids), "samples_per_parent": count, "differences": {}}
    rng = np.random.default_rng(2026)
    for name in ("verified_fraction", "unique_mutations_per_parent"):

        def value(rows, metric=name):
            return (
                sum(r["valid"] for r in rows) / count
                if metric == "verified_fraction"
                else len({r["dedup_key"] for r in rows if r["valid"]})
            )

        delta = np.array(
            [value(second["by_parent"][p]) - value(first["by_parent"][p]) for p in ids]
        )
        distribution = rng.choice(delta, size=(resamples, len(ids)), replace=True).mean(axis=1)
        result["differences"][name] = {
            "mean": float(delta.mean()),
            "parent_bootstrap_95_interval": np.quantile(distribution, [0.025, 0.975]).tolist(),
            "improved_parents": int((delta > 0).sum()),
            "worse_parents": int((delta < 0).sum()),
            "unchanged_parents": int((delta == 0).sum()),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diff", type=Path, required=True)
    parser.add_argument("--structured", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    first, second = load_probe(args.diff), load_probe(args.structured)
    if (
        first["manifest"]["representation"] != "diff"
        or second["manifest"]["representation"] != "structured"
    ):
        raise ValueError("Probe representations are reversed or unknown")
    result = {
        "diff": summary(first),
        "structured": summary(second),
        "paired": compare(first, second),
        "input_sha256": {"diff": first["input_sha256"], "structured": second["input_sha256"]},
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
