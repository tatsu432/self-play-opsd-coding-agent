"""Compare saved Task 13 probes by parent without executing candidate code."""

from __future__ import annotations

import argparse
import copy
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
        sorted(r["slot"] for r in values)
        != list(
            range(manifest.get("slot_start", 0), manifest.get("slot_start", 0) + manifest["slots"])
        )
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
        "unique_verified_in_checkpoint_history": len(
            {
                (r["parent_id"], r["dedup_key"])
                for r in rows
                if r["valid"] and r.get("in_training_history", False)
            }
        ),
        "unique_verified_not_in_checkpoint_history": len(
            {
                (r["parent_id"], r["dedup_key"])
                for r in rows
                if r["valid"] and not r.get("in_training_history", False)
            }
        ),
        "distinct_output_sequences": len({tuple(r["target_ids"]) for r in rows}),
        "reasons": dict(Counter(r["reason"] for r in rows)),
    }


def combine_samples(base: dict, extra: dict) -> dict:
    """Combine the prespecified first 16 parents' disjoint sampling slots."""
    if (
        extra["manifest"].get("slot_start")
        != base["manifest"].get("slot_start", 0) + base["manifest"]["slots"]
    ):
        raise ValueError("Additional sample slots are not contiguous")
    for field in ("model", "manifest_hash", "representation", "checkpoint"):
        if base["manifest"][field] != extra["manifest"][field]:
            raise ValueError(f"Additional sample {field} differs")
    if list(base["parents"])[: len(extra["parents"])] != list(extra["parents"]):
        raise ValueError("Additional samples do not use the prespecified parent prefix")
    if any(base["parents"][p] != source_hash for p, source_hash in extra["parents"].items()):
        raise ValueError("Additional sample parent sources differ")
    parents = extra["parents"]
    by_parent = {p: base["by_parent"][p] + extra["by_parent"][p] for p in parents}
    manifest = dict(base["manifest"])
    manifest["slots"] += extra["manifest"]["slots"]
    return {
        "manifest": manifest,
        "parents": parents,
        "by_parent": by_parent,
        "rows": [r for p in parents for r in by_parent[p]],
        "input_sha256": {"base": base["input_sha256"], "extra": extra["input_sha256"]},
    }


def apply_header_replay(probe: dict, path: Path) -> dict:
    """Overlay checksum-bound worker verification without changing original results."""
    complete = json.loads((path / "complete.json").read_text())
    rows = json.loads((path / "results.json").read_text())
    if (
        complete["source_hashes"] != probe["input_sha256"]
        or complete["count"] != len(rows)
        or complete["valid"] != sum(row["valid"] for row in rows)
        or complete["change"] != "remove exactly one leading diff --git header"
    ):
        raise ValueError("Replay identity, transformation or completeness differs")
    if complete["unique"] != len({(r["parent_id"], r["dedup_key"]) for r in rows if r["valid"]}):
        raise ValueError("Replay uniqueness differs from completion record")
    result = copy.deepcopy(probe)
    original = {(r["parent_id"], r["slot"]): r for r in result["rows"]}
    seen = set()
    for row in rows:
        key = (row["parent_id"], row["slot"])
        if key in seen or key not in original or original[key]["valid"]:
            raise ValueError("Replay references an absent, repeated or already-valid sample")
        seen.add(key)
        if row["valid"]:
            original[key].update(
                valid=True, dedup_key=row["dedup_key"], reason="header replay valid"
            )
    result["input_sha256"] = {
        "original": probe["input_sha256"],
        "replay": {name: file_hash(path / name) for name in ("complete.json", "results.json")},
    }
    return result


def compare(
    first: dict, second: dict, *, resamples: int = 10000, allow_model_difference: bool = False
) -> dict:
    if first["parents"] != second["parents"]:
        raise ValueError("Probe parent identities or source hashes do not match")
    if first["manifest"]["model"] != second["manifest"]["model"] and not allow_model_difference:
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
    for name in (
        "verified_fraction",
        "unique_mutations_per_parent",
        "novel_verified_fraction",
        "unique_novel_mutations_per_parent",
    ):

        def value(rows, metric=name):
            novel = "novel" in metric
            valid = [
                r
                for r in rows
                if r["valid"] and (not novel or not r.get("in_training_history", False))
            ]
            return (
                len(valid) / count
                if metric.endswith("fraction")
                else len({r["dedup_key"] for r in valid})
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
    parser.add_argument("--diff-extra", type=Path)
    parser.add_argument("--structured-extra", type=Path)
    parser.add_argument("--second-header-replay", type=Path)
    parser.add_argument(
        "--allow-model-difference",
        action="store_true",
        help="Explicitly label a paired cross-model comparison",
    )
    parser.add_argument(
        "--allow-same-format",
        action="store_true",
        help="Compare the same representation across models",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    first, second = load_probe(args.diff), load_probe(args.structured)
    if not args.allow_same_format and (
        first["manifest"]["representation"] != "diff"
        or second["manifest"]["representation"] != "structured"
    ):
        raise ValueError("Probe representations are reversed or unknown")
    result = {
        "first": summary(first),
        "second": summary(second),
        "paired": compare(first, second, allow_model_difference=args.allow_model_difference),
        "comparison_identities": {
            "first": {
                k: first["manifest"].get(k)
                for k in (
                    "model",
                    "representation",
                    "inference_backend",
                    "adapter",
                    "chat_template_non_thinking",
                )
            },
            "second": {
                k: second["manifest"].get(k)
                for k in (
                    "model",
                    "representation",
                    "inference_backend",
                    "adapter",
                    "chat_template_non_thinking",
                )
            },
        },
        "input_sha256": {"first": first["input_sha256"], "second": second["input_sha256"]},
    }
    if args.second_header_replay:
        normalized = apply_header_replay(second, args.second_header_replay)
        result["posthoc_header_replay"] = {
            "second": summary(normalized),
            "paired": compare(
                first, normalized, allow_model_difference=args.allow_model_difference
            ),
            "input_sha256": normalized["input_sha256"],
        }
    if bool(args.diff_extra) != bool(args.structured_extra):
        parser.error("Provide both additional-sample directories")
    if args.diff_extra and args.structured_extra:
        diff_eight = combine_samples(first, load_probe(args.diff_extra))
        structured_eight = combine_samples(second, load_probe(args.structured_extra))
        result["eight_sample_subset"] = {
            "first": summary(diff_eight),
            "second": summary(structured_eight),
            "paired": compare(
                diff_eight, structured_eight, allow_model_difference=args.allow_model_difference
            ),
            "input_sha256": {
                "first": diff_eight["input_sha256"],
                "second": structured_eight["input_sha256"],
            },
        }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
