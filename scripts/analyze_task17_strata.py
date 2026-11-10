"""Descriptive strata of the immutable Task 17 retrospective audit; no execution."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from selfplay_opsd.analysis import write_csv
from selfplay_opsd.storage import atomic_json, file_hash


def summarize(rows):
    edits = [r for r in rows if r["action"] == "apply_patch"]
    rejected = [r for r in edits if r["parse_status"] == "tool_error"]
    return dict(
        actions=len(rows),
        edit_attempts=len(edits),
        accepted_edits=sum(r["category"] == "accepted_edit" for r in edits),
        rejected_edits=len(rejected),
        invalid_json_schema=sum(r["category"] == "json_schema" for r in rows),
        recovered=sum(r["admitted"] == "True" for r in rejected),
        categories=dict(Counter(r["category"] for r in rejected)),
        episodes=len({r["episode_id"] for r in rows}),
        parents=len({r["parent_id"] for r in rows}),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve prior analysis output")
    path = args.audit / "actions.csv"
    with path.open() as f:
        rows = [r for r in csv.DictReader(f) if r["method"] != "base"]
    for row in rows:
        row["mutation_site_count"] = len(json.loads(row["mutation_sites"]))
    results = []
    for split in ("final_id", "external"):
        selected = [r for r in rows if r["split"] == split]
        task_features = {r["task_id"]: r for r in selected}
        for feature in (
            "source_tokens",
            "spec_tokens",
            "branches",
            "public_cases",
            "mutation_site_count",
        ):
            cut = float(np.median([float(r[feature]) for r in task_features.values()]))
            for high in (False, True):
                group = [r for r in selected if (float(r[feature]) > cut) == high]
                if group:
                    results.append(
                        dict(
                            split=split,
                            feature=feature,
                            stratum="above_median" if high else "at_or_below_median",
                            threshold=cut,
                            **summarize(group),
                        )
                    )
        for feature in ("mutation_type", "checkpoint", "method", "turn"):
            for value in sorted({r[feature] for r in selected}):
                group = [r for r in selected if r[feature] == value]
                results.append(
                    dict(
                        split=split,
                        feature=feature,
                        stratum=value,
                        threshold=None,
                        **summarize(group),
                    )
                )
        for method in ("A", "B"):
            for seed in ("2026", "2027", "2028"):
                group = [r for r in selected if r["method"] == method and r["seed"] == seed]
                results.append(
                    dict(
                        split=split,
                        feature="snapshot",
                        stratum=f"{method}-{seed}",
                        threshold=None,
                        **summarize(group),
                    )
                )
    atomic_json(
        args.output / "summary.json",
        {
            "strata": results,
            "interpretation": "Descriptive action-level associations with overlapping/confounded task features; not causal effects or independent action trials",
        },
    )
    write_csv(
        args.output / "strata.csv",
        [{k: v for k, v in r.items() if k != "categories"} for r in results],
    )
    atomic_json(
        args.output / "input-manifest.json", {str(p): file_hash(p) for p in (path, Path(__file__))}
    )
    for split in ("final_id", "external"):
        print(
            json.dumps(
                [
                    r
                    for r in results
                    if r["split"] == split
                    and r["feature"] in ("source_tokens", "spec_tokens", "branches")
                ]
            )
        )


if __name__ == "__main__":
    main()
