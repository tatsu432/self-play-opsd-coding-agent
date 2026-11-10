"""Analyze complete Task 17 interface cells from fetched private artifacts."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from selfplay_opsd.analysis import write_csv
from selfplay_opsd.storage import atomic_json, file_hash
from selfplay_opsd.task16_analysis import ast_features, paired_statistics
from selfplay_opsd.task17_analysis import compare_interfaces, episode_metrics
from selfplay_opsd.task17_recovery import error_category


def describe(records):
    result = {"tasks": len(records)}
    for key in (
        "success",
        "applied_edit",
        "actions",
        "valid_json_actions",
        "valid_schema_actions",
        "edit_attempts",
        "accepted_edits",
        "rejected_edits",
        "test_calls",
        "public_pass_hidden_fail",
        "tokens",
        "turns",
        "repeated_failed_edits",
    ):
        result[key] = sum(r[key] for r in records)
    result["termination"] = dict(Counter(r["termination"] for r in records))
    for key in ("seconds", "tokens", "turns", "rejected_edits", "first_edit_turn"):
        values = [r[key] for r in records if r[key] is not None]
        result[key + "_distribution"] = {
            "n": len(values),
            "mean": float(np.mean(values)) if values else None,
            "quantiles_0_25_50_75_100": np.quantile(values, [0, 0.25, 0.5, 0.75, 1]).tolist()
            if values
            else [],
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="JSON with cells [{arm,run_dir,allocated_gpu_hours}]",
    )
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve earlier results; output exists")
    manifest, pool = json.loads(args.manifest.read_text()), json.loads(args.pool.read_text())
    info = {r["task"]["task_id"]: r for r in pool["rows"]}
    families = {r["task"]["parent_id"]: r["family_id"] for r in pool["rows"]}
    cells, rows, hashes = (
        [],
        [],
        {str(args.manifest): file_hash(args.manifest), str(args.pool): file_hash(args.pool)},
    )
    for item in manifest["cells"]:
        root = Path(item["run_dir"])
        for name in ("manifest.json", "complete.json", "episodes.json"):
            hashes[str(root / name)] = file_hash(root / name)
        cell = json.loads((root / "manifest.json").read_text())
        complete = json.loads((root / "complete.json").read_text())
        episodes = json.loads((root / "episodes.json").read_text())
        if (
            cell["arm"] != item["arm"]
            or complete["episodes"] != len(episodes)
            or complete["pool_sha256"] != file_hash(args.pool)
            or cell["pool_sha256"] != file_hash(args.pool)
        ):
            raise ValueError("Cell completion/manifest identity differs")
        cell["records"] = [
            episode_metrics(e)
            | {
                "parent_id": info[e["task_id"]]["task"]["parent_id"],
                "presentation": info[e["task_id"]]["presentation"],
                "family_id": info[e["task_id"]]["family_id"],
                **cell.get("task_features", {}).get(e["task_id"], {}),
                **ast_features(info[e["task_id"]]["task"]["initial_source"]),
            }
            for e in episodes
        ]
        rows.extend(r | {"arm": item["arm"]} for r in cell["records"])
        cells.append(cell)
    result = compare_interfaces(cells, families)
    result["allocated_gpu_hours"] = sum(item["allocated_gpu_hours"] for item in manifest["cells"])
    result["cells"] = {
        c["arm"]: {
            "allocated_gpu_hours": next(
                i["allocated_gpu_hours"] for i in manifest["cells"] if i["arm"] == c["arm"]
            ),
            "original": describe([r for r in c["records"] if r["presentation"] == "original"]),
            "presentations": {
                p: describe([r for r in c["records"] if r["presentation"] == p])
                for p in sorted({r["presentation"] for r in c["records"]})
            },
        }
        for c in cells
    }
    strata = []
    originals = [r for r in cells[0]["records"] if r["presentation"] == "original"]
    for feature in ("source_tokens", "spec_tokens", "branches", "public_cases"):
        threshold = float(np.median([r[feature] for r in originals]))
        for high in (False, True):
            selected = {r["task_id"] for r in originals if (r[feature] > threshold) == high}
            if not selected:
                continue
            for first, second in (("A", "B"), ("C", "D")):
                a = [
                    r
                    for c in cells
                    if c["arm"] == first
                    for r in c["records"]
                    if r["task_id"] in selected
                ]
                b = [
                    r
                    for c in cells
                    if c["arm"] == second
                    for r in c["records"]
                    if r["task_id"] in selected
                ]
                strata.append(
                    {
                        "feature": feature,
                        "threshold": threshold,
                        "above_median": high,
                        "snapshot": "base" if first == "A" else "B2026-final",
                        "repairs": paired_statistics(a, b, families),
                        "applied_edits": paired_statistics(
                            [r | {"success": r["applied_edit"]} for r in a],
                            [r | {"success": r["applied_edit"]} for r in b],
                            families,
                        ),
                    }
                )
    result["exploratory_strata"] = strata
    presentations = []
    for cell in cells:
        original_by_parent = {
            r["parent_id"]: r for r in cell["records"] if r["presentation"] == "original"
        }
        for presentation in sorted({r["presentation"] for r in cell["records"]} - {"original"}):
            variants = [r for r in cell["records"] if r["presentation"] == presentation]
            first = [original_by_parent[r["parent_id"]] for r in variants]
            second = [
                r | {"task_id": original_by_parent[r["parent_id"]]["task_id"]} for r in variants
            ]
            presentations.append(
                {
                    "arm": cell["arm"],
                    "presentation": presentation,
                    "repairs": paired_statistics(first, second, families),
                    "applied_edits": paired_statistics(
                        [r | {"success": r["applied_edit"]} for r in first],
                        [r | {"success": r["applied_edit"]} for r in second],
                        families,
                    ),
                }
            )
    result["exploratory_presentation_effects"] = presentations
    errors = []
    for item in manifest["cells"]:
        for episode in json.loads((Path(item["run_dir"]) / "episodes.json").read_text()):
            for turn in episode["turns"]:
                if turn["parse_status"] == "valid":
                    continue
                message = (turn.get("observation") or {}).get("error", "")
                category = (
                    "json_or_schema"
                    if turn["parse_status"] == "invalid"
                    else error_category(message)
                )
                if "span" in message.lower() or "match" in message.lower() and item["arm"] in "BD":
                    category = "text_span"
                errors.append(
                    {"arm": item["arm"], "task_id": episode["task_id"], "category": category}
                )
    result["tool_errors"] = {
        arm: dict(Counter(r["category"] for r in errors if r["arm"] == arm)) for arm in "ABCD"
    }
    atomic_json(args.output / "summary.json", result)
    atomic_json(args.output / "input-manifest.json", hashes)
    write_csv(args.output / "episodes.csv", rows)
    write_csv(args.output / "tool-errors.csv", errors)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    for ax, metric, title in zip(
        axes,
        ("applied_edit", "success"),
        ("Tasks receiving an edit", "Verified repairs"),
        strict=True,
    ):
        values = [
            sum(r[metric] for r in c["records"] if r["presentation"] == "original") for c in cells
        ]
        ax.bar(
            [
                ("Base" if c["arm"] in "AB" else "B final")
                + "\n"
                + ("Diff" if c["arm"] in "AC" else "Text")
                for c in cells
            ],
            values,
        )
        ax.set(title=title, ylabel="Original tasks (denominator 40)")
    for suffix in ("png", "pdf"):
        fig.savefig(args.output / f"interface-outcomes.{suffix}")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained")
    for ax, metric, title in zip(
        axes,
        ("applied_edit", "success"),
        ("Edit application by source length", "Verified repair by source length"),
        strict=True,
    ):
        threshold = float(np.median([r["source_tokens"] for r in originals]))
        for cell in cells:
            selected = [r for r in cell["records"] if r["presentation"] == "original"]
            values = [
                np.mean([r[metric] for r in selected if (r["source_tokens"] > threshold) == high])
                for high in (False, True)
            ]
            ax.plot([0, 1], values, marker="o", label=cell["arm"])
        ax.set(
            title=title,
            xticks=[0, 1],
            xticklabels=[f"≤ {threshold:g} tokens", f"> {threshold:g} tokens"],
            ylabel="Fraction of original tasks",
            ylim=(-0.03, 1.03),
        )
        ax.legend(title="A/B base; C/D B final", fontsize=8)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output / f"interface-length.{suffix}")
    plt.close(fig)


if __name__ == "__main__":
    main()
