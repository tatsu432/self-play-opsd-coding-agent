"""Read-only Task 18 audit of preserved events and paired interface artifacts."""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np

from selfplay_opsd.storage import active_events, atomic_json, file_hash, provenance
from selfplay_opsd.task16_analysis import ast_features, paired_statistics
from selfplay_opsd.task17_analysis import verify_cell_execution
from selfplay_opsd.task18 import load_native_probe
from selfplay_opsd.task18_analysis import metrics as episode_metrics
from selfplay_opsd.task18_analysis import native_expansion_gate, paired_exposure


def analysis_code_hashes():
    root = Path(__file__).resolve().parents[1]
    names = (
        "scripts/analyze_task18.py",
        "src/selfplay_opsd/task18_analysis.py",
        "src/selfplay_opsd/task18.py",
        "src/selfplay_opsd/task18_protocol.py",
        "src/selfplay_opsd/core.py",
        "src/selfplay_opsd/mutations.py",
        "src/selfplay_opsd/task17_analysis.py",
        "src/selfplay_opsd/task16_analysis.py",
        "src/selfplay_opsd/task17_recovery.py",
        "src/selfplay_opsd/analysis.py",
        "uv.lock",
    )
    return {str(root / name): file_hash(root / name) for name in names}


def audit(manifest_path: Path, pool_path: Path):
    manifest = json.loads(manifest_path.read_text())
    pool = json.loads(pool_path.read_text())
    hashes = {str(p): file_hash(p) for p in (manifest_path, pool_path)}
    runs = []
    for item in manifest["runs"]:
        root = Path(item["run_dir"])
        complete = json.loads((root / "complete.json").read_text())
        events = [e for e in active_events(root) if e["kind"] == "iteration_commit"]
        if [e["completed_iteration"] for e in events] != list(range(1, 11)) or complete[
            "episodes"
        ] != 80:
            raise ValueError("Historical run lacks its ten committed iterations and endpoint")
        batches = []
        for event in events:
            payload = event["payload"]
            metric = payload["metrics"]["student"]
            source = root / payload["episodes_path"]
            if not source.resolve().is_relative_to(root.resolve()):
                raise ValueError("Unsafe trajectory path")
            batches.append(
                {
                    "iteration": event["completed_iteration"],
                    "snapshot": payload["snapshot"],
                    "episodes_available": source.is_file(),
                    **{k: v for k, v in metric.items() if k.startswith("pre_update_logprob")},
                }
            )
        for p in [
            root / "manifest.json",
            root / "complete.json",
            *sorted((root / "events").glob("*.jsonl")),
        ]:
            hashes[str(p)] = file_hash(p)
        worst = max(batches, key=lambda b: b.get("pre_update_logprob_abs_max", 0))
        runs.append(
            {
                "arm": item["arm"],
                "seed": item["seed"],
                "run_id": root.name,
                "episodes": complete["episodes"],
                "allocated_gpu_hours": item["allocated_gpu_hours"],
                "batches": batches,
                "worst_batch": worst,
                "available_training_shards": sum(b["episodes_available"] for b in batches),
            }
        )
    return {
        "kind": "saved-event audit; no model execution or new repair evaluation",
        "runs": runs,
        "historical_allocated_gpu_hours": sum(r["allocated_gpu_hours"] for r in runs),
        "new_allocated_gpu_hours": 0,
        "pool_families": len({r["family_id"] for r in pool["rows"]}),
        "source_hashes": hashes,
        "numerical_training_decision": "pending exact-snapshot GPU diagnostic; aggregate events cannot establish causal effects",
    }


def interfaces(manifest_path: Path, pool_path: Path):
    manifest, pool = json.loads(manifest_path.read_text()), json.loads(pool_path.read_text())
    info = {
        r["task"]["task_id"]: r
        for r in pool["rows"]
        if r["purpose"] == "interface" and r["presentation"] == "original"
    }
    cells = []
    hashes = {str(p): file_hash(p) for p in (manifest_path, pool_path)}
    for item in manifest["cells"]:
        root = Path(item["run_dir"])
        paths = [root / name for name in ("manifest.json", "complete.json", "episodes.json")]
        cell, complete, episodes = [json.loads(p.read_text()) for p in paths]
        if not item.get("status_path"):
            raise ValueError("Completed interface analysis requires controller status evidence")
        status_path = Path(item["status_path"])
        status = json.loads(status_path.read_text())
        attempt = status["attempts"][-1]["scheduler"]
        if attempt.get("state") != "COMPLETED" or attempt.get("exit_code") != "0:0":
            raise ValueError("Interface allocation is not terminal exit zero")
        executed_code = verify_cell_execution(
            cell, status, status["ledger"]["specification"]["operation"]
        )
        if cell.get("scientific_code_hash", executed_code) != executed_code:
            raise ValueError("Interface code differs from its immutable controller identity")
        cell["scientific_code_hash"] = executed_code
        hashes[str(status_path)] = file_hash(status_path)
        if cell["pool_sha256"] != file_hash(pool_path) or complete["episodes"] != len(episodes):
            raise ValueError("Interface completion or pool identity differs")
        if complete.get("episodes_sha256", file_hash(paths[2])) != file_hash(paths[2]):
            raise ValueError("Episode artifact checksum differs")
        if reference := cell.get("reused_probe"):
            probe_root = Path(item["reused_probe_dir"])
            probe_status_path = Path(item["reused_probe_status_path"])
            probe_status = json.loads(probe_status_path.read_text())
            scheduler = probe_status["attempts"][-1]["scheduler"]
            if scheduler.get("state") != "COMPLETED" or scheduler.get("exit_code") != "0:0":
                raise ValueError("Reused native probe is not terminal exit zero")
            previous = json.loads((probe_root / "manifest.json").read_text())
            if (
                verify_cell_execution(previous, probe_status, "task18-native")
                != reference["scientific_code_hash"]
            ):
                raise ValueError("Reused native probe code identity differs")
            reused = load_native_probe(
                reference | {"run_dir": str(probe_root)}, cell, set(cell["task_ids"][:8])
            )
            if (
                set(cell["new_task_ids"]) != set(cell["task_ids"]) - {e.task_id for e in reused}
                or complete.get("reused_episodes") != 8
                or complete.get("generated_episodes") != 32
            ):
                raise ValueError("Native new/reused exposure differs")
            by_task = {e["task_id"]: e for e in episodes}
            if any(by_task[e.task_id] != e.model_dump(mode="json") for e in reused):
                raise ValueError("Copied native probe trajectory differs")
            hashes[str(probe_status_path)] = file_hash(probe_status_path)
            hashes.update(
                {str(probe_root / n): file_hash(probe_root / n) for n in reference["sha256"]}
            )
        selected = [e for e in episodes if e["task_id"] in info]
        if {e["task_id"] for e in selected} != set(cell["task_ids"]) & set(info):
            raise ValueError("Manifest task coverage differs")
        if len(selected) != len({e["task_id"] for e in selected}):
            raise ValueError("Duplicate task outcomes")
        records = [
            episode_metrics(e)
            | {
                "parent_id": info[e["task_id"]]["task"]["parent_id"],
                "family_id": info[e["task_id"]]["family_id"],
                **cell.get("task_features", {}).get(e["task_id"], {}),
                **ast_features(info[e["task_id"]]["task"]["initial_source"]),
            }
            for e in selected
        ]
        cells.append(
            {
                "name": item["name"],
                "manifest": cell,
                "records": records,
                "allocated_gpu_hours": item["allocated_gpu_hours"],
            }
        )
        hashes.update({str(p): file_hash(p) for p in paths})
    comparisons = []
    families = {r["task"]["parent_id"]: r["family_id"] for r in info.values()}
    for a, b in itertools.combinations(cells, 2):
        left, right = a["manifest"], b["manifest"]
        if left["snapshot"] != right["snapshot"]:
            continue
        if any(left[k] != right[k] for k in ("model", "seed", "dataset_hash", "pool_sha256")):
            raise ValueError("Paired model/data/seed differs")
        lbudget = {
            k: v
            for k, v in left["episode"].items()
            if k not in {"edit_protocol", "tool_serialization"}
        }
        rbudget = {
            k: v
            for k, v in right["episode"].items()
            if k not in {"edit_protocol", "tool_serialization"}
        }
        if lbudget != rbudget:
            raise ValueError("Paired generation budgets differ")
        ids = {r["task_id"] for r in a["records"]} & {r["task_id"] for r in b["records"]}
        if not ids:
            raise ValueError("No common families")
        ar, br = ([r for r in c["records"] if r["task_id"] in ids] for c in (a, b))
        exposure = {"left": paired_exposure(ar), "right": paired_exposure(br)}
        serialization_pair = {
            left["episode"].get("tool_serialization", "plain_json"),
            right["episode"].get("tool_serialization", "plain_json"),
        } == {"plain_json", "qwen_native"} and all(
            cell["episode"]["edit_protocol"] == "replace_text" for cell in (left, right)
        )
        gate = None
        expansion = None
        if serialization_pair:
            plain, native = (
                (exposure["left"], exposure["right"])
                if right["episode"].get("tool_serialization") == "qwen_native"
                else (exposure["right"], exposure["left"])
            )
            gate = native_expansion_gate(plain, native)
            new_ids = set(left.get("new_task_ids", right.get("new_task_ids", []))) & ids
            if new_ids:
                new_ar, new_br = ([r for r in rows if r["task_id"] in new_ids] for rows in (ar, br))
                expansion = {
                    "common_families": len(new_ids),
                    "success": paired_statistics(new_ar, new_br, families),
                    "paired_exposure": {
                        "left": paired_exposure(new_ar),
                        "right": paired_exposure(new_br),
                    },
                    "interpretation": "Prespecified remaining tasks after the eight-task screen; all families were previously inspected.",
                }
        strata = []
        for feature in ("source_tokens", "spec_tokens", "branches"):
            if not all(feature in row for row in ar + br):
                continue
            threshold = float(np.median([row[feature] for row in ar]))
            for high in (False, True):
                selected = {row["task_id"] for row in ar if (row[feature] > threshold) == high}
                if selected:
                    strata.append(
                        {
                            "feature": feature,
                            "threshold": threshold,
                            "above_median": high,
                            "families": len(selected),
                            "paired": paired_statistics(
                                [r for r in ar if r["task_id"] in selected],
                                [r for r in br if r["task_id"] in selected],
                                families,
                            ),
                        }
                    )
        comparisons.append(
            {
                "left": a["name"],
                "right": b["name"],
                "common_families": len(ids),
                "paired_exposure": exposure,
                "native_expansion_gate": gate,
                "native_expansion_new_tasks": expansion,
                "editing_changed": left["episode"]["edit_protocol"]
                != right["episode"]["edit_protocol"],
                "serialization_changed": left["episode"].get("tool_serialization", "plain_json")
                != right["episode"].get("tool_serialization", "plain_json"),
                "exploratory_strata": strata,
                "success": paired_statistics(ar, br, families),
                "applied_edit": paired_statistics(
                    [r | {"success": r["applied_edit"]} for r in ar],
                    [r | {"success": r["applied_edit"]} for r in br],
                    families,
                ),
                "historical_control_contextual": left.get("scientific_code_hash")
                != right.get("scientific_code_hash")
                or left.get("prompt_hash") != right.get("prompt_hash"),
            }
        )
    return {
        "kind": "paired frozen-snapshot interface analysis",
        "comparisons": comparisons,
        "cells": {
            c["name"]: {
                "tasks": len(c["records"]),
                "repairs": sum(r["success"] for r in c["records"]),
                "actions": sum(r["actions"] for r in c["records"]),
                "valid_schema_actions": sum(r["valid_schema_actions"] for r in c["records"]),
                "accepted_edits": sum(r["accepted_edits"] for r in c["records"]),
                "metadata_recovered_edits": sum(
                    r["metadata_recovered_edits"] for r in c["records"]
                ),
                "rejected_edits": sum(r["rejected_edits"] for r in c["records"]),
                "tokens": sum(r["tokens"] for r in c["records"]),
                "turns": sum(r["turns"] for r in c["records"]),
                "public_successes": sum(r["public_pass"] for r in c["records"]),
                "hidden_successes": sum(r["hidden_pass"] for r in c["records"]),
                "tasks_with_applied_edit": sum(r["applied_edit"] for r in c["records"]),
                "failures_after_applied_edit": sum(
                    r["semantic_failure_after_applied_edit"] for r in c["records"]
                ),
                "rejection_taxonomy": dict(
                    sum((Counter(r["rejection_taxonomy"]) for r in c["records"]), Counter())
                ),
                "allocated_gpu_hours": c["allocated_gpu_hours"],
            }
            for c in cells
        },
        "source_hashes": hashes,
        "limitations": "Previously inspected training families; family intervals are not seed uncertainty. Historical controls require behavioral parity before causal attribution.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", choices=("audit", "interfaces"), default="audit")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve previous analysis; output already exists")
    report = (audit if args.operation == "audit" else interfaces)(args.manifest, args.pool)
    report["source_hashes"].update(analysis_code_hashes())
    report["analysis_provenance"] = provenance()
    atomic_json(args.output / "summary.json", report)


if __name__ == "__main__":
    main()
