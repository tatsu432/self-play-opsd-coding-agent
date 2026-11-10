"""Paired interface outcomes; action counts never substitute for verified repairs."""

from __future__ import annotations

from collections import Counter

from selfplay_opsd.core import digest
from selfplay_opsd.task16_analysis import paired_statistics


def episode_metrics(episode: dict) -> dict:
    turns = episode["turns"]
    edits = [(i, t) for i, t in enumerate(turns) if t["action"] in {"apply_patch", "replace_text"}]
    accepted = [i for i, t in edits if t["parse_status"] == "valid"]
    failed = Counter(t["raw_text"] for _, t in edits if t["parse_status"] != "valid")
    return {
        "task_id": episode["task_id"],
        "episode_id": episode["episode_id"],
        "success": bool(episode["reward"]),
        "public_pass": episode["public_pass"],
        "hidden_pass": episode["hidden_pass"],
        "public_pass_hidden_fail": episode["public_pass"] and not episode["hidden_pass"],
        "valid_json_actions": sum(t["parse_status"] != "invalid" for t in turns),
        "actions": len(turns),
        "edit_attempts": len(edits),
        "accepted_edits": len(accepted),
        "rejected_edits": len(edits) - len(accepted),
        "applied_edit": bool(accepted),
        "first_edit_turn": accepted[0] + 1 if accepted else None,
        "repeated_failed_edits": sum(max(0, n - 1) for n in failed.values()),
        "test_calls": episode["test_calls"],
        "termination": episode["termination"],
        "tokens": sum(len(t["target_ids"]) for t in turns),
        "seconds": episode["seconds"],
    }


def compare_interfaces(cells: list[dict], families: dict[str, str]) -> dict:
    if len(cells) != 4 or len({c["arm"] for c in cells}) != 4:
        raise ValueError("Require exactly four distinct frozen-model/interface cells")
    by_arm = {c["arm"]: c for c in cells}
    if set(by_arm) != set("ABCD"):
        raise ValueError("Unknown interface cell")
    for cell in cells[1:]:
        for field in ("pool_sha256", "dataset_hash", "seed", "model", "task_ids", "prompt_hash"):
            if cell[field] != cells[0][field]:
                raise ValueError(f"Cross-snapshot {field} differs")
    comparisons = []
    for first, second in (("A", "B"), ("C", "D")):
        a, b = by_arm[first], by_arm[second]
        for field in (
            "pool_sha256",
            "dataset_hash",
            "seed",
            "model",
            "snapshot",
            "task_ids",
            "prompt_hash",
        ):
            if a[field] != b[field]:
                raise ValueError(f"Paired {field} differs")
        if a["interface"] != "unified_diff" or b["interface"] != "replace_text":
            raise ValueError("Incorrect control/intervention assignment")
        if {k: v for k, v in a["episode"].items() if k != "edit_protocol"} != {
            k: v for k, v in b["episode"].items() if k != "edit_protocol"
        }:
            raise ValueError("Paired episode budgets differ")
        for cell in (a, b):
            if len(cell["records"]) != len(cell["task_ids"]) or {
                r["task_id"] for r in cell["records"]
            } != set(cell["task_ids"]):
                raise ValueError("Incomplete or duplicate paired outcomes")
        a_original = [r for r in a["records"] if r["presentation"] == "original"]
        b_original = [r for r in b["records"] if r["presentation"] == "original"]
        comparisons.append(
            {
                "snapshot": "base" if first == "A" else "B2026-final",
                "verified_repairs_original": paired_statistics(a_original, b_original, families),
                "applied_edit_original": paired_statistics(
                    [r | {"success": r["applied_edit"]} for r in a_original],
                    [r | {"success": r["applied_edit"]} for r in b_original],
                    families,
                ),
                "all_presentations_exploratory": paired_statistics(
                    a["records"], b["records"], families
                ),
            }
        )
    return {
        "comparison_identity": digest(
            [{k: v for k, v in c.items() if k != "records"} for c in cells]
        ),
        "comparisons": comparisons,
    }
