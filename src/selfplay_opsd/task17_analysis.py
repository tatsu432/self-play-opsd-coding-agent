"""Paired interface outcomes; action counts never substitute for verified repairs."""

from __future__ import annotations

from collections import Counter

from selfplay_opsd.core import digest
from selfplay_opsd.mutations import strict_json_object
from selfplay_opsd.task16_analysis import paired_statistics


def verify_cell_execution(cell: dict, status: dict, operation: str = "task17-interface") -> str:
    """Bind allocated-node provenance to its controller's immutable specification."""
    spec = status["ledger"]["specification"]
    expected = {
        "operation": operation,
        "method": cell["arm"],
        "study_id": cell["study_id"],
        "seed": cell["seed"],
        "commit": cell["provenance"]["git_commit"],
    }
    if any(spec[key] != value for key, value in expected.items()):
        raise ValueError("Cell execution differs from immutable scheduler specification")
    identity = spec["runtime_identity"]
    if (
        identity["prompt_hash"] != cell["prompt_hash"]
        or identity["manifest_hash"] != cell["dataset_hash"]
        or str(status["attempts"][-1]["job_id"]) != cell["provenance"]["allocation"]["SLURM_JOB_ID"]
    ):
        raise ValueError("Cell prompt/data/allocation identity differs")
    return identity["scientific_code_hash"]


def episode_metrics(episode: dict) -> dict:
    turns = episode["turns"]
    raw_actions: list[dict | None] = []
    for turn in turns:
        try:
            raw_actions.append(strict_json_object(turn["raw_text"]))
        except ValueError:
            raw_actions.append(None)
    edits = [
        (i, t)
        for i, t in enumerate(turns)
        if t["action"] in {"apply_patch", "replace_text"}
        or (raw_actions[i] or {}).get("action") in {"apply_patch", "replace_text"}
    ]
    accepted = [i for i, t in edits if t["parse_status"] == "valid"]
    failed = Counter(t["raw_text"] for _, t in edits if t["parse_status"] != "valid")
    return {
        "task_id": episode["task_id"],
        "episode_id": episode["episode_id"],
        "success": bool(episode["reward"]),
        "public_pass": episode["public_pass"],
        "hidden_pass": episode["hidden_pass"],
        "public_pass_hidden_fail": episode["public_pass"] and not episode["hidden_pass"],
        "runtime_failures": episode.get("runtime_failures", 0),
        "semantic_failures": episode.get("semantic_failures", 0),
        "verifier_status": episode.get("verifier_status", "unknown"),
        "generation_seconds": episode.get("timings", {}).get("generate", 0),
        "public_test_seconds": episode.get("timings", {}).get("public_tests", 0),
        "reward_verifier_seconds": episode.get("timings", {}).get("reward_eval", 0),
        "valid_json_actions": sum(a is not None for a in raw_actions),
        "valid_schema_actions": sum(t["parse_status"] != "invalid" for t in turns),
        "actions": len(turns),
        "turns": len(turns),
        "edit_attempts": len(edits),
        "accepted_edits": len(accepted),
        "metadata_recovered_edits": sum(
            t.get("edit_application") == "metadata_recovered" for _, t in edits
        ),
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
