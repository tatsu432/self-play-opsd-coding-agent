"""Oracle-free mechanical diff reconstruction and retrospective action taxonomy.

This module parses source but never executes it. Reconstructed programs require
separate isolated semantic verification; recoverability is not repair success.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from selfplay_opsd.core import digest
from selfplay_opsd.mutations import apply_patch, parse_action, signature, validate_source


def recover_diff(source: str, patch: str) -> str:
    """Ignore only hunk coordinates/counts; require unique exact old hunk text."""
    if len(patch.encode()) > 65536:
        raise ValueError("Patch exceeds size limit")
    lines = patch.splitlines(keepends=True)
    if (
        len(lines) < 3
        or lines[0].rstrip() not in {"--- a/src/solution.py", "--- src/solution.py"}
        or lines[1].rstrip() not in {"+++ b/src/solution.py", "+++ src/solution.py"}
    ):
        raise ValueError("Unsafe or malformed file headers")
    hunks: list[tuple[str, str]] = []
    old: list[str] = []
    new: list[str] = []
    seen = False
    for line in lines[2:]:
        if re.fullmatch(r"@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@[^\n]*\n?", line):
            if seen:
                hunks.append(("".join(old), "".join(new)))
            seen = True
            old, new = [], []
        elif line[:1] in {" ", "+", "-"} and line.endswith("\n"):
            if not seen:
                raise ValueError("Hunk body before header")
            if line[0] in " -":
                old.append(line[1:])
            if line[0] in " +":
                new.append(line[1:])
        else:
            raise ValueError("Malformed hunk content")
    hunks.append(("".join(old), "".join(new)))
    spans: list[tuple[int, int, str]] = []
    for before, after in hunks:
        start = source.find(before) if before else -1
        if start < 0 or source.find(before, start + 1) >= 0:
            raise ValueError("Old hunk text absent or ambiguous")
        if spans and start < spans[-1][1]:
            raise ValueError("Overlapping or reordered hunks")
        spans.append((start, start + len(before), after))
    result = source
    for start, end, after in reversed(spans):
        result = result[:start] + after + result[end:]
    if result == source:
        raise ValueError("No effective edit")
    return result


def admit_repair(source: str, initial_source: str) -> None:
    """Match historical diff source/API admission without executing code."""
    try:
        tree = validate_source(source)
    except SyntaxError as exc:
        raise ValueError(f"Invalid Python syntax: {exc.msg}") from exc
    if signature(tree) != signature(validate_source(initial_source)):
        raise ValueError("Repair must preserve function APIs")


def error_category(message: str) -> str:
    text = message.lower()
    for category, terms in (
        ("syntax", ("syntax",)),
        ("api_preservation", ("function api",)),
        ("import_source_policy", ("import", "forbidden", "unsafe", "introspection", "unsupported")),
        ("diff_headers", ("diff headers",)),
        ("path", ("allowlisted", "only src/solution.py", "extra file")),
        ("hunk_counts", ("hunk count",)),
        ("hunk_offsets", ("offset", "out-of-range", "overlapping")),
        ("context", ("context mismatch",)),
        ("hunk_format", ("hunk", "no-newline", "empty patch")),
    ):
        if any(term in text for term in terms):
            return category
    return "unknown"


def audit_episode(task, episode: dict) -> tuple[list[dict], list[dict]]:
    """Replay historical accepted edits and audit each failed action independently."""
    source = task.initial_source
    rows, candidates = [], []
    repeated: Counter[str] = Counter()
    for index, turn in enumerate(episode["turns"]):
        row: dict[str, Any] = {
            "episode_id": episode["episode_id"],
            "task_id": task.task_id,
            "parent_id": task.parent_id,
            "turn": index,
            "action": turn["action"],
            "parse_status": turn["parse_status"],
            "before_hash": digest(source),
            "category": "non_edit",
            "recovered": False,
            "admitted": False,
            "semantic_verified": None,
        }
        try:
            action = parse_action(turn["raw_text"])
        except ValueError:
            row["category"] = "json_schema"
            rows.append(row)
            continue
        if action["action"] == "apply_patch":
            try:
                after = apply_patch(source, action["patch"])
                admit_repair(after, task.initial_source)
            except ValueError as exc:
                if turn["parse_status"] != "tool_error":
                    raise ValueError("Historical accepted edit no longer reproduces") from exc
                row["category"] = error_category(str(exc))
                row["error_evidence"] = "deterministic_static_replay"
                reason_key = digest([source, row["category"]])
                repeated[reason_key] += 1
                row["same_failure_reason_count"] = repeated[reason_key]
                key = digest([source, action["patch"], row["category"]])
                repeated[key] += 1
                row["same_failed_action_count"] = repeated[key]
                try:
                    recovered = recover_diff(source, action["patch"])
                    row["recovered"] = True
                    admit_repair(recovered, task.initial_source)
                    row["admitted"] = True
                    candidates.append(
                        {**row, "source": recovered, "source_hash": digest(recovered)}
                    )
                except ValueError as recovery_error:
                    row["recovery_error"] = str(recovery_error)
            else:
                if turn["parse_status"] != "valid":
                    raise ValueError("Historical rejection unexpectedly accepted")
                source = after
                row["category"] = "accepted_edit"
        rows.append(row)
    if digest(source) != episode["final_source_hash"]:
        raise ValueError("Reconstructed historical final source differs")
    return rows, candidates
