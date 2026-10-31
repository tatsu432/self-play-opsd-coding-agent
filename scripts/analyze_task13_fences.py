"""Measure syntactic recovery after removing a single Markdown JSON fence.

This does not execute generated code or establish semantic validity.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from task13_structured import mutation, sites

from selfplay_opsd.data import TaskRepository
from selfplay_opsd.mutations import strict_json_object, validate_mutation
from selfplay_opsd.storage import atomic_json, file_hash


def unwrap(text: str) -> str | None:
    match = re.fullmatch(r"\s*```(?:json)?\s*\n(.*?)\n```\s*", text, re.DOTALL | re.IGNORECASE)
    return match.group(1) if match else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", type=Path)
    parser.add_argument("--data", type=Path, default=Path("data/protocol-v3"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.probe / "manifest.json").read_text())
    rows = json.loads((args.probe / "proposals.json").read_text())
    complete = json.loads((args.probe / "complete.json").read_text())
    repo = TaskRepository(args.data)
    if len(rows) != complete["count"] or repo.hash != manifest["manifest_hash"]:
        raise ValueError("Incomplete probe or mismatched dataset")
    parents = {p.parent_id: p for p in repo.parents("train")}
    counts: Counter[str] = Counter()
    unique: set[tuple[str, str]] = set()
    for row in rows:
        text = unwrap(row["raw_text"])
        if text is None:
            continue
        counts["exact_fenced_outputs"] += 1
        parent = parents[row["parent_id"]]
        try:
            obj = strict_json_object(text)
            counts["unwrapped_json"] += 1
            if manifest["representation"] == "structured":
                available = sites(parent.source)
                if (
                    set(obj) != {"site", "operation"}
                    or type(obj["site"]) is not int
                    or not isinstance(obj["operation"], str)
                    or not 0 <= obj["site"] < len(available)
                ):
                    raise ValueError("Invalid structured proposal")
                change = mutation(parent.source, available[obj["site"]]["path"], obj["operation"])
            else:
                if set(obj) != {"patch"} or not isinstance(obj["patch"], str):
                    raise ValueError("Invalid patch proposal")
                change = validate_mutation(parent.source, obj["patch"])
            counts["static_validator_passed"] += 1
            unique.add((parent.parent_id, change.dedup_key))
        except (ValueError, SyntaxError, TypeError, KeyError, IndexError) as exc:
            counts["rejected: " + str(exc)[:100]] += 1
    result = {
        "input_sha256": {
            name: file_hash(args.probe / name)
            for name in ("manifest.json", "proposals.json", "complete.json")
        },
        "counts": dict(counts),
        "unique_static_valid_mutations": len(unique),
        "limitation": "Static validity only; semantic verification and token admission were not performed.",
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
