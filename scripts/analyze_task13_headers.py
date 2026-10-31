"""Measure static recovery after removing one exact optional diff header."""

import argparse
import json
from collections import Counter
from pathlib import Path

from selfplay_opsd.data import TaskRepository
from selfplay_opsd.mutations import strict_json_object, validate_mutation
from selfplay_opsd.storage import atomic_json, file_hash

HEADER = "diff --git a/src/solution.py b/src/solution.py\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", type=Path)
    parser.add_argument("--data", type=Path, default=Path("data/protocol-v3"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.probe / "manifest.json").read_text())
    records = json.loads((args.probe / "proposals.json").read_text())
    complete = json.loads((args.probe / "complete.json").read_text())
    repo = TaskRepository(args.data)
    if (
        manifest["representation"] != "diff"
        or repo.hash != manifest["manifest_hash"]
        or len(records) != complete["count"]
    ):
        raise ValueError("Wrong format, dataset, or incomplete probe")
    parents = {p.parent_id: p for p in repo.parents("train")}
    counts: Counter[str] = Counter()
    unique: set[tuple[str, str]] = set()
    for row in records:
        try:
            obj = strict_json_object(row["raw_text"])
            if (
                set(obj) != {"patch"}
                or not isinstance(obj["patch"], str)
                or not obj["patch"].startswith(HEADER)
            ):
                continue
            counts["exact_extra_header"] += 1
            change = validate_mutation(
                parents[row["parent_id"]].source, obj["patch"][len(HEADER) :]
            )
            counts["static_validator_passed"] += 1
            unique.add((row["parent_id"], change.dedup_key))
        except (ValueError, SyntaxError, TypeError, KeyError, IndexError) as exc:
            counts["rejected: " + str(exc)[:100]] += 1
    result = {
        "input_sha256": {
            n: file_hash(args.probe / n)
            for n in ("manifest.json", "proposals.json", "complete.json")
        },
        "counts": dict(counts),
        "unique_static_valid_mutations": len(unique),
        "limitation": "Static validity only; semantic verification and token admission were not performed.",
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
