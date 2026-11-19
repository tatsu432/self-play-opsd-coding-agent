"""Audit pinned PDB releases against the current worker's source admission policy."""

from __future__ import annotations

import argparse
import collections
import warnings
from pathlib import Path

import pyarrow.parquet as pq

from selfplay_opsd.mutations import validate_source
from selfplay_opsd.storage import atomic_json, file_hash

RELEASES = {
    "PDB-Single-Full": "dac1f9aea17f507b2df3f253f06ff545bbdf2ce1",
    "PDB-Single": "b3c0b911c5df4444ff3e8160b8dbe4ea297c08ec",
    "PDB-Multi": "cb13eb9a160a702745d22129149083c14bedf215",
}
SCORER = "fbfa901eeae5b5f6944daea479a4ce31fad0e4a6"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    results = {}
    for name, revision in RELEASES.items():
        path = args.data / f"{name}.parquet"
        table = pq.read_table(path, columns=["source_dataset", "gt_solution", "buggy_code"])
        counts: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            for row in table.to_pylist():
                try:
                    validate_source(row["gt_solution"])
                    validate_source(row["buggy_code"])
                    result = "both_admitted"
                except (ValueError, SyntaxError) as exc:
                    result = str(exc).split(" (", 1)[0]
                counts[row["source_dataset"]][result] += 1
        results[name] = {
            "revision": revision,
            "sha256": file_hash(path),
            "rows": table.num_rows,
            "source_admission": counts,
        }
    atomic_json(
        args.output,
        {"official_scorer_commit": SCORER, "executed_unit_tests": False, "releases": results},
    )


if __name__ == "__main__":
    main()
