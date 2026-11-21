"""Prepare an explicitly adapted BigCodeBench subset using pinned test calls."""

from __future__ import annotations

import argparse
import ast
import os
import sys
from collections import Counter
from pathlib import Path

from selfplay_opsd.codec import decode, encode
from selfplay_opsd.config import SandboxConfig
from selfplay_opsd.core import Status, digest, read_json
from selfplay_opsd.pdb_data import bcb_cases, parent_key, release_rows
from selfplay_opsd.sandbox import SandboxBackend
from selfplay_opsd.storage import atomic_json, file_hash

BCB_SHA = "6079ff7a9f41fb928b3b0750c9e1e160ac36c8cc7c58124efbf86138549131d6"


def recorder(source: str, test: str, entry: str) -> str:
    """Execute fixed upstream test code only with published canonical source."""
    return f"""
import io
import types
import unittest
from collections import Counter
from codec import encode
def argument(value):
    return ["bytes", value.hex()] if type(value) is bytes else ["value", value]
def collect():
    module = types.ModuleType("solution")
    exec(compile({source!r}, "src/solution.py", "exec"), module.__dict__)
    original = getattr(module, {entry!r})
    calls = []
    counts = []
    excluded = []
    def observe(*args, **kwargs):
        inputs = [list(map(argument, args)), {{key: argument(value) for key, value in kwargs.items()}}]
        try:
            value = original(*args, **kwargs)
        except Exception as exc:
            kind = type(exc).__name__
            if kind not in {{"ValueError", "TypeError", "KeyError", "JSONDecodeError", "IndexError"}}:
                excluded.append("unsupported exception")
            calls.append([inputs, ["raises", kind]])
            raise
        if type(value) is Counter:
            calls.append([inputs, ["counter", dict(value)]])
        else:
            try:
                encode(value)
            except ValueError:
                excluded.append("unsupported return")
            else:
                calls.append([inputs, ["equal", value]])
        return value
    class Records(unittest.TestResult):
        def startTest(self, test):
            super().startTest(test)
            counts.append(len(calls))
    setattr(module, {entry!r}, observe)
    exec(compile({test!r}, "tests/upstream.py", "exec"), module.__dict__)
    tests = unittest.defaultTestLoader.loadTestsFromTestCase(module.TestCases)
    results = tests.run(Records())
    used = [*counts, len(calls)]
    if (
        not results.testsRun or not results.wasSuccessful() or results.skipped or excluded
        or any(left == right for left, right in zip(used, used[1:]))
    ):
        return None
    return calls
"""


def available(row: dict) -> bool:
    libs = ast.literal_eval(row["libs"]) if isinstance(row["libs"], str) else row["libs"]
    if any(lib.split(".")[0] not in sys.stdlib_module_names for lib in libs):
        return False
    disallowed = {"os", "pathlib", "socket", "tempfile", "mock", "subprocess", "shutil"}
    if disallowed.intersection(libs):
        return False
    for node in ast.walk(ast.parse(row["test"])):
        if isinstance(node, ast.Import) and any(
            n.name.split(".")[0] not in sys.stdlib_module_names
            or any(part in disallowed for part in n.name.split("."))
            for n in node.names
        ):
            return False
        if isinstance(node, ast.ImportFrom) and (
            not node.module
            or node.module.split(".")[0] not in sys.stdlib_module_names
            or any(part in disallowed for part in node.module.split("."))
        ):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", type=Path, required=True)
    parser.add_argument("--bcb", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    if file_hash(args.bcb) != BCB_SHA:
        raise ValueError("Expected pinned BigCodeBench fixtures")
    releases = release_rows(args.pdb)
    rows = sorted(
        (row for row in releases["PDB-Single"] if row["source_dataset"] == "bigcodebench"),
        key=lambda row: digest([21, row["task_id"]]),
    )
    upstream = read_json(args.bcb)
    worker = SandboxBackend(SandboxConfig())
    proof = worker.preflight()
    records: dict = {}
    counts: Counter[str] = Counter()
    attempted: set[str] = set()
    for row in rows:
        parent = parent_key(row).split(":", 1)[1]
        if parent in attempted:
            continue
        attempted.add(parent)
        entry = upstream[parent]
        try:
            bcb_cases(entry)
        except (ValueError, SyntaxError, TypeError):
            pass
        else:
            counts["exact_assertions"] += 1
            continue
        try:
            supported = available(entry)
        except SyntaxError:
            supported = False
        if not supported:
            counts["unsupported_fixture"] += 1
            continue
        source = recorder(row["gt_solution"], entry["test"], entry["entry_point"])
        samples = []
        for _ in range(2):
            result = worker.execute(source, "collect", [encode([])], 20, isolated_harness=True)
            if result.status in {Status.INFRA, Status.DEPENDENCY}:
                raise RuntimeError("BigCodeBench preparation worker failed")
            if result.status != Status.OK or not result.results or not result.results[0]["ok"]:
                break
            calls = decode(result.results[0]["value"])
            if not calls:
                break
            samples.append(calls)
        if len(samples) < 2 or samples[0] != samples[1]:
            counts["nonreproducible_or_unsupported"] += 1
            continue
        unique: dict[str, dict] = {}
        for inputs, (comparison, expected) in samples[0]:
            key = digest(encode(inputs))
            case = {"args": encode(inputs), "expected": encode(expected), "comparison": comparison}
            previous = unique.setdefault(key, case)
            if previous != case:
                break
        else:
            if len(unique) >= 2:
                records[parent] = {
                    "entry": entry["entry_point"],
                    "source_sha256": digest(row["gt_solution"]),
                    "case_count": len(samples[0]),
                    "cases": list(unique.values()),
                }
                counts["adapted"] += 1
                continue
        counts["no_consistent_distinct_inputs"] += 1
    atomic_json(
        args.output,
        {
            "bcb_sha256": BCB_SHA,
            "worker": proof["image_digest"],
            "parents": records,
            "counts": dict(counts),
        },
    )
    print(dict(counts))


if __name__ == "__main__":
    main()
