"""Compare PDB worker verdicts against the pinned scoring routines on released source."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from selfplay_opsd.codec import decode, encode
from selfplay_opsd.config import SandboxConfig
from selfplay_opsd.core import Status, read_json
from selfplay_opsd.pdb_verifier import PdbOracle, verify_pdb
from selfplay_opsd.sandbox import SandboxBackend
from selfplay_opsd.storage import atomic_json, file_hash
from selfplay_opsd.task21 import PdbRepository

LCB_SCORER_SHA = "b7cb6a8a69807bb868150a61742e25d7bb5328bbe01d514471b6ec43c9fa9ed2"
BCB_SHA = "6079ff7a9f41fb928b3b0750c9e1e160ac36c8cc7c58124efbf86138549131d6"


def lcb_harness(scorer: str, source: str, entry: str, oracle: PdbOracle) -> str:
    cases = [*oracle.public_cases, *oracle.hidden_cases]
    import json

    if oracle.handler == "lcb_call":
        inputs = ["\n".join(json.dumps(arg) for arg in decode(case.args)[0]) for case in cases]
        outputs = [json.dumps(decode(case.expected)) for case in cases]
        command = f"grade_call_based({source!r}, {inputs!r}, {outputs!r}, {entry!r}, 8)"
    else:
        inputs = [decode(case.args)[0] for case in cases]
        outputs = [decode(case.expected) for case in cases]
        command = f"grade_stdio({source!r}, {inputs!r}, {outputs!r}, 8)"
    return (
        scorer
        + f"""
import contextlib
import tempfile
def verdict():
    signal.signal(signal.SIGALRM, timeout_handler)
    with tempfile.TemporaryFile(mode="w+") as error_stream, contextlib.redirect_stderr(error_stream):
        result = {command}
    return result is not None and len(result[0]) == {len(cases)} and all(item is True for item in result[0])
"""
    )


def bcb_harness(source: str, test: str) -> str:
    # Use the pinned upstream candidate-plus-TestCases arrangement in a disposable worker.
    return f"""
import io
import types
import unittest
def verdict():
    module = types.ModuleType("solution")
    exec(compile({(source + chr(10) + test)!r}, "pdb_bcb.py", "exec"), module.__dict__)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(module.TestCases)
    result = unittest.TextTestRunner(stream=io.StringIO()).run(suite)
    return bool(result.testsRun) and not result.skipped and result.wasSuccessful()
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    scorer_path = args.upstream / "lcb-testing-util.py"
    bcb_path = args.upstream / "bigcodebench.json"
    if file_hash(scorer_path) != LCB_SCORER_SHA or file_hash(bcb_path) != BCB_SHA:
        raise ValueError("Upstream calibration inputs do not match the pinned scorer")
    # NumPy is imported but unused by this pinned scorer's call and I/O graders.
    scorer = scorer_path.read_text().replace("import numpy as np\n", "")
    bcb = read_json(bcb_path)
    repository = PdbRepository(args.data)
    worker = SandboxBackend(SandboxConfig())
    worker.preflight()
    rows = []
    for task, oracle in repository.tasks("feasibility"):
        if not isinstance(oracle, PdbOracle):
            raise ValueError("Expected a PDB oracle")
        for label, source in (("canonical", oracle.correct_source), ("buggy", task.initial_source)):
            if oracle.handler == "bcb_call":
                harness = bcb_harness(source, bcb[task.parent_id.split(":", 1)[1]]["test"])
            else:
                harness = lcb_harness(scorer, source, task.entry_point, oracle)
            official = worker.execute(harness, "verdict", [encode([])], 30, isolated_harness=True)
            ran = (
                official.status == Status.OK
                and bool(official.results)
                and official.results[0]["ok"]
            )
            adapted = verify_pdb(worker, source, task, oracle)
            rows.append(
                {
                    "task": task.task_id,
                    "handler": oracle.handler,
                    "source": label,
                    "official_pass": bool(decode(official.results[0]["value"])) if ran else None,
                    "official_status": official.status.value,
                    "official_error": official.results[0].get("error")
                    if official.results
                    else None,
                    "adapted_pass": adapted.passed,
                }
            )
    matched = sum(row["official_pass"] is row["adapted_pass"] for row in rows)
    atomic_json(
        args.output,
        {
            "bundle": repository.hash,
            "worker": worker.identity(),
            "lcb_scorer_sha256": LCB_SCORER_SHA,
            "bcb_fixture_sha256": BCB_SHA,
            "rows": rows,
            "matches": matched,
        },
    )
    print(f"{matched}/{len(rows)} original-source verdicts agreed")


if __name__ == "__main__":
    main()
