from __future__ import annotations

import base64
import json
import pickle
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from selfplay_opsd.codec import decode, encode
from selfplay_opsd.config import SandboxConfig
from selfplay_opsd.core import ExperimentError, Status
from selfplay_opsd.pdb_data import bcb_assertions, build_task, private_lcb, write_bundle
from selfplay_opsd.pdb_verifier import (
    PdbEnvironment,
    PdbOracle,
    call_case,
    stdout_equal,
    verify_pdb,
)
from selfplay_opsd.sandbox import Execution, SandboxBackend
from selfplay_opsd.task21 import PdbRepository, Task21Control


def task_pair(*, mode="lcb_call"):
    return build_task(
        {
            "source_dataset": "livecodebench",
            "task_id": "fixture_0",
            "buggy_code": "class Solution:\n    def solve(self, n):\n        return n - 1\n",
            "gt_solution": "class Solution:\n    def solve(self, n):\n        return n + 1\n",
            "task_prompt": "Increment the input.",
            "is_buggy": True,
            "bug_type": "logic",
            "bug_subtype": "operator",
            "bug_count": 1,
        },
        (mode, "solve", [call_case([8], 9)], [call_case([174671], 174672)]),
        "fixture",
        "PDB-Single",
        "feasibility",
    )


def test_visibility_and_infrastructure_remain_distinct():
    task, oracle = task_pair()
    requests = []

    def execute(source, entry, inputs, seconds, *, isolated_harness):
        assert entry == "invoke" and isolated_harness
        assert "174672" not in source and "174672" not in str(inputs)
        assert "return n + 1" not in source
        requests.append(inputs)
        return Execution(Status.OK, [{"ok": True, "value": 9} for _ in inputs], 0.1)

    sandbox = SimpleNamespace(execute=execute)
    policy = SimpleNamespace(is_coordinator=True, broadcast=lambda x: x)
    episode = PdbEnvironment(task, oracle, sandbox, policy)
    assert "174671" not in str(task.student_files())
    assert episode.step({"action": "run_tests"})["public_passed"]
    assert requests[0] == [oracle.public_cases[0].args]
    assert not episode.verify().passed
    assert requests[1] == [oracle.public_cases[0].args, oracle.hidden_cases[0].args]
    sandbox.execute = lambda *a, **kw: Execution(Status.INFRA, [], 0.0, "offline")
    with pytest.raises(ExperimentError, match="offline"):
        episode.verify()


def test_pdb_exact_text_preserves_class_entry():
    task, oracle = task_pair()
    env = PdbEnvironment(task, oracle, None, None)
    with pytest.raises(ValueError, match="entry point"):
        env.step(
            {
                "action": "replace_text",
                "path": "src/solution.py",
                "old_text": "solve",
                "new_text": "other",
            }
        )
    assert env.source == task.initial_source
    env.step(
        {
            "action": "replace_text",
            "path": "src/solution.py",
            "old_text": "n - 1",
            "new_text": "n + 1",
        }
    )
    assert env.source == oracle.correct_source


def test_lcb_wire_and_stdio_comparisons():
    payload = [{"input": "2", "output": "3", "testtype": "functional"}]
    packed = base64.b64encode(zlib.compress(pickle.dumps(json.dumps(payload), protocol=4)))
    assert private_lcb(packed.decode()) == payload
    forbidden = base64.b64encode(zlib.compress(pickle.dumps({"not": "text"})))
    with pytest.raises(ValueError, match="pickled JSON string"):
        private_lcb(forbidden.decode())
    assert stdout_equal("1.00 2.0\n\nhello", "1 2\n\n hello\n")
    assert not stdout_equal("1.00000000001", "1")
    assert not stdout_equal("1\n\n2", "1\n2")


def test_bcb_translation_keeps_whole_suite():
    suite = """import unittest
import json
class TestCases(unittest.TestCase):
    def test_a(self):
        value = json.dumps({'n': 6}).encode('utf-8')
        out = task_func(value)
        self.assertEqual(out, 7)
    def test_b(self):
        self.assertEqual(task_func('2', amount=3), 5)
"""
    cases = bcb_assertions(suite, "task_func")
    assert len(cases) == 2
    assert decode(cases[0].args)[0][0] == ["bytes", b'{"n": 6}'.hex()]
    assert decode(cases[1].expected) == 5
    with pytest.raises(ValueError):
        bcb_assertions(
            suite
            + "    def test_unsupported(self):\n        self.assertGreater(task_func(1), 1)\n",
            "task_func",
        )


def test_bundle_identity_and_science(tmp_path: Path):
    task, oracle = task_pair()
    write_bundle(tmp_path, {"feasibility": [(task, oracle)]}, {})
    repo = PdbRepository(tmp_path)
    assert repo.tasks("feasibility") == [(task, oracle)]
    assert isinstance(repo.parent_oracle(task.parent_id), PdbOracle)
    with pytest.raises(ValueError, match="multiple partitions"):
        write_bundle(tmp_path, {"feasibility": [(task, oracle)], "screen": [(task, oracle)]}, {})
    control = Task21Control(qualification_hash="verified")
    assert control.science != Task21Control(seed=2027, qualification_hash="verified").science
    (tmp_path / "pdb.json").write_text("{}")
    with pytest.raises(ValueError, match="SHA-256"):
        PdbRepository(tmp_path)


@pytest.mark.sandbox
def test_worker_executes_class_and_stdin_programs():
    task, oracle = task_pair()
    backend = SandboxBackend(SandboxConfig())
    assert verify_pdb(backend, oracle.correct_source, task, oracle).passed
    assert verify_pdb(backend, task.initial_source, task, oracle).semantic_failures == 2
    stdio = oracle.model_copy(
        update={
            "handler": "lcb_stdio",
            "public_cases": [oracle.public_cases[0].model_copy(update={"args": encode(["2\n"])})],
            "hidden_cases": [
                oracle.hidden_cases[0].model_copy(update={"args": encode(["174671\n"])})
            ],
        }
    )
    stdio = stdio.model_copy(
        update={
            "public_cases": [stdio.public_cases[0].model_copy(update={"expected": encode("9")})],
            "hidden_cases": [
                stdio.hidden_cases[0].model_copy(update={"expected": encode("174678")})
            ],
        }
    )
    assert verify_pdb(backend, "print(int(input()) + 7)", task, stdio).passed
