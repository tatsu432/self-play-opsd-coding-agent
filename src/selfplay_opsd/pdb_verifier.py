"""PDB repair verification using the existing isolated candidate worker."""

from __future__ import annotations

import ast
from decimal import Decimal, InvalidOperation
from typing import Literal

from selfplay_opsd.codec import decode, encode
from selfplay_opsd.core import Case, ExperimentError, OracleSpec, Status, TaskSpec
from selfplay_opsd.environment import RepairEnvironment
from selfplay_opsd.mutations import replace_span
from selfplay_opsd.sandbox import SandboxBackend, Verification

PROTOCOL = "pdb-native-v1"
Handler = Literal["lcb_call", "lcb_stdio", "bcb_call"]
LCB_MODULES = (
    "string",
    "re",
    "datetime",
    "collections",
    "heapq",
    "bisect",
    "copy",
    "math",
    "random",
    "statistics",
    "itertools",
    "functools",
    "operator",
    "io",
    "sys",
    "json",
)
# Keep the import order of the pinned LCB scorer. Both forms are required by its harness.
LCB_PRELUDE = "\n".join(
    [f"from {name} import *" for name in (*LCB_MODULES, "builtins", "typing")]
    + ["import " + ", ".join(LCB_MODULES), "sys.setrecursionlimit(50000)"]
)


class PdbOracle(OracleSpec):
    handler: Handler


def entry_signature(source: str, entry: str, handler: Handler) -> tuple:
    tree = ast.parse(source)
    if handler == "lcb_stdio":
        return ()
    classes = [
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Solution"
    ]
    body = classes[-1].body if handler == "lcb_call" and classes else tree.body
    definitions = [
        node for node in body if isinstance(node, ast.FunctionDef) and node.name == entry
    ]
    if not definitions:
        raise ValueError("Repair must preserve the benchmark entry point")
    args = definitions[-1].args
    return (
        bool(body is not tree.body),
        tuple(arg.arg for arg in args.posonlyargs),
        tuple(arg.arg for arg in args.args),
        tuple(arg.arg for arg in args.kwonlyargs),
        args.vararg.arg if args.vararg else None,
        args.kwarg.arg if args.kwarg else None,
    )


def runner_source(source: str, entry: str, handler: Handler) -> str:
    """Prepare code for the worker. Only candidate inputs are sent with it."""
    if handler == "lcb_stdio":
        return f"""
import sys
from contextlib import redirect_stdout
from io import StringIO, TextIOWrapper, BytesIO
from unittest.mock import mock_open, patch
_program = compile({(LCB_PRELUDE + chr(10) + source)!r}, "src/solution.py", "exec")
def invoke(value):
    output = StringIO()
    stream = TextIOWrapper(BytesIO(value.encode()), encoding="utf-8")
    with patch("sys.stdin", stream), patch("builtins.open", mock_open(read_data=value)), redirect_stdout(output):
        try:
            exec(_program, {{"__name__": "__main__"}})
        except SystemExit:
            pass
    return output.getvalue()
"""
    prefix = LCB_PRELUDE + "\n\n" if handler == "lcb_call" else ""
    solution_class = handler == "lcb_call" and "class Solution" in source
    load = f"""
import types
_module = types.ModuleType("tmp_sol")
exec(compile({(prefix + source)!r}, "src/solution.py", "exec"), _module.__dict__)
_target = _module.Solution() if {solution_class!r} else _module
_function = getattr(_target, {entry!r})
"""
    if handler == "bcb_call":
        return (
            load
            + """
from collections import Counter
def argument(item):
    kind, value = item
    return bytes.fromhex(value) if kind == "bytes" else value
def invoke(args, kwargs):
    result = _function(*(argument(value) for value in args), **{key: argument(value) for key, value in kwargs.items()})
    if type(result) is Counter:
        return ["counter", dict(result)]
    return [type(result).__name__, result]
"""
        )
    return (
        load
        + """
def invoke(args, kwargs):
    return _function(*args, **kwargs)
"""
    )


def stdout_equal(output: str, expected: str) -> bool:
    actual_lines = output.strip().split("\n")
    expected_lines = expected.strip().split("\n")
    if len(actual_lines) != len(expected_lines):
        return False
    for actual, target in zip(actual_lines, expected_lines, strict=True):
        if actual.strip() == target.strip():
            continue
        try:
            if [Decimal(x) for x in actual.split()] == [Decimal(x) for x in target.split()]:
                continue
        except InvalidOperation:
            pass
        return False
    return True


def case_passes(case: Case, output, handler: Handler) -> bool:
    expected = decode(case.expected)
    try:
        if handler == "lcb_stdio":
            return isinstance(output, str) and stdout_equal(output, expected)
        if handler == "lcb_call" and isinstance(output, tuple):
            output = list(output)
        if handler == "bcb_call":
            if not isinstance(output, list) or len(output) != 2:
                return False
            kind, output = output
            if case.comparison == "counter":
                from collections import Counter

                return kind == "counter" and Counter(output) == Counter(expected)
            if case.comparison == "sum_values":
                return isinstance(output, dict) and sum(output.values()) == expected
            if case.comparison == "dict":
                return isinstance(output, dict) and output == expected
            if case.comparison == "list":
                return isinstance(output, list) and output == expected
        if case.comparison == "almost_equal":
            return output == expected or round(abs(output - expected), int(case.atol)) == 0
        if case.comparison == "delta":
            return output == expected or abs(output - expected) <= case.atol
        if case.comparison == "truth":
            return bool(output) == expected
        return output == expected
    except (ArithmeticError, TypeError, ValueError):
        return False


def verify_pdb(
    sandbox: SandboxBackend, source: str, task: TaskSpec, oracle: PdbOracle, *, public: bool = False
) -> Verification:
    cases = oracle.public_cases if public else [*oracle.public_cases, *oracle.hidden_cases]
    if task.task_id != oracle.task_id or not cases or (not public and not oracle.hidden_cases):
        raise ExperimentError("Mismatched task or empty PDB verifier suite", Status.ORACLE)
    result = sandbox.execute(
        runner_source(source, task.entry_point, oracle.handler),
        "invoke",
        [case.args for case in cases],
        task.public_seconds if public else task.verifier_seconds,
        isolated_harness=True,
    )
    if result.status in {Status.INFRA, Status.DEPENDENCY, Status.ORACLE}:
        raise ExperimentError(result.detail, result.status)
    if result.status != Status.OK:
        return Verification(
            result.status, False, [False] * len(cases), 0, len(cases), result.seconds
        )
    if len(result.results) != len(cases):
        raise ExperimentError("Incorrect number of PDB worker results", Status.INFRA)
    checks = []
    runtime = 0
    for row, case in zip(result.results, cases, strict=True):
        if case.comparison == "raises":
            checks.append(not row["ok"] and row.get("error") == decode(case.expected))
        elif row["ok"]:
            checks.append(case_passes(case, decode(row["value"]), oracle.handler))
        else:
            checks.append(False)
            runtime += 1
    passed = all(checks)
    return Verification(
        Status.OK if passed else Status.STUDENT_FAILURE,
        passed,
        checks,
        len(checks) - sum(checks) - runtime,
        runtime,
        result.seconds,
    )


class PdbEnvironment(RepairEnvironment):
    def verify_source(self, *, public: bool) -> Verification:
        if not isinstance(self.oracle, PdbOracle) or self.task.protocol_version != PROTOCOL:
            raise ExperimentError("PDB environment requires its matching oracle", Status.ORACLE)
        return verify_pdb(self.sandbox, self.source, self.task, self.oracle, public=public)

    def replace_source(self, old_text: str, new_text: str) -> str:
        if not isinstance(self.oracle, PdbOracle):
            raise ExperimentError("PDB environment requires its matching oracle", Status.ORACLE)
        replacement = replace_span(self.source, old_text, new_text)
        try:
            before = entry_signature(
                self.task.initial_source, self.task.entry_point, self.oracle.handler
            )
            after = entry_signature(replacement, self.task.entry_point, self.oracle.handler)
        except SyntaxError as exc:
            raise ValueError(f"Invalid Python syntax: {exc.msg} (line {exc.lineno})") from exc
        if before != after:
            raise ValueError("Repair must preserve the benchmark entry point")
        return replacement


def call_case(args, expected, *, kwargs=None, comparison="equal", atol=0.0) -> Case:
    return Case(
        args=encode([args, kwargs or {}]),
        expected=encode(expected),
        comparison=comparison,
        atol=atol,
    )


def bcb_case(args, expected, *, kwargs, comparison="equal", atol=0.0) -> Case:
    def argument(value):
        return ["bytes", value.hex()] if type(value) is bytes else ["value", value]

    return call_case(
        [argument(value) for value in args],
        expected,
        kwargs={key: argument(value) for key, value in kwargs.items()},
        comparison=comparison,
        atol=atol,
    )
