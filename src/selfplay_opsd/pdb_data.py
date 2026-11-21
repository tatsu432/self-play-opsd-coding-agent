"""Materialize reproducible, parent-disjoint PDB tasks from pinned releases."""

from __future__ import annotations

import ast
import base64
import json
import pickletools
import re
import sys
import zlib
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

import pyarrow.parquet as pq

from selfplay_opsd.codec import decode, encode
from selfplay_opsd.core import Case, ParentProgram, TaskSpec, digest
from selfplay_opsd.mutations import make_patch, normalized_family
from selfplay_opsd.pdb_verifier import (
    PROTOCOL,
    Handler,
    PdbOracle,
    bcb_case,
    call_case,
    entry_signature,
)
from selfplay_opsd.storage import atomic_json, file_hash

SCORER = "fbfa901eeae5b5f6944daea479a4ce31fad0e4a6"
LCB_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
RELEASES = {
    "PDB-Single-Full": (
        "dac1f9aea17f507b2df3f253f06ff545bbdf2ce1",
        "68805114e527c989c88dfc23a00b60df3a24a87b45394d5bb6d8752632b4c2bc",
    ),
    "PDB-Single": (
        "b3c0b911c5df4444ff3e8160b8dbe4ea297c08ec",
        "db512661e7aad3be0c080f8892f3f274e50e8ca9583d8bcb02374066ba090f37",
    ),
    "PDB-Multi": (
        "cb13eb9a160a702745d22129149083c14bedf215",
        "d51021e4bae2ff1a46d3ba32ecd0f9460f776c0708514148a0cfaff74426c648",
    ),
}
QUESTION_ID = re.compile(rb'"question_id"\s*:\s*"([^"\\]+)"')
MAX_FIXTURE_BYTES = 64 * 1024 * 1024


def parent_key(row: dict) -> str:
    task_id = row["task_id"]
    parent, suffix = task_id.rsplit("_", 1)
    if not suffix.isdecimal() or row["source_dataset"] not in {"bigcodebench", "livecodebench"}:
        raise ValueError("Unrecognized PDB task identifier")
    return f"{row['source_dataset']}:{parent}"


def release_rows(root: Path) -> dict[str, list[dict]]:
    rows = {}
    for name, (_, sha) in RELEASES.items():
        path = root / f"{name}.parquet"
        if file_hash(path) != sha:
            raise ValueError(f"Unexpected PDB release checksum for {name}")
        rows[name] = pq.read_table(path).to_pylist()
    return rows


def lcb_rows(stream: Iterable[bytes], wanted: set[str]) -> Iterable[dict]:
    for line in stream:
        match = QUESTION_ID.search(line)
        if match and match[1].decode() in wanted:
            row = json.loads(line)
            if row["question_id"] != match[1].decode():
                raise ValueError("LiveCodeBench question ID mismatch")
            yield row


def private_lcb(value: str) -> list[dict]:
    """Read the upstream JSON or a pickled JSON string without unpickling."""
    if value.startswith("["):
        return json.loads(value)
    decoder = zlib.decompressobj()
    raw = decoder.decompress(base64.b64decode(value, validate=True), MAX_FIXTURE_BYTES + 1)
    if (
        len(raw) > MAX_FIXTURE_BYTES
        or not decoder.eof
        or decoder.unused_data
        or decoder.unconsumed_tail
    ):
        raise ValueError("Invalid or oversized compressed LCB fixture")
    operations = list(pickletools.genops(raw))
    names = [op.name for op, _, _ in operations]
    if (
        len(names) != 5
        or names[:2] != ["PROTO", "FRAME"]
        or names[2] not in {"BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8"}
        or names[3:] != ["MEMOIZE", "STOP"]
        or operations[-1][2] != len(raw) - 1
    ):
        raise ValueError("Expected an LCB pickled JSON string")
    payload = operations[2][1]
    if not isinstance(payload, str):
        raise ValueError("Expected an LCB JSON string")
    cases = json.loads(payload)
    if not isinstance(cases, list):
        raise ValueError("Invalid LCB private fixture")
    return cases


def lcb_cases(row: dict) -> tuple[Handler, str, list[Case], list[Case]]:
    entry = json.loads(row["metadata"]).get("func_name")
    handler: Handler = "lcb_call" if entry else "lcb_stdio"
    splits = json.loads(row["public_test_cases"]), private_lcb(row["private_test_cases"])
    expected_type = "functional" if entry else "stdin"
    cases: list[list[Case]] = []
    for split in splits:
        if not split or any(case["testtype"] != expected_type for case in split):
            raise ValueError("Missing or unsupported LiveCodeBench case type")
        if entry:
            cases.append(
                [
                    call_case(
                        [json.loads(line) for line in case["input"].split("\n")],
                        json.loads(case["output"]),
                    )
                    for case in split
                ]
            )
        else:
            cases.append(
                [
                    Case(args=encode([case["input"]]), expected=encode(case["output"]))
                    for case in split
                ]
            )
    return handler, entry or "__main__", cases[0], cases[1]


def literal(node: ast.expr, names: dict):
    if isinstance(node, ast.Name) and node.id in names:
        return names[node.id]
    if isinstance(node, ast.Call) and not node.keywords:
        func = ast.unparse(node.func)
        if func == "Counter" and len(node.args) <= 1:
            return Counter(*(literal(arg, names) for arg in node.args))
        if func == "json.dumps" and len(node.args) == 1:
            return json.dumps(literal(node.args[0], names))
        if isinstance(node.func, ast.Attribute) and node.func.attr == "encode":
            value = literal(node.func.value, names)
            charset = [literal(arg, names) for arg in node.args]
            if isinstance(value, str) and charset in ([], ["utf-8"]):
                return value.encode()
    return ast.literal_eval(node)


def bcb_assertions(test: str, entry: str) -> list[Case]:
    """Translate a complete supported unittest suite to coordinator-side assertions."""
    tree = ast.parse(test)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    if (
        len(classes) != 1
        or classes[0].name != "TestCases"
        or classes[0].decorator_list
        or [ast.unparse(base) for base in classes[0].bases] != ["unittest.TestCase"]
        or any(
            not (
                node is classes[0]
                or isinstance(node, ast.Import)
                and all(
                    name.asname is None and name.name in sys.stdlib_module_names
                    for name in node.names
                )
                or isinstance(node, ast.ImportFrom)
                and (node.level == 0 and node.module in sys.stdlib_module_names)
                or isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
            )
            for node in tree.body
        )
    ):
        raise ValueError("BigCodeBench suite requires unsupported imports or fixtures")

    def target(node: ast.expr) -> bool:
        return (
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == entry
        )

    def arguments(node: ast.expr, names: dict) -> tuple[list, dict]:
        if not isinstance(node, ast.Call) or not target(node):
            raise ValueError("Unsupported BigCodeBench assertion target")
        kwargs = {kw.arg: literal(kw.value, names) for kw in node.keywords}
        if None in kwargs:
            raise ValueError("Unsupported unpacking in BigCodeBench test")
        return [literal(arg, names) for arg in node.args], kwargs

    def assertion(node, check: str) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
            and node.func.attr.startswith(check)
            and bool(node.args)
        )

    methods = [
        m
        for m in classes[0].body
        if not (isinstance(m, ast.Expr) and isinstance(m.value, ast.Constant))
    ]
    if not all(
        isinstance(m, ast.FunctionDef) and m.name.startswith("test_") and not m.decorator_list
        for m in methods
    ):
        raise ValueError("Unsupported BigCodeBench test method")
    cases: list[Case] = []
    for method in sorted(methods, key=lambda m: getattr(m, "name", "")):
        assert isinstance(method, ast.FunctionDef)
        names: dict = {}
        assignment = None
        start = len(cases)
        for stmt in method.body:
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
            ):
                name = stmt.targets[0].id
                if target(stmt.value):
                    if assignment is not None:
                        raise ValueError("Dependent BigCodeBench test statements")
                    assignment = (name, stmt.value)
                else:
                    names[name] = literal(stmt.value, names)
                continue
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                continue
            if isinstance(stmt, ast.With) and len(stmt.items) == len(stmt.body) == 1:
                context, body = stmt.items[0].context_expr, stmt.body[0]
                if (
                    assertion(context, "assertRaises")
                    and isinstance(context, ast.Call)
                    and isinstance(context.func, ast.Attribute)
                    and context.func.attr == "assertRaises"
                    and len(context.args) == 1
                    and not context.keywords
                    and stmt.items[0].optional_vars is None
                    and isinstance(body, (ast.Assign, ast.Expr))
                    and assignment is None
                ):
                    exc = ast.unparse(context.args[0]).rsplit(".", 1)[-1]
                    if exc not in {
                        "ValueError",
                        "TypeError",
                        "KeyError",
                        "JSONDecodeError",
                        "IndexError",
                    }:
                        raise ValueError("Unsupported BigCodeBench exception assertion")
                    args, kwargs = arguments(body.value, names)
                    cases.append(bcb_case(args, exc, kwargs=kwargs, comparison="raises"))
                    continue
            call = stmt.value if isinstance(stmt, ast.Expr) else None
            if (
                not assertion(call, "assert")
                or not isinstance(call, ast.Call)
                or not isinstance(call.func, ast.Attribute)
            ):
                raise ValueError("Unsupported BigCodeBench test statement")
            source_call = call.args[0]
            transform = None
            if assignment:
                name, pending = assignment
                if isinstance(source_call, ast.Name) and source_call.id == name:
                    source_call = pending
                    assignment = None
                elif ast.unparse(source_call) == f"sum({name}.values())":
                    source_call = pending
                    assignment = None
                    transform = "sum_values"
            args, kwargs = arguments(source_call, names)
            check = call.func.attr
            options = {kw.arg: literal(kw.value, names) for kw in call.keywords}
            options.pop("msg", None)
            if (
                check in {"assertTrue", "assertFalse"}
                and len(call.args) in {1, 2}
                and not options
                and not transform
            ):
                case = bcb_case(args, check == "assertTrue", kwargs=kwargs, comparison="truth")
            elif (
                check in {"assertEqual", "assertDictEqual", "assertListEqual"}
                and len(call.args) in {2, 3}
                and not options
            ):
                comparison = transform or {
                    "assertDictEqual": "dict",
                    "assertListEqual": "list",
                }.get(check, "equal")
                expected = literal(call.args[1], names)
                if isinstance(expected, Counter) and comparison == "equal":
                    comparison, expected = "counter", dict(expected)
                case = bcb_case(args, expected, kwargs=kwargs, comparison=comparison)
            elif check == "assertAlmostEqual" and len(call.args) in {2, 3, 4} and not transform:
                if set(options) == {"delta"} and len(call.args) == 2:
                    comparison, atol = "delta", options["delta"]
                elif not options or (set(options) == {"places"} and len(call.args) == 2):
                    comparison = "almost_equal"
                    atol = (
                        literal(call.args[2], names)
                        if len(call.args) >= 3
                        else options.get("places", 7)
                    )
                else:
                    raise ValueError("Unsupported BigCodeBench precision")
                case = bcb_case(
                    args,
                    literal(call.args[1], names),
                    kwargs=kwargs,
                    comparison=comparison,
                    atol=atol,
                )
            else:
                raise ValueError("Unsupported BigCodeBench assertion")
            cases.append(case)
        if assignment is not None or len(cases) == start:
            raise ValueError(
                "BigCodeBench method requires every candidate observation to be checked"
            )
    if len({digest(case.args) for case in cases}) < 2:
        raise ValueError("Need distinct public and hidden BigCodeBench inputs")
    return cases


def build_task(
    row: dict,
    info: tuple[Handler, str, list[Case], list[Case]],
    family: str,
    release: str,
    split: str,
) -> tuple[TaskSpec, PdbOracle]:
    handler, entry, public, private = info
    source, canonical = row["buggy_code"], row["gt_solution"]
    if not row.get("is_buggy") or not source or source == canonical:
        raise ValueError("Record does not contain a bug")
    if max(len(source.encode()), len(canonical.encode())) > 65536:
        raise ValueError("Source exceeds native editor limit")
    if entry_signature(source, entry, handler) != entry_signature(canonical, entry, handler):
        raise ValueError("Bug changes benchmark entry point")
    if sum(len(json.dumps(case.args)) for case in [*public, *private]) > 6 * 1024 * 1024:
        raise ValueError("Candidate input batch exceeds worker limit")
    source_hash = digest(source)
    parent = parent_key(row)
    task_id = digest([PROTOCOL, parent, row["task_id"], source_hash])[:24]
    public_examples = [
        {
            "input": decode(case.args),
            "expected": decode(case.expected),
            "comparison": case.comparison,
        }
        for case in public[:5]
    ]
    task = TaskSpec(
        task_id=task_id,
        parent_id=parent,
        split=split,
        specification=row["task_prompt"],
        initial_source=source,
        entry_point=entry,
        source_hash=source_hash,
        mutation_seed=0,
        protocol_version=PROTOCOL,
        public_tests="# Examples used by run_tests; additional public examples may be evaluated.\n"
        + "\n".join("# " + repr(example) for example in public_examples),
    )
    oracle = PdbOracle(
        task_id=task_id,
        correct_source=canonical,
        inverse_patch=make_patch(source, canonical),
        public_cases=public,
        hidden_cases=private,
        handler=handler,
        provenance={
            "release": release,
            "revision": RELEASES[release][0],
            "pdb_id": row["task_id"],
            "family_id": family,
            "source_dataset": row["source_dataset"],
            "bug_type": row["bug_type"],
            "bug_subtype": row["bug_subtype"],
            "bug_count": row["bug_count"],
            "scorer": SCORER,
            "verification": "adapted-worker-parity-pending",
        },
    )
    return task, oracle


def families(releases: dict[str, list[dict]]) -> dict[str, str]:
    """Union upstream parent IDs across duplicate canonical AST families."""
    roots: dict[str, str] = {}
    normalized: dict[str, str] = {}

    def root(parent: str) -> str:
        roots.setdefault(parent, parent)
        while roots[parent] != parent:
            roots[parent] = roots[roots[parent]]
            parent = roots[parent]
        return parent

    for release in ("PDB-Single-Full", "PDB-Multi"):
        for row in releases[release]:
            parent = parent_key(row)
            root(parent)
            try:
                key = normalized_family(row["gt_solution"])
            except (SyntaxError, ValueError):
                key = digest(row["gt_solution"])
            owner = normalized.setdefault(key, parent)
            left, right = sorted((root(parent), root(owner)))
            roots[right] = left
    return {parent: root(parent) for parent in roots}


def variant_order(row: dict) -> str:
    return digest([21, parent_key(row), row["task_id"]])


def eligible(
    releases: dict[str, list[dict]],
    lcb: dict[str, dict],
    bcb: dict,
) -> tuple[dict[str, list[tuple[TaskSpec, PdbOracle]]], Counter[str]]:
    ids = {
        (row["source_dataset"], row["task_id"], digest(row["buggy_code"]))
        for row in releases["PDB-Single"]
    }
    family = families(releases)
    prepared: dict[str, list[tuple[TaskSpec, PdbOracle]]] = defaultdict(list)
    outcomes: Counter[str] = Counter()
    fixtures: dict[str, tuple[Handler, str, list[Case], list[Case]] | str] = {}
    for release in ("PDB-Single-Full", "PDB-Multi"):
        tried: Counter[str] = Counter()
        for row in sorted(releases[release], key=variant_order):
            parent = parent_key(row)
            if tried[parent] >= 3:
                continue
            source, key = parent.split(":", 1)
            try:
                if parent not in fixtures:
                    try:
                        fixtures[parent] = (
                            lcb_cases(lcb[key])
                            if source == "livecodebench"
                            else bcb_cases(bcb[key])
                        )
                    except (ArithmeticError, KeyError, SyntaxError, TypeError, ValueError) as exc:
                        fixtures[parent] = (
                            str(exc)[:95] if not isinstance(exc, KeyError) else "missing fixture"
                        )
                info = fixtures[parent]
                if isinstance(info, str):
                    raise ValueError(info)
                membership = (row["source_dataset"], row["task_id"], digest(row["buggy_code"]))
                selected_release = (
                    "PDB-Single" if membership in ids and release == "PDB-Single-Full" else release
                )
                item = build_task(row, info, family[parent], selected_release, "unassigned")
            except (ArithmeticError, KeyError, SyntaxError, TypeError, ValueError) as exc:
                outcomes[
                    f"{source}:{str(exc)[:95] if not isinstance(exc, KeyError) else 'missing fixture'}"
                ] += 1
                continue
            prepared[release].append(item)
            tried[parent] += 1
    return dict(prepared), outcomes


def bcb_cases(upstream: dict) -> tuple[Handler, str, list[Case], list[Case]]:
    libs = (
        ast.literal_eval(upstream["libs"])
        if isinstance(upstream["libs"], str)
        else upstream["libs"]
    )
    if any(lib.split(".")[0] not in sys.stdlib_module_names for lib in libs):
        raise ValueError("BigCodeBench candidate requires third-party dependencies")
    entry = upstream["entry_point"]
    suite = bcb_assertions(upstream["test"], entry)
    if any(case.args == suite[0].args for case in suite[1:]):
        raise ValueError("Repeated BigCodeBench input would cross visibility")
    return "bcb_call", entry, suite[:1], suite[1:]


def write_bundle(
    root: Path, groups: dict[str, list[tuple[TaskSpec, PdbOracle]]], provenance: dict
) -> None:
    records: dict[str, list[dict]] = {}
    parents = {}
    owners: dict[str, str] = {}
    for split, items in sorted(groups.items()):
        records[split] = []
        for task, oracle in items:
            family = oracle.provenance["family_id"]
            existing = owners.setdefault(family, split)
            if existing != split:
                raise ValueError("PDB source family appears in multiple partitions")
            task = task.model_copy(update={"split": split})
            records[split].append(
                {"task": task.model_dump(mode="json"), "oracle": oracle.model_dump(mode="json")}
            )
            parents[task.parent_id] = ParentProgram(
                parent_id=task.parent_id,
                source=oracle.correct_source,
                specification=task.specification,
                entry_point=task.entry_point,
                dataset=oracle.provenance["source_dataset"],
                provenance=oracle.provenance,
                family_id=family,
            ).model_dump(mode="json")
    payload = {
        "benchmark": PROTOCOL,
        "provenance": provenance,
        "parents": parents,
        "partitions": records,
    }
    atomic_json(root / "pdb.json", {"hash": digest(payload), **payload})
    manifest = {
        "hash": digest(payload),
        "benchmark": PROTOCOL,
        "ready": True,
        "bundle_sha256": file_hash(root / "pdb.json"),
        "counts": {split: len(rows) for split, rows in records.items()},
    }
    atomic_json(root / "manifest.json", manifest)
    atomic_json(
        root / "validation_report.json",
        {
            "executed": True,
            "ready": True,
            "manifest_hash": digest(manifest),
            "provenance": provenance,
        },
    )
