"""Conservative deterministic Python mutation and unified-diff validation."""

from __future__ import annotations

import ast
import builtins
import difflib
import io
import json
import random
import re
import tokenize
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, cast

from selfplay_opsd.core import digest

SAFE_IMPORTS = frozenset(
    {
        "math",
        "cmath",
        "collections",
        "itertools",
        "functools",
        "operator",
        "heapq",
        "bisect",
        "statistics",
        "string",
        "re",
        "typing",
        "fractions",
        "decimal",
        "array",
        "copy",
        "enum",
        "numbers",
    }
)
FORBIDDEN_NAMES = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "open",
        "input",
        "print",
        "__import__",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
        "dir",
        "help",
        "breakpoint",
        "exit",
        "quit",
        "memoryview",
        "object",
        "super",
        "classmethod",
        "staticmethod",
        "property",
    }
)
DANGEROUS_ATTRIBUTES = frozenset(
    {
        "attrgetter",
        "methodcaller",
        "Formatter",
        "gi_frame",
        "gi_code",
        "cr_frame",
        "cr_code",
        "ag_frame",
        "ag_code",
        "f_globals",
        "f_locals",
        "f_builtins",
        "f_back",
        "f_code",
        "tb_frame",
        "tb_next",
        "mro",
    }
)


def normalized_ast(source: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body.pop(0)
    return ast.dump(tree, include_attributes=False)


def normalized_family(source: str) -> str:
    """Alpha-normalized AST for transparent exact family grouping."""
    tree = ast.parse(source)
    names: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.setdefault(node.name, f"id{len(names)}")
        elif isinstance(node, ast.arg):
            names.setdefault(node.arg, f"id{len(names)}")
        elif isinstance(node, ast.Name) and node.id not in dir(builtins):
            names.setdefault(node.id, f"id{len(names)}")
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            node.name = names.get(node.name, node.name)
        elif isinstance(node, ast.arg):
            node.arg = names.get(node.arg, node.arg)
        elif isinstance(node, ast.Name):
            node.id = names.get(node.id, node.id)
    return normalized_ast(ast.unparse(tree))


def validate_source(source: str) -> ast.Module:
    tree = ast.parse(source)
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    if len(source.encode()) > 131072:
        raise ValueError("Source exceeds size limit")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = (
                [node.module] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names]
            )
            if isinstance(node, ast.ImportFrom) and node.level:
                raise ValueError("Relative imports are forbidden")
            if any(not m or m.split(".")[0] not in SAFE_IMPORTS for m in modules):
                raise ValueError("Unsupported or unsafe import")
            if any(a.asname and a.asname.startswith("_") for a in node.names):
                raise ValueError("Private import alias")
            if any(a.name.startswith("_") or a.name == "*" for a in node.names):
                raise ValueError("Private or wildcard import")
            if any(a.name in DANGEROUS_ATTRIBUTES for a in node.names):
                raise ValueError("Introspection import is forbidden")
        if isinstance(node, ast.Attribute) and (
            node.attr.startswith("_") or node.attr in DANGEROUS_ATTRIBUTES
        ):
            raise ValueError("Private/introspection attributes are forbidden")
        if isinstance(node, ast.Name) and (node.id in FORBIDDEN_NAMES or node.id.startswith("__")):
            raise ValueError(f"Forbidden name: {node.id}")
        if isinstance(node, ast.Name) and node.id == "type":
            parent = parents.get(node)
            if (
                not isinstance(parent, ast.Call)
                or parent.func is not node
                or len(parent.args) != 1
                or parent.keywords
            ):
                raise ValueError("Only direct single-argument type checks are supported")
        if isinstance(
            node,
            (
                ast.AsyncFunctionDef,
                ast.Await,
                ast.With,
                ast.AsyncWith,
                ast.ClassDef,
                ast.Global,
                ast.Nonlocal,
            ),
        ):
            raise ValueError(f"Unsupported syntax: {type(node).__name__}")
        if isinstance(node, ast.FunctionDef) and (
            node.decorator_list or node.name.startswith("__")
        ):
            raise ValueError("Decorators/private functions are unsupported")
    # Admission supports normal function definitions, imports, constants and helpers;
    # it does not authorize arbitrary module-level execution.
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            continue
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            try:
                ast.literal_eval(node.value) if node.value is not None else None
                continue
            except (ValueError, TypeError):
                pass
        raise ValueError("Unsupported executable module-level statement")
    return tree


def signature(tree: ast.Module) -> tuple:
    return tuple(
        (
            n.name,
            ast.dump(n.args),
            ast.dump(n.returns) if n.returns else None,
            tuple(ast.dump(d) for d in n.decorator_list),
        )
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
    )


def changed_ast_nodes(a: Any, b: Any) -> int:
    """Count changed AST edit roots; a changed subtree kind counts once.

    Equal-kind nodes recurse; primitive-field changes count their containing node
    once. Inserted/deleted subtree roots each count once. Locations are ignored.
    """
    if type(a) is not type(b):
        return 1
    if isinstance(a, ast.AST):
        fields = [(getattr(a, k), getattr(b, k)) for k in a._fields]
        direct = any(not isinstance(x, (ast.AST, list)) and x != y for x, y in fields)
        return int(direct) + sum(
            changed_ast_nodes(x, y) for x, y in fields if isinstance(x, (ast.AST, list))
        )
    if isinstance(a, list):
        return sum(changed_ast_nodes(x, y) for x, y in zip(a, b, strict=False)) + abs(
            len(a) - len(b)
        )
    return int(a != b)


def make_patch(before: str, after: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(True),
            after.splitlines(True),
            fromfile="a/src/solution.py",
            tofile="b/src/solution.py",
        )
    )


def apply_patch(source: str, patch: str) -> str:
    """Strict single-file unified diff: exact context, no fuzz, rename or path escapes."""
    if len(patch.encode()) > 65536:
        raise ValueError("Patch exceeds size limit")
    lines = patch.splitlines(keepends=True)
    if (
        len(lines) < 3
        or lines[0].rstrip() not in {"--- a/src/solution.py", "--- src/solution.py"}
        or lines[1].rstrip() not in {"+++ b/src/solution.py", "+++ src/solution.py"}
    ):
        raise ValueError("Only src/solution.py may be patched; two exact diff headers required")
    original, output = source.splitlines(keepends=True), []
    cursor, i, seen = 0, 2, False
    while i < len(lines):
        match = re.fullmatch(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@[^\n]*\n?", lines[i])
        if not match:
            raise ValueError("Invalid hunk header")
        old_start, old_count, new_start, new_count = (
            int(match[1]),
            int(match[2] or 1),
            int(match[3]),
            int(match[4] or 1),
        )
        start = old_start - 1 if old_count else old_start
        if start < cursor or start > len(original):
            raise ValueError("Overlapping/out-of-range hunk")
        output.extend(original[cursor:start])
        cursor = start
        if new_start != len(output) + (1 if new_count else 0):
            raise ValueError("Inconsistent new hunk offset")
        removed = added = 0
        i += 1
        while i < len(lines) and not lines[i].startswith("@@ "):
            line = lines[i]
            if line.startswith("\\ No newline"):
                raise ValueError("No-newline markers unsupported; source must end in newline")
            if not line or line[0] not in " +-":
                raise ValueError("Invalid hunk line or extra file")
            body = line[1:]
            if line[0] in " -":
                if cursor >= len(original) or original[cursor] != body:
                    raise ValueError("Patch context mismatch")
                cursor += 1
                removed += 1
            if line[0] in " +":
                output.append(body)
                added += 1
            i += 1
        if (removed, added) != (old_count, new_count):
            raise ValueError("Hunk count mismatch")
        seen = True
    if not seen:
        raise ValueError("Empty patch")
    return "".join(output + original[cursor:])


@dataclass(frozen=True)
class Mutation:
    source: str
    patch: str
    inverse_patch: str
    dedup_key: str
    changed_nodes: int
    changed_lines: int
    family: str


def validate_mutation(original: str, patch: str) -> Mutation:
    source = apply_patch(original, patch)
    before, after = validate_source(original), validate_source(source)
    if signature(before) != signature(after):
        raise ValueError("API changed")

    def imports(tree):
        return [ast.dump(n) for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]

    if imports(before) != imports(after):
        raise ValueError("Imports changed")

    def comments(text):
        return [
            token.string
            for token in tokenize.generate_tokens(io.StringIO(text).readline)
            if token.type == tokenize.COMMENT
        ]

    if comments(original) != comments(source):
        raise ValueError("Proposer comments/metadata may not change")

    def strings(tree):
        return [
            node.value.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ]

    if strings(before) != strings(after):
        raise ValueError("Proposer docstrings/metadata may not change")
    if any(isinstance(node, ast.Pass) for node in ast.walk(after)):
        raise ValueError("Deleting implementation bodies is forbidden")
    if any(isinstance(n, ast.Raise) for n in ast.walk(after)):
        # Conservative initial study excludes new or pre-existing raises rather
        # than attempting to infer whether they are deliberately unconditional.
        raise ValueError("Exception-based mutants unsupported")
    changes = changed_ast_nodes(before, after)
    changed_lines = sum(
        bool(line[1:].strip())
        for line in patch.splitlines()
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    )
    if not 1 <= changes <= 3 or changed_lines > 12:
        raise ValueError("Mutation exceeds 3 AST edit roots / 12 changed nonblank lines")
    if any(len(n.body) == 0 for n in ast.walk(after) if isinstance(n, ast.FunctionDef)):
        raise ValueError("Deleted function")
    # Only admitted node substitutions/updates; top-level declaration structure
    # and the set of function definitions must remain fixed.
    if len(before.body) != len(after.body):
        raise ValueError("Top-level structure changed")
    family = (
        "comparison"
        if any(isinstance(n, ast.Compare) for n in ast.walk(after))
        else "operator_or_constant"
    )
    return Mutation(
        source,
        patch,
        make_patch(source, original),
        digest(normalized_ast(source)),
        changes,
        changed_lines,
        family,
    )


def fixed_mutation(source: str, seed: int) -> Mutation:
    tree = validate_source(source)
    # ast operator nodes can be shared singletons: make each site independent.
    for expression in ast.walk(tree):
        if isinstance(expression, ast.BinOp):
            expression.op = type(expression.op)()
        elif isinstance(expression, ast.Compare):
            expression.ops = [type(op)() for op in expression.ops]
    candidates: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            candidates.append(node)
        elif isinstance(
            node, (ast.Add, ast.Sub, ast.Mult, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)
        ):
            candidates.append(node)
    if not candidates:
        raise ValueError("No admitted mutation sites")
    rng = random.Random(seed)
    index = rng.randrange(len(candidates))
    target = candidates[index]
    replacements = {
        ast.Add: ast.Sub,
        ast.Sub: ast.Add,
        ast.Mult: ast.Add,
        ast.Eq: ast.NotEq,
        ast.NotEq: ast.Eq,
        ast.Lt: ast.LtE,
        ast.LtE: ast.Lt,
        ast.Gt: ast.GtE,
        ast.GtE: ast.Gt,
    }

    class Mutator(ast.NodeTransformer):
        def generic_visit(self, node):
            if node is target:
                if isinstance(node, ast.Constant):
                    return ast.copy_location(
                        ast.Constant(cast(int, node.value) + rng.choice([-1, 1])), node
                    )
                return replacements[type(node)]()
            return super().generic_visit(node)

    changed = ast.unparse(ast.fix_missing_locations(Mutator().visit(tree))) + "\n"
    return validate_mutation(source, make_patch(source, changed))


def strict_json_object(raw: str) -> dict:
    def unique_pairs(pairs):
        result = {}
        for k, v in pairs:
            if k in result:
                raise ValueError("Duplicate JSON field")
            result[k] = v
        return result

    def invalid_constant(value):
        raise ValueError("Nonfinite JSON constants are forbidden")

    try:
        value = json.loads(raw, object_pairs_hook=unique_pairs, parse_constant=invalid_constant)
    except RecursionError as exc:
        raise ValueError("JSON nesting exceeds parser limit") from exc
    if not isinstance(value, dict):
        raise ValueError("One JSON object required")
    return value


def parse_action(raw: str) -> dict:
    value = strict_json_object(raw)
    action = value.get("action")
    if not isinstance(action, str):
        raise ValueError("Action must be a string")
    fields = {
        "read_file": {"action", "path"},
        "apply_patch": {"action", "patch"},
        "run_tests": {"action"},
        "finish": {"action"},
    }
    if (
        action not in fields
        or set(value) != fields[action]
        or any(not isinstance(v, str) for v in value.values())
    ):
        raise ValueError("Unknown action or incorrect fields")
    if action == "read_file":
        path = value["path"]
        if str(PurePosixPath(path)) != path or path not in {
            "README.md",
            "src/solution.py",
            "tests/test_public.py",
        }:
            raise ValueError("File path is not allowlisted")
    return value
