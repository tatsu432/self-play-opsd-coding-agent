"""Deterministic structured mutations for the Task 13 inference probe.

This module only parses and transforms source; semantic execution remains the
responsibility of the existing isolated verifier.
"""

from __future__ import annotations

import ast
from typing import Any

from selfplay_opsd.mutations import Mutation, make_patch, validate_mutation

OPERATORS = {
    "Add": ast.Add,
    "Sub": ast.Sub,
    "Mult": ast.Mult,
    "Eq": ast.Eq,
    "NotEq": ast.NotEq,
    "Lt": ast.Lt,
    "LtE": ast.LtE,
    "Gt": ast.Gt,
    "GtE": ast.GtE,
}


def sites(source: str) -> list[dict[str, Any]]:
    """List stable AST field paths and compatible transformations."""
    tree = ast.parse(source)
    found: list[dict[str, Any]] = []

    def visit(value: Any, path: list[str | int]) -> None:
        if isinstance(value, ast.Constant) and type(value.value) is int:
            found.append({"path": path, "operations": ["increment", "decrement"]})
        elif isinstance(value, ast.BinOp):
            names = [name for name in ("Add", "Sub", "Mult") if name != type(value.op).__name__]
            if type(value.op).__name__ in {"Add", "Sub", "Mult"}:
                found.append({"path": path + ["op"], "operations": names})
        elif isinstance(value, ast.Compare):
            for index, op in enumerate(value.ops):
                names = [
                    name
                    for name in ("Eq", "NotEq", "Lt", "LtE", "Gt", "GtE")
                    if name != type(op).__name__
                ]
                if type(op).__name__ in OPERATORS:
                    found.append({"path": path + ["ops", index], "operations": names})
        if isinstance(value, ast.AST):
            for key, child in ast.iter_fields(value):
                visit(child, path + [key])
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, path + [index])

    visit(tree, [])
    return found


def mutation(source: str, path: list[str | int], operation: str) -> Mutation:
    """Apply only an enumerated operation and use the existing validator."""
    allowed = {tuple(item["path"]): item["operations"] for item in sites(source)}
    if tuple(path) not in allowed or operation not in allowed[tuple(path)]:
        raise ValueError("Unknown structured mutation site or transformation")
    tree = ast.parse(source)
    target: Any = tree
    for component in path[:-1]:
        target = target[component] if isinstance(component, int) else getattr(target, component)
    last = path[-1]
    current = target[last] if isinstance(last, int) else getattr(target, last)
    if operation in {"increment", "decrement"}:
        current.value += 1 if operation == "increment" else -1
    elif isinstance(last, int):
        target[last] = OPERATORS[operation]()
    else:
        setattr(target, last, OPERATORS[operation]())
    modified = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    return validate_mutation(source, make_patch(source, modified))
