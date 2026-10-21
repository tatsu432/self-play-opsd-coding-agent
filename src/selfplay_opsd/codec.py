"""Bounded, explicit JSON values for sandbox IPC. Never deserialize Python objects."""

from __future__ import annotations

import math
from typing import Any


def encode(value: Any, depth: int = 0) -> Any:
    if depth > 64:
        raise ValueError("Value nesting exceeds 64")
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else {"@": "float", "v": repr(value)}
    if type(value) is complex:
        return {"@": "complex", "v": [encode(value.real), encode(value.imag)]}
    if type(value) in (list, tuple, set, frozenset):
        items = [encode(v, depth + 1) for v in value]
        if type(value) in (set, frozenset):
            items.sort(key=repr)
        return {"@": type(value).__name__, "v": items}
    if type(value) is dict:
        return {
            "@": "dict",
            "v": [[encode(k, depth + 1), encode(v, depth + 1)] for k, v in value.items()],
        }
    # EvalPlus regex-return tasks require only not-None semantics. Never invoke
    # arbitrary serialization hooks, repr, equality, or properties in the supervisor.
    if type(value).__module__ == "re" and type(value).__name__ == "Match":
        return {"@": "non_none", "v": True}
    raise ValueError(f"Unsupported result type: {type(value).__module__}.{type(value).__name__}")


def decode(value: Any, depth: int = 0) -> Any:
    if depth > 64:
        raise ValueError("Value nesting exceeds 64")
    if value is None or type(value) in (bool, int, float, str):
        return value
    if not isinstance(value, dict) or set(value) != {"@", "v"}:
        raise ValueError("Expected tagged JSON value")
    kind, body = value["@"], value["v"]
    if kind == "float" and body in ("inf", "-inf", "nan"):
        return float(body)
    if kind == "complex" and isinstance(body, list) and len(body) == 2:
        return complex(decode(body[0], depth + 1), decode(body[1], depth + 1))
    if kind == "non_none" and body is True:
        return True
    if kind == "dict" and isinstance(body, list):
        return {decode(k, depth + 1): decode(v, depth + 1) for k, v in body}
    if kind in {"list", "tuple", "set", "frozenset"} and isinstance(body, list):
        constructors = {"list": list, "tuple": tuple, "set": set, "frozenset": frozenset}
        return constructors[kind](decode(v, depth + 1) for v in body)
    raise ValueError("Unknown tagged JSON value")
