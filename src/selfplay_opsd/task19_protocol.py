"""Native Qwen3.5 tool calls and reasoning-free conversation history."""

from __future__ import annotations

import re
from typing import Any

from selfplay_opsd.config import EpisodeConfig
from selfplay_opsd.core import ParentProgram, TaskSpec, canonical
from selfplay_opsd.protocol import budget
from selfplay_opsd.structured_mutations import sites
from selfplay_opsd.task18_protocol import native_tools

STUDENT_TOOLS = native_tools()
FIELD = re.compile(r"<parameter=([a-z_]+)>\n(.*?)\n</parameter>\n", re.DOTALL)
CALL = re.compile(r"<tool_call>\n<function=([a-z_]+)>\n(.*?)</function>\n</tool_call>\Z", re.DOTALL)
CHAT_CONTROL = re.compile(r"<\|[^<>\n]+\|>")
NATIVE_TAGS = re.compile(r"</?(?:tool_call|function|parameter|think)(?:[=>])")

PROPOSER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }
    for name, description, properties in (
        (
            "mutate_ast",
            "Apply one listed operator at its listed AST site",
            {"site": {"type": "integer"}, "operation": {"type": "string"}},
        ),
        (
            "mutate_text",
            "Introduce a semantic bug by replacing a single unique exact span",
            {"old_text": {"type": "string"}, "new_text": {"type": "string"}},
        ),
    )
]


def safe_json(value: Any) -> str:
    # Transport potentially model-authored text as JSON without role delimiters.
    return canonical(value).replace("<", "\\u003c")


def parse_call(text: str, tools: list[dict]) -> tuple[str, dict]:
    text = text.strip()
    start = text.find("<tool_call>")
    if start < 0 or CHAT_CONTROL.search(text) or NATIVE_TAGS.search(text[:start]):
        raise ValueError("Expected one native Qwen3.5 call")
    match = CALL.fullmatch(text[start:])
    if match is None:
        raise ValueError("Malformed native tool call")
    name, body = match.groups()
    schemas = {tool["function"]["name"]: tool["function"]["parameters"] for tool in tools}
    if name not in schemas:
        raise ValueError("Unknown native tool")
    schema = schemas[name]
    values: dict[str, str | int] = {}
    while body:
        field = FIELD.match(body)
        if field is None:
            raise ValueError("Malformed native parameter")
        key, value = field.groups()
        if key in values or key not in schema["properties"] or NATIVE_TAGS.search(value):
            raise ValueError("Unexpected, duplicate or reserved native parameter")
        prop = schema["properties"][key]
        if prop["type"] == "integer":
            if re.fullmatch(r"[0-9]{1,8}", value) is None:
                raise ValueError("Expected a nonnegative native integer")
            values[key] = int(value)
        else:
            if "enum" in prop and value not in prop["enum"]:
                raise ValueError("Value is outside the native tool allowlist")
            values[key] = value
        body = body[field.end() :]
    if set(values) != set(schema["required"]):
        raise ValueError("Missing native parameters")
    return name, values


def render_call(name: str, arguments: dict) -> str:
    fields = "".join(
        f"<parameter={key}>\n{value}\n</parameter>\n" for key, value in arguments.items()
    )
    return f"<tool_call>\n<function={name}>\n{fields}</function>\n</tool_call>"


def student_history(task: TaskSpec, limits: EpisodeConfig) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "Repair the Python behavior described in README.md while preserving the public "
                "API. Read the allowed files, use exact text edits, and run public tests as "
                "needed. Public tests are read-only. Call exactly one function per reply. "
                "Use finish when the repair is ready."
            ),
        },
        {
            "role": "user",
            "content": safe_json(
                {
                    "specification": task.specification,
                    "files": list(task.student_files()),
                    "budget": budget(limits),
                }
            ),
        },
    ]


def proposer_history(parent: ParentProgram) -> list[dict]:
    catalog = [
        {"site": index, "operations": site["operations"], "path": site["path"]}
        for index, site in enumerate(sites(parent.source))
    ]
    return [
        {
            "role": "system",
            "content": (
                "Introduce a small semantic bug into this verified Python implementation. "
                "Preserve its API, imports, comments and docstrings. Keep the bug local, "
                "solvable and observable on valid inputs. Use exactly one native tool call. "
                "For mutate_text use an exact unique span including indentation and no diff."
            ),
        },
        {
            "role": "user",
            "content": safe_json(
                {
                    "specification": parent.specification,
                    "source": parent.source,
                    "ast_sites": catalog,
                }
            ),
        },
    ]


def assistant_history(visible: str, call: tuple[str, dict] | None) -> dict:
    if call is None:
        # Invalid or truncated output must not become active template markup.
        clean = visible.replace("<", "\\u003c")
        return {"role": "assistant", "content": clean, "reasoning_content": ""}
    name, arguments = call
    content = visible[: visible.index("<tool_call>")].strip().replace("<", "\\u003c")
    return {
        "role": "assistant",
        "content": content,
        "reasoning_content": "",
        "tool_calls": [{"function": {"name": name, "arguments": arguments}}],
    }
