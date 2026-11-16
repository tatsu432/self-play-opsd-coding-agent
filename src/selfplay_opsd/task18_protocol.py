"""Opt-in Qwen-native serialization; editing and execution use existing validators."""

from __future__ import annotations

from typing import Literal

from selfplay_opsd.config import EpisodeConfig
from selfplay_opsd.core import canonical
from selfplay_opsd.mutations import parse_action, strict_json_object
from selfplay_opsd.protocol import student_messages


class Task18EpisodeConfig(EpisodeConfig):
    tool_serialization: Literal["plain_json", "qwen_native"] = "plain_json"


def native_tools() -> list[dict]:
    fields: dict[str, dict] = {
        "read_file": {
            "path": {
                "type": "string",
                "enum": ["README.md", "src/solution.py", "tests/test_public.py"],
            }
        },
        "replace_text": {
            "path": {"type": "string", "enum": ["src/solution.py"]},
            "old_text": {
                "type": "string",
                "description": "Nonempty exact unique span of the CURRENT source",
            },
            "new_text": {
                "type": "string",
                "description": "Replacement text, possibly multiline or empty",
            },
        },
        "run_tests": {},
        "finish": {},
    }
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": {
                    "read_file": "Read an allowed workspace file",
                    "replace_text": "Replace exactly one matching source span",
                    "run_tests": "Run public tests",
                    "finish": "Finish the repair",
                }[name],
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": list(properties),
                    "additionalProperties": False,
                },
            },
        }
        for name, properties in fields.items()
    ]


def episode_messages(task, limits):
    messages = student_messages(task, limits)
    if getattr(limits, "tool_serialization", "plain_json") == "qwen_native":
        if limits.edit_protocol != "replace_text":
            raise ValueError("The native serialization probe holds replace_text semantics fixed")
        messages[0]["content"] = (
            "You are a Python repair agent. Repair the behavior described in README.md "
            "while preserving the public API. You may inspect allowed files, edit allowed "
            "source files, and run public tests. Additional tests will check correctness. "
            "Invoke exactly one documented function per reply using the tool-call format. "
            "Public tests are read-only. Use finish when the repair is ready."
        )
        # Schemas are supplied through the pinned tokenizer's tools argument.
        value = strict_json_object(messages[1]["content"])
        del value["tools"]
        value["protocol_version"] = "repair-qwen-native-text-v1"
        messages[1]["content"] = canonical(value)
    return messages


def parse_episode_action(raw, limits):
    if getattr(limits, "tool_serialization", "plain_json") != "qwen_native":
        return parse_action(raw, limits.edit_protocol)
    call = native_tool_call(raw)
    return parse_action(
        canonical({"action": call["name"], **call["arguments"]}), limits.edit_protocol
    )


def native_tool_call(raw):
    # Qwen allows ordinary assistant content around tool calls. Preserve that
    # content in sampled IDs/history, but execute only the one delimited call.
    text = raw.strip()
    if text.count("<tool_call>") != 1 or text.count("</tool_call>") != 1:
        raise ValueError("Expected exactly one Qwen tool call")
    start, end = text.index("<tool_call>") + len("<tool_call>"), text.index("</tool_call>")
    if end < start:
        raise ValueError("Incorrect Qwen tool-call boundaries")
    body = text[start:end].strip()
    # strict_json_object also rejects duplicate keys and extra JSON objects.
    call = strict_json_object(body)
    if (
        set(call) != {"name", "arguments"}
        or not isinstance(call["name"], str)
        or not isinstance(call["arguments"], dict)
    ):
        raise ValueError("Incorrect Qwen tool-call fields")
    if "action" in call["arguments"]:
        raise ValueError("Arguments cannot override the tool name")
    return call


class NativeTokenizer:
    def __init__(self, tokenizer):
        self.wrapped = tokenizer

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def apply_chat_template(self, messages, **kwargs):
        return self.wrapped.apply_chat_template(messages, tools=native_tools(), **kwargs)


class NativePolicy:
    """Delegate sampling/scoring unchanged; inject schemas at every template boundary."""

    def __init__(self, policy):
        self.wrapped = policy
        self.tokenizer = NativeTokenizer(policy.tokenizer)

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def encode_history(self, messages):
        return self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
