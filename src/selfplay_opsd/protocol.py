"""Frozen prompts, JSON protocol and budget serialization shared by every workflow."""

from __future__ import annotations

from selfplay_opsd.config import EpisodeConfig
from selfplay_opsd.core import OracleSpec, ParentProgram, TaskSpec, canonical, digest

VERSION = "repair-json-v1"
STUDENT = "You are a Python repair agent. Repair the behavior described in README.md while preserving the public API. You may inspect allowed files, edit allowed source files, and run the public tests. Additional tests will check correctness. Each reply must be exactly one JSON object invoking read_file, apply_patch, run_tests, or finish with the documented fields. Do not include prose outside JSON. Public tests are read-only. Use finish when the repair is ready."
PROPOSER = "You create executable Python repair tasks. Given a verified correct implementation and its specification, introduce a small semantic bug while preserving the API. Change at most 3 executable AST nodes and 12 nonblank source lines. Preserve imports and function signatures. Do not modify tests, add I/O or introspection, deliberately crash, or delete the implementation. Reply with exactly one JSON object containing a patch field whose value is a unified diff."
TEACHER = "Training-only privileged information follows: <correct_source>{source}</correct_source><inverse_initial_patch>{patch}</inverse_initial_patch>. Use it to assess the next action token in the provided agent history. The history is the student's actual history, including any incorrect edits and tool failures. The correct implementation describes the desired final behavior; an inverse patch may need adaptation after subsequent edits."
TOOLS = [
    {"action": "read_file", "path": "README.md|src/solution.py|tests/test_public.py"},
    {"action": "apply_patch", "patch": "unified diff restricted to src/solution.py"},
    {"action": "run_tests"},
    {"action": "finish"},
]


def budget(config: EpisodeConfig, turns: int = 0, tokens: int = 0) -> dict:
    return {
        "remaining_turns": config.max_turns - turns,
        "remaining_generated_tokens": config.max_generated_tokens - tokens,
        "max_next_action_tokens": min(
            config.max_turn_generated_tokens, config.max_generated_tokens - tokens
        ),
    }


def student_messages(task: TaskSpec, config: EpisodeConfig) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": STUDENT},
        {
            "role": "user",
            "content": canonical(
                {
                    "specification": task.specification,
                    "allowed_files": list(task.student_files()),
                    "tools": TOOLS,
                    "budget": budget(config),
                    "protocol_version": VERSION,
                }
            ),
        },
    ]


def proposer_messages(parent: ParentProgram) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": PROPOSER},
        {
            "role": "user",
            "content": canonical(
                {
                    "specification": parent.specification,
                    "correct_source": parent.source,
                    "allowed_path": "src/solution.py",
                    "output": {"patch": "unified diff"},
                }
            ),
        },
    ]


def teacher_messages(history: list[dict[str, str]], oracle: OracleSpec) -> list[dict[str, str]]:
    if not history or history[0]["role"] != "system":
        raise ValueError("Teacher history must begin with frozen system prompt")
    return [
        {
            "role": "system",
            "content": history[0]["content"]
            + "\n"
            + TEACHER.format(source=oracle.correct_source, patch=oracle.inverse_patch),
        },
        *history[1:],
    ]


PROMPT_HASH = digest(
    {
        "student": STUDENT,
        "proposer": PROPOSER,
        "teacher": TEACHER,
        "tools": TOOLS,
        "budget_version": 1,
        "serialization": "append-exact-target-ids-v2",
    }
)
