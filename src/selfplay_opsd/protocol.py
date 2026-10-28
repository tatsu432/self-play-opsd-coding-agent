"""Frozen prompts, JSON protocol and budget serialization shared by every workflow."""

from __future__ import annotations

from selfplay_opsd.config import EpisodeConfig
from selfplay_opsd.core import OracleSpec, ParentProgram, TaskSpec, canonical, digest

VERSION = "repair-json-v2"
PATCH_FORMAT = (
    "A patch is a single-file unified diff. Begin with --- a/src/solution.py and "
    "+++ b/src/solution.py. Context lines start with one space, removed lines with -, "
    "and added lines with +. Hunk counts and context must match the actual source. "
    "End each patch line with a newline. Omit diff --git, index, markdown fences and "
    "explanatory prose. In JSON, escape patch newlines. This example demonstrates "
    "syntax only; do not copy it as the task solution. Use the supplied source and task.\n"
    "--- a/src/solution.py\n+++ b/src/solution.py\n@@ -1,2 +1,2 @@\n"
    " def f(x):\n-    return x + 1\n+    return x - 1\n"
)
STUDENT = "You are a Python repair agent. Repair the behavior described in README.md while preserving the public API. You may inspect allowed files, edit allowed source files, and run the public tests. Additional tests will check correctness. Each reply must be exactly one JSON object invoking read_file, apply_patch, run_tests, or finish with the documented fields. Do not include prose outside JSON. Public tests are read-only. Use finish when the repair is ready."
PROPOSER = "You create executable Python repair tasks. Given a verified correct implementation and its specification, introduce a small semantic bug while preserving the API. Change at most 3 executable AST nodes and 12 nonblank source lines. Preserve imports and function signatures. Do not modify tests, add I/O or introspection, deliberately crash, or delete the implementation. Reply with exactly one JSON object containing a patch field whose value is a unified diff."
TEACHER = "Training-only privileged information follows: <correct_source>{source}</correct_source><inverse_initial_patch>{patch}</inverse_initial_patch>. Use it to assess the next action token in the provided agent history. The history is the student's actual history, including any incorrect edits and tool failures. The correct implementation describes the desired final behavior; an inverse patch may need adaptation after subsequent edits."
TOOLS = [
    {"action": "read_file", "path": "README.md|src/solution.py|tests/test_public.py"},
    {
        "action": "apply_patch",
        "patch": "unified diff restricted to src/solution.py. " + PATCH_FORMAT,
    },
    {"action": "run_tests"},
    {"action": "finish"},
]

PROPOSER_OUTPUT = {"patch": "unified diff. " + PATCH_FORMAT}


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
                    "output": PROPOSER_OUTPUT,
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


# These rendered templates are also the identity input: field values and order,
# not just field names, affect model conditioning. Dynamic task data is separately
# bound by the immutable data manifest.
def protocol_definition() -> dict:
    from selfplay_opsd.core import ParentProgram, TaskSpec

    parent = ParentProgram(
        parent_id="{parent_id}",
        source="{correct_source}",
        specification="{specification}",
        entry_point="{entry_point}",
        dataset="{dataset}",
        family_id="{family_id}",
        provenance={},
    )
    task = TaskSpec(
        task_id="{task_id}",
        parent_id=parent.parent_id,
        split="{split}",
        specification=parent.specification,
        initial_source="{initial_source}",
        entry_point=parent.entry_point,
        public_tests="{public_tests}",
        source_hash="{source_hash}",
        mutation_seed=0,
    )
    import ast
    from pathlib import Path

    package = Path(__file__).parent
    serialization = {}
    for filename, names in {
        "protocol.py": {"budget", "student_messages", "proposer_messages", "teacher_messages"},
        "core.py": {"canonical"},
        "environment.py": {"append_context", "bounded_observation", "run_episode", "step"},
        "policy.py": {"encode_history"},
    }.items():
        tree = ast.parse((package / filename).read_text())
        # Comments and docstrings do not condition the model. Tool response and
        # subsequent-turn templates do, including the episode's error/budget
        # serialization, so bind those executable definitions too.
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.FunctionDef)
                and node.body
                and isinstance(node.body[0], ast.Expr)
            ):
                value = node.body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    node.body = node.body[1:]
        serialization[filename] = [
            ast.dump(node, include_attributes=False)
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]
    return {
        "serialization_functions": serialization,
        "version": VERSION,
        "student_template": student_messages(task, EpisodeConfig()),
        "proposer_template": proposer_messages(parent),
        "teacher_template": TEACHER,
        "serialization": {
            "json": "sorted-keys-compact-utf8-v1",
            "targets": "append-exact-target-ids-v2",
            "observations": "canonical-json-token-prefix-truncation-marker-v1",
            "budget": "remaining-turns-and-generated-tokens-v1",
        },
    }


PROMPT_HASH = digest(protocol_definition())
