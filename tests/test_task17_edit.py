"""Exact replacement must be atomic, local, deterministic, and protocol isolated."""

import json
from types import SimpleNamespace

import pytest

from selfplay_opsd.config import EpisodeConfig
from selfplay_opsd.core import TokenTurn, canonical
from selfplay_opsd.environment import RepairEnvironment
from selfplay_opsd.mutations import apply_patch, make_patch, parse_action, replace_text
from selfplay_opsd.protocol import student_messages

SOURCE = "import math\n\ndef f(x):\n    y = x + 1\n    return y\n"


def action(old="x + 1", new="x - 1", path="src/solution.py"):
    return {"action": "replace_text", "path": path, "old_text": old, "new_text": new}


def environment():
    return RepairEnvironment(SimpleNamespace(initial_source=SOURCE), None, None, None)


def test_single_multiline_sequential_and_deletion():
    env = environment()
    observation = env.step(action())
    assert observation["ok"]
    env.step(action("    y = x - 1\n    return y", "    return x * 2"))
    assert env.source == "import math\n\ndef f(x):\n    return x * 2\n"
    assert replace_text(env.source, "\n\n", "\n") == env.source.replace("\n\n", "\n")


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("", "x"),
        ("missing", "x"),
        ("x", "y"),
        ("return y", "return ("),
        ("def f(x)", "def f(x, y)"),
        ("import math", "import os"),
        ("import math", "import random"),
        ("return y", "return open('x')"),
        ("return y", "return x.__class__"),
        ("return y", "return " + "x" * 65536),
    ],
)
def test_rejection_is_atomic(old, new):
    env = environment()
    with pytest.raises(ValueError):
        env.step(action(old, new))
    assert env.source == SOURCE


def test_overlapping_occurrences_rejected():
    with pytest.raises(ValueError, match="ambiguous"):
        replace_text("# aaa\ndef f(x):\n    return x\n", "aa", "b")


@pytest.mark.parametrize(
    "path",
    [
        "../src/solution.py",
        "/src/solution.py",
        "tests/test_public.py",
        "README.md",
        "src//solution.py",
        "src/./solution.py",
    ],
)
def test_forbidden_paths(path):
    with pytest.raises(ValueError, match="allowlisted"):
        parse_action(json.dumps(action(path=path)), "replace_text")
    env = environment()
    with pytest.raises(ValueError):
        env.step(action(path=path))
    assert env.source == SOURCE


@pytest.mark.parametrize(
    "value",
    [
        None,
        1,
        [],
        {},
        {"action": "replace_text"},
        action() | {"extra": "x"},
        action() | {"old_text": None},
    ],
)
def test_malformed_schema(value):
    with pytest.raises(ValueError):
        parse_action(json.dumps(value), "replace_text")


def test_selected_protocol_and_determinism():
    raw = canonical(action())
    assert parse_action(raw, "replace_text") == action()
    with pytest.raises(ValueError):
        parse_action(raw)
    with pytest.raises(ValueError):
        parse_action('{"action":"apply_patch","patch":"x"}', "replace_text")
    first, second = environment(), environment()
    assert canonical(first.step(action())) == canonical(second.step(action()))
    assert first.source == second.source
    patch = make_patch(SOURCE, SOURCE.replace("x + 1", "x - 1"))
    assert apply_patch(SOURCE, patch) == first.source
    with pytest.raises(ValueError):
        apply_patch(SOURCE, patch.replace("@@ -", "@@ -999"))


def test_templates_and_historical_turn_loading():
    task = SimpleNamespace(
        specification="Repair f", student_files=lambda: {"src/solution.py": SOURCE}
    )
    diff = student_messages(task, EpisodeConfig())
    text = student_messages(task, EpisodeConfig(edit_protocol="replace_text"))
    assert '"protocol_version":"repair-json-v2"' in diff[1]["content"]
    assert "repair-text-v1" in text[1]["content"]
    assert "replace_text" in text[0]["content"] and "apply_patch" not in text[0]["content"]
    old = dict(
        prompt_ids=[],
        target_ids=[],
        old_logps=[],
        history=[],
        raw_text="",
        action="finish",
        parse_status="valid",
    )
    assert TokenTurn(**old).observation is None
    assert TokenTurn(**old, observation={"ok": True}).model_dump()["observation"] == {"ok": True}


def test_episode_identity_and_final_observation_are_protocol_specific(monkeypatch):
    import selfplay_opsd.environment as module
    from selfplay_opsd.core import OracleSpec, TaskSpec, digest

    source = "def f(x):\n    return x\n"
    task = TaskSpec(
        task_id="t",
        parent_id="p",
        split="train",
        specification="s",
        initial_source=source,
        entry_point="f",
        public_tests="",
        source_hash=digest(source),
        mutation_seed=1,
    )
    oracle = OracleSpec(
        task_id="t",
        correct_source=source,
        inverse_patch="",
        public_cases=[],
        hidden_cases=[],
        provenance={},
    )
    monkeypatch.setattr(module, "check_admission", lambda *args: None)
    policy = SimpleNamespace(
        is_coordinator=True,
        broadcast=lambda x: x,
        encode_history=lambda history: [11],
        version=lambda role: "version",
        generate=lambda *args, **kwargs: SimpleNamespace(
            ids=[1], logps=[-0.2], text='{"action":"finish"}'
        ),
    )
    verification = SimpleNamespace(
        passed=True,
        details=[],
        seconds=0,
        status=module.Status.OK,
        semantic_failures=0,
        runtime_failures=0,
    )
    sandbox = SimpleNamespace(verify=lambda *args: verification)
    first = module.run_episode(policy, sandbox, task, oracle, EpisodeConfig(), 2026, greedy=True)
    second = module.run_episode(
        policy,
        sandbox,
        task,
        oracle,
        EpisodeConfig(edit_protocol="replace_text"),
        2026,
        greedy=True,
    )
    assert first.episode_id != second.episode_id
    assert first.turns[-1].observation == {"ok": True, "finished": True}
