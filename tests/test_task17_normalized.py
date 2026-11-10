"""Same-prompt metadata correction preserves strict controls and source admission."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from selfplay_opsd.config import EpisodeConfig, load_config
from selfplay_opsd.core import TaskSpec, TokenTurn, digest
from selfplay_opsd.environment import RepairEnvironment
from selfplay_opsd.mutations import make_patch, parse_action
from selfplay_opsd.protocol import student_messages

SOURCE = "def f(x):\n    return x + 2\n"


def bad_metadata(before, after):
    patch = make_patch(before, after)
    header = next(line for line in patch.splitlines() if line.startswith("@@"))
    return patch.replace(header, "@@ -99,55 +98,44 @@")


def env(source=SOURCE, normalized=False):
    # The supplied oracle intentionally disagrees; editing must not consult it.
    return RepairEnvironment(
        SimpleNamespace(initial_source=source),
        SimpleNamespace(correct_source="def f(x):\n    return 999\n"),
        None,
        None,
        normalize_diff_metadata=normalized,
    )


def test_only_selected_mode_recovers_and_labels_without_changing_observation():
    target = "def f(x):\n    return x + 1\n"
    patch = bad_metadata(SOURCE, target)
    strict, normalized = env(), env(normalized=True)
    with pytest.raises(ValueError):
        strict.step({"action": "apply_patch", "patch": patch})
    assert strict.source == SOURCE
    result = normalized.step({"action": "apply_patch", "patch": patch})
    assert normalized.source == target
    assert normalized.last_edit_application == "metadata_recovered"
    ordinary = env()
    expected = ordinary.step({"action": "apply_patch", "patch": make_patch(SOURCE, target)})
    assert result == expected == {"ok": True, "source_hash": digest(target)}
    assert ordinary.last_edit_application == "strict"
    record = TokenTurn(
        prompt_ids=[1],
        target_ids=[2],
        old_logps=[-0.1],
        history=[],
        raw_text=json.dumps({"action": "apply_patch", "patch": patch}),
        action="apply_patch",
        parse_status="valid",
        observation=result,
        edit_application=normalized.last_edit_application,
    )
    assert TokenTurn.model_validate(record.model_dump()).edit_application == "metadata_recovered"


@pytest.mark.parametrize(
    "target",
    [
        "def f(x):\n    return (\n",
        "def g(x):\n    return x\n",
        "import os\ndef f(x):\n    return x\n",
    ],
)
def test_inadmissible_recovery_preserves_original_error_and_source(target):
    patch = bad_metadata(SOURCE, target)
    first, second = env(), env(normalized=True)
    with pytest.raises(ValueError) as original:
        first.step({"action": "apply_patch", "patch": patch})
    with pytest.raises(ValueError) as recovered:
        second.step({"action": "apply_patch", "patch": patch})
    assert str(original.value) == str(recovered.value)
    assert second.source == SOURCE and second.last_edit_application is None


def test_ambiguous_and_forbidden_paths_do_not_use_oracle():
    source = "def f(x):\n    return x\n\ndef g(x):\n    return x\n"
    patch = "--- a/src/solution.py\n+++ b/src/solution.py\n@@ -99,1 +99,1 @@\n-    return x\n+    return x + 1\n"
    subject = env(source, True)
    for value in (patch, patch.replace("src/solution.py", "tests/test_public.py")):
        with pytest.raises(ValueError):
            subject.step({"action": "apply_patch", "patch": value})
        assert subject.source == source
    # A safe but semantically arbitrary change is accepted independently of oracle.
    subject = env(normalized=True)
    wrong = "def f(x):\n    return x * 9\n"
    subject.step({"action": "apply_patch", "patch": bad_metadata(SOURCE, wrong)})
    assert subject.source == wrong


def test_conditioning_and_historical_science_are_unchanged():
    task = TaskSpec(
        task_id="t",
        parent_id="p",
        split="train",
        specification="Return x+1.",
        initial_source=SOURCE,
        entry_point="f",
        public_tests="",
        source_hash=digest(SOURCE),
        mutation_seed=1,
    )
    assert student_messages(task, EpisodeConfig()) == student_messages(
        task, EpisodeConfig(edit_protocol="normalized_diff")
    )
    raw = json.dumps(
        {"action": "apply_patch", "patch": make_patch(SOURCE, SOURCE.replace("+ 2", "+ 1"))}
    )
    assert parse_action(raw) == parse_action(raw, "normalized_diff")
    with pytest.raises(ValueError):
        parse_action(
            json.dumps(
                {
                    "action": "replace_text",
                    "path": "src/solution.py",
                    "old_text": "x",
                    "new_text": "y",
                }
            ),
            "normalized_diff",
        )
    cfg = load_config(
        Path("configs/task15-v2.yaml"), method="B", seed=2026, hardware_profile="single_gpu_80gb"
    )
    assert cfg.scientific_hash == "3fb8aaa585826f21695e89758e1a937fa371e1e83ed341b6ba3d1454d03ab709"
