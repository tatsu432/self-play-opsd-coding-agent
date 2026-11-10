from types import SimpleNamespace

import pytest

from selfplay_opsd.core import digest
from selfplay_opsd.mutations import make_patch
from selfplay_opsd.task17_recovery import audit_episode, recover_diff

BEFORE = "def f(x):\n    return x + 1\n"
AFTER = "def f(x):\n    return x - 1\n"


def test_recovers_only_metadata_errors():
    patch = make_patch(BEFORE, AFTER)
    header = next(line for line in patch.splitlines() if line.startswith("@@"))
    wrong = patch.replace(header, "@@ -999,42 +998,43 @@")
    assert recover_diff(BEFORE, wrong) == AFTER
    assert BEFORE == "def f(x):\n    return x + 1\n"


@pytest.mark.parametrize(
    "patch",
    [
        "--- a/tests/test_public.py\n+++ b/tests/test_public.py\n@@ -1 +1 @@\n-x\n+y\n",
        "--- a/src/solution.py\n+++ b/src/solution.py\n@@ -1 +1 @@\n-missing\n+x\n",
        "--- a/src/solution.py\n+++ b/src/solution.py\n@@ -0,0 +1 @@\n+x\n",
        "--- a/src/solution.py\n+++ b/src/solution.py\n@@ -1 +1 @@\n bad context\n+x\n",
        "--- a/src/solution.py\n+++ b/src/solution.py\n@@ -1 +1 @@\n-def f(x):\n+def g(x):",
        "--- a/src/solution.py\n+++ b/src/solution.py\n-def f(x):\n+def g(x):\n",
        "--- a/src/solution.py\n+++ b/src/solution.py\n@@ -1 +1 @@\n@@ -1 +1 @@\n-def f(x):\n+def g(x):\n",
    ],
)
def test_ambiguous_or_malformed_is_not_recovered(patch):
    with pytest.raises(ValueError):
        recover_diff(BEFORE, patch)


def test_reconstruction_never_advances_counterfactual_source():
    import json

    patch = make_patch(BEFORE, AFTER).replace("@@ -1,2 +1,2 @@", "@@ -9,2 +9,2 @@")
    turn = {
        "action": "apply_patch",
        "parse_status": "tool_error",
        "raw_text": json.dumps({"action": "apply_patch", "patch": patch}),
    }
    task = SimpleNamespace(initial_source=BEFORE, task_id="t", parent_id="p")
    episode = {"episode_id": "e", "turns": [turn, turn], "final_source_hash": digest(BEFORE)}
    rows, candidates = audit_episode(task, episode)
    assert len(candidates) == 2
    assert rows[1]["same_failed_action_count"] == 2
    assert candidates[1]["before_hash"] == digest(BEFORE)
    assert candidates[1]["source"] == AFTER
    assert candidates[1]["semantic_verified"] is None
