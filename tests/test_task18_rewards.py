import importlib.util
from pathlib import Path

import pytest


def module():
    spec = importlib.util.spec_from_file_location(
        "rewards", Path("scripts/analyze_task18_rewards.py")
    )
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_hidden_bounds_preserve_unknown_public_failures():
    episode = {
        "semantic_failures": 4,
        "runtime_failures": 0,
        "reward": 0,
        "public_pass": False,
        "hidden_pass": False,
        "turns": [],
        "final_source_hash": "final",
    }
    value = module().hidden_fraction(episode, 3, 10, "initial")
    assert value == {"lower": 0.7, "upper": 0.9, "exact": False}
    episode["turns"] = [
        {
            "parse_status": "valid",
            "action": "replace_text",
            "observation": {"source_hash": "final"},
        },
        {"parse_status": "valid", "action": "run_tests", "observation": {"passed": 2, "total": 3}},
    ]
    assert module().hidden_fraction(episode, 3, 10, "initial") == {
        "lower": 0.7,
        "upper": 0.7,
        "exact": True,
    }
    episode["turns"][0]["observation"]["source_hash"] = "other"
    assert module().hidden_fraction(episode, 3, 10, "initial") == value


def test_public_success_and_runtime_failure_allow_exact_hidden_counts():
    episode = {
        "semantic_failures": 4,
        "runtime_failures": 0,
        "reward": 0,
        "public_pass": True,
        "hidden_pass": False,
        "turns": [],
        "final_source_hash": "same",
    }
    assert module().hidden_fraction(episode, 3, 10, "same")["lower"] == 0.6
    episode.update(semantic_failures=0, runtime_failures=13, public_pass=False)
    assert module().hidden_fraction(episode, 3, 10, "same") == {
        "lower": 0,
        "upper": 0,
        "exact": True,
    }
    episode["hidden_pass"] = True
    with pytest.raises(ValueError, match="contradict"):
        module().hidden_fraction(episode, 3, 10, "same")
