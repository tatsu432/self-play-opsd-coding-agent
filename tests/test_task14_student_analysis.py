import importlib.util
from pathlib import Path

import pytest


def test_paired_student_analysis_rejects_mismatched_seed(monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    spec = importlib.util.spec_from_file_location(
        "task14_student_analysis", "scripts/analyze_task14_students.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    first = {("t", 1): {"reward": 0.0}}
    second = {("t", 1): {"reward": 1.0}}
    assert module.compare(first, second)["mean_success_difference"] == 1.0
    with pytest.raises(ValueError, match="matching tasks"):
        module.compare(first, {("t", 2): {"reward": 1.0}})
