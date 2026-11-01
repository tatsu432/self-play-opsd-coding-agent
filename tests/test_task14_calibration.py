import importlib.util
import sys
from pathlib import Path

import pytest


def test_calibration_matches_aggregate_strength_and_rejects_degenerate(monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    spec = importlib.util.spec_from_file_location("task14_calibrate", "scripts/task14_calibrate.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    coefficient, gated, ungated = module.matched_coefficient(
        [
            {"gated_norm": 0.3, "ungated_unit_norm": 3.0},
            {"gated_norm": 0.4, "ungated_unit_norm": 4.0},
        ]
    )
    assert coefficient == pytest.approx(0.1)
    assert gated == pytest.approx(0.25) and ungated == pytest.approx(25)
    assert module.matched_coefficient([{"gated_norm": 0, "ungated_unit_norm": 1}])[0] is None
