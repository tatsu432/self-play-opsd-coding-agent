import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "task13_counterfactual", Path("scripts/task13_counterfactual.py")
)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_actual_adapter_delta_comparison_and_restoration_equality():
    before = {
        "adapters": {
            "layer.proposer.weight": torch.tensor([1.0, 1.0]),
            "layer.student.weight": torch.tensor([5.0]),
        }
    }
    left = module.adapter_delta(
        before, {"adapters": {"layer.proposer.weight": torch.tensor([2.0, 1.0])}}
    )
    right = module.adapter_delta(
        before, {"adapters": {"layer.proposer.weight": torch.tensor([1.0, 3.0])}}
    )
    result = module.delta_comparison(left, right)
    assert result["rl_update_norm"] == 1
    assert result["combined_update_norm"] == 2
    assert result["cosine"] == pytest.approx(0)
    assert module.state_equal({"state": [torch.tensor([1.0])]}, {"state": [torch.tensor([1.0])]})
    assert not module.state_equal(
        {"state": [torch.tensor([1.0])]}, {"state": [torch.tensor([2.0])]}
    )
