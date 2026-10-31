import importlib.util
from pathlib import Path
from types import SimpleNamespace

from selfplay_opsd.hpc import TASK13_MODELS

spec = importlib.util.spec_from_file_location(
    "task13_model_policy", Path("scripts/task13_model_policy.py")
)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_model_revisions_match_controller_and_qwen3_disables_thinking():
    assert TASK13_MODELS["A"][:2] == module.MODELS["coder7b"]
    assert TASK13_MODELS["C"][:2] == module.MODELS["qwen3-8b"]
    calls = []
    policy = object.__new__(module.ModelProbePolicy)
    policy.choice = "qwen3-8b"
    policy.tokenizer = SimpleNamespace(
        apply_chat_template=lambda messages, **kwargs: calls.append(kwargs) or [1]
    )
    assert policy.encode_history([{"role": "user", "content": "x"}]) == [1]
    assert calls[0]["enable_thinking"] is False
