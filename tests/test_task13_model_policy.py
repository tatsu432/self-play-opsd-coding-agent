import importlib.util
import sys
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


def test_inference_policy_pins_revision_and_sampling_without_training(monkeypatch):
    engines = []

    class Engine:
        def __init__(self, **kwargs):
            engines.append(kwargs)

        def get_tokenizer(self):
            return object()

        def generate(self, prompts, params, use_tqdm):
            assert prompts == [{"prompt_token_ids": [1, 2]}]
            assert params["temperature"] == 1.0 and params["seed"] == 123
            return [
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            token_ids=[3],
                            logprobs=[{3: SimpleNamespace(logprob=-0.2)}],
                            text="x",
                            finish_reason="stop",
                        )
                    ]
                )
            ]

    monkeypatch.setitem(
        sys.modules, "vllm", SimpleNamespace(LLM=Engine, SamplingParams=lambda **x: x)
    )
    monkeypatch.setitem(sys.modules, "vllm.inputs", SimpleNamespace(TokensPrompt=lambda **x: x))
    cfg = SimpleNamespace(
        model=SimpleNamespace(),
        hardware=SimpleNamespace(rollout_memory_fraction=0.4),
        episode=SimpleNamespace(max_student_sequence_tokens=10240),
        seed=2026,
    )
    policy = module.ModelProbePolicy(cfg, "coder7b")
    response = policy.generate([1, 2], 512, 123)
    assert response.ids == [3] and response.logps == [-0.2]
    assert policy.last_finish_reason == "stop"
    assert engines[0]["revision"] == module.MODELS["coder7b"][1]
    assert engines[0]["trust_remote_code"] is False
