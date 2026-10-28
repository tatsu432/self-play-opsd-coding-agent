import io
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from selfplay_opsd import vllm_worker
from selfplay_opsd.core import ExperimentError
from selfplay_opsd.policy import GenerationRequest
from selfplay_opsd.verl_backend import VerlPolicy, rollout_environment


def test_engine_startup_failure_reaches_rpc_with_traceback(monkeypatch):
    response = io.StringIO()

    def fail(**kwargs):
        raise ValueError("invalid device mask")

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=fail, SamplingParams=object))
    monkeypatch.setitem(sys.modules, "vllm.inputs", SimpleNamespace(TokensPrompt=dict))
    monkeypatch.setitem(sys.modules, "vllm.lora.request", SimpleNamespace(LoRARequest=object))
    monkeypatch.setenv("SELFPLAY_RPC_FD", "7")
    monkeypatch.setattr(vllm_worker.os, "fdopen", lambda *a, **kw: response)
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"engine": {}}\n'))
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    vllm_worker.main()
    result = json.loads(response.getvalue())
    assert not result["ok"] and result["error"] == "ValueError: invalid device mask"
    assert "engine = LLM" in result["traceback"]


def test_batched_rpc_preserves_request_order_seeds_and_logprobs(monkeypatch):
    response = io.StringIO()
    calls = []

    class Engine:
        def __init__(self, **kwargs):
            pass

        def generate(self, prompts, settings, *, lora_request, use_tqdm):
            calls.append((prompts, settings, lora_request, use_tqdm))
            return [
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            token_ids=[setting.seed],
                            logprobs=[{setting.seed: SimpleNamespace(logprob=-0.25)}],
                            text=str(setting.seed),
                        )
                    ]
                )
                for setting in settings
            ]

    class Params:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=Engine, SamplingParams=Params))
    monkeypatch.setitem(sys.modules, "vllm.inputs", SimpleNamespace(TokensPrompt=lambda **kw: kw))
    monkeypatch.setitem(
        sys.modules, "vllm.lora.request", SimpleNamespace(LoRARequest=lambda *args: args)
    )
    monkeypatch.setenv("SELFPLAY_RPC_FD", "7")
    monkeypatch.setattr(vllm_worker.os, "fdopen", lambda *a, **kw: response)
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            '{"engine": {}}\n'
            '{"action":"generate_many","requests":['
            '{"prompt":[11],"max_tokens":5,"seed":17,"greedy":false},'
            '{"prompt":[22],"max_tokens":7,"seed":23,"greedy":false}],'
            '"version":"v","adapter_id":1,"adapter_path":"/adapter"}\n'
        ),
    )
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    vllm_worker.main()
    result = json.loads(response.getvalue().splitlines()[1])
    assert result["generations"] == [
        {"ids": [17], "logps": [-0.25], "text": "17"},
        {"ids": [23], "logps": [-0.25], "text": "23"},
    ]
    prompts, settings, adapter, progress = calls[0]
    assert prompts == [{"prompt_token_ids": [11]}, {"prompt_token_ids": [22]}]
    assert [item.seed for item in settings] == [17, 23]
    assert [item.max_tokens for item in settings] == [5, 7]
    assert adapter == ("v", 1, "/adapter") and progress is False


def test_verl_batch_rpc_carries_version_and_rejects_incomplete_result(monkeypatch):
    policy = object.__new__(VerlPolicy)
    policy._adapter_ids = {}
    policy._adapter_paths = {"v": "/adapter"}
    monkeypatch.setattr(policy, "_ensure_rollout", lambda: "v")
    requests = [GenerationRequest([11], 5, 17), GenerationRequest([22], 7, 23)]
    calls = []

    def rpc(payload):
        calls.append(payload)
        return {
            "generations": [
                {"ids": [17], "logps": [-0.25], "text": "17"},
                {"ids": [23], "logps": [-0.5], "text": "23"},
            ]
        }

    monkeypatch.setattr(policy, "_rpc", rpc)
    assert [row.ids for row in policy.generate_many(requests)] == [[17], [23]]
    assert calls[0]["version"] == "v"
    assert calls[0]["requests"] == [
        {"prompt": [11], "max_tokens": 5, "seed": 17, "greedy": False},
        {"prompt": [22], "max_tokens": 7, "seed": 23, "greedy": False},
    ]
    monkeypatch.setattr(policy, "_rpc", lambda payload: {"generations": []})
    with pytest.raises(ExperimentError, match="incomplete"):
        policy.generate_many(requests)


def test_rollout_starts_own_store_after_torchrun_agent_environment(monkeypatch):
    monkeypatch.setenv("TORCHELASTIC_USE_AGENT_STORE", "True")
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "parent-trainer")
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    env = rollout_environment(dict(os.environ))
    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert not any(k.startswith("TORCHELASTIC_") for k in env)
    assert "RANK" not in env and "WORLD_SIZE" not in env
    # A one-rank group must host its own TCPStore. An inherited agent-store
    # flag instead waits for a nonexistent server even when ranks are removed.
    script = """import datetime,socket,torch.distributed as d
s=socket.socket();s.bind(("127.0.0.1",0));port=s.getsockname()[1];s.close()
d.init_process_group("gloo",init_method=f"tcp://127.0.0.1:{port}",rank=0,world_size=1,
    timeout=datetime.timedelta(seconds=5))
d.barrier();d.destroy_process_group()
"""
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
