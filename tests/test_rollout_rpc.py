import io
import json
import os
import subprocess
import sys
from types import SimpleNamespace

from selfplay_opsd import vllm_worker
from selfplay_opsd.verl_backend import rollout_environment


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
