import io
import json
import sys
from types import SimpleNamespace

from selfplay_opsd import vllm_worker


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
