"""Trusted rollout subprocess; separate lifetime from sharded training CUDA state."""

from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Any


def main():
    # vLLM logging goes to stderr; stdout is exclusively the bounded RPC stream.
    rpc = os.fdopen(int(os.environ["SELFPLAY_RPC_FD"]), "w", buffering=1)
    sys.stdout = sys.stderr
    try:
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt
        from vllm.lora.request import LoRARequest

        request = json.loads(sys.stdin.readline())
        engine = LLM(**request["engine"])
    except Exception as exc:
        rpc.write(
            json.dumps(
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()[-4000:],
                }
            )
            + "\n"
        )
        rpc.flush()
        return
    rpc.write(json.dumps({"ok": True, "ready": True}) + "\n")
    rpc.flush()
    try:
        result: dict[str, Any]
        for line in sys.stdin:
            command = json.loads(line)
            try:
                if command["action"] == "close":
                    break
                if command["action"] == "sleep":
                    engine.sleep(level=1)
                    result = {"ok": True}
                elif command["action"] == "wake":
                    engine.wake_up()
                    result = {"ok": True}
                else:
                    requests = (
                        command["requests"] if command["action"] == "generate_many" else [command]
                    )
                    if not requests or len(requests) > 64:
                        raise ValueError("Generation batch size must be 1..64")
                    settings = [
                        SamplingParams(
                            temperature=0.0 if item["greedy"] else 1.0,
                            top_p=1.0,
                            top_k=-1,
                            min_p=0.0,
                            repetition_penalty=1.0,
                            max_tokens=item["max_tokens"],
                            seed=item["seed"],
                            logprobs=1,
                        )
                        for item in requests
                    ]
                    adapter = LoRARequest(
                        command["version"], command["adapter_id"], command["adapter_path"]
                    )
                    outputs = engine.generate(
                        [TokensPrompt(prompt_token_ids=item["prompt"]) for item in requests],
                        settings,
                        lora_request=adapter,
                        use_tqdm=False,
                    )
                    if len(outputs) != len(requests):
                        raise ValueError("vLLM returned an incomplete generation batch")
                    generations = []
                    for response in outputs:
                        output = response.outputs[0]
                        ids = list(output.token_ids)
                        logps = [
                            float(row[token].logprob)
                            for row, token in zip(output.logprobs, ids, strict=True)
                        ]
                        generations.append({"ids": ids, "logps": logps, "text": output.text})
                    result = (
                        {"ok": True, "generations": generations}
                        if command["action"] == "generate_many"
                        else {"ok": True, **generations[0]}
                    )
            except Exception as exc:
                result = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()[-4000:],
                }
            rpc.write(json.dumps(result, allow_nan=False) + "\n")
            rpc.flush()
    finally:
        del engine


if __name__ == "__main__":
    main()
