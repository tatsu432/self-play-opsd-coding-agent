"""Inference-only vLLM policy for pinned Task 13 model comparisons."""

from contextlib import nullcontext

from selfplay_opsd.policy import Generation

MODELS = {
    "coder7b": ("Qwen/Qwen2.5-Coder-7B-Instruct", "c03e6d358207e414f1eca0bb1891e29f1db0e242"),
    "qwen3-8b": ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"),
}


class ModelProbePolicy:
    def __init__(self, config, choice):
        from vllm import LLM

        name, revision = MODELS[choice]
        config.model.name = name
        config.model.revision = config.model.tokenizer_revision = revision
        self.config = config
        self.choice = choice
        self.last_finish_reason = None
        self.engine = LLM(
            model=name,
            revision=revision,
            tokenizer_revision=revision,
            dtype="bfloat16",
            tensor_parallel_size=1,
            gpu_memory_utilization=config.hardware.rollout_memory_fraction,
            max_model_len=config.episode.max_student_sequence_tokens,
            enable_prefix_caching=False,
            enforce_eager=True,
            trust_remote_code=False,
            disable_log_stats=True,
            seed=config.seed,
        )
        self.tokenizer = self.engine.get_tokenizer()

    def active(self, _role):
        return nullcontext()

    def encode_history(self, messages):
        options = {"enable_thinking": False} if self.choice == "qwen3-8b" else {}
        ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, **options
        )
        if not isinstance(ids, list) or not ids:
            raise ValueError("Model chat template did not return token IDs")
        return ids

    def generate(self, prompt, max_tokens, seed):
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        params = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            min_p=0.0,
            repetition_penalty=1.0,
            max_tokens=max_tokens,
            seed=seed,
            logprobs=1,
        )
        response = self.engine.generate(
            [TokensPrompt(prompt_token_ids=prompt)], params, use_tqdm=False
        )[0].outputs[0]
        ids = list(response.token_ids)
        logps = [
            float(row[token].logprob) for row, token in zip(response.logprobs, ids, strict=True)
        ]
        self.last_finish_reason = response.finish_reason
        return Generation(ids, logps, response.text)

    def close(self):
        del self.engine
