"""One Qwen3.5 adapter, an explicit sampling distribution and exact-prefix scoring."""

from __future__ import annotations

import hashlib
import importlib
import re
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from selfplay_opsd.task19 import MODEL, REVISION, Task19Control

TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True)
class Sampling:
    temperature: float
    top_p: float
    top_k: int = 20


STUDENT_SAMPLING = Sampling(0.6, 0.95)
PROPOSER_SAMPLING = Sampling(0.7, 0.8)


def sampling_distribution(logits: Tensor, sampling: Sampling) -> tuple[Tensor, Tensor]:
    top, ids = logits.float().topk(min(sampling.top_k, logits.shape[-1]), dim=-1)
    top = top / sampling.temperature
    probabilities = top.softmax(-1)
    excluded = probabilities.cumsum(-1) - probabilities >= sampling.top_p
    return ids, top.masked_fill(excluded, -torch.inf).log_softmax(-1)


def target_logps(logits: Tensor, targets: Tensor, sampling: Sampling | None) -> Tensor:
    if sampling is None:
        return -nn.functional.cross_entropy(logits.float(), targets, reduction="none")
    ids, logps = sampling_distribution(logits, sampling)
    return logps.masked_fill(ids != targets.unsqueeze(-1), -torch.inf).amax(-1)


@dataclass(frozen=True)
class Sample:
    ids: list[int]
    logps: list[float]
    text: str
    visible: str
    token_kinds: list[str]
    stopped: bool


class SharedPolicy:
    is_coordinator = True

    def __init__(self, control: Task19Control):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoTokenizer

        qwen_model = importlib.import_module("transformers").Qwen3_5ForConditionalGeneration

        self.device = torch.device("cuda:0")
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Task 19 requires one allocated BF16 CUDA device")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Task 19 requires exactly one visible GPU")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        self.markers = {
            marker: self.tokenizer.convert_tokens_to_ids(marker)
            for marker in (
                "<think>",
                "</think>",
                "<tool_call>",
                "</tool_call>",
                "<|im_end|>",
                "<|endoftext|>",
            )
        }
        if any(
            self.tokenizer.encode(marker, add_special_tokens=False) != [index]
            for marker, index in self.markers.items()
        ):
            raise RuntimeError("Unexpected native delimiters in pinned tokenizer")
        self.eos = {self.markers["<|im_end|>"], self.markers["<|endoftext|>"]}
        base, loading = qwen_model.from_pretrained(
            MODEL,
            revision=REVISION,
            dtype=torch.float32 if control.precision == "float32" else torch.bfloat16,
            attn_implementation="sdpa",
            output_loading_info=True,
        )
        if loading.get("missing_keys") or loading.get("mismatched_keys"):
            raise RuntimeError("Qwen3.5 checkpoint did not completely populate the model")
        language_linear = {
            name: module
            for name, module in base.named_modules()
            if ".language_model.layers." in name and isinstance(module, nn.Linear)
        }
        targets = sorted(name for name in language_linear if name.rsplit(".", 1)[-1] in TARGETS)
        if {name.rsplit(".", 1)[-1] for name in targets} != set(TARGETS):
            raise RuntimeError("Qwen3.5 language projection inventory differs")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(control.seed)
            self.model: Any = get_peft_model(
                base,
                LoraConfig(
                    r=32,
                    lora_alpha=64,
                    lora_dropout=0,
                    bias="none",
                    target_modules="(?:" + "|".join(re.escape(name) for name in targets) + ")",
                ),
                adapter_name="shared",
            )
        self.model.to(self.device)
        self.model.eval()
        self.base: Any = base
        self.base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        self.named_parameters = sorted(
            (name, value) for name, value in self.model.named_parameters() if value.requires_grad
        )
        if len(self.named_parameters) != 2 * len(targets) or any(
            ".shared." not in name or tensor.dtype != torch.float32
            for name, tensor in self.named_parameters
        ):
            raise RuntimeError("Unexpected shared trainable coordinates")
        self.parameters = [tensor for _, tensor in self.named_parameters]
        self.optimizer = torch.optim.AdamW(
            self.parameters, lr=control.learning_rate, weight_decay=0, eps=1e-8
        )

    @staticmethod
    def broadcast(value):
        return value

    def encode(self, history: list[dict], tools: list[dict], thinking: bool) -> list[int]:
        return self.tokenizer.apply_chat_template(
            history,
            tools=tools,
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )

    def _forward(self, ids: Tensor, *, start: int = 0, past=None, cache: bool = False):
        positions = torch.arange(start, start + ids.shape[1], device=self.device).unsqueeze(0)
        return self.base.model(
            input_ids=ids, position_ids=positions, past_key_values=past, use_cache=cache
        )

    def _last(self, context: Tensor, position: int, past, cached: bool):
        chunk = context if not cached or past is None else context[:, -1:]
        outputs = self._forward(chunk, start=position + 1 - chunk.shape[1], past=past, cache=cached)
        return self.base.lm_head(outputs.last_hidden_state[:, -1]), outputs.past_key_values

    @torch.inference_mode()
    def generate(
        self,
        prompt: list[int],
        maximum: int,
        seed: int,
        *,
        thinking: bool,
        sampling: Sampling,
        cached: bool,
    ) -> Sample:
        self.model.eval()
        generator = torch.Generator(device=self.device).manual_seed(seed)
        context = torch.tensor([prompt], device=self.device)
        ids, probabilities = [], []
        past = None
        for offset in range(maximum):
            logits, past = self._last(context, len(prompt) + offset - 1, past, cached)
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite Qwen3.5 generation logits")
            shortlist, logps = sampling_distribution(logits, sampling)
            choice = torch.multinomial(logps.exp(), 1, generator=generator)
            next_id = shortlist.gather(1, choice)
            selected, logp = int(next_id.item()), float(logps.gather(1, choice).item())
            ids.append(selected)
            probabilities.append(logp)
            if selected in self.eos:
                break
            context = torch.cat((context, next_id), dim=1)
        kinds, visible = visibility(ids, self.markers, self.eos, thinking)
        return Sample(
            ids,
            probabilities,
            self.tokenizer.decode(
                ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            ),
            self.tokenizer.decode(
                visible, skip_special_tokens=False, clean_up_tokenization_spaces=False
            ),
            kinds,
            bool(ids and ids[-1] in self.eos),
        )

    @torch.inference_mode()
    def replay(
        self,
        prompt: list[int],
        targets: list[int],
        sampling: Sampling,
        *,
        cached: bool,
    ) -> Tensor:
        self.model.eval()
        context = torch.tensor([prompt], device=self.device)
        past, values = None, []
        for index, token in enumerate(targets):
            logits, past = self._last(context, len(prompt) + index - 1, past, cached)
            target = torch.tensor([token], device=self.device)
            values.append(target_logps(logits, target, sampling))
            context = torch.cat((context, target.view(1, 1)), dim=1)
        return torch.cat(values) if values else torch.empty(0, device=self.device)

    def score(
        self,
        prompt: list[int],
        targets: list[int],
        sampling: Sampling | None,
        *,
        gradients: bool = False,
        exact_prefix: bool = False,
        raw_also: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        if not targets:
            empty = torch.empty(0, device=self.device)
            return empty, empty if raw_also else None
        self.model.train(gradients)
        with torch.set_grad_enabled(gradients):
            # The saved prompt is the actual conditioning context for this turn.
            context = torch.tensor([prompt + targets[:-1]], device=self.device)
            target = torch.tensor(targets, device=self.device)
            if exact_prefix:

                def prefix_logits(tokens: Tensor) -> Tensor:
                    states = self._forward(tokens).last_hidden_state
                    return self.base.lm_head(states[:, -1])

                def score_prefix(tokens: Tensor, label: Tensor) -> Tensor:
                    logits = prefix_logits(tokens)
                    value = target_logps(logits, label, sampling)
                    return (
                        torch.stack((value, target_logps(logits, label, None)))
                        if raw_also
                        else value
                    )

                rows = [
                    checkpoint(
                        score_prefix,
                        context[:, : len(prompt) + offset],
                        label.view(1),
                        use_reentrant=False,
                    )
                    if gradients
                    else score_prefix(context[:, : len(prompt) + offset], label.view(1))
                    for offset, label in enumerate(target)
                ]
            else:
                hidden = self._forward(context).last_hidden_state[0, len(prompt) - 1 :]

                def score_block(states: Tensor, labels: Tensor) -> Tensor:
                    logits = self.base.lm_head(states)
                    value = target_logps(logits, labels, sampling)
                    return (
                        torch.stack((value, target_logps(logits, labels, None)))
                        if raw_also
                        else value
                    )

                rows = [
                    checkpoint(score_block, states, labels, use_reentrant=False)
                    if gradients
                    else score_block(states, labels)
                    for states, labels in zip(hidden.split(32), target.split(32), strict=True)
                ]
            scores = torch.cat(rows, dim=-1)
            return (scores[0], scores[1]) if raw_also else (scores, None)

    def state(self) -> dict:
        return {
            "parameter_names": [name for name, _ in self.named_parameters],
            "shared": [p.detach().cpu().clone() for p in self.parameters],
            "optimizer": self.optimizer.state_dict(),
        }

    def restore(self, state: dict) -> None:
        names = [name for name, _ in self.named_parameters]
        if state["parameter_names"] != names:
            raise ValueError("Shared LoRA checkpoint coordinate ordering differs")
        with torch.no_grad():
            for target, value in zip(self.parameters, state["shared"], strict=True):
                if value.shape != target.shape or value.dtype != target.dtype:
                    raise ValueError("Shared LoRA checkpoint coordinate shape or dtype differs")
                target.copy_(value)
        self.optimizer.load_state_dict(state["optimizer"])
        self.optimizer.zero_grad(set_to_none=True)

    def version(self) -> str:
        identity = hashlib.sha256()
        for name, tensor in self.named_parameters:
            identity.update(name.encode())
            identity.update(tensor.detach().cpu().numpy().tobytes())
        return identity.hexdigest()


def visibility(
    ids: list[int],
    markers: dict[str, int],
    eos: set[int],
    thinking: bool,
) -> tuple[list[str], list[int]]:
    kinds, visible = [], []
    tool = False
    for token in ids:
        if token in eos:
            kinds.append("assistant")
            continue
        if token == markers["<think>"]:
            thinking = True
        if thinking:
            kinds.append("reasoning")
            if token == markers["</think>"]:
                thinking = False
            continue
        if token == markers["<tool_call>"]:
            tool = True
        kinds.append("tool" if tool else "assistant")
        visible.append(token)
        if token == markers["</tool_call>"]:
            tool = False
    return kinds, visible
