"""Token-preserving HF policy and backend-independent training examples."""

from __future__ import annotations

import contextlib
import hashlib
from collections.abc import Iterator
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch import Tensor
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from selfplay_opsd.config import RunConfig
from selfplay_opsd.core import ExperimentError, TokenTurn


class ByteTokenizer:
    """Offline tiny-model tokenizer. Production always uses the pinned tokenizer."""

    pad_token_id, bos_token_id, eos_token_id = 0, 1, 2
    vocab_size = 259

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [b + 3 for b in text.encode("latin1", errors="replace")]

    def decode(self, tokens, skip_special_tokens: bool = True) -> str:
        return bytes(i - 3 for i in tokens if 3 <= i < 259).decode("latin1")

    def apply_chat_template(
        self, messages: list[dict], tokenize: bool = True, add_generation_prompt: bool = True
    ):
        ids = [self.bos_token_id]
        for message in messages:
            ids += self.encode(f"[{message['role']}]\n{message['content']}")
            if message["role"] == "assistant":
                ids += [self.eos_token_id]
            ids += self.encode("\n")
        if add_generation_prompt:
            ids += self.encode("[assistant]\n")
        return ids if tokenize else self.decode(ids)


@dataclass(frozen=True)
class Generation:
    ids: list[int]
    logps: list[float]
    text: str


@dataclass(frozen=True)
class LossExample:
    prompt: list[int]
    targets: list[int]
    old: list[float]
    advantage: float
    coefficient: list[float]
    # Each turn is weighted by 1/(episode token count * episodes/task * tasks).
    token_weight: float
    kind: str = "combined"
    base_logps: list[float] | None = None
    kl_coefficient: float = 0.0


def tokenizer_for(config: RunConfig):
    if config.model.tiny:
        return ByteTokenizer()
    if config.model.tokenizer_revision == "UNRESOLVED":
        raise ExperimentError("Pin tokenizer revision before model loading")
    return AutoTokenizer.from_pretrained(
        config.model.name, revision=config.model.tokenizer_revision, trust_remote_code=False
    )


class HFPolicy:
    distributed = False
    is_coordinator = True
    world_size = 1

    def __init__(self, config: RunConfig, *, device: str | None = None):
        self.config = config
        self.tokenizer = tokenizer_for(config)
        self.device = torch.device(
            device
            or ("cuda:0" if torch.cuda.is_available() and config.hardware.gpu_count else "cpu")
        )
        self.role = "student"
        base: torch.nn.Module
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            if config.model.tiny:
                base = Qwen3ForCausalLM(
                    Qwen3Config(
                        vocab_size=259,
                        hidden_size=32,
                        intermediate_size=64,
                        num_hidden_layers=2,
                        num_attention_heads=4,
                        num_key_value_heads=2,
                        head_dim=8,
                        max_position_embeddings=16384,
                        attention_dropout=0.0,
                        pad_token_id=0,
                        bos_token_id=1,
                        eos_token_id=2,
                    )
                )
            else:
                if config.model.revision == "UNRESOLVED":
                    raise ExperimentError("Pin model revision before loading")
                base = AutoModelForCausalLM.from_pretrained(
                    config.model.name,
                    revision=config.model.revision,
                    dtype=torch.bfloat16 if config.model.dtype == "bfloat16" else torch.float32,
                    trust_remote_code=False,
                    attn_implementation="sdpa",
                )
            modules = {n.rsplit(".", 1)[-1] for n, _ in base.named_modules()}
            if not set(config.model.target_modules) <= modules:
                raise ExperimentError("Configured LoRA targets missing from actual model")
            lora = LoraConfig(
                r=config.model.rank,
                lora_alpha=config.model.alpha,
                lora_dropout=0.0,
                target_modules=config.model.target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(base, lora, adapter_name="student")
            self.model.add_adapter("proposer", lora)
            # Explicitly identical initialization; each role owns independent tensors.
            params = dict(self.model.named_parameters())
            with torch.no_grad():
                for name, parameter in params.items():
                    if ".student." in name:
                        params[name.replace(".student.", ".proposer.")].copy_(parameter)
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
            if hasattr(module, "attention_dropout"):
                module.attention_dropout = 0.0  # type: ignore[assignment]
        self.model.to(self.device)
        self.model.eval()
        if config.hardware.activation_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            self.model.enable_input_require_grads()
        self.optimizers = {
            role: torch.optim.AdamW(
                self.parameters(role),
                lr=config.objective.student_lr
                if role == "student"
                else config.objective.proposer_lr,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.0,
            )
            for role in ("student", "proposer")
        }

    def named_parameters(self, role: str) -> list[tuple[str, Tensor]]:
        return sorted(
            (name, param) for name, param in self.model.named_parameters() if f".{role}." in name
        )

    def parameters(self, role: str) -> list[Tensor]:
        return [p for _, p in self.named_parameters(role)]

    def activate(self, role: str):
        if role not in self.optimizers:
            raise ValueError(role)
        self.role = role
        self.model.set_adapter(role)
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_(".student." in name or ".proposer." in name)
        self.model.eval()

    @contextlib.contextmanager
    def active(self, role: str) -> Iterator[None]:
        previous = self.role
        self.activate(role)
        try:
            yield
        finally:
            self.activate(previous)

    def version(self, role: str) -> str:
        h = hashlib.sha256()
        for name, param in self.named_parameters(role):
            h.update(name.replace(f".{role}.", ".adapter.").encode())
            h.update(param.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()

    def encode_history(self, messages: list[dict[str, str]]) -> list[int]:
        ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        if not isinstance(ids, list) or not ids:
            raise ExperimentError("Tokenizer chat template must return nonempty token IDs")
        return ids

    def score(self, prompt: list[int], targets: list[int], *, gradients: bool = False) -> Tensor:
        return self.score_many([(prompt, targets)], gradients=gradients)[0]

    def score_many(
        self, requests: list[tuple[list[int], list[int]]], *, gradients: bool = False
    ) -> list[Tensor]:
        results: list[Tensor] = [
            torch.empty(0, device=self.device, dtype=torch.float32) for _ in requests
        ]
        nonempty = [
            (index, prefix, targets) for index, (prefix, targets) in enumerate(requests) if targets
        ]
        microbatch = self.config.hardware.microbatch_episodes
        for start in range(0, len(nonempty), microbatch):
            rows = nonempty[start : start + microbatch]
            if any(not prefix for _, prefix, _ in rows):
                raise ExperimentError("Causal targets require a prefix token")
            length = max(len(prefix) + len(targets) for _, prefix, targets in rows)
            count = max(len(targets) for _, _, targets in rows)
            ids = torch.full(
                (len(rows), length),
                self.tokenizer.pad_token_id or 0,
                dtype=torch.long,
                device=self.device,
            )
            mask = torch.zeros_like(ids)
            for row, (_, prefix, targets) in enumerate(rows):
                combined = prefix + targets
                ids[row, -len(combined) :] = torch.tensor(combined, device=self.device)
                mask[row, -len(combined) :] = 1
            positions = (mask.cumsum(-1) - 1).clamp_min(0)
            with torch.set_grad_enabled(gradients):
                logits = self.model(
                    input_ids=ids,
                    attention_mask=mask,
                    position_ids=positions,
                    use_cache=False,
                    logits_to_keep=count + 1,
                ).logits
                for row, (index, _, targets) in enumerate(rows):
                    selected = logits[row, -len(targets) - 1 : -1]
                    target = torch.tensor(targets, device=self.device)
                    blocks = [
                        -F.cross_entropy(
                            selected[k : k + 128].float(), target[k : k + 128], reduction="none"
                        )
                        for k in range(0, len(targets), 128)
                    ]
                    values = torch.cat(blocks)
                    if not torch.isfinite(values).all():
                        raise FloatingPointError("Nonfinite target log probabilities")
                    results[index] = values
        return results

    def score_base(self, prompt: list[int], targets: list[int]) -> Tensor:
        """Optional named KL variant: frozen adapter-disabled base identity."""
        with self.model.disable_adapter():
            return self.score(prompt, targets).detach()

    def generate(
        self, prompt: list[int], max_tokens: int, seed: int, *, greedy: bool = False
    ) -> Generation:
        if max_tokens < 1:
            return Generation([], [], "")
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        settings = GenerationConfig(
            do_sample=not greedy,
            temperature=1.0 if not greedy else None,
            top_p=1.0 if not greedy else None,
            top_k=0 if not greedy else None,
            max_new_tokens=max_tokens,
            repetition_penalty=1.0,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            torch.manual_seed(seed)
            if devices:
                torch.cuda.manual_seed_all(seed)
            inputs = torch.tensor([prompt], device=self.device)
            output = self.model.generate(
                input_ids=inputs, attention_mask=torch.ones_like(inputs), generation_config=settings
            )
            ids = output[0, len(prompt) :].tolist()
            old = self.score(prompt, ids).tolist()
        return Generation(ids, old, self.tokenizer.decode(ids, skip_special_tokens=True))

    def gradients(
        self, examples: list[LossExample], *, step: bool = False
    ) -> tuple[list[Tensor], dict]:
        optimizer = self.optimizers[self.role]
        parameters = self.parameters(self.role)
        previous = [p.grad for p in parameters] if not step else []
        was_training = self.model.training
        self.model.train()  # Transformers checkpointing is active only in training mode.
        optimizer.zero_grad(set_to_none=True)
        try:
            metrics = self._backward_examples(examples)
            grads, finished = self._gradient_result(parameters, step)
            return grads, metrics | finished
        finally:
            self.model.train(was_training)
            optimizer.zero_grad(set_to_none=True)
            if not step:
                for parameter, gradient in zip(parameters, previous, strict=True):
                    parameter.grad = gradient

    def _backward_examples(self, examples: list[LossExample]) -> dict:
        from selfplay_opsd.objectives import grpo, signed_gap

        metrics = {"rl_loss": 0.0, "sd_loss": 0.0, "kl_loss": 0.0, "total_loss": 0.0}
        nonempty = [e for e in examples if e.targets]
        size = self.config.hardware.microbatch_episodes
        for start in range(0, len(nonempty), size):
            batch = nonempty[start : start + size]
            scores = self.score_many([(e.prompt, e.targets) for e in batch], gradients=True)
            total = torch.zeros((), device=self.device)
            for example, score in zip(batch, scores, strict=True):
                current = score.unsqueeze(0)
                mask = torch.ones_like(current)
                rl = current.sum() * 0
                sd = current.sum() * 0
                kl = current.sum() * 0
                if example.kind in {"rl", "combined"}:
                    rl = grpo(
                        current,
                        torch.tensor([example.old], device=self.device),
                        torch.tensor([example.advantage], device=self.device),
                        mask,
                        self.config.objective.clip,
                    )
                if example.kind in {"sd", "combined"}:
                    sd = signed_gap(
                        current, torch.tensor([example.coefficient], device=self.device), mask
                    )
                if example.kind == "ce":
                    rl = -current.mean()
                if example.kl_coefficient:
                    if example.base_logps is None or len(example.base_logps) != len(
                        example.targets
                    ):
                        raise ExperimentError("Fixed-base KL targets missing")
                    ratio = (
                        torch.tensor([example.base_logps], device=self.device).detach() - current
                    )
                    kl = (ratio.exp() - ratio - 1).mean() * example.kl_coefficient
                scale = len(example.targets) * example.token_weight
                loss = (rl + sd + kl) * scale
                total = total + loss
                metrics["rl_loss"] += float(rl.detach()) * scale
                metrics["sd_loss"] += float(sd.detach()) * scale
                metrics["kl_loss"] += float(kl.detach()) * scale
                metrics["total_loss"] += float(loss.detach())
            total.backward()
        return metrics

    def _gradient_result(self, parameters: list[Tensor], step: bool) -> tuple[list[Tensor], dict]:
        grads = [
            torch.zeros_like(p, dtype=torch.float32)
            if p.grad is None
            else p.grad.detach().float().clone()
            for p in parameters
        ]
        square = sum(
            (g.double().square().sum() for g in grads),
            torch.zeros((), device=self.device, dtype=torch.float64),
        )
        if not torch.isfinite(square):
            raise FloatingPointError("Nonfinite training/scoring gradient")
        norm = square.sqrt().item()
        metrics = {
            "grad_norm": norm,
            "stepped": int(step and norm > 0),
            "clip_coefficient": min(1.0, self.config.objective.grad_clip / (norm + 1e-6)),
            "update_norm": 0.0,
        }
        if step and norm > 0:
            before = [p.detach().clone() for p in parameters]
            torch.nn.utils.clip_grad_norm_(
                parameters, self.config.objective.grad_clip, error_if_nonfinite=True
            )
            self.optimizers[self.role].step()
            delta = sum(
                (
                    (p.detach() - old).double().square().sum()
                    for p, old in zip(parameters, before, strict=True)
                ),
                torch.zeros((), device=self.device, dtype=torch.float64),
            )
            metrics["update_norm"] = delta.sqrt().item()
        return grads, metrics

    def state_dict(self) -> dict:
        return {
            "adapters": {
                n: p.detach().cpu().clone()
                for n, p in self.model.named_parameters()
                if ".student." in n or ".proposer." in n
            },
            "optimizers": {role: opt.state_dict() for role, opt in self.optimizers.items()},
            "parameter_names": {
                role: [n for n, _ in self.named_parameters(role)] for role in self.optimizers
            },
        }

    def load_state_dict(self, state: dict) -> None:
        for role in self.optimizers:
            if state["parameter_names"][role] != [n for n, _ in self.named_parameters(role)]:
                raise ExperimentError("Checkpoint adapter coordinate names/order changed")
        parameters = dict(self.model.named_parameters())
        with torch.no_grad():
            for name, tensor in state["adapters"].items():
                parameters[name].copy_(tensor.to(parameters[name]))
        for role, optimizer in self.optimizers.items():
            optimizer.load_state_dict(state["optimizers"][role])

    def sync_adapter(self, role: str) -> str:
        return self.version(role)

    def prepare_rollout(self):
        return None

    def generate_unicast(self, prompt, max_tokens, seed, *, greedy=False):
        return self.generate(prompt, max_tokens, seed, greedy=greedy)

    def broadcast(self, value):
        return value

    def coordinator_call(self, function):
        """Execute trusted IO once and propagate failures to every training rank."""
        result, error = None, None
        if self.is_coordinator:
            try:
                result = function()
            except ValueError as exc:
                error = ("invalid", str(exc), "")
            except Exception as exc:
                status = (
                    exc.status.value
                    if isinstance(exc, ExperimentError)
                    else "sandbox_infrastructure_error"
                )
                error = ("fatal", str(exc), status)
        result, error = self.broadcast((result, error))
        if error:
            if error[0] == "invalid":
                raise ValueError(error[1])
            from selfplay_opsd.core import Status

            raise ExperimentError(error[1], Status(error[2]))
        return result

    def barrier(self):
        return None

    def close(self):
        return None

    def prune_rollout_exports(self):
        return None


def turn_from_generation(
    policy: HFPolicy,
    history: list[dict[str, str]],
    generation: Generation,
    action: str,
    parse_status: str,
) -> TokenTurn:
    return TokenTurn(
        prompt_ids=policy.encode_history(history),
        target_ids=generation.ids,
        old_logps=generation.logps,
        history=history,
        raw_text=generation.text,
        action=action,
        parse_status=parse_status,
    )
