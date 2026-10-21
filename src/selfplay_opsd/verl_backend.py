"""Thin verl actor extension, FSDP2 coordinates, and phase-separated vLLM rollout.

Pinned upstream: verl bec9ef74768dd201881cd4e54cd0385e87caae27 (v0.7.1).
All ranks evaluate the same mathematical microbatches. FSDP averages identical
contributions, so uneven/empty data ranks cannot alter objective normalization.
Only parameter-coordinate shards differ between ranks. This prioritizes a clear
correctness baseline over additional data-parallel throughput.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import selectors
import shutil
import subprocess
import sys
import time
from typing import Any, cast

import torch

from selfplay_opsd.config import RunConfig
from selfplay_opsd.core import ExperimentError, canonical
from selfplay_opsd.policy import Generation, HFPolicy, LossExample


class VerlPolicy(HFPolicy):
    def __init__(self, config: RunConfig):
        if not torch.cuda.is_available():
            raise ExperimentError("verl production backend requires allocated CUDA devices")
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import MixedPrecisionPolicy
        from verl.utils.fsdp_utils import apply_fsdp2

        started = time.monotonic()
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        if not torch.distributed.is_initialized():
            if "RANK" not in os.environ:
                raise ExperimentError(
                    "Launch every GPU profile with torchrun, including world size 1"
                )
            torch.distributed.init_process_group("nccl")
        self.world_size = torch.distributed.get_world_size()
        self.rank = torch.distributed.get_rank()
        self.is_coordinator = self.rank == 0
        self.distributed = self.world_size > 1
        if self.world_size != config.hardware.gpu_count:
            raise ExperimentError("torchrun world size does not match hardware profile")
        super().__init__(config, device=f"cuda:{local_rank}")
        self.mesh = init_device_mesh("cuda", (self.world_size,))
        for role in ("student", "proposer"):
            for parameter in self.parameters(role):
                parameter.requires_grad_(True)
        apply_fsdp2(
            self.model,
            {
                "mesh": self.mesh,
                "mp_policy": MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16, reduce_dtype=torch.float32
                ),
            },
            {"wrap_policy": {"transformer_layer_cls_to_wrap": ["Qwen3DecoderLayer"]}},
        )
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
        self._versions: dict[str, str] = {}
        self._adapter_paths: dict[str, str] = {}
        self._rollout: Any = None
        self._awake = False
        self._offloaded = False
        self._adapter_ids: dict[str, int] = {}
        self._log: Any = None
        self._rpc_stream: Any = None
        self.actor = self._actor_extension()
        self.initialization_seconds = time.monotonic() - started

    def _actor_extension(self):
        from verl.workers.actor.base import BasePPOActor

        owner = self

        class ExactTokenActor(BasePPOActor):
            """Official BasePPOActor port; project loss metadata remains explicit."""

            def compute_log_prob(self, data):
                return owner.score(data.meta_info["prompt_ids"], data.meta_info["target_ids"])

            def update_policy(self, data):
                return owner._gradient_examples(
                    [LossExample(**x) for x in data.meta_info["examples"]], step=True
                )[1]

            def collect_gradients(self, examples, step=False):
                return owner._gradient_examples(examples, step=step)

        return ExactTokenActor(self.config)

    def broadcast(self, value):
        values = [value if self.is_coordinator else None]
        torch.distributed.broadcast_object_list(values, src=0, device=self.device)
        return values[0]

    def barrier(self):
        torch.distributed.barrier()

    def _full(self, value):
        if hasattr(value, "full_tensor"):
            value = value.full_tensor()
        return value.detach().cpu().clone()

    def version(self, role: str) -> str:
        if role not in self._versions:
            self._ensure_scoring()
            h = hashlib.sha256()
            for name, parameter in self.named_parameters(role):
                h.update(name.replace(f".{role}.", ".adapter.").encode())
                full = self._full(parameter)
                h.update(full.contiguous().view(torch.uint8).numpy().tobytes())
            self._versions[role] = h.hexdigest()
        return self._versions[role]

    def _rpc(self, payload: dict, timeout: float = 600, *, collective: bool = True):
        result, error = None, None
        if self.is_coordinator:
            try:
                self._rollout.stdin.write(canonical(payload) + "\n")
                self._rollout.stdin.flush()
                selector = selectors.DefaultSelector()
                try:
                    selector.register(self._rpc_stream, selectors.EVENT_READ)
                    if not selector.select(timeout):
                        raise TimeoutError("vLLM RPC deadline exceeded")
                    line = self._rpc_stream.readline(16 * 1024 * 1024)
                finally:
                    selector.close()
                result = json.loads(line)
                if not result.get("ok"):
                    raise ExperimentError(result.get("error", "vLLM worker failed"))
            except Exception as exc:
                error = str(exc)
        if collective:
            result, error = self.broadcast((result, error))
        if error:
            raise ExperimentError(error)
        return result

    def _ensure_scoring(self):
        from verl.utils.fsdp_utils import load_fsdp_model_to_gpu

        if getattr(self, "_awake", False):
            self._rpc({"action": "sleep"})
            self._awake = False
        if getattr(self, "_offloaded", False):
            load_fsdp_model_to_gpu(self.model)
            for optimizer in self.optimizers.values():
                for state in optimizer.state.values():
                    for key, value in state.items():
                        if torch.is_tensor(value) and value.ndim:
                            state[key] = value.to(self.device)
            self._offloaded = False

    def sync_adapter(self, role: str) -> str:
        from safetensors.torch import save_file

        self._ensure_scoring()
        version = self.version(role)
        if version not in self._adapter_paths:
            path = self.config.run_dir / "state/rollout_adapters" / version
            tensors = {
                name.replace(f".{role}.", "."): self._full(parameter)
                for name, parameter in self.named_parameters(role)
            }
            if self.is_coordinator:
                path.mkdir(parents=True, exist_ok=True)
                save_file(tensors, str(path / "adapter_model.safetensors"))
                self.model.peft_config[role].save_pretrained(str(path))
            self.barrier()
            self._adapter_paths[version] = str(path.resolve())
        return version

    def _ensure_rollout(self):
        from verl.utils.fsdp_utils import offload_fsdp_model_to_cpu

        version = (
            self.sync_adapter(self.role)
            if self.role not in self._versions
            or self._versions[self.role] not in self._adapter_paths
            else self._versions[self.role]
        )
        if not self._offloaded:
            offload_fsdp_model_to_cpu(self.model, empty_cache=False)
            for optimizer in self.optimizers.values():
                for state in optimizer.state.values():
                    for key, value in state.items():
                        if torch.is_tensor(value):
                            state[key] = value.cpu()
            torch.cuda.empty_cache()  # One phase boundary, never per microbatch.
            self._offloaded = True
        self.barrier()
        if self._rollout is None:
            error = None
            if self.is_coordinator:
                try:
                    path = self.config.run_dir / "state/vllm.log"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    self._log = path.open("a")
                    environment = {
                        k: v
                        for k, v in os.environ.items()
                        if k
                        not in {
                            "RANK",
                            "WORLD_SIZE",
                            "LOCAL_RANK",
                            "LOCAL_WORLD_SIZE",
                            "MASTER_ADDR",
                            "MASTER_PORT",
                            "GROUP_RANK",
                            "ROLE_RANK",
                            "ROLE_WORLD_SIZE",
                            "TORCHELASTIC_RUN_ID",
                        }
                    }
                    read_fd, write_fd = os.pipe()
                    environment["SELFPLAY_RPC_FD"] = str(write_fd)
                    environment["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
                    self._rollout = subprocess.Popen(
                        [sys.executable, "-m", "selfplay_opsd.vllm_worker"],
                        stdin=subprocess.PIPE,
                        stdout=self._log,
                        stderr=self._log,
                        pass_fds=(write_fd,),
                        text=True,
                        bufsize=1,
                        env=environment,
                    )
                    os.close(write_fd)
                    self._rpc_stream = os.fdopen(read_fd, "r", buffering=1)
                except Exception as exc:
                    error = str(exc)
            error = self.broadcast(error)
            if error:
                raise ExperimentError(error)
            # Non-coordinator ranks need a non-null sentinel for lifecycle checks.
            if not self.is_coordinator:
                self._rollout = True
            self._rpc(
                {
                    "engine": {
                        "model": self.config.model.name,
                        "revision": self.config.model.revision,
                        "tokenizer_revision": self.config.model.tokenizer_revision,
                        "dtype": "bfloat16",
                        "tensor_parallel_size": self.config.hardware.tensor_parallel,
                        "gpu_memory_utilization": self.config.hardware.rollout_memory_fraction,
                        "max_model_len": self.config.episode.max_student_sequence_tokens,
                        "enable_lora": True,
                        "max_lora_rank": self.config.model.rank,
                        "max_loras": 2,
                        "max_cpu_loras": 4,
                        "enable_sleep_mode": True,
                        "enable_prefix_caching": False,
                        "enforce_eager": True,
                        "trust_remote_code": False,
                        "disable_log_stats": True,
                        "seed": self.config.seed,
                    }
                },
                timeout=900,
            )
            self._awake = True
        elif not self._awake:
            self._rpc({"action": "wake"})
            self._awake = True
        return version

    def generate(
        self, prompt: list[int], max_tokens: int, seed: int, *, greedy: bool = False
    ) -> Generation:
        if max_tokens <= 0:
            return Generation([], [], "")
        version = self._ensure_rollout()
        result = self._rpc(
            {
                "action": "generate",
                "prompt": prompt,
                "max_tokens": max_tokens,
                "seed": seed,
                "greedy": greedy,
                "version": version,
                "adapter_id": self._adapter_ids.setdefault(version, len(self._adapter_ids) + 1),
                "adapter_path": self._adapter_paths[version],
            }
        )
        return Generation(result["ids"], result["logps"], result["text"])

    def prepare_rollout(self):
        self._ensure_rollout()

    def generate_unicast(self, prompt, max_tokens, seed, *, greedy=False):
        if not self.is_coordinator or not self._awake:
            raise ExperimentError("Unicast rollout requires the prepared coordinator engine")
        version = self._versions[self.role]
        result = self._rpc(
            {
                "action": "generate",
                "prompt": prompt,
                "max_tokens": max_tokens,
                "seed": seed,
                "greedy": greedy,
                "version": version,
                "adapter_id": self._adapter_ids.setdefault(version, len(self._adapter_ids) + 1),
                "adapter_path": self._adapter_paths[version],
            },
            collective=False,
        )
        return Generation(result["ids"], result["logps"], result["text"])

    def score(self, prompt, targets, *, gradients=False):
        self._ensure_scoring()
        return super().score(prompt, targets, gradients=gradients)

    def score_many(self, requests, *, gradients=False):
        self._ensure_scoring()
        return super().score_many(requests, gradients=gradients)

    def gradients(self, examples, *, step=False):
        return self.actor.collect_gradients(examples, step=step)

    def _gradient_examples(self, examples, *, step=False):
        self._ensure_scoring()
        return HFPolicy.gradients(self, examples, step=step)

    def _gradient_result(self, parameters, step):
        gradients = []
        for parameter in parameters:
            local = parameter.to_local() if hasattr(parameter, "to_local") else parameter
            grad = parameter.grad
            if grad is not None:
                grad = grad.to_local() if hasattr(grad, "to_local") else grad
            if not hasattr(parameter, "placements") or not any(
                p.is_shard() for p in parameter.placements
            ):
                raise ExperimentError("Every scored coordinate must be a unique FSDP shard")
            gradients.append(
                torch.zeros_like(local, dtype=torch.float32)
                if grad is None
                else grad.detach().float().clone()
            )
        norm_square = sum(
            (g.double().square().sum() for g in gradients),
            torch.zeros((), device=self.device, dtype=torch.float64),
        )
        torch.distributed.all_reduce(norm_square)
        norm = norm_square.sqrt().item()
        if not torch.isfinite(norm_square):
            raise FloatingPointError("Nonfinite distributed gradient")
        metrics = {
            "grad_norm": norm,
            "stepped": int(step and norm > 0),
            "clip_coefficient": min(1.0, self.config.objective.grad_clip / (norm + 1e-6)),
        }
        if step and norm > 0:
            coefficient = min(1.0, self.config.objective.grad_clip / (norm + 1e-6))
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad.mul_(coefficient)
            self.optimizers[self.role].step()
            self._versions.pop(self.role, None)
        return gradients, metrics

    def state_dict(self):
        self._ensure_scoring()

        def full_tree(value):
            if torch.is_tensor(value):
                return self._full(value)
            if isinstance(value, dict):
                return {k: full_tree(v) for k, v in value.items()}
            if isinstance(value, list):
                return [full_tree(v) for v in value]
            return copy.deepcopy(value)

        return {
            "adapters": {
                name: self._full(p)
                for role in self.optimizers
                for name, p in self.named_parameters(role)
            },
            "optimizers": {
                role: full_tree(opt.state_dict()) for role, opt in self.optimizers.items()
            },
            "parameter_names": {
                role: [n for n, _ in self.named_parameters(role)] for role in self.optimizers
            },
        }

    def load_state_dict(self, state):
        from torch.distributed.tensor import distribute_tensor

        self._ensure_scoring()
        for role in self.optimizers:
            pairs = self.named_parameters(role)
            if state["parameter_names"][role] != [n for n, _ in pairs]:
                raise ExperimentError("Adapter coordinate order changed")
            with torch.no_grad():
                for name, parameter in pairs:
                    parameter = cast(Any, parameter)
                    parameter.copy_(
                        distribute_tensor(
                            state["adapters"][name].to(self.device),
                            device_mesh=parameter.device_mesh,
                            placements=parameter.placements,
                        )
                    )
            restored = copy.deepcopy(state["optimizers"][role])
            ids = [i for group in restored["param_groups"] for i in group["params"]]
            for param_id, (_, parameter) in zip(ids, pairs, strict=True):
                parameter = cast(Any, parameter)
                for key, value in restored["state"].get(param_id, {}).items():
                    if torch.is_tensor(value) and value.shape == parameter.shape:
                        restored["state"][param_id][key] = distribute_tensor(
                            value.to(self.device),
                            device_mesh=parameter.device_mesh,
                            placements=parameter.placements,
                        )
            self.optimizers[role].load_state_dict(restored)
        self._versions.clear()

    def close(self):
        if self._rollout is not None and self.is_coordinator:
            try:
                self._rollout.stdin.write('{"action":"close"}\n')
                self._rollout.stdin.flush()
                self._rollout.wait(timeout=20)
            except (OSError, subprocess.TimeoutExpired):
                self._rollout.kill()
                self._rollout.wait(timeout=10)
            self._rollout.stdin.close()
            if self._rpc_stream:
                self._rpc_stream.close()
            if self._log:
                self._log.close()
        self._rollout = None
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    def prune_rollout_exports(self):
        keep = set(self._versions.values())
        for version in list(self._adapter_paths):
            if version not in keep:
                if self.is_coordinator:
                    shutil.rmtree(self._adapter_paths[version])
                del self._adapter_paths[version]
        self.barrier()
