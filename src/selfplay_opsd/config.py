"""Strict method/resource configuration with explicit scientific compatibility."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from selfplay_opsd.core import digest


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class ModelConfig(Strict):
    name: str = "Qwen/Qwen3-4B-Instruct-2507"
    revision: str = "UNRESOLVED"
    tokenizer_revision: str = "UNRESOLVED"
    dtype: Literal["bfloat16", "float32"] = "bfloat16"
    rank: int = Field(default=32, gt=0)
    alpha: int = Field(default=64, gt=0)
    dropout: float = 0.0
    target_modules: list[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    backend: Literal["hf", "verl"] = "verl"
    tiny: bool = False


class EpisodeConfig(Strict):
    max_turns: int = Field(default=8, ge=1)
    max_generated_tokens: int = Field(default=4096, ge=1)
    max_turn_generated_tokens: int = Field(default=1024, ge=1)
    max_student_sequence_tokens: int = Field(default=10240, ge=1)
    max_teacher_sequence_tokens: int = Field(default=12288, ge=1)
    privilege_reserve: int = Field(default=2048, ge=1)
    max_observation_tokens: int = Field(default=1024, ge=1)

    @model_validator(mode="after")
    def limits(self):
        if (
            self.max_teacher_sequence_tokens
            < self.max_student_sequence_tokens + self.privilege_reserve
        ):
            raise ValueError(
                "Teacher sequence must accommodate the student sequence plus privilege reserve"
            )
        if self.max_turn_generated_tokens > self.max_generated_tokens:
            raise ValueError("Per-turn budget exceeds episode budget")
        return self


class ObjectiveConfig(Strict):
    clip: float = Field(default=0.2, gt=0, lt=1)
    sd_lambda: float = Field(default=0.1, ge=0)
    gate_beta: float = Field(default=1.0, gt=0)
    gap_clip: float = Field(default=5.0, gt=0)
    student_lr: float = Field(default=1e-5, gt=0)
    proposer_lr: float = Field(default=5e-6, gt=0)
    grad_clip: float = Field(default=1.0, gt=0)
    near_zero: float = Field(default=1e-12, gt=0)
    fixed_base_kl: float = Field(default=0.0, ge=0)


class HardwareConfig(Strict):
    profile: str = "dual_gpu_80gb"
    gpu_count: int = Field(default=2, ge=0)
    allocated_gpu_count: int | None = Field(default=None, ge=0)
    min_vram_gib: int = Field(default=79, ge=0)
    microbatch_episodes: int = Field(default=1, ge=1)
    cpu_workers: int = Field(default=8, ge=1)
    rollout_concurrency: int = Field(default=4, ge=1)
    host_ram_gib: int = Field(default=128, ge=1)
    tensor_parallel: int = Field(default=2, ge=1)
    activation_checkpointing: bool = True
    offload: bool = True
    rollout_memory_fraction: float = Field(default=0.5, gt=0, lt=1)

    @model_validator(mode="after")
    def parallelism(self):
        if self.allocated_gpu_count is not None and self.allocated_gpu_count < self.gpu_count:
            raise ValueError("Allocated GPU count cannot be less than used GPU count")
        if self.gpu_count and (
            self.tensor_parallel > self.gpu_count or self.gpu_count % self.tensor_parallel
        ):
            raise ValueError("Tensor parallelism must divide the allocated GPU count")
        if self.gpu_count == 0 and self.tensor_parallel != 1:
            raise ValueError("CPU debug requires tensor_parallel=1")
        return self


class SandboxConfig(Strict):
    backend: Literal["docker", "apptainer", "remote"] = "docker"
    image: str = "selfplay-worker:local"
    runtime: str = "docker"
    remote_url: str | None = None
    remote_token_env: str = "SELFPLAY_WORKER_TOKEN"
    cpus: int = Field(default=1, ge=1)
    memory_mib: int = Field(default=1024, ge=64)
    pids: int = Field(default=32, ge=8)
    output_bytes: int = Field(default=4 * 1024 * 1024 + 4096, ge=1024)
    disk_mib: int = Field(default=64, ge=8)
    retries: int = Field(default=2, ge=0, le=5)
    public_seconds: float = Field(default=10.0, gt=0)
    verifier_seconds: float = Field(default=30.0, gt=0)

    @model_validator(mode="after")
    def remote_transport(self):
        if self.remote_url:
            from urllib.parse import urlparse

            url = urlparse(self.remote_url)
            if url.username or url.password or url.scheme not in {"https", "http"}:
                raise ValueError(
                    "Remote worker URL must use HTTPS or loopback HTTP, without credentials"
                )
            if url.scheme == "http" and url.hostname not in {"127.0.0.1", "localhost", "::1"}:
                raise ValueError("Use HTTPS or an SSH-forwarded loopback worker endpoint")
        return self


class LoggingConfig(Strict):
    mode: Literal["offline", "online", "disabled"] = "offline"
    project: str = "self-play-opsd-coding-agent"
    entity: str | None = None
    sample_every: int = Field(default=20, ge=1)
    table_rows: int = Field(default=32, ge=1)
    upload_checkpoints: bool = False


class RunConfig(Strict):
    study_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")
    method: Literal["A", "B", "C", "D", "GRPO", "GRPO_KL"] = "B"
    seed: int = 2026
    iterations: int = Field(default=200, ge=1)
    parents_per_iteration: int = Field(default=4, ge=1)
    proposals_per_parent: int = Field(default=4, ge=2)
    episodes_per_task: int = Field(default=4, ge=2)
    proposer_max_tokens: int = Field(default=512, ge=1)
    checkpoint_every: int = Field(default=10, ge=1)
    evaluate_every: int = Field(default=20, ge=1)
    diagnostic_attempts: int = Field(default=320, ge=16)
    output_root: Path = Path("runs")
    data_root: Path = Path("data/prepared")
    max_active_hours: float | None = Field(default=None, gt=0)
    max_gpu_hours: float | None = Field(default=None, gt=0)
    model: ModelConfig = ModelConfig()
    episode: EpisodeConfig = EpisodeConfig()
    objective: ObjectiveConfig = ObjectiveConfig()
    hardware: HardwareConfig = HardwareConfig()
    sandbox: SandboxConfig = SandboxConfig()
    logging: LoggingConfig = LoggingConfig()

    @model_validator(mode="after")
    def named_variants(self):
        if self.method == "GRPO_KL" and self.objective.fixed_base_kl <= 0:
            raise ValueError("GRPO_KL requires an explicit positive fixed_base_kl coefficient")
        if self.method != "GRPO_KL" and self.objective.fixed_base_kl != 0:
            raise ValueError("Fixed-base KL is permitted only in the named GRPO_KL variant")
        return self

    @property
    def charged_gpu_count(self) -> int:
        return (
            self.hardware.gpu_count
            if self.hardware.allocated_gpu_count is None
            else self.hardware.allocated_gpu_count
        )

    @property
    def gate(self) -> str:
        return "none" if self.method in {"A", "GRPO", "GRPO_KL"} else "sigmoid_gap"

    @property
    def candidate_objective(self) -> str | None:
        return {"C": "rl_only", "D": "rl_plus_gated_sd"}.get(self.method)

    @property
    def scientific_hash(self) -> str:
        values = self.model_dump(
            mode="json",
            exclude={
                "hardware",
                "logging",
                "output_root",
                "data_root",
                "max_active_hours",
                "max_gpu_hours",
                "checkpoint_every",
            },
        )
        values["model"].pop("backend")
        values["sandbox"] = {
            k: values["sandbox"][k]
            for k in (
                "cpus",
                "memory_mib",
                "pids",
                "output_bytes",
                "disk_mib",
                "retries",
                "public_seconds",
                "verifier_seconds",
            )
        }
        return digest(values)

    @property
    def run_dir(self) -> Path:
        return self.output_root / self.study_id / f"{self.method}-seed{self.seed}"

    def resolved(self) -> dict:
        return self.model_dump(mode="json") | {
            "gate": self.gate,
            "proposer_reward": "gradient_alignment" if self.candidate_objective else "difficulty",
            "candidate_objective": self.candidate_objective,
            "reference_objective": "canonical_repair_ce",
            "scientific_hash": self.scientific_hash,
        }


PROFILES: dict[str, dict] = {
    "cpu_debug": {
        "gpu_count": 0,
        "min_vram_gib": 0,
        "tensor_parallel": 1,
        "cpu_workers": 1,
        "host_ram_gib": 4,
        "activation_checkpointing": False,
    },
    "single_gpu_80gb": {"gpu_count": 1, "tensor_parallel": 1},
    "dual_gpu_80gb": {"gpu_count": 2, "tensor_parallel": 2},
    "dual_gpu_141gb": {"gpu_count": 2, "tensor_parallel": 2, "min_vram_gib": 130},
    "multi_gpu_single_node_4": {
        "gpu_count": 4,
        "tensor_parallel": 4,
        "cpu_workers": 16,
        "host_ram_gib": 256,
    },
    "multi_gpu_single_node_8": {
        "gpu_count": 8,
        "tensor_parallel": 8,
        "cpu_workers": 16,
        "host_ram_gib": 256,
    },
}


def load_config(path: Path | None = None, **overrides) -> RunConfig:
    values = yaml.safe_load(path.read_text()) if path else {}
    values = values or {}
    original_profile = values.get("hardware", {}).get("profile", "dual_gpu_80gb")
    profile = overrides.pop("hardware_profile", original_profile)
    if profile not in PROFILES:
        raise ValueError(f"Unknown hardware profile: {profile}; choose {list(PROFILES)}")
    if profile != original_profile:
        values["hardware"] = {**values.get("hardware", {}), **PROFILES[profile], "profile": profile}
    else:
        values["hardware"] = {**PROFILES[profile], **values.get("hardware", {}), "profile": profile}
    if "SELFPLAY_ALLOCATED_GPU_COUNT" in os.environ:
        values["hardware"]["allocated_gpu_count"] = int(os.environ["SELFPLAY_ALLOCATED_GPU_COUNT"])
    # Allocation paths are operational and excluded from the scientific hash.
    for environment, field in (
        ("SELFPLAY_DATA_ROOT", "data_root"),
        ("SELFPLAY_OUTPUT_ROOT", "output_root"),
    ):
        if environment in os.environ:
            value = Path(os.environ[environment])
            if not value.is_absolute():
                raise ValueError(f"{environment} must be an absolute path")
            values[field] = value
    values.update({k: v for k, v in overrides.items() if v is not None})
    if profile == "cpu_debug":
        debug_defaults = {"backend": "hf", "dtype": "float32", "tiny": True, "rank": 4, "alpha": 8}
        values["model"] = {
            "backend": "hf",
            "dtype": "float32",
            "tiny": True,
            "rank": 4,
            "alpha": 8,
            **values.get("model", {}),
            **(debug_defaults if profile != original_profile else {}),
        }
    return RunConfig.model_validate(values)
