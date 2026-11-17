"""Versioned scientific control and objective for the Task 19 shared policy."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from pydantic import Field, model_validator
from torch import Tensor

from selfplay_opsd.config import EpisodeConfig, SandboxConfig, Strict
from selfplay_opsd.core import digest

MODEL = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
STUDY = "repair-selfplay-2026-task19-v1"
Arm = Literal["control", "A", "B", "C", "D"]
ARMS = ("control", "A", "B", "C", "D")
GROUP = 4
PILOT = 32


class Task19Control(Strict):
    study: str = STUDY
    seed: int = 2026
    calibration_hash: str | None = None
    qualification_hash: str | None = None
    endpoint: int | None = Field(default=None, gt=PILOT, le=1024, multiple_of=32)
    episode: EpisodeConfig = EpisodeConfig(
        edit_protocol="replace_text",
        max_generated_tokens=8192,
        max_turn_generated_tokens=2048,
        max_student_sequence_tokens=16384,
        max_teacher_sequence_tokens=20480,
        privilege_reserve=4096,
    )
    proposer_thinking: bool = False
    proposal_tokens: int = Field(default=1024, gt=0)
    cached_rollout: bool = True
    sd_scale: float | None = Field(default=None, gt=0)
    proposer_scale: float | None = Field(default=None, gt=0)
    learning_rate: float = Field(default=1e-5, gt=0)
    clip: float = Field(default=0.2, gt=0, lt=1)
    grad_clip: float = Field(default=1.0, gt=0)
    sandbox: SandboxConfig = SandboxConfig(
        backend="remote",
        process_policy="single_process",
        resource_backend="slurm",
        remote_url="http://127.0.0.1:8765",
    )

    @model_validator(mode="after")
    def validate_study(self):
        if self.seed != 2026 or self.episode.edit_protocol != "replace_text":
            raise ValueError("Task 19 requires seed 2026 and exact-text repair")
        if not self.study.startswith("repair-selfplay-2026-task19-"):
            raise ValueError("Task 19 requires a distinct study namespace")
        if self.endpoint is not None and not self.ready:
            raise ValueError("The endpoint requires executed calibration and qualification")
        return self

    @property
    def ready(self) -> bool:
        return (
            self.sd_scale is not None
            and self.proposer_scale is not None
            and self.calibration_hash is not None
            and self.qualification_hash is not None
        )

    @property
    def science(self) -> str:
        return digest(
            {
                "model": MODEL,
                "revision": REVISION,
                "control": self.model_dump(mode="json", exclude={"endpoint"}),
            }
        )

    @classmethod
    def read(cls, path: Path) -> Task19Control:
        return cls.model_validate_json(path.read_text())

    def milestones(self) -> list[int]:
        if self.endpoint is None:
            return [0, PILOT]
        return sorted({0, PILOT, self.endpoint, *(self.endpoint * n // 4 for n in (1, 2, 3))})


def centered_rewards(rewards: list[float]) -> Tensor:
    values = torch.tensor(rewards)
    return values - values.mean()


def clipped_sum(current: Tensor, behavior: Tensor, advantage: float, clip: float) -> Tensor:
    # A sampled token can leave the current top-k/top-p support. Its exact
    # ratio is zero, including its derivative through the rejected logit.
    ratio = (current - behavior.detach()).exp()
    return -torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage).sum()


def opsd_sum(current: Tensor, gap: Tensor, *, gated: bool) -> Tensor:
    gap = gap.detach().float()
    weight = gap.sigmoid() if gated else torch.ones_like(gap)
    return -(current * (weight * gap.clamp(-5, 5))).sum()
