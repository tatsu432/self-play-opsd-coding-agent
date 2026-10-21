"""The scientific objective: no framework defaults, no optimizer side effects."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor


def advantages(rewards: Tensor) -> Tensor:
    values = rewards.detach().float()
    centered = values - values.mean(dim=-1, keepdim=True)
    sigma = centered.square().mean(dim=-1, keepdim=True).sqrt()
    return torch.where(sigma == 0, torch.zeros_like(centered), centered / (sigma + 1e-6))


def episode_means(values: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(values.dtype)
    return (values * weights).sum(-1) / weights.sum(-1).clamp_min(1)


def grpo(
    current: Tensor, old: Tensor, advantage: Tensor, mask: Tensor, clip: float = 0.2
) -> Tensor:
    ratio = (current - old.detach()).exp()
    a = advantage.detach().unsqueeze(-1)
    surrogate = torch.minimum(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a)
    return -episode_means(surrogate, mask).mean()


def sd_coefficients(
    teacher: Tensor, old: Tensor, gated: bool, beta: float = 1.0, gap_clip: float = 5.0
) -> tuple[Tensor, Tensor, Tensor]:
    gap = (teacher.detach().float() - old.detach().float()).detach()
    gate = torch.sigmoid(beta * gap) if gated else torch.ones_like(gap)
    return gap, gate.detach(), (gate * gap.clamp(-gap_clip, gap_clip)).detach()


def signed_gap(current: Tensor, coefficients: Tensor, mask: Tensor) -> Tensor:
    return -episode_means(current * coefficients.detach(), mask).mean()


def reference_ce(current: Tensor, edit_mask: Tensor) -> Tensor:
    return -episode_means(current, edit_mask).mean()


def difficulty(rewards: Iterable[float]) -> float:
    values = list(rewards)
    p = sum(values) / len(values)
    return 4 * p * (1 - p)


@dataclass(frozen=True)
class Alignment:
    cosine: float | None
    reward: float
    reference_norm: float
    candidate_norm: float
    exact_zero: bool
    near_zero: bool


def alignment(
    candidate: Iterable[Tensor],
    reference: Iterable[Tensor],
    threshold: float = 1e-12,
    distributed: bool = False,
) -> Alignment:
    """Inputs are unique coordinate shards if distributed; replicated values are forbidden."""
    pairs = list(zip(candidate, reference, strict=True))
    if not pairs:
        raise ValueError("No trainable gradient coordinates")
    device = pairs[0][0].device
    stats = torch.zeros(3, device=device, dtype=torch.float64)
    for c, r in pairs:
        if c.shape != r.shape:
            raise ValueError("Gradient coordinate mismatch")
        c, r = c.detach().double(), r.detach().double()
        stats += torch.stack([(c * r).sum(), c.square().sum(), r.square().sum()])
    if distributed:
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
    if not torch.isfinite(stats).all():
        raise FloatingPointError("Nonfinite alignment gradient")
    dot, cn, rn = stats.tolist()
    cn, rn = cn**0.5, rn**0.5
    near_zero = min(cn, rn) <= threshold
    if near_zero:
        return Alignment(None, 0.5, rn, cn, min(cn, rn) == 0, True)
    cosine = dot / (cn * rn)
    if not -1.000001 <= cosine <= 1.000001:
        raise FloatingPointError(f"Invalid cosine {cosine}")
    cosine = min(1.0, max(-1.0, cosine))
    return Alignment(cosine, (1 + cosine) / 2, rn, cn, False, False)


def collect_gradients(loss: Tensor, parameters: list[Tensor]) -> list[Tensor]:
    """Independent non-mutating reference collector for unsharded models."""
    grads = torch.autograd.grad(loss, parameters, allow_unused=True)
    return [
        torch.zeros_like(p, dtype=torch.float32) if g is None else g.detach().float().clone()
        for p, g in zip(parameters, grads, strict=True)
    ]
