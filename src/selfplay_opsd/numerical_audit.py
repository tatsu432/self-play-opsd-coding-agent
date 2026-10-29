"""Bounded saved-token numerical audit; no generation, candidate execution or steps."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from selfplay_opsd.config import RunConfig
from selfplay_opsd.core import EpisodeRecord, ExperimentError, ProposalRecord
from selfplay_opsd.objectives import advantages, alignment, sd_coefficients
from selfplay_opsd.storage import CheckpointManager, atomic_json, file_hash, read_json


def error_statistics(deltas: list[float], clip: float) -> dict:
    values = np.asarray(deltas, dtype=float)
    if not np.isfinite(values).all():
        raise FloatingPointError("Nonfinite numerical audit")
    absolute = np.abs(values)
    return {
        "tokens": len(values),
        "mean_signed": float(values.mean()) if len(values) else None,
        "mean_absolute": float(absolute.mean()) if len(values) else None,
        "p99_absolute": float(np.quantile(absolute, 0.99)) if len(values) else None,
        "max_absolute": float(absolute.max()) if len(values) else None,
        "outside_clip": int(((values < np.log1p(-clip)) | (values > np.log1p(clip))).sum()),
    }


def score_role(policy, turns: list, role: str, version: str, clip: float) -> tuple[list, dict]:
    bins: dict[str, list] = defaultdict(list)
    rescored = []
    with policy.active(role):
        if policy.version(role) != version:
            raise ExperimentError("Replay policy version mismatch")
        for turn in turns:
            score = policy.score(turn.prompt_ids, turn.target_ids).detach().float().cpu()
            old = torch.tensor(turn.old_logps)
            if len(score) != len(old):
                raise ExperimentError("Replay target alignment mismatch")
            delta = (score - old).tolist()
            rescored.append(turn.model_copy(update={"old_logps": score.tolist()}))
            bins["all"].extend(delta)
            for index, value in enumerate(delta):
                bins["position/first" if index == 0 else "position/later"].append(value)
                bins[f"length/{'short' if len(delta) <= 128 else 'long'}"].append(value)
                bins[f"context/{'short' if len(turn.prompt_ids) <= 2048 else 'long'}"].append(value)
    return rescored, {name: error_statistics(values, clip) for name, values in bins.items()}


def compare_gradients(left, right, policy, threshold):
    result = asdict(alignment(left, right, threshold, policy.distributed))
    difference = sum(
        (a.double() - b.double()).square().sum() for a, b in zip(left, right, strict=True)
    )
    if policy.distributed:
        torch.distributed.all_reduce(difference)
    result["relative_l2"] = (
        float(difference.sqrt() / result["reference_norm"]) if result["reference_norm"] else None
    )
    return result


def assess(
    policy, config: RunConfig, episodes: list[dict], proposals: list[dict], anchors: list[dict]
) -> dict:
    from selfplay_opsd.trainer import proposer_examples, reference_examples, student_examples

    groups: dict[str, list] = defaultdict(list)
    for row in episodes:
        episode = EpisodeRecord.model_validate(row)
        groups[episode.task_id].append(episode)
    if not groups:
        return {"available": False}
    parsed = [
        ProposalRecord.model_validate({k: v for k, v in row.items() if k != "reward"})
        for row in proposals
    ]
    if any(e.policy_version != policy.version("student") for g in groups.values() for e in g):
        raise ExperimentError("Student replay snapshot mismatch")
    if any(p.policy_version != policy.version("proposer") for p in parsed):
        raise ExperimentError("Proposer replay snapshot mismatch")
    flat = [e for group in groups.values() for e in group]
    turns, student_stats = score_role(
        policy,
        [t for e in flat for t in e.turns],
        "student",
        flat[0].policy_version,
        config.objective.clip,
    )
    changed = []
    cursor = 0
    gaps, weights, clip_changes = [], [], []
    for episode in flat:
        new_turns = turns[cursor : cursor + len(episode.turns)]
        cursor += len(episode.turns)
        changed.append(episode.model_copy(update={"turns": new_turns}))
        for a, b in zip(episode.turns, new_turns, strict=True):
            for gated in (False, True):
                before = sd_coefficients(
                    torch.tensor(a.teacher_logps),
                    torch.tensor(a.old_logps),
                    gated,
                    config.objective.gate_beta,
                    config.objective.gap_clip,
                )
                after = sd_coefficients(
                    torch.tensor(b.teacher_logps),
                    torch.tensor(b.old_logps),
                    gated,
                    config.objective.gate_beta,
                    config.objective.gap_clip,
                )
                if gated:
                    gaps.extend((before[0] - after[0]).tolist())
                    weights.extend((before[1] - after[1]).tolist())
                    clip_changes.extend(
                        (
                            (before[0].abs() > config.objective.gap_clip)
                            != (after[0].abs() > config.objective.gap_clip)
                        ).tolist()
                    )
    altered: dict[str, list] = defaultdict(list)
    for e in changed:
        altered[e.task_id].append(e)
    proposer_turns, proposer_stats = score_role(
        policy,
        [p.turn for p in parsed],
        "proposer",
        parsed[0].policy_version,
        config.objective.clip,
    )
    result: dict = {
        "student": student_stats,
        "proposer": proposer_stats,
        "opsd_gap_delta": error_statistics(gaps, config.objective.clip),
        "opsd_gate_abs_delta_mean": float(np.abs(weights).mean()) if weights else None,
        "opsd_gate_abs_delta_max": float(np.abs(weights).max()) if weights else None,
        "opsd_gap_clip_changed_tokens": sum(clip_changes),
        "methods": {},
    }
    with policy.active("student"):
        reference, _ = policy.gradients(reference_examples(anchors))
        for method in "ABCD":
            cfg = config.model_copy(update={"method": method}, deep=True)
            left_examples: list = []
            right_examples: list = []
            before_rewards, after_rewards = {}, {}
            candidate_deltas = []
            for task, group in groups.items():
                if len(group) != config.episodes_per_task:
                    raise ExperimentError("Incomplete reward group")
                before_examples, _ = student_examples(group, cfg, tasks=len(groups))
                after_examples, _ = student_examples(altered[task], cfg, tasks=len(groups))
                left_examples.extend(before_examples)
                right_examples.extend(after_examples)
                if method in "CD":
                    kind = "rl" if method == "C" else "combined"
                    a, _ = student_examples(group, cfg, kind=kind)
                    b, _ = student_examples(altered[task], cfg, kind=kind)
                    ga, _ = policy.gradients(a)
                    gb, _ = policy.gradients(b)
                    aa = alignment(ga, reference, cfg.objective.near_zero, policy.distributed)
                    ab = alignment(gb, reference, cfg.objective.near_zero, policy.distributed)
                    before_rewards[task], after_rewards[task] = aa.reward, ab.reward
                    candidate_deltas.append(
                        {
                            "before": asdict(aa),
                            "after": asdict(ab),
                            "gradient_change": compare_gradients(
                                ga, gb, policy, cfg.objective.near_zero
                            ),
                        }
                    )
            comparisons = {}
            for kind in ("rl", "sd", "combined"):
                from dataclasses import replace

                ga, _ = policy.gradients([replace(e, kind=kind) for e in left_examples])
                gb, _ = policy.gradients([replace(e, kind=kind) for e in right_examples])
                comparisons[kind] = compare_gradients(ga, gb, policy, cfg.objective.near_zero)
            result["methods"][method] = {
                "student_gradients": comparisons,
                "candidate_alignments": candidate_deltas,
            }
            if method in "CD":
                # All valid task rewards are recomputed under this method.
                original_rewards = [before_rewards.get(p.task_id or "", -1.0) for p in parsed]
                alternate_rewards = [after_rewards.get(p.task_id or "", -1.0) for p in parsed]
            else:
                from selfplay_opsd.objectives import difficulty

                original_rewards = [
                    difficulty(e.reward for e in groups[p.task_id or ""]) if p.valid else -1.0
                    for p in parsed
                ]
                alternate_rewards = original_rewards
            changed_proposals = [
                p.model_copy(update={"turn": t})
                for p, t in zip(parsed, proposer_turns, strict=True)
            ]
            with policy.active("proposer"):
                ga, _ = policy.gradients(proposer_examples(parsed, original_rewards, cfg))
                gb, _ = policy.gradients(
                    proposer_examples(changed_proposals, alternate_rewards, cfg)
                )
            result["methods"][method]["proposer_gradient_change"] = compare_gradients(
                ga, gb, policy, cfg.objective.near_zero
            )
            result["methods"][method]["proposer_reward_max_delta"] = max(
                abs(a - b) for a, b in zip(original_rewards, alternate_rewards, strict=True)
            )
    # Actual active clipping depends on advantage sign, not just outside interval.
    active = tokens = 0
    for task, group in groups.items():
        adv = advantages(torch.tensor([e.reward for e in group])).tolist()
        for e, new, advantage in zip(group, altered[task], adv, strict=True):
            for a, b in zip(e.turns, new.turns, strict=True):
                ratio = (torch.tensor(b.old_logps) - torch.tensor(a.old_logps)).exp()
                active += int(
                    (
                        (ratio > 1 + config.objective.clip)
                        if advantage > 0
                        else (ratio < 1 - config.objective.clip)
                        if advantage < 0
                        else torch.zeros_like(ratio, dtype=torch.bool)
                    ).sum()
                )
                tokens += len(ratio)
    result.update(
        active_clipped_tokens=active,
        tokens_in_complete_groups=tokens,
        mixed_groups=sum(len({e.reward for e in group}) > 1 for group in groups.values()),
    )
    return result


def run(config: RunConfig, output: Path) -> dict:
    from selfplay_opsd.cli import make_policy
    from selfplay_opsd.data import TaskRepository
    from selfplay_opsd.protocol import PROMPT_HASH

    policy = make_policy(config)
    reports: dict[str, Any] = {}
    sources = [
        ("initial", "repair-selfplay-2026-efficiency-v1-pilot-h200", 1, None),
        ("later", "repair-selfplay-2026-efficiency-v1-pilot-h200", 10, "checkpoints/000009"),
        (
            "mixed",
            "repair-selfplay-2026-efficiency-v2-pilot-h200",
            2,
            "acceptance/iteration-one-checkpoint",
        ),
    ]
    try:
        for label, study, iteration, checkpoint in sources:
            root = config.output_root / study / f"D-seed{config.seed}"
            paths = list((root / "trajectories").glob(f"*/{iteration:06d}-episodes.json"))
            if len(paths) != 1:
                raise ExperimentError(f"Expected one saved {label} shard")
            if checkpoint:
                state, metadata = CheckpointManager.load(root / checkpoint)
                if (
                    metadata["prompt_hash"] != PROMPT_HASH
                    or metadata["manifests_hash"] != TaskRepository(config.data_root).hash
                    or metadata["resolved_config"]["model"] != config.model.model_dump(mode="json")
                ):
                    raise ExperimentError("Saved checkpoint protocol/model/data differs")
                policy.load_state_dict(state["policy"])
                anchors = state["reference_anchors"]
            else:
                # Initial snapshot references are in the saved iteration-zero state.
                state, metadata = CheckpointManager.load(root / "checkpoints/000000")
                policy.load_state_dict(state["policy"])
                anchors = state["reference_anchors"]
            episodes = read_json(paths[0])
            proposals_path = paths[0].with_name(f"{iteration:06d}-proposals.json")
            # Initial/later snapshots need token stratification; the complete
            # mixed workload supplies bounded expensive gradient comparisons.
            if label == "mixed":
                report = assess(policy, config, episodes, read_json(proposals_path), anchors)
            else:
                turns = [EpisodeRecord.model_validate(e) for e in episodes]
                _, stats = score_role(
                    policy,
                    [t for e in turns for t in e.turns],
                    "student",
                    turns[0].policy_version,
                    config.objective.clip,
                )
                proposals = [
                    ProposalRecord.model_validate({k: v for k, v in p.items() if k != "reward"})
                    for p in read_json(proposals_path)
                ]
                _, proposer = score_role(
                    policy,
                    [p.turn for p in proposals],
                    "proposer",
                    proposals[0].policy_version,
                    config.objective.clip,
                )
                report = {"student": stats, "proposer": proposer}
            reports[label] = report | {
                "episode_sha256": file_hash(paths[0]),
                "proposal_sha256": file_hash(proposals_path),
            }
            atomic_json(output / "numerical-audit.json", reports)
        reports["semantics"] = (
            "Diagnostic counterfactual only; no optimizer steps, generation or candidate execution. Original behavior denominators remain unchanged."
        )
        atomic_json(output / "numerical-audit.json", reports)
        return reports
    finally:
        policy.close()
