"""Training-only calibration and repeat-controlled numerical qualification."""

from __future__ import annotations

import copy
import math
import statistics
from pathlib import Path

import torch

from selfplay_opsd.core import keyed_seed
from selfplay_opsd.objectives import alignment, difficulty
from selfplay_opsd.storage import atomic_json, file_hash, restore_rng, rng_state
from selfplay_opsd.task19 import GROUP, Task19Control, centered_rewards, clipped_sum
from selfplay_opsd.task19_policy import STUDENT_SAMPLING
from selfplay_opsd.task19_train import Study, joint_step, norm


def qualify(study: Study) -> Task19Control:
    output = study.output
    cfg = study.cfg
    references = study.references()
    study.reference_gradient(references)

    def proposals():
        return study.collect(32)

    groups, tasks, _ = proposals()
    attempts = [row for group in groups for row in group]
    screen = {
        "thinking": cfg.proposer_thinking,
        "attempts": len(attempts),
        "valid": sum(row.task is not None for row in attempts),
        "unique": len(tasks),
    }
    proposed = [screen]
    if not cfg.proposer_thinking and (
        screen["valid"] < 0.25 * screen["attempts"] or screen["unique"] < 4
    ):
        thinking = cfg.model_copy(update={"proposer_thinking": True, "proposal_tokens": 2048})
        study.cfg = thinking
        study.proposal_groups, study.seen = 0, set()
        other_groups, other_tasks, _ = proposals()
        other_rows = [row for group in other_groups for row in group]
        valid = sum(row.task is not None for row in other_rows)
        proposed.append(
            {
                "thinking": True,
                "attempts": len(other_rows),
                "valid": valid,
                "unique": len(other_tasks),
            }
        )
        if (len(other_tasks), valid / len(other_rows)) > (
            len(tasks),
            screen["valid"] / screen["attempts"],
        ):
            cfg, groups, tasks = thinking, other_groups, other_tasks
        study.cfg = cfg
    extra_groups, extra_tasks, _ = proposals()
    groups.extend(extra_groups)
    tasks.update(extra_tasks)
    if not tasks:
        raise RuntimeError("Bounded proposer calibration yielded no verified tasks")
    rolls, _ = study.rollouts(tasks, distill=True)
    episodes = [row for rows in rolls.values() for row in rows]
    capped = sum(
        row.termination in {"turn_tokens", "episode_tokens", "context_limit", "turn_limit"}
        for row in episodes
    )
    before_caps = {"episodes": len(episodes), "capped": capped}
    if capped > 0.1 * len(episodes):
        episode = cfg.episode.model_copy(
            update={
                "max_turn_generated_tokens": 4096,
                "max_generated_tokens": 12288,
                "max_student_sequence_tokens": 20480,
                "max_teacher_sequence_tokens": 24576,
            }
        )
        cfg = cfg.model_copy(update={"episode": episode})
        study.cfg = cfg
        # A different limit changes the prompt. Collect fresh samples at the final setting.
        study.proposal_groups, study.seen = 0, set()
        groups, tasks, _ = proposals()
        extra_groups, extra_tasks, _ = proposals()
        groups.extend(extra_groups)
        tasks.update(extra_tasks)
        rolls, _ = study.rollouts(tasks, distill=True)
    atomic_json(
        output / "frozen-training-batch.json",
        {
            "snapshot": study.snapshot,
            "proposals": [[row.model_dump(mode="json") for row in group] for group in groups],
            "repairs": {
                key: [row.model_dump(mode="json") for row in values]
                for key, values in rolls.items()
            },
        },
    )

    diagnostic = numerical(study, rolls)
    atomic_json(output / "numerical.json", diagnostic)
    if not diagnostic["passed"]:
        if not cfg.cached_rollout:
            raise RuntimeError("Task 19 uncached execution failed numerical qualification")
        atomic_json(
            output / "uncached-control.json",
            cfg.model_copy(update={"cached_rollout": False}).model_dump(mode="json"),
        )
        raise RuntimeError(
            "Cached execution failed qualification; use the emitted uncached control"
        )

    ratios, rl_norms = [], []
    for rows in rolls.values():
        rl, sd, _ = study.student_gradient(rows, gated=False, use_sd=True)
        rl_norm, sd_norm = norm(rl), norm(sd)
        rl_norms.append(rl_norm)
        if min(rl_norm, sd_norm) > 1e-12:
            ratios.append(0.3 * rl_norm / sd_norm)
    proposer_norms = []
    for group in groups:
        for row in group:
            if row.task is not None:
                row.reward = difficulty(float(item.success) for item in rolls[row.task])
        gradient, _ = study.proposer_gradient([group])
        proposer_norms.append(norm(gradient))
    proposer_norms = [value for value in proposer_norms if value > 1e-12]
    rl_nonzero = [value for value in rl_norms if value > 1e-12]
    if not ratios or not proposer_norms or not rl_nonzero:
        raise RuntimeError("Training-only batch did not identify both loss scales")
    sd_scale = statistics.median(ratios)
    proposer_scale = 0.5 * statistics.median(rl_nonzero) / statistics.median(proposer_norms)
    episode_metrics = [row.metrics() for group in rolls.values() for row in group]
    report = {
        "identity": study.identity,
        "proposer_screen": proposed,
        "student_cap_screen": before_caps,
        "final_rollouts": episode_metrics,
        "sd_scale": sd_scale,
        "proposer_scale": proposer_scale,
        "sd_norm_targets": ratios,
        "rl_norms": rl_norms,
        "proposer_norms": proposer_norms,
    }
    atomic_json(output / "calibration.json", report)
    result = cfg.model_copy(
        update={
            "sd_scale": sd_scale,
            "proposer_scale": proposer_scale,
            "calibration_hash": file_hash(output / "calibration.json"),
            "qualification_hash": file_hash(output / "numerical.json"),
        }
    )
    atomic_json(output / "calibrated-control.json", result.model_dump(mode="json"))
    return result


def numerical(study: Study, rolls: dict) -> dict:
    policy = study.policy
    cfg = study.cfg
    gradient = study.zeros()
    counterfactual = study.zeros()
    mismatch = []
    sampled_replay_max = 0.0
    clip_changes = 0
    compared = 0
    sampled_support_exits = 0
    selected_groups: dict[str, str] = {}
    for key, episodes in rolls.items():
        rewards = [float(row.success) for row in episodes]
        category = "mixed" if min(rewards) != max(rewards) else "uniform"
        if category in selected_groups:
            continue
        selected_groups[category] = key
        for episode, advantage in zip(episodes, centered_rewards(rewards).tolist(), strict=True):
            # Include the first and last turn. Replaying the actual sampled path
            # includes the prompt, thinking markers, tool/EOS and the late prefix.
            positions = sorted({0, len(episode.turns) - 1}) if episode.turns else []
            for position in positions:
                turn = episode.turns[position]
                if not turn.ids:
                    continue
                if cfg.cached_rollout:
                    replayed = policy.replay(turn.prompt, turn.ids, STUDENT_SAMPLING, cached=True)
                    replay_targets = replayed[-8:]
                else:
                    prefix = turn.prompt + turn.ids[:-8]
                    replay_targets = policy.replay(
                        prefix, turn.ids[-8:], STUDENT_SAMPLING, cached=False
                    )
                prefix = turn.prompt + turn.ids[:-8]
                ids = turn.ids[-8:]
                current = policy.score(
                    prefix,
                    ids,
                    STUDENT_SAMPLING,
                    gradients=True,
                    exact_prefix=not cfg.cached_rollout,
                )[0]
                cached = replay_targets.to(device=current.device).clone()
                old = current.new_tensor(turn.logps[-8:])
                if torch.isfinite(cached).all():
                    sampled_replay_max = max(sampled_replay_max, float((cached - old).abs().max()))
                else:
                    sampled_support_exits += int((~cached.isfinite()).sum())
                prod_ratio, cache_ratio = (current.detach() - old).exp(), (cached - old).exp()
                finite = torch.isfinite(current) & torch.isfinite(cached)
                mismatch.extend((current.detach()[finite] - cached[finite]).abs().tolist())
                if advantage:
                    compared += len(ids)
                    if advantage > 0:
                        clip_changes += int(
                            ((prod_ratio > 1 + cfg.clip) != (cache_ratio > 1 + cfg.clip)).sum()
                        )
                    else:
                        clip_changes += int(
                            ((prod_ratio < 1 - cfg.clip) != (cache_ratio < 1 - cfg.clip)).sum()
                        )
                    scale = 1 / (GROUP * cfg.episode.max_generated_tokens)
                    loss = clipped_sum(current, old, advantage, cfg.clip) * scale
                    study.accumulate(gradient, loss, retain_graph=True)
                    # Change the forward ratio for a sensitivity check while
                    # holding the saved behavior denominator and local Jacobian.
                    stable = current.masked_fill(~finite, 0)
                    measured = cached + (stable - stable.detach())
                    compare = clipped_sum(measured, old, advantage, cfg.clip) * scale
                    study.accumulate(counterfactual, compare)
        if len(selected_groups) == 2:
            break
    if norm(gradient) <= 1e-12:
        return {
            "passed": False,
            "reason": "No informative replay gradients",
            "groups": selected_groups,
        }
    state, rng = copy.deepcopy(policy.state()), rng_state()
    anchor = [p.detach().clone() for p in policy.parameters]

    def displacement(values):
        policy.restore(copy.deepcopy(state))
        restore_rng(rng)
        joint_step(policy, values, study.zeros(), 0.0, cfg.grad_clip)
        return [p.detach() - old for p, old in zip(policy.parameters, anchor, strict=True)]

    try:
        repeats = [displacement(gradient) for _ in range(3)]
        candidate = displacement(counterfactual)
    finally:
        policy.restore(state)
        restore_rng(rng)

    def difference(a, b):
        return {
            "relative": norm([x - y for x, y in zip(a, b, strict=True)]) / max(norm(a), 1e-12),
            "cosine": alignment(a, b).cosine,
        }

    repeat = max(difference(repeats[0], row)["relative"] for row in repeats[1:])
    update, grad = difference(repeats[0], candidate), difference(gradient, counterfactual)
    clip_fraction = clip_changes / max(compared, 1)
    passed = (
        set(selected_groups) == {"mixed", "uniform"}
        and sampled_support_exits == 0
        and sampled_replay_max <= 0.02
        and repeat <= 0.01
        and clip_fraction <= 0.01
        and all(
            row["relative"] <= 0.05 and (row["cosine"] or 0.0) >= 0.99 for row in (update, grad)
        )
        and all(map(math.isfinite, mismatch))
    )
    return {
        "passed": passed,
        "cached_rollout": cfg.cached_rollout,
        "groups": selected_groups,
        "snapshot": study.snapshot,
        "sample_seed": keyed_seed(cfg.seed, "qualification"),
        "comparisons": compared,
        "sampled_replay_max_abs": sampled_replay_max,
        "sampled_support_exits": sampled_support_exits,
        "mean_logprob_difference": statistics.mean(mismatch) if mismatch else None,
        "clip_change_fraction": clip_fraction,
        "repeat_relative": repeat,
        "gradient_difference": grad,
        "update_difference": update,
    }


def verify_evidence(
    cfg: Task19Control, root: Path, execution: dict, dataset: str, initial: str
) -> None:
    from selfplay_opsd.core import read_json

    if not cfg.ready:
        raise ValueError("Task 19 has no executed qualification and calibration")
    if (
        file_hash(root / "numerical.json") != cfg.qualification_hash
        or file_hash(root / "calibration.json") != cfg.calibration_hash
    ):
        raise ValueError("Task 19 evidence checksum differs")
    numerical_report = read_json(root / "numerical.json")
    calibration = read_json(root / "calibration.json")
    identity = calibration["identity"]
    if (
        not numerical_report["passed"]
        or numerical_report["cached_rollout"] != cfg.cached_rollout
        or calibration["sd_scale"] != cfg.sd_scale
        or calibration["proposer_scale"] != cfg.proposer_scale
        or identity["execution"] != execution
        or identity["data"] != dataset
        or identity["initial"] != initial
    ):
        raise ValueError("Task 19 calibration is not bound to this execution and dataset")
    qualified = Task19Control.read(root / "calibrated-control.json")
    if cfg.science != qualified.science:
        raise ValueError("Task 19 scientific settings differ from the qualified control")
