"""Saved-token interventions separating the PPO denominator from the SD gap.

The rescored PPO branch is a diagnostic intervention, never a training estimator.
Every branch starts from the same full policy/optimizer/RNG state.
"""

from __future__ import annotations

import copy
import dataclasses
from collections import defaultdict

from selfplay_opsd.core import ExperimentError
from selfplay_opsd.numerical_audit import compare_gradients, error_statistics
from selfplay_opsd.storage import restore_rng, rng_state
from selfplay_opsd.trainer import student_examples


def intervention_examples(
    episodes, cfg, *, denominator="behavior", gap="scorer", no_privilege=False
):
    if denominator not in {"behavior", "scorer"} or gap not in {"behavior", "scorer"}:
        raise ValueError("Unknown numerical intervention")
    examples = []
    groups = defaultdict(list)
    for episode in episodes:
        turns = []
        for turn in episode.turns:
            if len(turn.frozen_unprivileged_logps) != len(turn.target_ids):
                raise ExperimentError("Numerical replay needs frozen unprivileged scores")
            turns.append(
                turn.model_copy(update={"teacher_logps": turn.frozen_unprivileged_logps})
                if no_privilege
                else turn
            )
        groups[episode.task_id].append(episode.model_copy(update={"turns": turns}))
    config = cfg.model_copy(
        update={
            "sd_gap_definition": "same_scorer_v2" if gap == "scorer" else "teacher_behavior_v1"
        },
        deep=True,
    )
    for group in groups.values():
        current, _ = student_examples(group, config, tasks=len(groups))
        turns = [t for e in group for t in e.turns if t.target_ids]
        if denominator == "scorer":
            current = [
                dataclasses.replace(example, old=turn.frozen_unprivileged_logps)
                for example, turn in zip(current, turns, strict=True)
            ]
        examples.extend(current)
    return examples


def audit_updates(policy, cfg, episodes):
    if not episodes or any(e.policy_version != policy.version("student") for e in episodes):
        raise ExperimentError("Saved episodes do not match the audit snapshot")
    initial, initial_rng = copy.deepcopy(policy.state_dict()), rng_state()
    branches = {}
    # A second real same-prefix scoring operation measures no-privilege repeat
    # error. Do not manufacture a zero gap by subtracting one array from itself.
    no_privilege_episodes = []
    repeat_deltas: list[float] = []
    with policy.active("student"):
        for episode in episodes:
            scores = policy.score_many([(t.prompt_ids, t.target_ids) for t in episode.turns])
            turns = []
            for turn, score in zip(episode.turns, scores, strict=True):
                values = score.detach().float().tolist()
                repeat_deltas.extend(
                    a - b for a, b in zip(values, turn.frozen_unprivileged_logps, strict=True)
                )
                turns.append(turn.model_copy(update={"teacher_logps": values}))
            no_privilege_episodes.append(episode.model_copy(update={"turns": turns}))
    cases = [
        ("behavior:behavior", "behavior", "behavior", False),
        ("scorer:behavior", "scorer", "behavior", False),
        ("behavior:scorer", "behavior", "scorer", False),
        ("scorer:scorer", "scorer", "scorer", False),
        ("repeat", "behavior", "scorer", False),
        ("no_privilege:scorer", "behavior", "scorer", True),
        ("no_privilege:behavior", "behavior", "behavior", True),
    ]
    try:
        with policy.active("student"):
            for label, denominator, gap, no_privilege in cases:
                policy.load_state_dict(copy.deepcopy(initial))
                restore_rng(initial_rng)
                examples = intervention_examples(
                    no_privilege_episodes if no_privilege else episodes,
                    cfg,
                    denominator=denominator,
                    gap=gap,
                )
                gradient, metrics = policy.gradients(examples)
                # Measure SD alone for the no-privilege spurious-update control.
                if no_privilege:
                    examples = [dataclasses.replace(e, kind="sd") for e in examples]
                    gradient, metrics = policy.gradients(examples)
                policy.gradients(examples, step=True)
                after = policy.state_dict()
                displacement = [
                    tensor.double() - initial["adapters"][name].double()
                    for name, tensor in after["adapters"].items()
                    if ".student." in name
                ]
                branches[label] = {"gradient": gradient, "update": displacement, "metrics": metrics}
        comparisons = {}
        for left, right in [
            ("behavior:behavior", "scorer:behavior"),
            ("behavior:scorer", "scorer:scorer"),
            ("behavior:behavior", "behavior:scorer"),
            ("scorer:behavior", "scorer:scorer"),
            ("behavior:scorer", "repeat"),
            ("no_privilege:scorer", "no_privilege:behavior"),
        ]:
            comparisons[left + " -> " + right] = {
                kind: compare_gradients(
                    branches[left][kind], branches[right][kind], policy, cfg.objective.near_zero
                )
                for kind in ("gradient", "update")
            }
        return {
            "available": True,
            "no_privilege_scorer_repeat": error_statistics(repeat_deltas, cfg.objective.clip),
            "metrics": {name: value["metrics"] for name, value in branches.items()},
            "comparisons": comparisons,
            "note": "Denominator substitutions are diagnostic only. Main PPO retains actual behavior probabilities.",
        }
    finally:
        policy.load_state_dict(initial)
        restore_rng(initial_rng)
        from selfplay_opsd.diagnostics import _optimizer_digest

        restored = policy.state_dict()
        if _optimizer_digest(restored) != _optimizer_digest(initial):
            raise ExperimentError("Numerical audit did not restore parameters and optimizers")


def teacher_action_discrimination(policy, cfg, episodes, oracles):
    """Does privilege favor a verified inverse edit over premature finish?

    This is initial-state action discrimination, not generated-teacher repair
    accuracy. Each training task contributes once, independent of its outcome.
    """
    from selfplay_opsd.core import canonical
    from selfplay_opsd.protocol import teacher_messages

    rows = []
    seen = set()
    with policy.active("student"):
        for episode in episodes:
            if episode.task_id in seen or not episode.turns:
                continue
            seen.add(episode.task_id)
            oracle = oracles[episode.task_id]
            first = episode.turns[0]
            targets = [
                policy.tokenizer.encode(canonical(action), add_special_tokens=False)
                + [policy.tokenizer.eos_token_id]
                for action in (
                    {"action": "apply_patch", "patch": oracle.inverse_patch},
                    {"action": "finish"},
                )
            ]
            if any(len(ids) > cfg.episode.max_turn_generated_tokens for ids in targets):
                continue
            teacher = policy.encode_history(teacher_messages(first.history, oracle))
            if any(
                len(teacher) + len(ids) > cfg.episode.max_teacher_sequence_tokens for ids in targets
            ):
                continue
            student_scores = policy.score_many([(first.prompt_ids, ids) for ids in targets])
            teacher_scores = policy.score_many([(teacher, ids) for ids in targets])
            unprivileged = float(student_scores[0].mean() - student_scores[1].mean())
            privileged = float(teacher_scores[0].mean() - teacher_scores[1].mean())
            rows.append(
                {
                    "task_id": episode.task_id,
                    "unprivileged_repair_vs_finish": unprivileged,
                    "privileged_repair_vs_finish": privileged,
                    "privilege_difference": privileged - unprivileged,
                }
            )
    return {
        "tasks": len(rows),
        "skipped_budget": len(seen) - len(rows),
        "rows": rows,
        "mean_discrimination_difference": sum(row["privilege_difference"] for row in rows)
        / len(rows)
        if rows
        else None,
        "improved_tasks": sum(row["privilege_difference"] > 0 for row in rows),
        "interpretation": "length-normalized initial-action discrimination; not teacher rollout success",
    }
