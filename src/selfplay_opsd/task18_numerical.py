"""Read-only saved-token analysis and state-restored Task 18 update diagnostics."""

from __future__ import annotations

import copy
import math
from collections import defaultdict

import numpy as np
import torch

from selfplay_opsd.core import ExperimentError
from selfplay_opsd.diagnostics import _optimizer_digest
from selfplay_opsd.numerical_audit import compare_gradients
from selfplay_opsd.objectives import advantages
from selfplay_opsd.storage import restore_rng, rng_state
from selfplay_opsd.task15_audit import intervention_examples


def mismatch_statistics(values):
    signed = np.asarray(values, dtype=float)
    if not np.isfinite(signed).all():
        raise FloatingPointError("Nonfinite rollout/rescoring difference")
    absolute = np.abs(signed)
    return {
        "tokens": len(signed),
        "mean_absolute": float(absolute.mean()) if len(signed) else None,
        "quantiles_0_50_90_95_99_100": np.quantile(absolute, [0, 0.5, 0.9, 0.95, 0.99, 1]).tolist()
        if len(signed)
        else [],
        "above_005": int((absolute > 0.05).sum()),
        "above_02": int((absolute > 0.2).sum()),
        "above_02_fraction": float((absolute > 0.2).mean()) if len(signed) else None,
    }


def saved_token_report(episodes, clip=0.2):
    groups = defaultdict(list)
    for episode in episodes:
        groups[episode.task_id].append(episode)
    deltas, branch_changes = [], 0
    bins = defaultdict(list)
    for group in groups.values():
        advs = advantages(torch.tensor([e.reward for e in group])).tolist()
        for episode, advantage in zip(group, advs, strict=True):
            for turn in episode.turns:
                n = len(turn.target_ids)
                if len(turn.old_logps) != n or len(turn.frozen_unprivileged_logps) != n:
                    raise ExperimentError("Incomplete saved token probability mapping")
                for i, (old, score) in enumerate(
                    zip(turn.old_logps, turn.frozen_unprivileged_logps, strict=True)
                ):
                    delta = score - old
                    if not math.isfinite(delta):
                        raise FloatingPointError("Nonfinite probability difference")
                    deltas.append(delta)
                    bins["first" if i == 0 else "later"].append(delta)
                    # Ratio-one counterfactual has an unclipped gradient. Actual
                    # positive/negative advantages clip on opposite ratio sides.
                    branch_changes += int(
                        (advantage > 0 and delta > math.log1p(clip))
                        or (advantage < 0 and delta < math.log1p(-clip))
                    )
    return mismatch_statistics(deltas) | {
        "clipped_surrogate_branch_changes": branch_changes,
        "by_position": {key: mismatch_statistics(value) for key, value in bins.items()},
        "groups": len(groups),
        "mixed_groups": sum(len({e.reward for e in group}) > 1 for group in groups.values()),
        "uniform_success_groups": sum(
            all(e.reward == 1 for e in group) for group in groups.values()
        ),
        "uniform_failure_groups": sum(
            all(e.reward == 0 for e in group) for group in groups.values()
        ),
        "interpretation": "Frozen-score versus actual behavior; targeted records, not a population estimate",
    }


def select_groups(episodes, maximum=4):
    """Worst mismatch first, then mixed/uniform controls; retain whole groups."""
    if maximum < 1:
        raise ValueError("Positive group limit required")
    groups = defaultdict(list)
    for episode in episodes:
        groups[episode.task_id].append(episode)
    if not groups:
        raise ValueError("No diagnostic groups")
    errors = {
        key: max(
            (
                abs(a - b)
                for e in group
                for t in e.turns
                for a, b in zip(t.old_logps, t.frozen_unprivileged_logps, strict=True)
            ),
            default=0,
        )
        for key, group in groups.items()
    }
    ordered = sorted(groups, key=lambda key: (-errors[key], key))
    selected = ordered[:1]
    for mixed in (True, False):
        candidates = [
            key
            for key in ordered
            if (len({e.reward for e in groups[key]}) > 1) == mixed and key not in selected
        ]
        if candidates and len(selected) < maximum:
            selected.append(candidates[0])
    selected.extend(key for key in ordered if key not in selected)
    return [episode for key in selected[:maximum] for episode in groups[key]]


def audit_updates(policy, cfg, episodes, relative_screen=0.01, cosine_screen=0.99):
    """Three restored branches isolate denominator influence under the current SD gap."""
    if not episodes or any(e.policy_version != policy.version("student") for e in episodes):
        raise ExperimentError("Audit requires the complete generating snapshot")
    initial, random_state = copy.deepcopy(policy.state_dict()), rng_state()
    initial_mode = policy.model.training
    results = {}
    try:
        with policy.active("student"):
            for label, denominator in (
                ("behavior", "behavior"),
                ("repeat", "behavior"),
                ("rescored_diagnostic", "scorer"),
            ):
                policy.load_state_dict(copy.deepcopy(initial))
                restore_rng(random_state)
                policy.model.train(initial_mode)
                examples = intervention_examples(
                    episodes, cfg, denominator=denominator, gap="scorer"
                )
                gradient, _ = policy.gradients(examples)
                _, metrics = policy.gradients(examples, step=True)
                after = policy.state_dict()
                names = initial["parameter_names"]["student"]
                update = [
                    after["adapters"][name].double() - initial["adapters"][name].double()
                    for name in names
                ]
                results[label] = {"gradient": gradient, "update": update, "metrics": metrics}
        comparisons: dict[str, dict] = {}
        for label in ("repeat", "rescored_diagnostic"):
            comparisons[label] = {}
            for kind in ("gradient", "update"):
                left, right = results["behavior"][kind], results[label][kind]
                comparison = compare_gradients(left, right, policy, cfg.objective.near_zero)
                comparison["coordinate_sign_changes"] = sum(
                    int(((a * b) < 0).sum()) for a, b in zip(left, right, strict=True)
                )
                comparisons[label][kind] = comparison
        repeat, causal = (
            comparisons["repeat"]["update"],
            comparisons["rescored_diagnostic"]["update"],
        )
        material = (
            causal["relative_l2"] is not None
            and causal["relative_l2"] > max(relative_screen, repeat["relative_l2"] or 0)
        ) or (
            causal["cosine"] is not None
            and causal["cosine"] < min(cosine_screen, repeat["cosine"] or cosine_screen)
        )
        return {
            "comparisons": comparisons,
            "metrics": {key: value["metrics"] for key, value in results.items()},
            "requires_investigation": material,
            "training_ready": False,
            "note": "Screen only: alignment, synchronization and numerical-mode checks also required. Original behavior denominators are unchanged.",
        }
    finally:
        policy.load_state_dict(initial)
        restore_rng(random_state)
        policy.model.train(initial_mode)
        if _optimizer_digest(policy.state_dict()) != _optimizer_digest(initial):
            raise ExperimentError("Task 18 audit failed full state restoration")


def cached_scoring_probe(policy, episodes):
    """Teacher force the exact saved IDs incrementally; no sampling or updates."""
    import torch.nn.functional as F

    if not episodes or any(e.policy_version != policy.version("student") for e in episodes):
        raise ExperimentError("Cached scoring requires the exact generating snapshot")
    requests = [
        (turn.prompt_ids, turn.target_ids) for e in episodes for turn in e.turns if turn.target_ids
    ]
    old = [value for e in episodes for turn in e.turns for value in turn.old_logps]
    frozen = [
        value for e in episodes for turn in e.turns for value in turn.frozen_unprivileged_logps
    ]
    version = policy.version("student")
    was_training = policy.model.training
    cached = []
    try:
        with policy.active("student"):
            policy.model.eval()
            full = [
                float(value)
                for scores in policy.score_many(requests)
                for value in scores.detach().float().cpu().tolist()
            ]
            with torch.no_grad():
                for prompt, targets in requests:
                    output = policy.model(
                        input_ids=torch.tensor([prompt], device=policy.device),
                        attention_mask=torch.ones(
                            (1, len(prompt)), device=policy.device, dtype=torch.long
                        ),
                        position_ids=torch.arange(len(prompt), device=policy.device).unsqueeze(0),
                        use_cache=True,
                        logits_to_keep=1,
                    )
                    for index, target in enumerate(targets):
                        cached.append(
                            float(
                                -F.cross_entropy(
                                    output.logits[:, -1].float(),
                                    torch.tensor([target], device=policy.device),
                                )
                            )
                        )
                        if index + 1 < len(targets):
                            output = policy.model(
                                input_ids=torch.tensor([[target]], device=policy.device),
                                attention_mask=torch.ones(
                                    (1, len(prompt) + index + 1),
                                    device=policy.device,
                                    dtype=torch.long,
                                ),
                                position_ids=torch.tensor(
                                    [[len(prompt) + index]], device=policy.device
                                ),
                                past_key_values=output.past_key_values,
                                use_cache=True,
                                logits_to_keep=1,
                            )
                    del output
            repeat = [
                float(value)
                for scores in policy.score_many(requests)
                for value in scores.detach().float().cpu().tolist()
            ]
    finally:
        policy.model.train(was_training)
    if policy.version("student") != version:
        raise ExperimentError("Cached scoring mutated model weights")
    if not len(full) == len(cached) == len(repeat) == len(old) == len(frozen):
        raise ExperimentError("Cached scoring lost target alignment")
    outliers = [i for i, (a, b) in enumerate(zip(full, old, strict=True)) if abs(a - b) > 0.2]
    return {
        "tokens": len(old),
        "full_vs_behavior": mismatch_statistics([a - b for a, b in zip(full, old, strict=True)]),
        "cached_vs_behavior": mismatch_statistics(
            [a - b for a, b in zip(cached, old, strict=True)]
        ),
        "cached_vs_full": mismatch_statistics([a - b for a, b in zip(cached, full, strict=True)]),
        "full_vs_saved_frozen": mismatch_statistics(
            [a - b for a, b in zip(full, frozen, strict=True)]
        ),
        "full_repeat": mismatch_statistics([a - b for a, b in zip(repeat, full, strict=True)]),
        "cached_vs_behavior_at_full_outliers": mismatch_statistics(
            [cached[i] - old[i] for i in outliers]
        ),
        "mode": {
            "attention": getattr(policy.model.config, "_attn_implementation", None),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_tf32": torch.backends.cuda.matmul.allow_tf32,
            "model_parameter_dtypes": sorted({str(p.dtype) for p in policy.model.parameters()}),
        },
        "note": "Exact saved target IDs, including EOS. No constrained decoding, sampling, denominator substitution or optimizer update.",
    }


def frozen_rollout_replay(policy, episodes, maximum=4):
    """Recreate bounded unprivileged turns in a fresh production rollout engine."""
    if policy.config.model.backend != "verl" or not 1 <= maximum <= 4:
        raise ValueError("Frozen rollout replay requires the production sampler and 1..4 turns")
    candidates = []
    for episode in episodes:
        if episode.policy_version != policy.version("student"):
            raise ExperimentError("Frozen rollout replay snapshot differs")
        for index, turn in enumerate(episode.turns):
            if turn.target_ids:
                error = max(
                    abs(a - b)
                    for a, b in zip(turn.old_logps, turn.frozen_unprivileged_logps, strict=True)
                )
                candidates.append((error, episode.episode_id, index, episode.seed + index, turn))
    selected = sorted(candidates, key=lambda row: (-row[0], row[1], row[2]))[:maximum]
    records = []
    with policy.active("student"):
        for _, episode_id, index, seed, turn in selected:
            output = policy.generate(turn.prompt_ids, len(turn.target_ids), seed, greedy=False)
            matched = 0
            for a, b in zip(turn.target_ids, output.ids, strict=False):
                if a != b:
                    break
                matched += 1
            records.append(
                {
                    "episode_id": episode_id,
                    "turn": index,
                    "seed": seed,
                    "original_tokens": len(turn.target_ids),
                    "new_tokens": len(output.ids),
                    "identical_target_ids": turn.target_ids == output.ids,
                    "identical_prefix_tokens": matched,
                    "probability_difference_on_identical_prefix": mismatch_statistics(
                        [
                            a - b
                            for a, b in zip(
                                output.logps[:matched], turn.old_logps[:matched], strict=True
                            )
                        ]
                    ),
                }
            )
    return {
        "turns": records,
        "new_generated_tokens": sum(r["new_tokens"] for r in records),
        "selection": "up to four largest saved mismatch turns; original sampling seeds and exact unprivileged prefixes",
        "note": "No generated action or Python program is executed. Probability comparisons stop at the first differing target ID.",
    }
