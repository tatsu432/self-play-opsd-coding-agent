"""Saved-workload proposer ratios and state-restored Adam update diagnostics."""

from __future__ import annotations

import copy
from pathlib import Path

import torch

from selfplay_opsd.core import ProposalRecord
from selfplay_opsd.numerical_audit import compare_gradients, error_statistics, score_role
from selfplay_opsd.storage import CheckpointManager, atomic_json, file_hash, read_json, restore_rng


def compare_updates(policy, config, proposals, rewards, checkpoint_state):
    from selfplay_opsd.trainer import proposer_examples

    policy.load_state_dict(copy.deepcopy(checkpoint_state["policy"]))
    turns, scores = score_role(
        policy,
        [p.turn for p in proposals],
        "proposer",
        proposals[0].policy_version,
        config.objective.clip,
    )
    changed = [p.model_copy(update={"turn": t}) for p, t in zip(proposals, turns, strict=True)]
    outputs, gradients, updates = {}, {}, {}
    for name, rows in (("behavior", proposals), ("trainer_rescored", changed)):
        policy.load_state_dict(copy.deepcopy(checkpoint_state["policy"]))
        restore_rng(checkpoint_state["rng"])
        with policy.active("proposer"):
            examples = proposer_examples(rows, rewards, config)
            training_deltas, active_clip = [], 0
            was_training = policy.model.training
            policy.model.train()
            try:
                for e in examples:
                    if e.advantage == 0 or not e.targets:
                        continue
                    current = (
                        policy.score(e.prompt, e.targets, gradients=True).detach().float().cpu()
                    )
                    delta = current - torch.tensor(e.old)
                    training_deltas.extend(delta.tolist())
                    ratios = delta.exp()
                    active_clip += int(
                        (
                            (ratios > 1 + config.objective.clip)
                            if e.advantage > 0
                            else (ratios < 1 - config.objective.clip)
                        ).sum()
                    )
            finally:
                policy.model.train(was_training)
            first, metrics = policy.gradients(examples)
            second, _ = policy.gradients(examples)
            outputs[name] = {
                "gradient_repeat": compare_gradients(
                    first, second, policy, config.objective.near_zero
                ),
                "forward_training_ratio_log_delta": error_statistics(
                    training_deltas, config.objective.clip
                ),
                "active_clipped_tokens": active_clip,
                "nonzero_advantage_examples": sum(e.advantage != 0 for e in examples),
                "metrics": metrics,
            }
            gradients[name] = first
            _, update_metrics = policy.gradients(examples, step=True)
            delta_parameters = []
            for parameter_name, p in policy.named_parameters("proposer"):
                value = p.to_local() if hasattr(p, "to_local") else p
                delta_parameters.append(
                    value.detach().float().cpu()
                    - checkpoint_state["policy"]["adapters"][parameter_name].float()
                )
            updates[name] = delta_parameters
            outputs[name]["optimizer_update_metrics"] = update_metrics
    outputs["gradient_comparison"] = compare_gradients(
        gradients["behavior"], gradients["trainer_rescored"], policy, config.objective.near_zero
    )
    outputs["adam_update_comparison"] = compare_gradients(
        updates["behavior"], updates["trainer_rescored"], policy, config.objective.near_zero
    )
    outputs["eval_scoring_discrepancy"] = scores
    policy.load_state_dict(copy.deepcopy(checkpoint_state["policy"]))
    restore_rng(checkpoint_state["rng"])
    return outputs


def run(config, output: Path):
    from selfplay_opsd.cli import make_policy
    from selfplay_opsd.data import TaskRepository
    from selfplay_opsd.objectives import difficulty
    from selfplay_opsd.protocol import PROMPT_HASH

    root = (
        config.output_root
        / "repair-selfplay-2026-efficiency-v2-pilot-h200"
        / f"D-seed{config.seed}"
    )
    checkpoint = root / "acceptance/iteration-one-checkpoint"
    state, metadata = CheckpointManager.load(checkpoint)
    if (
        metadata["prompt_hash"] != PROMPT_HASH
        or metadata["manifests_hash"] != TaskRepository(config.data_root).hash
    ):
        raise ValueError("Checkpoint protocol/data mismatch")
    shard = next((root / "trajectories").glob("*/000002-proposals.json"))
    original = read_json(shard)
    proposals = [
        ProposalRecord.model_validate({k: v for k, v in row.items() if k != "reward"})
        for row in original
    ]
    episodes = read_json(shard.with_name("000002-episodes.json"))
    difficulty_rewards = [
        difficulty(e["reward"] for e in episodes if e["task_id"] == p.task_id) if p.valid else -1.0
        for p in proposals
    ]
    policy = make_policy(config)
    try:
        report = {
            "source_optimizer_steps": [
                float(value["step"])
                for value in state["policy"]["optimizers"]["proposer"]["state"].values()
                if "step" in value
            ],
            "checkpoint_marker_sha256": file_hash(checkpoint / "complete.json"),
            "proposal_sha256": file_hash(shard),
            "no_persistent_optimizer_update": True,
        }
        report["actual_saved_D_rewards"] = compare_updates(
            policy, config, proposals, [row["reward"] for row in original], state
        )
        atomic_json(output / "proposer-audit.json", report)
        report["difficulty_rewards"] = compare_updates(
            policy, config, proposals, difficulty_rewards, state
        )
        report["restored_proposer_version"] = policy.version("proposer")
        if report["restored_proposer_version"] != metadata["proposer_version"]:
            raise ValueError("Proposer state restoration differs")
        atomic_json(output / "proposer-audit.json", report)
        return report
    finally:
        policy.load_state_dict(state["policy"])
        restore_rng(state["rng"])
        policy.close()
