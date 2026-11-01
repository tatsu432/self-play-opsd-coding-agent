"""Compare four proposer rewards on identical training-only rollouts."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import os
from pathlib import Path

import torch
from task13_counterfactual import adapter_delta, delta_comparison, state_equal

from selfplay_opsd.cli import make_policy
from selfplay_opsd.config import load_config
from selfplay_opsd.core import keyed_seed
from selfplay_opsd.data import TaskRepository
from selfplay_opsd.environment import score_teacher
from selfplay_opsd.objectives import advantages, alignment, difficulty
from selfplay_opsd.pilot import training_reference_proxies
from selfplay_opsd.rollout import episode_wave
from selfplay_opsd.sandbox import SandboxBackend
from selfplay_opsd.storage import atomic_json, restore_rng, rng_state
from selfplay_opsd.trainer import (
    SelfPlayTrainer,
    proposer_examples,
    reference_examples,
    student_examples,
)


def examples_for(proposals, rewards, cfg, centered):
    examples = proposer_examples(proposals, rewards, cfg)
    if centered:
        size = cfg.proposals_per_parent
        for start in range(0, len(proposals), size):
            group = torch.tensor(rewards[start : start + size], dtype=torch.float32)
            values = (group - group.mean()).tolist()
            for offset, value in enumerate(values):
                examples[start + offset] = dataclasses.replace(
                    examples[start + offset], advantage=value
                )
    return examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--representation", choices=("unified_diff", "structured_ast"), required=True
    )
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID") or not os.getenv("SELFPLAY_WORKER_TOKEN"):
        raise RuntimeError("Counterfactual requires allocated isolated worker")
    if args.output.exists():
        raise ValueError("Output already exists")
    cfg = load_config(args.settings, method="C", hardware_profile="single_gpu_80gb")
    cfg.proposer_representation = args.representation
    cfg.study_id = "repair-diagnosis-2026-task14-counterfactual-v1"
    cfg.output_root = args.output
    cfg.logging.mode = "disabled"
    repo = TaskRepository(cfg.data_root)
    sandbox = SandboxBackend(cfg.sandbox)
    sandbox.preflight()
    policy = make_policy(cfg)
    try:
        trainer = SelfPlayTrainer(cfg, policy, repo, sandbox)
        original = copy.deepcopy(policy.state_dict())
        original_rng = rng_state()
        anchors = training_reference_proxies(trainer)
        proposals, tasks = [], {}
        for iteration in range(1, 5):
            if len(tasks) >= 12:
                break
            batch, batch_tasks = trainer.proposals(iteration)
            proposals.extend(batch)
            tasks.update(batch_tasks)
            for row in batch:
                if row.valid:
                    trainer.history.setdefault(row.parent_id, set()).add(row.dedup_key)
            atomic_json(
                args.output / "proposals.json", [p.model_dump(mode="json") for p in proposals]
            )
        groups = {}
        with policy.active("student"):
            for task_id, (task, oracle) in tasks.items():
                jobs = [(task, oracle, keyed_seed(2026, "task14-cf", task_id, i)) for i in range(4)]
                episodes = episode_wave(policy, sandbox, jobs, cfg.episode, concurrency=1)
                groups[task_id] = [
                    score_teacher(policy, episode, oracle, cfg.episode)[0] for episode in episodes
                ]
                atomic_json(
                    args.output / "episodes.json",
                    {
                        key: [episode.model_dump(mode="json") for episode in value]
                        for key, value in groups.items()
                    },
                )
            reference, _ = policy.gradients(reference_examples(anchors))
            first_reference, _ = policy.gradients(reference_examples(anchors[::2]))
            second_reference, _ = policy.gradients(reference_examples(anchors[1::2]))
            reference_stability = dataclasses.asdict(
                alignment(
                    first_reference, second_reference, cfg.objective.near_zero, policy.distributed
                )
            )
            rewards = {
                key: [] for key in ("validity", "difficulty", "rl_alignment", "combined_alignment")
            }
            for proposal in proposals:
                if not proposal.valid:
                    for values in rewards.values():
                        values.append(-1.0)
                    continue
                episodes = groups[proposal.task_id]
                rl, _ = student_examples(episodes, cfg, kind="rl")
                combined, _ = student_examples(episodes, cfg, kind="combined")
                grad_rl, _ = policy.gradients(rl)
                grad_combined, _ = policy.gradients(combined)
                rewards["validity"].append(1.0)
                rewards["difficulty"].append(difficulty(e.reward for e in episodes))
                rewards["rl_alignment"].append(
                    alignment(
                        grad_rl, reference, cfg.objective.near_zero, policy.distributed
                    ).reward
                )
                rewards["combined_alignment"].append(
                    alignment(
                        grad_combined, reference, cfg.objective.near_zero, policy.distributed
                    ).reward
                )
        cfg.parents_per_iteration = len(proposals) // cfg.proposals_per_parent
        result = {
            "rewards": rewards,
            "groups": [],
            "branches": {},
            "snapshot": "base",
            "representation": args.representation,
            "reference_stability": reference_stability,
        }
        for start in range(0, len(proposals), cfg.proposals_per_parent):
            result["groups"].append(
                {
                    "parent_id": proposals[start].parent_id,
                    "valid": sum(
                        p.valid for p in proposals[start : start + cfg.proposals_per_parent]
                    ),
                    "advantages": {
                        name: advantages(
                            torch.tensor(values[start : start + cfg.proposals_per_parent])
                        ).tolist()
                        for name, values in rewards.items()
                    },
                }
            )
        with policy.active("proposer"):
            for centered in (False, True):
                for name, values in rewards.items():
                    policy.load_state_dict(copy.deepcopy(original))
                    restore_rng(original_rng)
                    examples = examples_for(proposals, values, cfg, centered)
                    grad, metrics = policy.gradients(examples)
                    policy.gradients(examples, step=True)
                    delta = adapter_delta(original, policy.state_dict())
                    result["branches"][f"{'centered' if centered else 'standardized'}:{name}"] = {
                        "grad_norm": metrics["grad_norm"],
                        "gradient": grad,
                        "delta": delta,
                    }
            for normalization in ("standardized", "centered"):
                base = result["branches"][f"{normalization}:validity"]
                for name in rewards:
                    row = result["branches"][f"{normalization}:{name}"]
                    row["gradient_vs_validity"] = (
                        dataclasses.asdict(
                            alignment(
                                row.pop("gradient"),
                                base["gradient"],
                                cfg.objective.near_zero,
                                policy.distributed,
                            )
                        )
                        if name != "validity"
                        else None
                    )
                    row["update_vs_validity"] = delta_comparison(row["delta"], base["delta"])
                for name in rewards:
                    result["branches"][f"{normalization}:{name}"].pop("delta")
                base.pop("gradient")
        policy.load_state_dict(original)
        restore_rng(original_rng)
        if not state_equal(policy.state_dict(), original):
            raise RuntimeError("Counterfactual failed to restore policy and optimizer")
        args.output.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output / "proposals.json", [p.model_dump(mode="json") for p in proposals])
        atomic_json(
            args.output / "episodes.json",
            {key: [e.model_dump(mode="json") for e in value] for key, value in groups.items()},
        )
        atomic_json(args.output / "counterfactual.json", result)
        atomic_json(
            args.output / "complete.json",
            {"groups": len(result["groups"]), "valid": len(tasks), "restoration_verified": True},
        )
    finally:
        policy.close()


if __name__ == "__main__":
    main()
