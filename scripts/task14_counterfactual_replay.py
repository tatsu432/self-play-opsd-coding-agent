"""Compare all saved proposer reward gradients and updates pairwise."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
from itertools import combinations
from pathlib import Path

from task13_counterfactual import adapter_delta, delta_comparison, state_equal
from task14_counterfactual import examples_for

from selfplay_opsd.cli import make_policy
from selfplay_opsd.config import load_config
from selfplay_opsd.core import ProposalRecord
from selfplay_opsd.objectives import alignment
from selfplay_opsd.storage import atomic_json, file_hash, restore_rng, rng_state

SOURCE = Path("/runs/repair-diagnosis-2026-task14-counterfactual-v1/e74ecdee52efe104590c60b3")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID"):
        raise RuntimeError("Counterfactual replay requires an allocated GPU")
    if args.output.exists():
        raise ValueError("Output already exists")
    complete = json.loads((SOURCE / "complete.json").read_text())
    report = json.loads((SOURCE / "counterfactual.json").read_text())
    proposals = [
        ProposalRecord.model_validate(row)
        for row in json.loads((SOURCE / "proposals.json").read_text())
    ]
    rewards = report["rewards"]
    if len(proposals) != complete["groups"] * 4 or any(
        len(r) != len(proposals) for r in rewards.values()
    ):
        raise ValueError("Counterfactual source is incomplete")
    cfg = load_config(args.settings, method="C", hardware_profile="single_gpu_80gb")
    cfg.proposer_representation = report["representation"]
    cfg.parents_per_iteration = len(proposals) // cfg.proposals_per_parent
    cfg.logging.mode = "disabled"
    policy = make_policy(cfg)
    try:
        original = copy.deepcopy(policy.state_dict())
        original_rng = rng_state()
        if any(p.policy_version != policy.version("proposer") for p in proposals):
            raise ValueError("Saved proposals do not match the reconstructed snapshot")
        branches = {}
        with policy.active("proposer"):
            for centered in (False, True):
                for name in (*rewards, "validity_repeat"):
                    values = rewards["validity" if name == "validity_repeat" else name]
                    policy.load_state_dict(copy.deepcopy(original))
                    restore_rng(original_rng)
                    examples = examples_for(proposals, values, cfg, centered)
                    gradient, metrics = policy.gradients(examples)
                    policy.gradients(examples, step=True)
                    branches[(centered, name)] = {
                        "gradient": gradient,
                        "norm": metrics["grad_norm"],
                        "delta": adapter_delta(original, policy.state_dict()),
                    }
            comparisons = {}
            for centered in (False, True):
                for first, second in combinations((*rewards, "validity_repeat"), 2):
                    a, b = branches[(centered, first)], branches[(centered, second)]
                    comparisons[
                        f"{'centered' if centered else 'standardized'}:{first}:{second}"
                    ] = {
                        "gradient": dataclasses.asdict(
                            alignment(
                                a["gradient"],
                                b["gradient"],
                                cfg.objective.near_zero,
                                policy.distributed,
                            )
                        ),
                        "updates": delta_comparison(a["delta"], b["delta"]),
                        "gradient_norms": [a["norm"], b["norm"]],
                    }
        policy.load_state_dict(original)
        restore_rng(original_rng)
        if not state_equal(policy.state_dict(), original):
            raise RuntimeError("Counterfactual replay failed to restore state")
        args.output.mkdir(parents=True, mode=0o700)
        atomic_json(
            args.output / "replay.json",
            {
                "source_hashes": {
                    name: file_hash(SOURCE / name)
                    for name in (
                        "complete.json",
                        "counterfactual.json",
                        "proposals.json",
                        "episodes.json",
                    )
                },
                "comparisons": comparisons,
                "restoration_verified": True,
            },
        )
        atomic_json(
            args.output / "complete.json",
            {"groups": complete["groups"], "restoration_verified": True},
        )
    finally:
        policy.close()


if __name__ == "__main__":
    main()
