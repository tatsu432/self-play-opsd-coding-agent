"""Measure behavior-rescoring impact on student gradients and Adam updates."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import torch
from task13_counterfactual import delta_comparison, state_equal
from task13_static_continue import SOURCE, restore_pool
from task14_checkpoint import load_b200

from selfplay_opsd.cli import make_policy
from selfplay_opsd.config import load_config
from selfplay_opsd.core import EpisodeRecord
from selfplay_opsd.data import TaskRepository
from selfplay_opsd.environment import score_teacher
from selfplay_opsd.numerical_audit import score_role
from selfplay_opsd.objectives import advantages, alignment
from selfplay_opsd.sandbox import SandboxBackend
from selfplay_opsd.storage import atomic_json, file_hash, restore_rng, rng_state
from selfplay_opsd.trainer import student_examples


def clipping_impact(original, rescored, clip):
    groups = defaultdict(list)
    for index, episode in enumerate(original):
        groups[episode.task_id].append(index)
    affected = eligible = 0
    for indices in groups.values():
        values = advantages(torch.tensor([original[i].reward for i in indices])).tolist()
        for index, advantage in zip(indices, values, strict=True):
            if advantage == 0:
                continue
            for first, second in zip(original[index].turns, rescored[index].turns, strict=True):
                for old, current in zip(first.old_logps, second.old_logps, strict=True):
                    delta = current - old
                    eligible += 1
                    affected += int(
                        (advantage > 0 and delta > math.log1p(clip))
                        or (advantage < 0 and delta < math.log1p(-clip))
                    )
    return {
        "eligible_tokens": eligible,
        "clipped_tokens": affected,
        "clipped_fraction": affected / eligible if eligible else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID") or not os.getenv("SELFPLAY_WORKER_TOKEN"):
        raise RuntimeError("Numerical diagnostic requires an allocated isolated worker")
    if args.output.exists():
        raise ValueError("Output already exists")
    manifest = json.loads((args.source / "manifest.json").read_text())
    complete = json.loads((args.source / "complete.json").read_text())
    records = json.loads((args.source / "episodes.json").read_text())
    if complete["episodes"] != len(records) or len(records) != len(manifest["tasks"]) * 4:
        raise ValueError("Source student probe is incomplete")
    cfg = load_config(args.settings, method="B", hardware_profile="single_gpu_80gb")
    cfg.episode.edit_protocol = manifest["protocol"]
    cfg.logging.mode = "disabled"
    repo = TaskRepository(cfg.data_root)
    if (
        manifest["dataset"] != repo.hash
        or manifest["pool_sha256"] != file_hash(SOURCE / "pool.json")
        or manifest["snapshot"] not in {"base", "b200"}
    ):
        raise ValueError("Student probe provenance differs")
    pool = {task.task_id: (task, oracle) for task, oracle, _ in restore_pool(SOURCE, repo)}
    selected = {task["task_id"] for task in manifest["tasks"][:8]}
    episodes = [EpisodeRecord.model_validate(row) for row in records if row["task_id"] in selected]
    sandbox = SandboxBackend(cfg.sandbox)
    sandbox.preflight()
    policy = make_policy(cfg)
    try:
        if manifest["snapshot"] == "b200":
            state, _, marker = load_b200(cfg, repo)
            if marker != manifest["checkpoint_marker_sha256"]:
                raise ValueError("Student probe checkpoint marker differs")
            policy.load_state_dict(state["policy"])
            restore_rng(state["rng"])
        original = copy.deepcopy(policy.state_dict())
        original_rng = rng_state()
        with policy.active("student"):
            if any(e.policy_version != policy.version("student") for e in episodes):
                raise ValueError("Saved episodes do not match initial student snapshot")
            scored = [
                score_teacher(policy, e, pool[e.task_id][1], cfg.episode)[0] for e in episodes
            ]
            turns, stats = score_role(
                policy,
                [t for e in scored for t in e.turns],
                "student",
                policy.version("student"),
                cfg.objective.clip,
            )
            altered = []
            cursor = 0
            for episode in scored:
                changed = turns[cursor : cursor + len(episode.turns)]
                cursor += len(episode.turns)
                altered.append(episode.model_copy(update={"turns": changed}))
            branches = {}
            for label, data in (("behavior", scored), ("rescored_diagnostic", altered)):
                policy.load_state_dict(copy.deepcopy(original))
                restore_rng(original_rng)
                grouped = defaultdict(list)
                for episode in data:
                    grouped[episode.task_id].append(episode)
                examples = []
                for group in grouped.values():
                    current, _ = student_examples(group, cfg, tasks=len(grouped), kind="combined")
                    examples.extend(current)
                gradient, metrics = policy.gradients(examples)
                policy.gradients(examples, step=True)
                after = policy.state_dict()
                delta = {
                    name: tensor.double() - original["adapters"][name].double()
                    for name, tensor in after["adapters"].items()
                    if ".student." in name
                }
                branches[label] = {
                    "gradient": gradient,
                    "gradient_norm": metrics["grad_norm"],
                    "delta": delta,
                }
            report = {
                "source_sha256": file_hash(args.source / "episodes.json"),
                "protocol": manifest["protocol"],
                "tasks": len(selected),
                "episodes": len(episodes),
                "mismatch": stats,
                "sign_sensitive_clipping": clipping_impact(scored, altered, cfg.objective.clip),
                "gradient_comparison": dataclasses.asdict(
                    alignment(
                        branches["behavior"]["gradient"],
                        branches["rescored_diagnostic"]["gradient"],
                        cfg.objective.near_zero,
                        policy.distributed,
                    )
                ),
                "adam_update_comparison": delta_comparison(
                    branches["behavior"]["delta"], branches["rescored_diagnostic"]["delta"]
                ),
                "note": "Rescored denominators are diagnostic only; training retains behavior probabilities.",
            }
        policy.load_state_dict(original)
        restore_rng(original_rng)
        if not state_equal(policy.state_dict(), original):
            raise RuntimeError("Numerical probe failed to restore optimizer state")
        args.output.mkdir(parents=True, mode=0o700)
        atomic_json(args.output / "numerical.json", report)
        atomic_json(
            args.output / "complete.json", {"episodes": len(episodes), "restoration_verified": True}
        )
    finally:
        policy.close()


if __name__ == "__main__":
    main()
