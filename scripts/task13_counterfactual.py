"""Matched C/D candidate and proposer gradient comparison at one checkpoint."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import os
from pathlib import Path

import torch

from selfplay_opsd.cli import make_policy
from selfplay_opsd.config import load_config
from selfplay_opsd.core import ProposalRecord, TokenTurn, digest, keyed_seed
from selfplay_opsd.data import TaskRepository, build_task
from selfplay_opsd.environment import check_admission, score_teacher
from selfplay_opsd.mutations import strict_json_object, validate_mutation
from selfplay_opsd.objectives import advantages, alignment
from selfplay_opsd.protocol import PROMPT_HASH, proposer_messages
from selfplay_opsd.rollout import episode_wave
from selfplay_opsd.sandbox import SandboxBackend
from selfplay_opsd.storage import (
    CheckpointManager,
    atomic_json,
    restore_rng,
    rng_state,
    scientific_code_hash,
)
from selfplay_opsd.trainer import proposer_examples, reference_examples, student_examples


def adapter_delta(before: dict, after: dict) -> dict:
    delta = {}
    for name, tensor in before["adapters"].items():
        if ".proposer." in name:
            delta[name] = after["adapters"][name].double() - tensor.double()
    return delta


def delta_comparison(first: dict, second: dict) -> dict:
    dot = sum(float((first[k] * second[k]).sum()) for k in first)
    left = sum(float(t.square().sum()) for t in first.values()) ** 0.5
    right = sum(float(t.square().sum()) for t in second.values()) ** 0.5
    return {
        "rl_update_norm": left,
        "combined_update_norm": right,
        "cosine": dot / (left * right) if left and right else None,
    }


def state_equal(first, second) -> bool:
    if torch.is_tensor(first) or torch.is_tensor(second):
        return torch.is_tensor(first) and torch.is_tensor(second) and torch.equal(first, second)
    if isinstance(first, dict) and isinstance(second, dict):
        return first.keys() == second.keys() and all(
            state_equal(first[k], second[k]) for k in first
        )
    if isinstance(first, (list, tuple)) and isinstance(second, (list, tuple)):
        return len(first) == len(second) and all(
            state_equal(a, b) for a, b in zip(first, second, strict=True)
        )
    return first == second


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SELFPLAY_WORKER_TOKEN"):
        raise RuntimeError("Counterfactual requires allocated Slurm and isolated worker")
    if args.output.exists():
        raise RuntimeError("Counterfactual output already exists")
    cfg = load_config(args.settings, method="C", hardware_profile="single_gpu_80gb")
    repo = TaskRepository(cfg.data_root)
    state, metadata = CheckpointManager.load(args.checkpoint)
    if (
        metadata["scientific_hash"] != cfg.scientific_hash
        or metadata["manifests_hash"] != repo.hash
        or metadata["prompt_hash"] != PROMPT_HASH
        or metadata["scientific_code_hash"] != scientific_code_hash()
        or metadata["reference_hash"] != digest(state["reference_anchors"])
    ):
        raise ValueError("Checkpoint scientific, data, prompt, code or reference identity differs")
    sandbox = SandboxBackend(cfg.sandbox)
    sandbox.preflight()
    policy = make_policy(cfg)
    try:
        policy.load_state_dict(state["policy"])
        restore_rng(state["rng"])
        initial_versions = {role: policy.version(role) for role in ("student", "proposer")}
        args.output.mkdir(parents=True, mode=0o700)
        parents = sorted(repo.parents("train"), key=lambda p: digest([2026, p.parent_id]))[:16]
        proposals, tasks = [], {}
        history = state["history"]
        with policy.active("proposer"):
            for parent in parents:
                if len(tasks) >= 13:
                    break  # Finish the current group and never exceed 16 candidates.
                pending = set()
                prompt = policy.encode_history(proposer_messages(parent))
                for slot in range(4):
                    seed = keyed_seed(2026, "task13-cd", parent.parent_id, slot)
                    result = policy.generate(prompt, cfg.proposer_max_tokens, seed)
                    valid, reason, task_id, key = False, "invalid", None, None
                    try:
                        value = strict_json_object(result.text)
                        if set(value) != {"patch"} or not isinstance(value["patch"], str):
                            raise ValueError("Invalid patch proposal")
                        change = validate_mutation(parent.source, value["patch"])
                        key = change.dedup_key
                        if key in history.get(parent.parent_id, []) or key in pending:
                            raise ValueError("duplicate")
                        public, hidden = repo.parent_cases(parent.parent_id)
                        task, oracle = build_task(
                            parent, change, public, hidden, "train", seed, sandbox
                        )
                        check_admission(policy, task, oracle, cfg.episode)
                        task_id = task.task_id
                        tasks[task_id] = task, oracle
                        pending.add(key)
                        valid, reason = True, "valid"
                    except (ValueError, SyntaxError, KeyError, TypeError) as exc:
                        reason = str(exc)[:300]
                    turn = TokenTurn(
                        prompt_ids=prompt,
                        target_ids=result.ids,
                        old_logps=result.logps,
                        history=proposer_messages(parent),
                        raw_text=result.text,
                        action="propose_patch",
                        parse_status="valid" if valid else "invalid",
                    )
                    proposals.append(
                        ProposalRecord(
                            proposal_id=digest([parent.parent_id, slot, seed]),
                            parent_id=parent.parent_id,
                            slot=slot,
                            seed=seed,
                            policy_version=policy.version("proposer"),
                            turn=turn,
                            valid=valid,
                            reason=reason,
                            task_id=task_id,
                            dedup_key=key,
                        )
                    )
                atomic_json(
                    args.output / "proposals.json", [p.model_dump(mode="json") for p in proposals]
                )
        groups = {}
        with policy.active("student"):
            for task_id, (task, oracle) in tasks.items():
                jobs = [
                    (task, oracle, keyed_seed(2026, "task13-cd-student", task_id, i))
                    for i in range(4)
                ]
                episodes = episode_wave(policy, sandbox, jobs, cfg.episode, concurrency=1)
                groups[task_id] = [
                    score_teacher(policy, e, oracle, cfg.episode)[0] for e in episodes
                ]
            reference, _ = policy.gradients(reference_examples(state["reference_anchors"]))
            rewards_c, rewards_d, scores = [], [], []
            for proposal in proposals:
                if not proposal.valid:
                    rewards_c.append(-1.0)
                    rewards_d.append(-1.0)
                    continue
                episodes = groups[proposal.task_id]
                rl, _ = student_examples(episodes, cfg, kind="rl")
                combined, _ = student_examples(episodes, cfg, kind="combined")
                gc, _ = policy.gradients(rl)
                gd, _ = policy.gradients(combined)
                c = alignment(gc, reference, cfg.objective.near_zero, policy.distributed)
                d = alignment(gd, reference, cfg.objective.near_zero, policy.distributed)
                rewards_c.append(c.reward)
                rewards_d.append(d.reward)
                scores.append(
                    {
                        "task_id": proposal.task_id,
                        "rewards": [e.reward for e in episodes],
                        "rl": dataclasses.asdict(c),
                        "combined": dataclasses.asdict(d),
                    }
                )
        cfg.parents_per_iteration = len(proposals) // 4
        comparisons = []
        for start in range(0, len(proposals), 4):
            a = advantages(torch.tensor(rewards_c[start : start + 4]))
            b = advantages(torch.tensor(rewards_d[start : start + 4]))
            comparisons.append(
                {
                    "parent_id": proposals[start].parent_id,
                    "valid": sum(p.valid for p in proposals[start : start + 4]),
                    "advantage_l2": float((a - b).norm()),
                    "rl_advantages": a.tolist(),
                    "combined_advantages": b.tolist(),
                }
            )
        report = {
            "checkpoint_iteration": metadata["iteration"],
            "checkpoint_scientific_hash": metadata["scientific_hash"],
            "reference_hash": metadata["reference_hash"],
            "groups": comparisons,
            "candidate_scores": scores,
            "proposer_gradient": None,
            "adam_update": None,
        }
        if proposals:
            with policy.active("proposer"):
                gc, mc = policy.gradients(proposer_examples(proposals, rewards_c, cfg))
                gd, md = policy.gradients(proposer_examples(proposals, rewards_d, cfg))
                report["proposer_gradient"] = {
                    "rl_norm": mc["grad_norm"],
                    "combined_norm": md["grad_norm"],
                    "comparison": dataclasses.asdict(
                        alignment(gc, gd, cfg.objective.near_zero, policy.distributed)
                    ),
                }
                selected = max(
                    range(len(comparisons)), key=lambda i: comparisons[i]["advantage_l2"]
                )
                if comparisons[selected]["advantage_l2"] > 1e-5:
                    mini = cfg.model_copy(deep=True)
                    mini.parents_per_iteration = 1
                    start = selected * 4
                    original_rng = rng_state()
                    deltas = []
                    for rewards in (rewards_c, rewards_d):
                        policy.load_state_dict(copy.deepcopy(state["policy"]))
                        restore_rng(original_rng)
                        policy.gradients(
                            proposer_examples(
                                proposals[start : start + 4], rewards[start : start + 4], mini
                            ),
                            step=True,
                        )
                        deltas.append(adapter_delta(state["policy"], policy.state_dict()))
                    report["adam_update"] = {
                        "parent_id": comparisons[selected]["parent_id"],
                        **delta_comparison(*deltas),
                    }
                    policy.load_state_dict(state["policy"])
                    restore_rng(original_rng)
                    restored = policy.state_dict()
                    if not state_equal(restored, state["policy"]) or any(
                        policy.version(role) != version
                        for role, version in initial_versions.items()
                    ):
                        raise RuntimeError(
                            "Counterfactual parameter or optimizer restoration failed"
                        )
                    report["restoration_verified"] = True
        atomic_json(args.output / "counterfactual.json", report)
        atomic_json(
            args.output / "complete.json", {"groups": len(comparisons), "valid": len(tasks)}
        )
    finally:
        policy.close()


if __name__ == "__main__":
    main()
