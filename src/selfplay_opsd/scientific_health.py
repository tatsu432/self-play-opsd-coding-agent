"""Read-only scientific-health summary of committed training evidence."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from selfplay_opsd.storage import active_events, read_json


def summarize(root: Path, allocated_gpu_hours: float | None = None) -> dict:
    events = active_events(root)
    commits = [event["payload"] for event in events if event["kind"] == "iteration_commit"]
    invalid: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    groups: Counter[int] = Counter()
    phases: Counter[str] = Counter()
    attempted = valid = episodes = successes = alignment_count = near_zero = 0
    rl_diagnostics = zero_rl = student_steps = proposer_steps = 0
    duplicates = raw_episode_shards = raw_proposal_shards = 0
    distinct_mutations: set[str] = set()
    for event in events:
        if event["kind"] == "phase_time":
            phases[event["phase"]] += event["payload"]["seconds"]
    for commit in commits:
        metrics = commit["metrics"]
        attempted += commit["attempted"]
        valid += commit["valid"]
        episodes += commit["episodes"]
        student_steps += commit["student_steps"]
        proposer_steps += commit["proposer_steps"]
        proposer, student, alignment = (
            metrics["proposer"],
            metrics["student"],
            metrics["alignment"],
        )
        invalid.update(proposer["invalid_reasons"])
        duplicates += proposer["duplicate_count"]
        groups.update({int(k): v for k, v in student["success_group_mix"].items()})
        successes += student["success_numerator"]
        alignment_count += alignment["scored_count"]
        near_zero += sum(score["near_zero"] for score in alignment["scores"])
        interaction = student["gradient_interaction"]
        if interaction.get("sampled"):
            rl_diagnostics += 1
            zero_rl += interaction.get("rl_norm") == 0
        shard = root / commit["episodes_path"]
        if shard.exists():
            raw_episode_shards += 1
            for episode in read_json(shard):
                actions.update(
                    f"{turn['action']}:{turn['parse_status']}" for turn in episode["turns"]
                )
        proposal_shard = root / commit["proposals_path"]
        if proposal_shard.exists():
            raw_proposal_shards += 1
            distinct_mutations.update(
                row["dedup_key"]
                for row in read_json(proposal_shard)
                if row["valid"] and row.get("dedup_key")
            )
    evaluations = [event["payload"] for event in events if event["kind"] == "evaluation"]
    complete_eval = [row for row in evaluations if row.get("success") is not None]
    result = {
        "committed_iterations": len(commits),
        "attempted_proposals": attempted,
        "valid_proposals": valid,
        "valid_fraction": valid / attempted if attempted else None,
        "invalid_reasons": dict(invalid.most_common()),
        "duplicate_proposals": duplicates,
        "distinct_valid_mutations_in_available_shards": len(distinct_mutations),
        "raw_episode_shards_available": raw_episode_shards,
        "raw_proposal_shards_available": raw_proposal_shards,
        "solver_episodes": episodes,
        "successful_episodes": successes,
        "outcome_groups": {
            "all_zero": groups[0],
            "all_one": sum(count for wins, count in groups.items() if wins == 4),
            "mixed": sum(count for wins, count in groups.items() if 0 < wins < 4),
        },
        "student_actions": dict(actions.most_common()),
        "student_steps": student_steps,
        "proposer_steps": proposer_steps,
        "sampled_rl_gradients": rl_diagnostics,
        "sampled_zero_rl_gradients": zero_rl,
        "alignment_scores": alignment_count,
        "near_zero_alignment_scores": near_zero,
        "phase_seconds": dict(phases),
        "sampled_rl_norms": [
            c["metrics"]["student"]["gradient_interaction"]["rl_norm"]
            for c in commits
            if c["metrics"]["student"]["gradient_interaction"].get("sampled")
        ],
        "sampled_lambda_opsd_norms": [
            c["metrics"]["student"]["gradient_interaction"]["lambda_sd_norm"]
            for c in commits
            if c["metrics"]["student"]["gradient_interaction"].get("sampled")
        ],
        "development_evaluations": len(complete_eval),
        "development_successes": sum(row["success"] for row in complete_eval),
        "evaluation_progression": [
            {"iteration": row["iteration"], "task_id": row["task_id"], "success": row["success"]}
            for row in complete_eval
        ],
        "allocated_gpu_hours": allocated_gpu_hours,
        "valid_proposals_per_allocated_gpu_hour": (
            valid / allocated_gpu_hours if allocated_gpu_hours else None
        ),
        "successful_episodes_per_allocated_gpu_hour": (
            successes / allocated_gpu_hours if allocated_gpu_hours else None
        ),
    }
    if commits:
        result["latest_iteration"] = commits[-1]["iteration"]
        result["recent_iterations"] = [
            {
                "iteration": c["iteration"],
                "valid": c["valid"],
                "successes": c["metrics"]["student"]["success_numerator"],
                "student_grad_norm": c["metrics"]["student"]["grad_norm"],
                "pre_update_logprob_abs_max": c["metrics"]["student"].get(
                    "pre_update_logprob_abs_max"
                ),
                "pre_update_logprob_abs_mean": c["metrics"]["student"].get(
                    "pre_update_logprob_abs_mean"
                ),
                "pre_update_logprob_abs_above_02_fraction": c["metrics"]["student"].get(
                    "pre_update_logprob_abs_above_02_fraction"
                ),
                "alignment_near_zero_fraction": c["metrics"]["alignment"]["near_zero_fraction"],
            }
            for c in commits[-10:]
        ]
        valid_counts = [c["valid"] for c in commits]
        result["valid_per_iteration_median"] = statistics.median(valid_counts)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--allocated-gpu-hours", type=float)
    args = parser.parse_args()
    if args.allocated_gpu_hours is not None and args.allocated_gpu_hours <= 0:
        parser.error("--allocated-gpu-hours must be positive")
    print(json.dumps(summarize(args.run_dir, args.allocated_gpu_hours), indent=2))


if __name__ == "__main__":
    main()
