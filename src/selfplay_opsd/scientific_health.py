"""Read-only scientific-health summary of committed training evidence."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

from selfplay_opsd.storage import active_events, read_json


def regression(root: Path, output: Path) -> dict:
    """Summarize active committed evidence without assuming raw shards exist."""
    events = active_events(root)
    commits = [e["payload"] for e in events if e["kind"] == "iteration_commit"]
    declared_evaluations = {
        (
            e["segment_id"],
            e["completed_iteration"],
            e["payload"].get("checkpoint"),
            e["payload"].get("endpoint"),
        ): e["payload"]["expected"]
        for e in events
        if e["kind"] == "evaluation_complete"
        and e["payload"].get("split") == "dev_monitor"
        and e["payload"].get("expected") is not None
        and e["payload"].get("completed") == e["payload"]["expected"]
    }
    grouped_evaluations: dict[tuple, list[dict]] = {}
    for e in events:
        if e["kind"] == "evaluation" and e["payload"].get("split") == "dev_monitor":
            identity = (
                e["segment_id"],
                e["completed_iteration"],
                e["payload"].get("checkpoint"),
                e["payload"].get("endpoint"),
            )
            grouped_evaluations.setdefault(identity, []).append(e["payload"])
    complete_evaluations = {
        identity
        for identity, expected in declared_evaluations.items()
        if len(grouped_evaluations.get(identity, [])) == expected
        and len({row.get("task_id") for row in grouped_evaluations.get(identity, [])}) == expected
        and all(row.get("success") is not None for row in grouped_evaluations.get(identity, []))
    }
    incomplete_evaluation_rows = sum(
        e["kind"] == "evaluation"
        and e["payload"].get("split") == "dev_monitor"
        and (
            e["segment_id"],
            e["completed_iteration"],
            e["payload"].get("checkpoint"),
            e["payload"].get("endpoint"),
        )
        not in complete_evaluations
        for e in events
    )
    evaluations = [
        e["payload"]
        for e in events
        if e["kind"] == "evaluation"
        and e["payload"].get("split") == "dev_monitor"
        and (
            e["segment_id"],
            e["completed_iteration"],
            e["payload"].get("checkpoint"),
            e["payload"].get("endpoint"),
        )
        in complete_evaluations
    ]
    windows = []
    for start, end in ((1, 20), (21, 40), (41, 80), (81, 100), (101, 140), (141, 200)):
        rows = [c for c in commits if start <= c["iteration"] <= end]
        groups: Counter[int] = Counter()
        reasons: Counter[str] = Counter()
        terminations: Counter[str] = Counter()
        for row in rows:
            groups.update(
                {int(k): v for k, v in row["metrics"]["student"]["success_group_mix"].items()}
            )
            reasons.update(row["metrics"]["proposer"].get("invalid_reasons", {}))
            terminations.update(row["metrics"]["student"].get("termination", {}))
        uniform_steps = sum(
            bool(c["student_steps"])
            and not any(
                0 < int(k) < 4 and v
                for k, v in c["metrics"]["student"]["success_group_mix"].items()
            )
            for c in rows
        )
        gap_tokens = sum(
            c["metrics"].get("distill", {}).get("gap", {}).get("count", 0) for c in rows
        )
        gap_mean = (
            sum(
                c["metrics"]["distill"]["gap"]["mean"] * c["metrics"]["distill"]["gap"]["count"]
                for c in rows
                if c["metrics"].get("distill", {}).get("gap", {}).get("count", 0)
            )
            / gap_tokens
            if gap_tokens
            else None
        )
        clip_fraction = (
            sum(
                c["metrics"]["distill"]["gap_clip_fraction"]
                * c["metrics"]["distill"]["gap"]["count"]
                for c in rows
                if c["metrics"].get("distill", {}).get("gap", {}).get("count", 0)
            )
            / gap_tokens
            if gap_tokens
            else None
        )
        windows.append(
            {
                "start": start,
                "end": end,
                "commits": len(rows),
                "attempted": sum(c["attempted"] for c in rows),
                "valid": sum(c["valid"] for c in rows),
                "episodes": sum(c["episodes"] for c in rows),
                "successes": sum(c["metrics"]["student"]["success_numerator"] for c in rows),
                "all_failure": groups[0],
                "mixed": sum(v for k, v in groups.items() if 0 < k < 4),
                "all_success": groups[4],
                "student_steps": sum(c["student_steps"] for c in rows),
                "zero_rl_signal_steps": uniform_steps,
                "sampled_gradient_batches": sum(
                    bool(c["metrics"]["student"]["gradient_interaction"].get("sampled"))
                    for c in rows
                ),
                "gap_tokens": gap_tokens,
                "gap_mean": gap_mean,
                "gap_clip_fraction": clip_fraction,
                "student_clipped_steps": sum(
                    bool(c["student_steps"])
                    and c["metrics"]["student"].get("clip_coefficient", 1) < 1
                    for c in rows
                ),
                "invalid_reasons": dict(sorted(reasons.items())),
                "terminations": dict(sorted(terminations.items())),
                "sampled_gradients": [
                    c["metrics"]["student"]["gradient_interaction"]
                    for c in rows
                    if c["metrics"]["student"]["gradient_interaction"].get("sampled")
                ],
            }
        )
    tasks: dict[str, dict] = {}
    for row in evaluations:
        task = tasks.setdefault(
            row["task_id"],
            {
                "task_id": row["task_id"],
                "parent_id": row["parent_id"],
                "results": {},
                "metrics": {},
            },
        )
        task["results"][str(row["iteration"])] = row["success"]
        task["metrics"][str(row["iteration"])] = row.get("metrics", {})
        path = row.get("episode_path")
        if path and (root / path).is_file():
            episodes = read_json(root / path)
            if len(episodes) != 1 or episodes[0].get("task_id") != row["task_id"]:
                raise ValueError(f"Evaluation shard does not match task: {path}")
            episode = episodes[0]
            counts = Counter(
                f"{t.get('action', 'unknown')}:{t.get('parse_status', 'unknown')}"
                for t in episode.get("turns", [])
            )
            task.setdefault("actions", {})[str(row["iteration"])] = dict(sorted(counts.items()))
            if row.get("success") is False:
                escaped = False
                for turn in episode.get("turns", []):
                    if (
                        turn.get("action") == "apply_patch"
                        and turn.get("parse_status") == "tool_error"
                    ):
                        try:
                            escaped |= "\\n" in json.loads(turn["raw_text"])["patch"]
                        except (ValueError, KeyError, TypeError):
                            pass
                if episode.get("public_pass") and not episode.get("hidden_pass"):
                    reason = "public_pass_hidden_fail"
                elif escaped:
                    reason = "double_escaped_patch"
                elif counts.get("apply_patch:tool_error"):
                    reason = "patch_rejected"
                elif any(t.get("parse_status") == "invalid" for t in episode.get("turns", [])):
                    reason = "invalid_action"
                elif counts.get("apply_patch:valid"):
                    reason = "unsuccessful_edit"
                elif episode.get("termination") in {"token_limit", "turn_limit", "context_limit"}:
                    reason = "budget_exhausted"
                else:
                    reason = "unknown_or_other"
                task.setdefault("failure_category", {})[str(row["iteration"])] = reason
    transitions = []
    for task in sorted(tasks.values(), key=lambda x: x["task_id"]):
        if task["results"].get("20") is True:
            later = sorted((int(k), v) for k, v in task["results"].items() if int(k) > 20)
            failure = next((k for k, v in later if v is False), None)
            task["first_failure"] = failure
            task["recovered"] = any(
                v is True for k, v in later if failure is not None and k > failure
            )
            transitions.append(task)
    missing = {
        kind: sum(not (root / c[f"{kind}_path"]).is_file() for c in commits)
        for kind in ("episodes", "proposals")
    }
    shard_actions: Counter[str] = Counter()
    shard_proposals: Counter[str] = Counter()
    duplicate_ownership: Counter[str] = Counter()
    accepted_by_parent: dict[str, set[str]] = {}
    gaps: list[float] = []
    gap_strata: dict[str, list[float]] = {}
    window_gaps: dict[tuple[int, int], list[float]] = {(w["start"], w["end"]): [] for w in windows}
    window_patches: dict[tuple[int, int], Counter[str]] = {
        (w["start"], w["end"]): Counter() for w in windows
    }
    for commit in sorted(commits, key=lambda row: row["iteration"]):
        window_key = next(
            (key for key in window_gaps if key[0] <= commit["iteration"] <= key[1]), None
        )
        episode_path = root / commit["episodes_path"]
        if episode_path.is_file():
            for episode in read_json(episode_path):
                for turn_index, turn in enumerate(episode.get("turns", [])):
                    shard_actions[
                        f"{turn.get('action', 'unknown')}:{turn.get('parse_status', 'unknown')}"
                    ] += 1
                    old, teacher = turn.get("old_logps", []), turn.get("teacher_logps", [])
                    if len(old) != len(teacher):
                        raise ValueError(f"Teacher/behavior token count mismatch: {episode_path}")
                    values = [t - o for t, o in zip(teacher, old, strict=True)]
                    gaps.extend(values)
                    if window_key is not None:
                        window_gaps[window_key].extend(values)
                        if turn.get("action") == "apply_patch":
                            try:
                                patch = json.loads(turn["raw_text"])["patch"]
                                window_patches[window_key]["examined"] += 1
                                window_patches[window_key]["literal_backslash_n"] += "\\n" in patch
                                window_patches[window_key]["actual_newline"] += "\n" in patch
                            except (ValueError, KeyError, TypeError):
                                window_patches[window_key]["unavailable"] += 1
                    for key in (
                        "success" if episode.get("reward") else "failure",
                        f"{turn.get('action', 'unknown')}:{turn.get('parse_status', 'unknown')}",
                        "first_turn" if turn_index == 0 else "later_turn",
                    ):
                        gap_strata.setdefault(key, []).extend(values)
        proposal_path = root / commit["proposals_path"]
        if proposal_path.is_file():
            pending: dict[str, set[str]] = {}
            for proposal in read_json(proposal_path):
                shard_proposals["valid" if proposal["valid"] else proposal["reason"]] += 1
                parent, key = proposal.get("parent_id"), proposal.get("dedup_key")
                if proposal.get("reason") == "duplicate":
                    duplicate_ownership[
                        "previous_iteration"
                        if key in accepted_by_parent.get(parent, set())
                        else "same_iteration"
                        if key in pending.get(parent, set())
                        else "unknown"
                    ] += 1
                if proposal["valid"] and parent and key:
                    pending.setdefault(parent, set()).add(key)
            for parent, keys in pending.items():
                accepted_by_parent.setdefault(parent, set()).update(keys)
    for window in windows:
        window_id = (window["start"], window["end"])
        values = window_gaps[window_id]
        window["shard_gap_tokens"] = len(values)
        window["positive_gap_fraction"] = (
            sum(g > 0 for g in values) / len(values) if values else None
        )
        window["negative_gap_fraction"] = (
            sum(g < 0 for g in values) / len(values) if values else None
        )
        window["patch_newlines"] = dict(sorted(window_patches[window_id].items()))
    evaluation_summary = []
    for iteration in sorted({e["iteration"] for e in evaluations}):
        rows = [e for e in evaluations if e["iteration"] == iteration]
        action_counts: Counter[str] = Counter()
        tool_errors: Counter[str] = Counter()
        patch_newlines: Counter[str] = Counter()
        normalized_patch_results: Counter[str] = Counter()
        available = 0
        for row in rows:
            path = row.get("episode_path")
            if not path or not (root / path).is_file():
                continue
            available += 1
            for episode in read_json(root / path):
                turns = episode.get("turns", [])
                for index, turn in enumerate(turns):
                    action_counts[
                        f"{turn.get('action', 'unknown')}:{turn.get('parse_status', 'unknown')}"
                    ] += 1
                    if (
                        turn.get("action") == "apply_patch"
                        and turn.get("parse_status") == "tool_error"
                    ):
                        try:
                            patch = json.loads(turn["raw_text"])["patch"]
                            patch_newlines["literal_backslash_n"] += "\\n" in patch
                            patch_newlines["actual_newline"] += "\n" in patch
                            patch_newlines["examined"] += 1
                            if "\\n" in patch:
                                from selfplay_opsd.mutations import apply_patch

                                source = None
                                for message in turn.get("history", []):
                                    if message.get("role") != "tool":
                                        continue
                                    try:
                                        observation_value = json.loads(message["content"])
                                    except (ValueError, KeyError):
                                        continue
                                    if (
                                        observation_value.get("ok")
                                        and observation_value.get("path") == "src/solution.py"
                                    ):
                                        source = observation_value.get("content")
                                if source is None:
                                    normalized_patch_results["source_unavailable"] += 1
                                else:
                                    try:
                                        apply_patch(source, patch.replace("\\n", "\n").lstrip("\n"))
                                        normalized_patch_results["applies"] += 1
                                    except ValueError as exc:
                                        normalized_patch_results[str(exc)] += 1
                        except (KeyError, TypeError, ValueError):
                            patch_newlines["unavailable"] += 1
                        history = (
                            turns[index + 1].get("history", []) if index + 1 < len(turns) else []
                        )
                        observation = next(
                            (m["content"] for m in reversed(history) if m.get("role") == "tool"),
                            None,
                        )
                        try:
                            error = (
                                json.loads(observation).get("error", "unknown")
                                if observation
                                else "unrecorded"
                            )
                        except (ValueError, AttributeError):
                            error = "unknown"
                        tool_errors[error] += 1
        evaluation_summary.append(
            {
                "iteration": iteration,
                "tasks": len(rows),
                "successes": sum(e["success"] is True for e in rows),
                "tokens": sum(e["metrics"].get("tokens", 0) for e in rows),
                "available_shards": available,
                "actions": dict(sorted(action_counts.items())),
                "patch_errors": dict(sorted(tool_errors.items())),
                "failed_patch_newlines": dict(sorted(patch_newlines.items())),
                "normalized_patch_diagnostic": dict(sorted(normalized_patch_results.items())),
                "failure_categories": dict(
                    sorted(
                        Counter(
                            t.get("failure_category", {}).get(
                                str(iteration), "unknown_or_missing_shard"
                            )
                            for t in tasks.values()
                            if t["results"].get(str(iteration)) is False
                        ).items()
                    )
                ),
                "terminations": dict(
                    sorted(
                        Counter(e["metrics"].get("termination", "unknown") for e in rows).items()
                    )
                ),
            }
        )
    pointer_path = root / "checkpoints/latest.json"
    pointer = read_json(pointer_path) if pointer_path.is_file() else None
    checkpoint_name = pointer.get("path") if isinstance(pointer, dict) else None
    checkpoint_present = bool(
        isinstance(checkpoint_name, str)
        and checkpoint_name.isdigit()
        and (root / "checkpoints" / checkpoint_name / "state.pt").is_file()
    )
    result = {
        "committed_iterations": len(commits),
        "iterations": [
            {
                "iteration": c["iteration"],
                "attempted": c["attempted"],
                "valid": c["valid"],
                "episodes": c["episodes"],
                "student_steps": c["student_steps"],
                "proposer_steps": c.get("proposer_steps"),
                "student": c["metrics"]["student"],
                "proposer": c["metrics"]["proposer"],
                "distill": c["metrics"].get("distill", {}),
                "alignment": c["metrics"].get("alignment", {}),
            }
            for c in sorted(commits, key=lambda row: row["iteration"])
        ],
        "local_latest_checkpoint": pointer,
        "local_checkpoint_payload_present": checkpoint_present,
        "incomplete_evaluation_rows": incomplete_evaluation_rows,
        "windows": windows,
        "initial_success_transitions": transitions,
        "evaluation_tasks": sorted(tasks.values(), key=lambda row: row["task_id"]),
        "evaluations": evaluation_summary,
        "missing_training_shards": missing,
        "shard_actions": dict(sorted(shard_actions.items())),
        "shard_proposal_reasons": dict(sorted(shard_proposals.items())),
        "duplicate_ownership_in_available_shards": dict(sorted(duplicate_ownership.items())),
        "shard_gap_statistics": {
            "tokens": len(gaps),
            "positive_fraction": sum(g > 0 for g in gaps) / len(gaps) if gaps else None,
            "negative_fraction": sum(g < 0 for g in gaps) / len(gaps) if gaps else None,
            "mean": statistics.mean(gaps) if gaps else None,
        },
        "shard_gap_strata": {
            key: {
                "tokens": len(values),
                "mean": statistics.mean(values) if values else None,
                "positive_fraction": sum(g > 0 for g in values) / len(values) if values else None,
                "clipped_fraction_at_5": sum(abs(g) > 5 for g in values) / len(values)
                if values
                else None,
            }
            for key, values in sorted(gap_strata.items())
        },
        "missing_evaluation_shards": sum(
            not e.get("episode_path") or not (root / e["episode_path"]).is_file()
            for e in evaluations
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "regression.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (output / "windows.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(windows[0]))
        writer.writeheader()
        writer.writerows(windows)
    with (output / "task_transitions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["task_id", "parent_id", "first_failure", "recovered", "results"]
        )
        writer.writeheader()
        for row in transitions:
            writer.writerow(
                {
                    key: json.dumps(row[key], sort_keys=True) if key == "results" else row[key]
                    for key in writer.fieldnames
                }
            )
    lines = [
        "# Regression summary",
        "",
        "| Window | Valid/attempted | Mixed groups | Student steps | Zero-RL-signal steps |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in windows:
        lines.append(
            f"| {row['start']}–{row['end']} | {row['valid']}/{row['attempted']} | {row['mixed']} | {row['student_steps']} | {row['zero_rl_signal_steps']} |"
        )
    lines.extend(
        [
            "",
            f"Missing training shards: {missing}; incomplete evaluation rows excluded: {incomplete_evaluation_rows}.",
            "Zero current RL signal does not exclude optimizer momentum from earlier gradients.",
            "",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines))
    return result


def summarize(
    root: Path, allocated_gpu_hours: float | None = None, through_iteration: int | None = None
) -> dict:
    events = active_events(root)
    if through_iteration is not None:
        events = [e for e in events if e["completed_iteration"] <= through_iteration]
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
    evaluations = [
        event["payload"]
        for event in events
        if event["kind"] == "evaluation"
        and event["payload"].get("split", "dev_monitor") == "dev_monitor"
    ]
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
    observations = []
    if attempted and valid / attempted < 0.25:
        observations.append(
            "Invalid proposals dominate; inspect the recorded reason/action patterns."
        )
    if rl_diagnostics and zero_rl == rl_diagnostics:
        observations.append(
            "All sampled RL gradients are zero; optimizer steps alone do not establish useful RL learning."
        )
    elif rl_diagnostics:
        observations.append(
            f"{zero_rl}/{rl_diagnostics} sampled RL gradients are zero; inspect mixed-reward yield."
        )
    if alignment_count and near_zero / alignment_count > 0.8:
        observations.append(
            "Most candidate alignments are near zero; inspect reference/candidate signal."
        )
    if not any(row.get("iteration", 0) > 0 for row in complete_eval):
        observations.append(
            "No post-training development evaluation is available; held-out improvement is not established."
        )
    if student_steps or proposer_steps:
        observations.append(
            "Mechanical optimizer updates occurred; this is distinct from demonstrated repair improvement."
        )
    result["interpretation"] = observations
    result["scientific_underperformance_action"] = (
        "review_and_recommend; no automatic scientific stopping gate"
    )
    result["requested_through_iteration"] = through_iteration
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--allocated-gpu-hours", type=float)
    parser.add_argument("--regression", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--through-iteration", type=int, choices=(5, 10, 20))
    args = parser.parse_args()
    if args.regression:
        if args.output is None:
            parser.error("--regression requires --output")
        print(json.dumps(regression(args.run_dir, args.output), indent=2))
        return
    if args.allocated_gpu_hours is not None and args.allocated_gpu_hours <= 0:
        parser.error("--allocated-gpu-hours must be positive")
    print(
        json.dumps(
            summarize(args.run_dir, args.allocated_gpu_hours, args.through_iteration), indent=2
        )
    )


if __name__ == "__main__":
    main()
