"""Aggregate-only, checkpoint-backed projections of authoritative local events.

No trajectories, model state, arbitrary strings or configuration dictionaries are
accepted by the upload projection. Canonical tables replace superseded lineage;
scalar history is append-only audit data, never the authority after rollback.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from selfplay_opsd.config import RunConfig
from selfplay_opsd.core import digest
from selfplay_opsd.storage import active_events, file_hash, read_json

METHOD_NAMES = {
    "A": "A-difficulty-ungated-opsd",
    "B": "B-difficulty-gated-opsd",
    "C": "C-rl-alignment-gated-opsd",
    "D": "D-combined-alignment-gated-opsd",
}
DISTRIBUTION_FIELDS = {"count", "mean", "std", "min", "max", "p05", "p50", "p90", "p95"}
# Deliberately explicit: neither arbitrary metric names nor strings are uploaded.
FIELDS = {
    "train": "outer_iteration student_steps proposer_steps",
    "compute": "active_seconds allocated_gpu_hours",
    "student": "repair_success_fraction public_pass_fraction hidden_full_pass_fraction "
    "success_numerator success_denominator mixed_group_fraction cumulative_generated_tokens "
    "cumulative_episodes tokens test_calls public_pass_hidden_fail invalid_actions "
    "invalid_action_fraction pre_update_logprob_abs_max pre_update_logprob_abs_mean "
    "pre_update_logprob_abs_above_005_fraction pre_update_logprob_abs_above_02_fraction "
    "grad_norm rl_loss sd_loss kl_loss total_loss stepped clip_fraction",
    "proposer": "nonzero_advantage_count attempted_count valid_count valid_fraction "
    "distinct_parent_count generated_tokens infrastructure_retries duplicate_count "
    "grad_norm rl_loss sd_loss kl_loss total_loss stepped clip_fraction",
    "distill": "gating_enabled mapped_tokens mapping_error_count gap_clip_fraction "
    "gate_saturation_fraction positive_gap_fraction",
    "alignment": "available scored_count reference_norm exact_zero_fraction "
    "near_zero_fraction negative_alignment_fraction",
    "perf": "proposer_generation_batches proposer_generated_tokens_per_second "
    "student_generation_requests student_generated_tokens_per_generation_second "
    "propose_seconds validate_seconds rollout_seconds reward_eval_seconds "
    "old_student_score_seconds teacher_score_seconds reference_gradient_seconds "
    "candidate_gradient_seconds student_update_seconds proposer_update_seconds "
    "checkpoint_seconds development_eval_seconds",
}
DISTRIBUTIONS = {
    "student": "sampled_token_surprisal generated_tokens_per_episode turns",
    "proposer": "rewards patch_changed_lines patch_changed_nodes",
    "distill": "gap gate signed_coefficient",
    "alignment": "cosine candidate_norm",
}
INTERACTION_FIELDS = {
    "sampled",
    "rl_norm",
    "sd_norm",
    "lambda_sd_norm",
    "combined_norm",
    "cosine",
    "lambda_sd_to_rl_norm_ratio",
    "rl_sd_cosine",
    "total_norm",
    "ratio",
}
SYSTEM_FIELDS = {
    "host_rss_bytes",
    "descendant_rss_bytes",
    "host_available_bytes",
    "disk_free_bytes",
    "cpu_percent",
    "timestamp",
}


def numeric(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def aggregate_metrics(metrics: dict) -> dict:
    output = {}
    for group, fields in FIELDS.items():
        values = metrics.get(group, {})
        for field in fields.split():
            value = values.get(field)
            if numeric(value):
                output[f"{group}/{field}"] = value
    for group, distributions in DISTRIBUTIONS.items():
        for name in distributions.split():
            for field, value in metrics.get(group, {}).get(name, {}).items():
                if field in DISTRIBUTION_FIELDS and numeric(value):
                    output[f"{group}/{name}/{field}"] = value
    for field, value in metrics.get("student", {}).get("gradient_interaction", {}).items():
        if field in INTERACTION_FIELDS and numeric(value):
            output[f"student/gradient_interaction/{field}"] = value
    return output


def checkpoint_watermark(root: Path) -> tuple[int, dict]:
    """Read the atomically published marker without rehashing large model state.

    The trainer already verified all files before publication. The reconciliation
    CLI additionally verifies the complete checkpoint before making network calls.
    """
    latest = root / "checkpoints/latest.json"
    if not latest.exists():
        return -1, {}
    pointer = read_json(latest)
    name = pointer["path"]
    if name != f"{int(pointer['iteration']):06d}":
        raise ValueError("Invalid checkpoint pointer")
    checkpoint = root / "checkpoints" / name
    complete = read_json(checkpoint / "complete.json")
    metadata = read_json(checkpoint / "metadata.json")
    if (
        complete["iteration"] != pointer["iteration"]
        or metadata["iteration"] != pointer["iteration"]
        or complete["files"].get("metadata.json") != file_hash(checkpoint / "metadata.json")
    ):
        raise ValueError("Inconsistent checkpoint metadata")
    return int(pointer["iteration"]), metadata


def canonical_snapshot(root: Path, config: RunConfig) -> dict:
    watermark, metadata = checkpoint_watermark(root)
    events = active_events(root)
    # An older latest pointer can still exist after resuming a prior checkpoint.
    # Until the new segment saves, it cannot authorize that segment's new rows.
    starts = [e for e in events if e["kind"] == "segment_start"]
    if starts and metadata.get("segment_id") != starts[-1]["segment_id"]:
        rollback = starts[-1]["payload"].get("rollback_to")
        if rollback is not None:
            watermark = min(watermark, rollback)
    commits = [
        e
        for e in events
        if e["kind"] == "iteration_commit" and e["completed_iteration"] <= watermark
    ]
    allocation_samples = []
    allocation_path = root / "state/allocation-samples.jsonl"
    if allocation_path.exists():
        for line in allocation_path.read_text().splitlines():
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue  # A concurrently appended final sample is retried later.
            if numeric(sample.get("allocated_gpu_hours")) and numeric(sample.get("timestamp")):
                allocation_samples.append(sample)

    def allocation_at(timestamp):
        return next((s for s in reversed(allocation_samples) if s["timestamp"] <= timestamp), None)

    rows: list[dict] = []
    attempted = valid = successful = 0
    previous_active = 0.0
    durations: list[float] = []
    systems = {
        (e["segment_id"], e["completed_iteration"]): e["payload"]
        for e in events
        if e["kind"] == "system"
    }
    phases: dict[tuple, dict] = defaultdict(dict)
    for event in events:
        if event["kind"] == "phase_time" and numeric(event["payload"].get("seconds")):
            phases[(event["segment_id"], event["completed_iteration"])][event["phase"]] = event[
                "payload"
            ]["seconds"]
    for event in commits:
        payload = event["payload"]
        metrics = payload["metrics"]
        row = aggregate_metrics(metrics)
        iteration = payload["iteration"]
        row.update(
            {
                "train/outer_iteration": iteration,
                "train/target_iterations": config.iterations,
                "run/event_id": f"{event['segment_id']}:{event['event_index']}",
                "run/segment_id": event["segment_id"],
                "run/last_local_commit_timestamp": event["wall_timestamp"],
            }
        )
        attempted += payload["attempted"]
        valid += payload["valid"]
        successful += metrics["student"]["success_numerator"]
        active = metrics.get("compute", {}).get("active_seconds", 0.0)
        if active > previous_active:
            durations.append(active - previous_active)
        previous_active = active
        if durations:
            recent = sum(durations[-10:]) / len(durations[-10:])
            row["perf/recent_iteration_seconds"] = recent
            if len(durations) >= 5:
                row["perf/remaining_active_hours_estimate"] = (
                    max(0, config.iterations - iteration) * recent / 3600
                )
        row["compute/active_training_hours"] = active / 3600
        estimate = row.pop("compute/allocated_gpu_hours", None)
        if estimate is not None:
            row["compute/trainer_gpu_hours_estimate"] = estimate
        sample = allocation_at(event["wall_timestamp"])
        if sample:
            row["compute/allocated_gpu_hours"] = sample["allocated_gpu_hours"]
            row["compute/accounting_sample_age_seconds"] = (
                event["wall_timestamp"] - sample["timestamp"]
            )
            row["run/slurm_job_id"] = str(sample["job_id"])
        row["proposer/cumulative_attempted"] = attempted
        row["proposer/cumulative_valid"] = valid
        row["student/cumulative_successful_episodes"] = successful
        hours = row.get("compute/allocated_gpu_hours", 0)
        if hours > 0:
            row["compute/valid_tasks_per_trainer_gpu_hour"] = valid / hours
            row["compute/successful_episodes_per_trainer_gpu_hour"] = successful / hours
        proposer = metrics["proposer"]
        row["proposer/duplicate_fraction"] = proposer.get("duplicate_count", 0) / max(
            1, payload["attempted"]
        )
        # A fixed taxonomy exposes useful reasons without arbitrary exception text.
        reasons = {
            "Hunk count mismatch": "hunk_count",
            "duplicate": "duplicate",
            "Invalid hunk line or extra file": "hunk_line_or_file",
            "Patch context mismatch": "patch_context",
            "Mutation exceeds 3 AST edit roots / 12 changed nonblank lines": "mutation_limit",
            "Mutant has no semantic failure (crash/timeout-only mutants invalid)": "no_semantic_failure",
            "Only src/solution.py may be patched; two exact diff headers required": "diff_headers",
        }
        for reason, count in proposer.get("invalid_reasons", {}).items():
            category = reasons.get(
                reason,
                "json_parse"
                if reason.startswith(
                    ("Extra data:", "Expecting value:", "Invalid control character")
                )
                else "other",
            )
            name = f"proposer/invalid_reason/{category}"
            if numeric(count):
                row[name] = row.get(name, 0) + count
        # Hash labels rather than upload arbitrary exception strings.
        for reason, count in proposer.get("invalid_reasons", {}).items():
            if numeric(count):
                row[f"proposer/invalid_reason_hash/{digest(reason)[:12]}"] = count
        mix = metrics["student"].get("success_group_mix", {})
        row["student/all_zero_groups"] = mix.get("0", mix.get(0, 0))
        row["student/all_success_groups"] = mix.get(
            str(config.episodes_per_task), mix.get(config.episodes_per_task, 0)
        )
        if config.method in {"A", "B"}:
            row = {key: value for key, value in row.items() if not key.startswith("alignment/")}
            row["alignment/available"] = False
        else:
            scores = metrics.get("alignment", {}).get("scores", [])
            if scores:
                row["alignment/neutral_reward_fraction"] = sum(
                    s.get("reward") == 0.5 for s in scores
                ) / len(scores)
        denominator = metrics["student"].get("success_denominator", payload["episodes"])
        if denominator:
            probability = metrics["student"]["success_numerator"] / denominator
            row.update(
                {
                    "student/reward_mean": probability,
                    "student/reward_std": math.sqrt(probability * (1 - probability)),
                }
            )
        shard = root / payload.get("episodes_path", "missing")
        if shard.is_file() and shard.resolve().is_relative_to(root.resolve()):
            gaps = [
                teacher - old
                for e in read_json(shard)
                for t in e["turns"]
                for teacher, old in zip(t["teacher_logps"], t["old_logps"], strict=True)
            ]
            if gaps:
                row["distill/positive_gap_fraction"] = sum(g > 0 for g in gaps) / len(gaps)
                row["distill/negative_gap_fraction"] = sum(g < 0 for g in gaps) / len(gaps)
        row["compute/gpu_hours_measurement"] = (
            "Slurm allocation samples; missing samples leave the GPU-hour axis unset"
        )
        key = event["segment_id"], event["completed_iteration"]
        for field, value in systems.get(key, {}).items():
            if field in SYSTEM_FIELDS and numeric(value):
                row[f"system/{field}"] = value
        for device, fields in systems.get(key, {}).get("gpu", {}).items():
            if str(device).isdigit():
                for field in (
                    "allocated_bytes",
                    "reserved_bytes",
                    "peak_bytes",
                    "device_used_bytes_at_sample",
                ):
                    if numeric(fields.get(field)):
                        row[f"system/gpu/{device}/{field}"] = fields[field]
        for name in ("checkpoint", "development_eval", "adapter_sync"):
            if name in phases[key]:
                row[f"perf/{name}_seconds"] = phases[key][name]
        rows.append(row)
    evaluations = []
    # A completed evaluation is atomic at the event level; incomplete evaluations
    # must never enter the canonical dashboard or best-checkpoint summary.
    for event in events:
        if event["kind"] != "evaluation_complete" or event["completed_iteration"] > watermark:
            continue
        payload = event["payload"]
        if payload.get("split") != "dev_monitor":
            continue
        iteration = event["completed_iteration"]
        records = [
            e["payload"]
            for e in events
            if e["kind"] == "evaluation"
            and e["segment_id"] == event["segment_id"]
            and e["event_index"] < event["event_index"]
            and e["payload"].get("iteration") == iteration
            and e["payload"].get("split") == "dev_monitor"
        ]
        parents: dict[str, list[bool]] = defaultdict(list)
        for record in records:
            if record.get("success") is not None:
                parents[record["parent_id"]].append(record["success"])
        if not parents:
            continue
        macro = sum(sum(v) / len(v) for v in parents.values()) / len(parents)
        success = sum(sum(v) for v in parents.values()) / sum(map(len, parents.values()))
        sample = allocation_at(event["wall_timestamp"])
        hours = sample["allocated_gpu_hours"] if sample else None
        evaluations.append(
            {
                "train/outer_iteration": iteration,
                "compute/allocated_gpu_hours": hours,
                "eval_by_iteration/dev_parent_macro": macro,
                "eval_by_iteration/dev_success": success,
                "eval_by_gpu_hours/dev_parent_macro": macro,
                "eval_by_gpu_hours/dev_success": success,
            }
        )
    summary = {
        "run/checkpoint_iteration": watermark,
        "run/canonical_rows": len(rows),
        "run/last_checkpoint": f"{watermark:06d}" if watermark >= 0 else "unavailable",
        "alignment/applicability": "applicable"
        if config.method in {"C", "D"}
        else "not_applicable",
    }
    if evaluations:
        best = max(evaluations, key=lambda r: r["eval_by_iteration/dev_parent_macro"])
        summary.update(
            {
                "eval/best_dev_parent_macro": best["eval_by_iteration/dev_parent_macro"],
                "eval/best_checkpoint_iteration": best["train/outer_iteration"],
            }
        )
        baseline = next((r for r in evaluations if r["train/outer_iteration"] == 0), None)
        if baseline:
            initial = baseline["eval_by_iteration/dev_parent_macro"]
            summary.update(
                {
                    "eval/initial_dev_parent_macro": initial,
                    "eval/improvement_over_initial": evaluations[-1][
                        "eval_by_iteration/dev_parent_macro"
                    ]
                    - initial,
                }
            )
    if rows:
        summary.update(rows[-1])
    if allocation_samples:
        summary["compute/current_allocated_gpu_hours"] = allocation_samples[-1][
            "allocated_gpu_hours"
        ]
        summary["compute/accounting_timestamp"] = allocation_samples[-1]["timestamp"]
    # Include lineage even when it has no surviving row (rollback to zero).
    lineage = [
        {"segment": e["segment_id"], "rollback_to": e["payload"].get("rollback_to")}
        for e in events
        if e["kind"] == "segment_start"
    ]
    return {
        "rows": rows,
        "evaluations": evaluations,
        "summary": summary,
        "checkpoint": metadata,
        "revision": digest([rows, evaluations, lineage, summary]),
    }


def table_data(rows: list[dict]) -> tuple[list[str], list[list]]:
    columns = sorted({key for row in rows for key in row})
    return columns, [[row.get(key) for key in columns] for row in rows]


DASHBOARD_CHARTS = {
    "progress": ("training", "train/outer_iteration", "train/student_steps"),
    "student_success": ("training", "train/outer_iteration", "student/repair_success_fraction"),
    "proposer_validity": ("training", "train/outer_iteration", "proposer/valid_fraction"),
    "student_surrogate_loss": ("training", "train/outer_iteration", "student/total_loss"),
    "student_gradient_norm": ("training", "train/outer_iteration", "student/grad_norm"),
    "opsd_gate": ("training", "train/outer_iteration", "distill/gate/mean"),
    "opsd_gap": ("training", "train/outer_iteration", "distill/gap/mean"),
    "alignment": ("training", "train/outer_iteration", "alignment/cosine/mean"),
    "numerical_disagreement": (
        "training",
        "train/outer_iteration",
        "student/pre_update_logprob_abs_mean",
    ),
    "dev_by_iteration": (
        "development",
        "train/outer_iteration",
        "eval_by_iteration/dev_parent_macro",
    ),
    "dev_by_gpu_hours": (
        "development",
        "compute/allocated_gpu_hours",
        "eval_by_gpu_hours/dev_parent_macro",
    ),
    "gpu_hours": ("training", "train/outer_iteration", "compute/allocated_gpu_hours"),
    "iteration_duration": ("training", "train/outer_iteration", "perf/recent_iteration_seconds"),
    "host_memory": ("training", "train/outer_iteration", "system/host_rss_bytes"),
}


def dashboard_charts(wandb, tables: dict):
    charts = {}
    for name, (table_name, x, y) in DASHBOARD_CHARTS.items():
        table = tables[table_name]
        if x in table.columns and y in table.columns:
            charts[f"dashboard/{name}"] = wandb.plot.line(table, x, y, title=name.replace("_", " "))
    return charts
