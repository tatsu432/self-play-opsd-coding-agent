"""Conditional engineering projections from real committed learning workloads."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict


def quantile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def project_method(
    events: list[dict],
    *,
    allocated_seconds: float,
    baseline_seconds: float,
    iterations: int = 200,
    attempts_per_iteration: int = 16,
    checkpoint_every: int = 1,
    evaluate_every: int = 20,
    acceptance_only_seconds: float = 0.0,
    diagnostic_seconds_per_valid: float | None = None,
) -> dict:
    commits = [e["payload"] for e in events if e["kind"] == "iteration_commit"]
    if not commits or not any(c["valid"] and c["episodes"] for c in commits):
        raise ValueError("Empty workload cannot establish a learning cost projection")
    phases: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    phase_rows = [e for e in events if e["kind"] == "phase_time"]
    for e in phase_rows:
        phases[e["completed_iteration"]][e["phase"]] += e["payload"]["seconds"]
    learned = [c for c in commits if c["valid"] > 0 and c["episodes"] > 0]
    checkpoints = [e["payload"]["seconds"] for e in phase_rows if e["phase"] == "checkpoint"]
    checkpoint = quantile(checkpoints, 0.9)
    # This conservative amortization retains startup, continuation, logger and
    # shutdown costs rather than presenting phase time as allocated GPU time.
    recorded_seconds = sum(e["payload"]["seconds"] for e in phase_rows)
    unattributed = max(0.0, allocated_seconds - recorded_seconds - acceptance_only_seconds)
    overhead_per_iteration = unattributed / len(commits)
    special = {
        "checkpoint",
        "gradient_diagnostic",
        "pilot_reference_preparation",
        "development_eval",
    }
    observed_steady = statistics.mean(
        sum(v for k, v in phases[c["iteration"]].items() if k not in special) for c in commits
    )
    observed_diagnostic = statistics.mean(
        phases[c["iteration"]].get("gradient_diagnostic", 0.0) for c in commits
    )
    if observed_diagnostic == 0 and diagnostic_seconds_per_valid is not None:
        observed_diagnostic = diagnostic_seconds_per_valid * statistics.mean(
            c["valid"] for c in commits
        )
    observed_checkpoint = statistics.mean(checkpoints) if checkpoints else 0.0
    monitor_count = iterations // evaluate_every + 1  # Includes main iteration zero.
    monitor_seconds = monitor_count * baseline_seconds

    def forecast(steady: float, diagnostic: float, cp: float) -> float:
        return (
            iterations * (steady + overhead_per_iteration)
            + (iterations // 20) * diagnostic
            + (math.ceil(iterations / checkpoint_every) + 1) * cp
            + monitor_seconds
        ) / 3600

    def full_valid(fraction: float) -> tuple[float, float]:
        fixed, variable, diagnostics = [], [], []
        for c in learned:
            p = phases[c["iteration"]]
            perf = c["metrics"]["perf"]
            propose = perf.get("propose_seconds") or 0.0
            validate = perf.get("validate_seconds") or 0.0
            reference = perf.get("reference_gradient_seconds") or 0.0
            fixed.append(
                propose
                + max(0.0, p.get("propose_validate", 0.0) - propose - validate)
                + reference
                + p.get("proposer_update", 0.0)
                + p.get("adapter_sync", 0.0)
            )
            variable.append(
                (
                    validate
                    + p.get("rollout_reward", 0.0)
                    + p.get("old_student_teacher_score", 0.0)
                    + max(0.0, p.get("alignment", 0.0) - reference)
                    + p.get("student_update", 0.0)
                )
                / c["valid"]
            )
            diagnostics.append(
                p.get("gradient_diagnostic", 0.0) / c["valid"]
                or diagnostic_seconds_per_valid
                or 0.0
            )
        return (
            quantile(fixed, fraction) + attempts_per_iteration * quantile(variable, fraction),
            attempts_per_iteration * quantile(diagnostics, fraction),
        )

    full_median, diagnostic_median = full_valid(0.5)
    full_p90, diagnostic_p90 = full_valid(0.9)
    observed = forecast(observed_steady, observed_diagnostic, observed_checkpoint)
    dense = forecast(full_median, diagnostic_median, checkpoint)
    conservative = max(observed, forecast(full_p90, diagnostic_p90, checkpoint))
    planning_iteration = full_p90 + diagnostic_p90 + checkpoint + overhead_per_iteration
    # Allow twice the conditional high-load iteration plus checkpoint margin.
    signal_seconds = max(300, math.ceil(2 * planning_iteration + checkpoint + 120))
    if signal_seconds > 65535 or signal_seconds >= 23 * 3600:
        raise ValueError("Measured workload does not fit the proposed Slurm continuation envelope")
    segment_hours = 23 - signal_seconds / 3600
    continuations = max(1, math.ceil(conservative / segment_hours) + 1)
    if continuations > 32:
        raise ValueError("Projected workload exceeds the existing continuation ceiling")
    return {
        "committed_iterations": len(commits),
        "learning_iterations": len(learned),
        "attempted": sum(c["attempted"] for c in commits),
        "valid": sum(c["valid"] for c in commits),
        "episodes": sum(c["episodes"] for c in commits),
        "tokens": sum(c["metrics"]["student"]["tokens"] for c in commits),
        "actual_student_steps": sum(c["metrics"]["student"]["stepped"] for c in commits),
        "actual_proposer_steps": sum(c["metrics"]["proposer"]["stepped"] for c in commits),
        "allocated_seconds": allocated_seconds,
        "acceptance_only_seconds": acceptance_only_seconds,
        "diagnostic_seconds_per_valid_p90": quantile(
            [
                phases[c["iteration"]].get("gradient_diagnostic", 0.0) / c["valid"]
                for c in learned
                if phases[c["iteration"]].get("gradient_diagnostic", 0.0) > 0
            ],
            0.9,
        )
        or diagnostic_seconds_per_valid,
        "recorded_phase_seconds": recorded_seconds,
        "unattributed_seconds": unattributed,
        "observed_load_200_hours": observed,
        "all_valid_observed_lengths_200_hours": dense,
        "planning_projection_200_hours": conservative,
        "planning_iteration_seconds": planning_iteration,
        "signal_seconds": signal_seconds,
        "max_continuations": continuations,
        "monitoring_events": monitor_count,
        "monitoring_seconds_charged": monitor_seconds,
        "checkpoint_p90_seconds": checkpoint,
        "assumptions": [
            "Conditional scenarios, not confidence intervals or guaranteed future throughput",
            "All-valid scenario retains observed episode lengths and per-task phase rates",
            "Higher response/context lengths may exceed these scenarios",
            "Only nonempty learning iterations estimate variable-workload rates",
            "Full baseline allocated time is conservatively charged for each monitoring event",
            "Pilot gradient diagnostics are charged at the main frequency of every 20 iterations",
            "Unattributed allocation time is conservatively amortized per training iteration",
            "Canonical pilot reference proxies can differ in token length from main references",
            "Methods without sampled gradient diagnostics use the measured D diagnostic rate",
        ],
    }
