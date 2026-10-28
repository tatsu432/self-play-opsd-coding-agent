"""Conditional workload projections must retain overhead and reject empty runs."""

import pytest

from selfplay_opsd.workload import project_method


def events():
    result = [
        {
            "kind": "iteration_commit",
            "payload": {
                "iteration": 1,
                "attempted": 16,
                "valid": 1,
                "episodes": 4,
                "metrics": {
                    "student": {"tokens": 100, "stepped": 1},
                    "proposer": {"stepped": 1},
                    "perf": {
                        "propose_seconds": 2,
                        "validate_seconds": 2,
                        "reference_gradient_seconds": 1,
                    },
                },
            },
        }
    ]
    for phase, seconds in {
        "propose_validate": 4,
        "rollout_reward": 10,
        "old_student_teacher_score": 4,
        "alignment": 3,
        "student_update": 2,
        "proposer_update": 1,
        "adapter_sync": 1,
        "gradient_diagnostic": 2,
        "checkpoint": 1,
    }.items():
        result.append(
            {
                "kind": "phase_time",
                "phase": phase,
                "completed_iteration": 1,
                "payload": {"seconds": seconds},
            }
        )
    return result


def test_sparse_workload_forecast_does_not_multiply_fixed_cost_or_omit_monitoring():
    report = project_method(events(), allocated_seconds=28, baseline_seconds=60)
    # 200 steady iterations, ten main diagnostics, 201 checkpoints and 11 monitors.
    assert report["observed_load_200_hours"] == pytest.approx(
        (200 * 25 + 10 * 2 + 201 + 11 * 60) / 3600
    )
    # Full yield changes per-task work; sampling/reference/adapter costs stay fixed.
    assert report["all_valid_observed_lengths_200_hours"] == pytest.approx(
        (200 * 325 + 10 * 32 + 201 + 11 * 60) / 3600
    )
    assert report["planning_projection_200_hours"] >= report["observed_load_200_hours"]
    assert report["monitoring_events"] == 11
    assert report["signal_seconds"] >= report["planning_iteration_seconds"]


def test_acceptance_fixture_is_charged_to_readiness_but_excluded_from_main_projection():
    plain = project_method(events(), allocated_seconds=28, baseline_seconds=60)
    fixture = project_method(
        events(), allocated_seconds=128, acceptance_only_seconds=100, baseline_seconds=60
    )
    assert fixture["allocated_seconds"] == 128
    assert fixture["planning_projection_200_hours"] == plain["planning_projection_200_hours"]


def test_empty_workload_cannot_be_projected():
    with pytest.raises(ValueError, match="Empty workload"):
        project_method([], allocated_seconds=30, baseline_seconds=60)
