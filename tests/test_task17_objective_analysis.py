"""Check that the objective report rejects incomplete or mismatched studies."""

import importlib.util
import json
from pathlib import Path

import pytest


def test_objective_analysis_requires_complete_matched_evidence(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "task17_objective_analysis", Path("scripts/analyze_task17_objectives.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="explicit compatibility"):
        module.validate_seed_code_hashes({2026: "old", 2027: "new"}, None, None)
    with pytest.raises(ValueError, match="declared seed revision"):
        module.validate_seed_code_hashes({2026: "old", 2027: "new"}, {"2026": "old"}, "reason")
    module.validate_seed_code_hashes(
        {2026: "old", 2027: "new"}, {"2026": "old", "2027": "new"}, "seed record fix"
    )
    manifest = {
        "arm": "D",
        "seed": 2026,
        "pool_sha256": "pool",
        "control_sha256": "control",
        "interface": "replace_text",
        "dataset_hash": "data",
        "episode": {},
        "model": {},
        "snapshot": "base",
        "task_ids": ["train"],
        "prompt_hash": "prompt",
        "scientific_code_hash": "code",
    }
    config = {
        "method": "GRPO",
        "objective": {"sd_lambda": 0.1},
        "sd_gap_definition": "same_scorer_v2",
        "episodes_per_task": 4,
        "iterations": 10,
        "parents_per_iteration": 2,
        "proposals_per_parent": 2,
        "gradient_diagnostics": True,
        "hardware": {},
        "sandbox": {},
    }
    for name, value in (
        ("manifest.json", manifest),
        ("resolved_config.json", config),
        ("complete.json", {"episodes": 80, "proposer_steps": 0, "student_steps": 5}),
    ):
        (tmp_path / name).write_text(json.dumps(value))
    pool = {"dev": {"purpose": "development", "task": {"parent_id": "parent"}}}
    events = []
    for k in range(1, 11):
        student = {
            "success_numerator": 3,
            "success_denominator": 8,
            "cumulative_generated_tokens": k * 10,
            "pre_update_logprob_abs_max": 0.1,
        }
        events.append(
            {
                "kind": "iteration_commit",
                "completed_iteration": k,
                "payload": {"metrics": {"student": student}},
            }
        )
    for episodes in (0, 40, 80):
        name = f"dev-{episodes}.json"
        (tmp_path / name).write_text(json.dumps([{"task_id": "dev", "reward": 1}]))
        events.append(
            {"kind": "task17_development", "payload": {"episodes": episodes, "path": name}}
        )
        for task in range(41):
            events.append(
                {
                    "kind": "evaluation",
                    "completed_iteration": episodes // 8,
                    "payload": {"split": "dev_monitor", "task_id": str(task), "success": True},
                }
            )
    monkeypatch.setattr(module, "active_events", lambda _: events)
    item = {"arm": "D", "seed": 2026, "run_dir": str(tmp_path), "allocated_gpu_hours": 1}
    control = {"calibration_coefficient": 0.03}
    summary, records, _, _ = module.load_run(item, pool, "pool", "control", control)
    assert summary["rollout_episodes"] == 80 and set(records) == {0, 40, 80}
    events = [
        e
        for e in events
        if not (e["kind"] == "task17_development" and e["payload"]["episodes"] == 80)
    ]
    monkeypatch.setattr(module, "active_events", lambda _: events)
    with pytest.raises(ValueError, match="lacks required"):
        module.load_run(item, pool, "pool", "control", control)
    config["method"] = "B"
    (tmp_path / "resolved_config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="incorrect objective"):
        module.load_run(item, pool, "pool", "control", control)
