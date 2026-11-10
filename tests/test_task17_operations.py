"""Task 17 controls must preserve pairing, independent budgets and resume identity."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from selfplay_opsd.config import load_config
from selfplay_opsd.task17 import configure_operation, partition_parents
from selfplay_opsd.task17_analysis import compare_interfaces


def script(name):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_command_keeps_operation_arm_seed_and_resume():
    job = script("hpc_job")
    env = {
        "SELFPLAY_OPERATION": "task17-interface",
        "SELFPLAY_METHOD": "D",
        "SELFPLAY_SETTINGS": "configs/task17.yaml",
        "SELFPLAY_OUTPUT": "/runs/interface/id",
        "SELFPLAY_SEED": "2026",
    }
    command = job.task13_command_args(env)
    assert command[0] == "torchrun"
    assert command[command.index("--operation") + 1] == "interface"
    assert command[command.index("--arm") + 1] == "D"
    env.update(SELFPLAY_OPERATION="task17-prepare")
    assert job.task13_command_args(env)[0] == "python"
    env.update(SELFPLAY_OPERATION="task17-replay")
    replay = job.task13_command_args(env)
    assert replay[0] == "python" and "--candidates" in replay
    env.update(SELFPLAY_OPERATION="task17-student", SELFPLAY_RESUME="/runs/checkpoints/000004")
    assert job.task13_command_args(env)[-2:] == ["--resume", "/runs/checkpoints/000004"]


def test_budget_authorizations_do_not_touch_historical_readiness(tmp_path, monkeypatch):
    remote = script("hpc_remote")
    monkeypatch.setattr(remote, "workspace", lambda _: tmp_path)
    paths = []
    for operation, limit in (("task17-interface", 10), ("task17-student", 24)):
        payload = {"study_id": "task17", "operation": operation}
        paths.append(remote.budget_path(payload))
        budget = remote.refreshed_budget(payload)
        assert budget["limit"] == limit
        assert budget["committed_gpu_hours"] == 0
    assert len(set(paths)) == 2
    assert not any(path.name.startswith("budget-") for path in paths)
    assert remote.budget_path({"study_id": "old"}).name == "budget-readiness-cpsc4880.json"


def test_phase1_budget_reserves_and_enforces_ceiling(tmp_path, monkeypatch):
    remote = script("hpc_remote")
    monkeypatch.setattr(remote, "workspace", lambda _: tmp_path)
    payload = {
        "study_id": "task17",
        "operation": "task17-interface",
        "readiness_budget": 10,
        "run_id": "fixture",
        "gpu_count": 1,
        "time": "02:00:00",
    }
    for i in range(5):
        remote.reserve_budget(payload, str(i))
    with pytest.raises(ValueError, match="exhausted"):
        remote.reserve_budget(payload, "sixth")


def test_effective_objective_configs_are_distinct_and_common_within_seed(tmp_path):
    control = tmp_path / "control.json"
    control.write_text(
        json.dumps({"interface": "replace_text", "episodes": 128, "calibration_coefficient": 0.03})
    )
    configs = [
        configure_operation(load_config(Path("configs/task17.yaml")), "student", arm, control)
        for arm in "ABCD"
    ]
    assert [cfg.method for cfg in configs] == ["A", "B", "A", "GRPO"]
    assert [cfg.objective.sd_lambda for cfg in configs] == [0.1, 0.1, 0.03, 0.1]
    assert len({cfg.run_dir for cfg in configs}) == 4
    assert all(
        cfg.sd_gap_definition == "same_scorer_v2" and cfg.iterations == 16 for cfg in configs
    )
    with pytest.raises(ValueError, match="Freeze"):
        configure_operation(configs[0], "student", "A", tmp_path / "missing")


def test_disjoint_partition_is_deterministic(monkeypatch):
    import selfplay_opsd.task17 as module
    from selfplay_opsd.core import ParentProgram

    monkeypatch.setattr(
        module, "COUNTS", {"interface": 2, "curriculum": 2, "calibration": 1, "development": 1}
    )
    parents = [
        ParentProgram(
            parent_id=f"p{i}",
            family_id=f"f{i // 2}",
            source=f"def f(x):\n    return x + {i + 1}\n",
            specification="s",
            entry_point="f",
            dataset="mbpp",
            provenance={},
        )
        for i in range(16)
    ]
    parents.append(parents[0].model_copy(update={"parent_id": "Mbpp/590", "family_id": "excluded"}))
    repo = SimpleNamespace(parents=lambda split: parents)
    first = partition_parents(repo)
    parents.reverse()
    assert partition_parents(repo) == first
    assert len(set(p for rows in first.values() for p in rows)) == 6
    by_id = {p.parent_id: p for p in parents}
    assert len({by_id[p].family_id for rows in first.values() for p in rows}) == 6


def test_pairing_refuses_snapshot_budget_and_completion_drift():
    cells = []
    for arm in "ABCD":
        cells.append(
            {
                "arm": arm,
                "pool_sha256": "p",
                "dataset_hash": "d",
                "seed": 2026,
                "model": {},
                "snapshot": "base" if arm in "AB" else "b",
                "task_ids": ["t"],
                "prompt_hash": "h",
                "interface": "unified_diff" if arm in "AC" else "replace_text",
                "episode": {"max_turns": 8},
                "records": [
                    {
                        "task_id": "t",
                        "parent_id": "p",
                        "success": arm in "BD",
                        "applied_edit": True,
                        "presentation": "original",
                    }
                ],
            }
        )
    assert (
        compare_interfaces(cells, {"p": "f"})["comparisons"][0]["verified_repairs_original"][
            "second_only"
        ]
        == 1
    )
    cells[1]["snapshot"] = "different"
    with pytest.raises(ValueError, match="snapshot"):
        compare_interfaces(cells, {"p": "f"})


@pytest.mark.parametrize(
    "operation,count,budget,time_limit",
    [("task17-prepare", 0, 10, "01:00:00"), ("task17-interface", 1, 10, "02:00:00")],
)
def test_controller_binds_effective_interface_and_resources(
    monkeypatch, operation, count, budget, time_limit
):
    from selfplay_opsd import hpc

    class Remote:
        def call(self, name, *args):
            if name == "context":
                return {
                    "storage": {
                        "project": {"project_cpsc4880": "/project/cpsc4880"},
                        "scratch": {"scratch_cpsc4880": "/scratch/cpsc4880"},
                    },
                    "slurm": {
                        "associations": {"exit_code": 0, "stdout": "cpsc4880||normal|normal\n"}
                    },
                }
            assert name == "study"
            return {
                "images": {
                    "trainer": {"path": "/trainer.sif", "sha256": "a" * 64},
                    "worker": {"path": "/worker.sif", "sha256": "b" * 64},
                },
                "reports": {"worker": {}, "freeze": {}},
            }

    monkeypatch.setattr(
        hpc, "snapshot", lambda *a: {"commit": "a" * 40, "manifest_sha256": "c" * 64}
    )
    monkeypatch.setattr(
        "selfplay_opsd.readiness.execution_identity",
        lambda cfg: {"manifest_hash": "d" * 64, "interface": cfg.episode.edit_protocol},
    )
    result = hpc.run_payload(
        Remote(),
        "yale-hpc-cpsc4880",
        operation=operation,
        method="D",
        profile="single_gpu_80gb",
        gpu_type="h200",
        account="cpsc4880",
        image="",
        image_sha256="",
        settings=Path("configs/task17.yaml"),
        seed=2026,
    )
    assert result["gpu_count"] == count
    assert result["readiness_budget"] == budget
    assert result["time"] == time_limit
    assert result["runtime_identity"]["interface"] == (
        "replace_text" if operation == "task17-interface" else "unified_diff"
    )


def test_phase2_backend_uses_phase2_budget_without_operation_key(tmp_path, monkeypatch):
    remote = script("hpc_remote")
    monkeypatch.setattr(remote, "workspace", lambda _: tmp_path)
    phase2 = {"study_id": "repair-selfplay-2026-task17-objectives-B-v1"}
    assert remote.refreshed_budget(phase2)["limit"] == 24
    assert remote.budget_path(phase2).name == "task17-phase2-budget.json"
    phase1 = {"study_id": "repair-selfplay-2026-task17-interface-v1", "operation": "backend-check"}
    assert remote.refreshed_budget(phase1)["limit"] == 10
    assert remote.budget_path(phase1).name == "task17-phase1-budget.json"


def test_historical_checkpoint_verifier_rejects_wrong_prompt_before_import(monkeypatch):
    monkeypatch.syspath_prepend("scripts")
    runner = script("task17_experiment")
    cfg = load_config(Path("configs/task17.yaml"), hardware_profile="single_gpu_80gb")
    old = load_config(
        Path("configs/task15-v2.yaml"), method="B", seed=2026, hardware_profile="single_gpu_80gb"
    )
    assert old.scientific_hash == runner.HISTORICAL_B2026["scientific_hash"]
    metadata = runner.HISTORICAL_B2026 | {
        "resolved_config": {"model": cfg.model.model_dump(mode="json")},
        "prompt_hash": "wrong",
    }
    monkeypatch.setattr(runner.CheckpointManager, "load", lambda _: ({}, metadata))
    with pytest.raises(ValueError, match="provenance"):
        runner.load_historical_weights(
            None, cfg, SimpleNamespace(hash=metadata["manifests_hash"]), Path("unused")
        )
