"""Checkpoint, lineage and upload-boundary regressions for Task 10."""

import copy
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from selfplay_opsd.config import load_config
from selfplay_opsd.observability import WandbSink
from selfplay_opsd.storage import CheckpointManager, EventStore, active_events
from selfplay_opsd.telemetry import METHOD_NAMES, aggregate_metrics, canonical_snapshot


def fixture(tmp_path, method="B"):
    cfg = load_config(
        study_id="telemetry", method=method, hardware_profile="cpu_debug", output_root=tmp_path
    )
    store = EventStore(cfg.run_dir, {"logical_run_id": "logical"})
    return cfg, store


def commit(store, iteration, extra=None):
    store.iteration = iteration
    metrics = {
        "train": {"outer_iteration": iteration},
        "compute": {"active_seconds": iteration * 10, "allocated_gpu_hours": iteration / 10},
        "proposer": {
            "attempted_count": 16,
            "valid_count": 1,
            "duplicate_count": 0,
            "invalid_reasons": {"SECRET raw patch": 15},
        },
        "student": {
            "success_numerator": 1,
            "success_group_mix": {"1": 1},
            "repair_success_fraction": 0.25,
        },
        "alignment": {"available": True, "scored_count": 0},
    }
    if extra:
        metrics.update(extra)
    store.emit(
        "iteration_commit",
        {"iteration": iteration, "metrics": metrics, "attempted": 16, "valid": 1, "episodes": 4},
        committed=True,
    )


def checkpoint(store, iteration):
    CheckpointManager(store.root).save(
        iteration, {}, {"iteration": iteration, "segment_id": store.segment}
    )


def test_no_checkpoint_no_cloud_progress_and_resume_old_pointer(tmp_path):
    cfg, store = fixture(tmp_path)
    commit(store, 1)
    assert canonical_snapshot(store.root, cfg)["rows"] == []
    checkpoint(store, 1)
    assert len(canonical_snapshot(store.root, cfg)["rows"]) == 1
    commit(store, 2)
    assert len(canonical_snapshot(store.root, cfg)["rows"]) == 1
    checkpoint(store, 2)
    resumed = EventStore(store.root, store.identity, rollback_to=1)
    commit(resumed, 2)
    snap = canonical_snapshot(store.root, cfg)
    assert [r["train/outer_iteration"] for r in snap["rows"]] == [1]
    checkpoint(resumed, 2)
    snap2 = canonical_snapshot(store.root, cfg)
    assert [r["train/outer_iteration"] for r in snap2["rows"]] == [1, 2]
    assert snap2["rows"][-1]["run/segment_id"] == resumed.segment
    assert snap2["revision"] != snap["revision"]


def test_aggregate_allowlist_excludes_strings_arbitrary_keys_and_raw_tables(tmp_path):
    cfg, store = fixture(tmp_path, "A")
    commit(
        store,
        1,
        {
            "student": {
                "success_numerator": 1,
                "raw_text": "SECRET",
                "SECRET": 99,
                "loss": "SECRET",
                "grad_norm": 1.0,
            }
        },
    )
    checkpoint(store, 1)
    snap = canonical_snapshot(store.root, cfg)
    assert "SECRET" not in str(snap)
    assert snap["rows"][0]["alignment/available"] is False
    assert "alignment/scored_count" not in snap["rows"][0]
    assert aggregate_metrics({"student": {"grad_norm": float("nan")}}) == {}


def test_only_complete_development_evaluation_is_published(tmp_path):
    cfg, store = fixture(tmp_path)
    checkpoint(store, 0)
    for split in ("dev_monitor", "final_id"):
        store.emit(
            "evaluation",
            {
                "split": split,
                "iteration": 0,
                "parent_id": "private-id",
                "success": True,
                "expected": "SECRET",
            },
            committed=True,
        )
    assert not canonical_snapshot(store.root, cfg)["evaluations"]
    store.emit("evaluation_complete", {"split": "dev_monitor"}, committed=True)
    snap = canonical_snapshot(store.root, cfg)
    assert len(snap["evaluations"]) == 1
    assert snap["summary"]["eval/initial_dev_parent_macro"] == 1
    assert "private-id" not in str(snap)


class FakeRun:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.summary = {}
        self.logs = []
        self.url = "https://wandb.ai/authorized/project/logical"

    def define_metric(self, *args, **kwargs):
        pass

    def log(self, row):
        self.logs.append(copy.deepcopy(row))

    def finish(self):
        pass


def test_sink_groups_resume_retry_and_never_uploads_raw_content(tmp_path, monkeypatch):
    runs, calls = [], []

    def init(**kwargs):
        calls.append(kwargs)
        run = FakeRun(**kwargs)
        runs.append(run)
        return run

    sdk = SimpleNamespace(init=init, Settings=lambda **kw: kw, Table=lambda **kw: kw)
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    cfg, store = fixture(tmp_path)
    cfg.logging.mode = "online"
    cfg.logging.upload_checkpoints = True
    commit(store, 1)
    checkpoint(store, 1)
    sink = WandbSink(cfg, store)
    sink.table("private", ["raw_text"], [["SECRET"]])
    sink.finish()
    assert calls[0]["id"] == "logical"
    assert calls[0]["name"] == METHOD_NAMES["B"]
    assert calls[0]["group"] == "telemetry"
    assert calls[0]["resume"] == "allow"
    assert "SECRET" not in str(runs[0].logs)
    assert runs[0].logs[-1]["train/outer_iteration"] == 1
    resumed = EventStore(store.root, store.identity, rollback_to=1)
    again = WandbSink(cfg, resumed)
    again.finish()
    assert calls[1]["id"] == calls[0]["id"]
    assert any(e["kind"] == "sample_table" for e in active_events(store.root))


def test_sdk_exception_text_is_never_persisted(tmp_path, monkeypatch):
    def init(**kwargs):
        raise OSError("secret-key-and-private-context")

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init, Settings=lambda **kw: kw))
    cfg, store = fixture(tmp_path)
    sink = WandbSink(cfg, store)
    time.sleep(0.05)
    sink.finish()
    warnings = [e for e in active_events(store.root) if e["kind"] == "logging_warning"]
    assert warnings
    assert "secret-key" not in str(warnings)
    assert warnings[0]["payload"]["error_type"] == "OSError"


def test_real_offline_sdk_does_not_change_scientific_rng(tmp_path):
    cfg, store = fixture(tmp_path)
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    sink = WandbSink(cfg, store)
    sink.finish()
    after = random.getstate(), np.random.get_state(), torch.get_rng_state()
    assert before[0] == after[0]
    assert np.array_equal(before[1][1], after[1][1])
    assert before[1][2:] == after[1][2:]
    assert torch.equal(before[2], after[2])


def test_online_config_changes_no_science_except_study():
    old = load_config(__import__("pathlib").Path("configs/task09-diagnostics.yaml"))
    new = load_config(__import__("pathlib").Path("configs/task10-online.yaml"))
    assert old.scientific_settings(common=True, calibration=True) == new.scientific_settings(
        common=True, calibration=True
    )
    assert old.hardware == new.hardware
    assert old.logging.mode == "offline" and new.logging.mode == "online"


def test_actual_gpu_hour_axis_uses_slurm_not_trainer_estimate(tmp_path):
    import json

    cfg, store = fixture(tmp_path)
    sample_path = store.root / "state/allocation-samples.jsonl"
    sample_path.write_text(
        json.dumps({"timestamp": 1, "allocated_gpu_hours": 7.0, "job_id": "42"}) + "\n"
    )
    commit(store, 1)
    checkpoint(store, 1)
    row = canonical_snapshot(store.root, cfg)["rows"][0]
    assert row["compute/allocated_gpu_hours"] == 7.0
    assert row["compute/trainer_gpu_hours_estimate"] == 0.1
    assert row["run/slurm_job_id"] == "42"
