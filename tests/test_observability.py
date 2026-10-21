import os

import pytest

from selfplay_opsd.config import load_config
from selfplay_opsd.observability import WandbSink
from selfplay_opsd.storage import EventStore, active_events


def test_disabled_logging_has_no_network_or_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    cfg = load_config(study_id="log", hardware_profile="cpu_debug", output_root=tmp_path)
    cfg.logging.mode = "disabled"
    store = EventStore(cfg.run_dir, {"logical_run_id": "log"})
    sink = WandbSink(cfg, store)
    sink.log({"train": {"outer_iteration": 1}})
    sink.finish()
    assert sink.thread is None


def test_offline_wandb_actual_sdk(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    cfg = load_config(study_id="offline", hardware_profile="cpu_debug", output_root=tmp_path)
    cfg.logging.mode = "offline"
    store = EventStore(cfg.run_dir, {"logical_run_id": "offline"})
    sink = WandbSink(cfg, store)
    sink.log(
        {
            "train": {"outer_iteration": 1},
            "student": {"repair_success_fraction": 0.5},
            "compute": {"allocated_gpu_hours": 0.0},
        }
    )
    sink.finish()
    assert list(cfg.run_dir.glob("wandb/offline-run-*/*.wandb"))


def test_wandb_failure_preserves_local_events(tmp_path, monkeypatch):
    import wandb

    def unavailable(**kwargs):
        raise OSError("injected outage")

    monkeypatch.setattr(wandb, "init", unavailable)
    cfg = load_config(study_id="outage", hardware_profile="cpu_debug", output_root=tmp_path)
    store = EventStore(cfg.run_dir, {"logical_run_id": "outage"})
    store.emit("iteration_commit", {"x": 1}, committed=True)
    sink = WandbSink(cfg, store)
    sink.thread.join(timeout=5)
    sink.log({"student": {"loss": 1.0}})
    sink.finish()
    assert any(e["kind"] == "iteration_commit" for e in active_events(cfg.run_dir))
    assert any(e["kind"] == "logging_warning" for e in active_events(cfg.run_dir))


@pytest.mark.wandb_online
@pytest.mark.skipif(
    not os.getenv("WANDB_API_KEY"), reason="No online W&B credentials in environment"
)
def test_online_wandb_smoke(tmp_path):
    cfg = load_config(
        study_id="integration-smoke", hardware_profile="cpu_debug", output_root=tmp_path
    )
    cfg.logging.mode = "online"
    store = EventStore(cfg.run_dir, {"logical_run_id": "integration-smoke"})
    sink = WandbSink(cfg, store)
    sink.log({"train": {"outer_iteration": 0}, "integration_smoke": True})
    sink.finish()
    assert not any(e["kind"] == "logging_warning" for e in active_events(cfg.run_dir))
