import pytest
import torch

from selfplay_opsd.analysis import analyze, diagnostic_statistics, paired_bootstrap
from selfplay_opsd.storage import (
    CheckpointManager,
    EventStore,
    active_events,
    restore_rng,
    rng_state,
)


def test_atomic_checkpoints_corruption_and_previous_preserved(tmp_path, monkeypatch):
    manager = CheckpointManager(tmp_path)
    first = manager.save(0, {"x": torch.arange(3)}, {"iteration": 0})
    original = manager.verify

    def fail(path):
        raise OSError("injected before checkpoint commit")

    monkeypatch.setattr(manager, "verify", fail)
    with pytest.raises(OSError):
        manager.save(1, {"x": torch.arange(4)}, {"iteration": 1})
    assert first.exists() and not (manager.root / "000001").exists()
    monkeypatch.setattr(manager, "verify", original)
    state, _ = manager.load(first)
    assert torch.equal(state["x"], torch.arange(3))
    (first / "state.pt").write_bytes(b"corrupted")
    with pytest.raises(RuntimeError, match="checksum"):
        manager.load(first)


def test_rollback_supersedes_multiple_iterations_but_keeps_compute(tmp_path):
    store = EventStore(tmp_path, {"logical_run_id": "x"})
    for i in range(1, 5):
        store.iteration = i
        store.emit("iteration_commit", {"version": "old"}, committed=True)
        store.emit("phase_time", {"seconds": 1.0}, committed=True)
    resumed = EventStore(tmp_path, {"logical_run_id": "x"}, rollback_to=1)
    resumed.iteration = 2
    resumed.emit("iteration_commit", {"version": "new"}, committed=True)
    rows = active_events(tmp_path)
    assert [
        (r["completed_iteration"], r["payload"]["version"])
        for r in rows
        if r["kind"] == "iteration_commit"
    ] == [(1, "old"), (2, "new")]
    assert sum(r["payload"]["seconds"] for r in rows if r["kind"] == "phase_time") == 4


def test_rng_restore():
    import random

    import numpy as np

    saved = rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(2))
    restore_rng(saved)
    assert random.random() == expected[0] and np.random.rand() == expected[1]
    assert torch.equal(torch.rand(2), expected[2])


def test_parent_paired_bootstrap_and_incomplete_denominators(tmp_path):
    first = [{"task_id": str(i), "parent_id": str(i // 2), "success": False} for i in range(6)]
    second = [r | {"success": True} for r in first]
    result = paired_bootstrap(first, second)
    assert result["difference"] == result["ci_low"] == result["ci_high"] == 1.0
    assert result["parents"] == 3 and result["tasks"] == 6
    assert paired_bootstrap(first, second[:-1])["status"] == "incomplete_or_unpaired"
    assert diagnostic_statistics([0.5] * 4, [1, 2, 3, 4])["spearman"] is None
    study = tmp_path / "study"
    study.mkdir()
    result = analyze(study)
    assert len(result["summary"]) == 10 and all(r["status"] == "unrun" for r in result["summary"])
