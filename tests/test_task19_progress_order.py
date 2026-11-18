import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "task19_progress_order", Path(__file__).parents[1] / "scripts/task19_progress_order.py"
)
assert spec and spec.loader
order = importlib.util.module_from_spec(spec)
spec.loader.exec_module(order)


def fixture():
    return {
        "completed_waves": 0,
        "groups": [[{"task": "z"}, {"task": "a"}]],
        "tasks": {"a": {"task": 2}, "z": {"task": 1}},
        "repairs": {"z": [1, 2, 3, 4]},
        "identity": {"id": "bound"},
    }


def test_reconstructs_task_order_without_changing_data():
    original = fixture()
    result = order.reordered(original)
    assert result == original
    assert list(result["tasks"]) == ["z", "a"]
    assert list(result["repairs"]) == ["z"]
    original["completed_waves"] = 1
    with pytest.raises(ValueError, match="first wave"):
        order.reordered(original)


def test_rewrite_requires_terminal_matching_job_and_archives_original(tmp_path, monkeypatch):
    run = tmp_path / "run"
    (run / "state").mkdir(parents=True)
    progress = run / "qualification-progress.json"
    raw = json.dumps(fixture(), sort_keys=True).encode()
    progress.write_bytes(raw)
    checksum = hashlib.sha256(raw).hexdigest()
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"run_dir": str(run), "attempts": [{"job_id": "123"}]}))
    (run / "state/health.json").write_text(
        json.dumps({"allocation": "123", "phase": "stopped", "identity": {"id": "bound"}})
    )
    args = argparse.Namespace(
        progress=progress, ledger=ledger, job_id="123", expected_sha256=checksum
    )
    monkeypatch.setattr(
        order.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="123|RUNNING\n")
    )
    with pytest.raises(ValueError, match="terminal"):
        order.rewrite_stopped(args)
    assert progress.read_bytes() == raw
    monkeypatch.setattr(
        order.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="123|TIMEOUT\n")
    )
    order.rewrite_stopped(args)
    assert json.loads(progress.read_bytes()) == json.loads(raw)
    assert list(json.loads(progress.read_bytes())["tasks"]) == ["z", "a"]
    assert (run / f"qualification-progress-before-order-{checksum}.json").read_bytes() == raw
