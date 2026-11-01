import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_b200_requires_exact_historical_identity(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "task14_checkpoint", Path("scripts/task14_checkpoint.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = {"name": "fixture"}
    metadata = module.EXPECTED | {"resolved_config": {"model": model}}
    state = {"policy": "fixture"}
    monkeypatch.setattr(module.CheckpointManager, "load", lambda path: (state, metadata))
    monkeypatch.setattr(module, "file_hash", lambda path: "marker")
    cfg = SimpleNamespace(model=SimpleNamespace(model_dump=lambda **kwargs: model))
    repo = SimpleNamespace(hash=module.EXPECTED["manifests_hash"])
    assert module.load_b200(cfg, repo) == (state, metadata, "marker")
    metadata["prompt_hash"] = "wrong"
    with pytest.raises(ValueError, match="identity"):
        module.load_b200(cfg, repo)
