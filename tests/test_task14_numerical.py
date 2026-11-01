import importlib.util
from pathlib import Path
from types import SimpleNamespace


def test_sign_sensitive_clipping_counts_only_boundaries_that_change_loss(monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    spec = importlib.util.spec_from_file_location("task14_numerical", "scripts/task14_numerical.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = [
        SimpleNamespace(task_id="t", reward=r, turns=[SimpleNamespace(old_logps=[0.0, 0.0])])
        for r in (1.0, 0.0)
    ]
    rescored = [SimpleNamespace(turns=[SimpleNamespace(old_logps=[0.5, -0.5])]) for _ in original]
    result = module.clipping_impact(original, rescored, 0.2)
    assert result == {"eligible_tokens": 4, "clipped_tokens": 2, "clipped_fraction": 0.5}
    assert module.outlier_logps([0.0, 0.0], [0.1, 0.5]) == [0.0, 0.5]
    manifest = {"tasks": [{"task_id": str(i)} for i in range(10)]}
    records = [
        {"task_id": str(i), "reward": r}
        for i in range(10)
        for r in ((0, 1) if i in {8, 9} else (0, 0))
    ]
    assert module.select_tasks(manifest, records, True)[:2] == ["8", "9"]
    assert module.select_tasks(manifest, records, False) == [str(i) for i in range(8)]
