import importlib.util
from pathlib import Path


def test_unwrap_requires_a_single_complete_json_fence(monkeypatch):
    script_dir = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(script_dir))
    spec = importlib.util.spec_from_file_location(
        "task13_fences", script_dir / "analyze_task13_fences.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.unwrap('```json\n{"site": 1}\n```') == '{"site": 1}'
    assert module.unwrap("Explanation\n```json\n{}\n```") is None
    assert module.unwrap("```json\n{}\n```\nextra") is None
