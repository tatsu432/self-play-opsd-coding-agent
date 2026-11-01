import importlib.util
from pathlib import Path


def test_mixed_group_comparison_uses_shared_parents():
    spec = importlib.util.spec_from_file_location(
        "proposal_students", Path("scripts/analyze_task14_proposal_students.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def group(parent, rewards):
        return {"parent_id": parent, "episodes": [{"reward": r} for r in rewards]}

    first = [group("a", [0, 0, 0, 0]), group("b", [0, 1, 0, 1])]
    second = [group("a", [0, 1, 0, 1]), group("c", [0, 0, 0, 0])]
    assert module.summarize(first)["mixed_fraction"] == 0.5
    result = module.compare(first, second)
    assert result["shared_parents"] == 1 and result["mean_difference"] == 1.0
