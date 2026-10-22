import pytest

from selfplay_opsd.codec import encode
from selfplay_opsd.core import Case, ParentProgram
from selfplay_opsd.data import assign_splits, family_groups, public_arguments
from selfplay_opsd.evaluation import validate_data


@pytest.mark.sandbox
def test_actual_isolation_and_fresh_verifier(sandbox, repository):
    attestation = sandbox.preflight()
    assert attestation["ready"]
    report = validate_data(repository, sandbox)
    assert report["split_audit"] == "pass"
    source = "def f(x):\n    while True:\n        x += 1\n"
    result = sandbox.verify(source, "f", [Case(args=encode([1]), expected=encode(2))], 0.1)
    assert not result.passed and result.runtime_failures == 1
    # A fresh execution sees no previous episode's filesystem or module state.
    source = "state = []\ndef f(x):\n    state.append(x)\n    return len(state)\n"
    case = [Case(args=encode([1]), expected=encode(1))]
    assert sandbox.verify(source, "f", case, 2).passed
    assert sandbox.verify(source, "f", case, 2).passed


def test_example_extraction_and_family_partition():
    args = public_arguments(">>> f([1, 2], (3,4))\n5\nassert f([1,2],(3,4))==5\n", "f")
    assert args == [[[1, 2], (3, 4)]]
    a = ParentProgram(
        parent_id="a",
        source="def f(x):\n return x + 1\n",
        specification="x",
        entry_point="f",
        dataset="mbpp",
        provenance={},
    )
    b = a.model_copy(
        update={"parent_id": "b", "source": "def g(y):\n return y + 1\n", "entry_point": "g"}
    )
    h = a.model_copy(update={"parent_id": "human", "dataset": "humaneval"})
    families, decisions = family_groups([a, b, h])
    assert families["a"] == families["b"] == families["human"]
    roles = assign_splits([a, b, h], families, 2026)
    assert roles["a"] == roles["b"] == "excluded_external_overlap"
    assert roles["human"] == "external"


def test_target_validation_can_preserve_frozen_data(repository, sandbox, tmp_path):
    existing = repository.root / "validation_report.json"
    before = existing.read_bytes() if existing.exists() else None
    output = tmp_path / "readiness/validation.json"
    report = validate_data(repository, sandbox, execute=False, output=output)
    assert output.is_file() and report["manifest_hash"] == repository.hash
    assert (existing.read_bytes() if existing.exists() else None) == before
