import pytest

from selfplay_opsd.data import construct_oracle, download_inventory
from selfplay_opsd.mutations import fixed_mutation


@pytest.mark.download
@pytest.mark.sandbox
def test_pinned_actual_data_and_oracle(tmp_path, sandbox):
    rows = download_inventory(tmp_path)
    assert sum(r["parent"]["dataset"] == "mbpp" for r in rows) == 378
    assert sum(r["parent"]["dataset"] == "humaneval" for r in rows) == 164
    for parent_id in ("Mbpp/17", "HumanEval/0"):
        raw = next(r for r in rows if r["parent"]["parent_id"] == parent_id)
        parent, public, hidden = construct_oracle(raw, sandbox, tmp_path / "cache")
        assert public and hidden
        assert sandbox.verify(parent.source, parent.entry_point, public + hidden, 30).passed
        mutation = fixed_mutation(parent.source, 2026)
        result = sandbox.verify(mutation.source, parent.entry_point, public + hidden, 30)
        # A fixed operator need not produce a valid task on every attempt; the
        # bounded builder uses verifier behavior to accept/reject it.
        assert len(result.details) == len(public) + len(hidden)
