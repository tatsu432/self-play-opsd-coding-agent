import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "analyze_task13", Path("scripts/analyze_task13_probe.py")
)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def make_probe(start, count, parents=("a", "b")):
    by_parent = {
        p: [
            {
                "parent_id": p,
                "slot": i,
                "seed": 10 + i,
                "valid": i % 2 == 0,
                "dedup_key": str(i),
                "target_ids": [i],
                "reason": "valid",
            }
            for i in range(start, start + count)
        ]
        for p in parents
    }
    return {
        "parents": {p: p + "-hash" for p in parents},
        "by_parent": by_parent,
        "rows": [r for values in by_parent.values() for r in values],
        "input_sha256": {},
        "manifest": {
            "slots": count,
            "slot_start": start,
            "model": {"name": "test"},
            "manifest_hash": "hash",
            "representation": "diff",
            "checkpoint": None,
        },
    }


def test_additional_samples_are_disjoint_and_prespecified():
    base, extra = make_probe(0, 4), make_probe(4, 4, ("a",))
    combined = module.combine_samples(base, extra)
    assert combined["manifest"]["slots"] == 8
    assert len(combined["by_parent"]["a"]) == 8
    with pytest.raises(ValueError, match="contiguous"):
        module.combine_samples(base, make_probe(3, 4, ("a",)))
    with pytest.raises(ValueError, match="prefix"):
        module.combine_samples(base, make_probe(4, 4, ("b",)))


def test_paired_comparison_requires_matching_seeds():
    first, second = make_probe(0, 4), make_probe(0, 4)
    second["by_parent"]["a"][0]["seed"] = 999
    with pytest.raises(ValueError, match="seeds"):
        module.compare(first, second)


def test_cross_model_comparison_requires_explicit_opt_in():
    first, second = make_probe(0, 4), make_probe(0, 4)
    second["manifest"]["model"] = {"name": "other"}
    with pytest.raises(ValueError, match="Model identity"):
        module.compare(first, second)
    assert module.compare(first, second, allow_model_difference=True)["parents"] == 2


def test_replay_is_checksum_bound_and_preserves_original(tmp_path):
    probe = make_probe(0, 2, ("a",))
    probe["input_sha256"] = {"proposals.json": "source-hash"}
    rows = [{"parent_id": "a", "slot": 1, "valid": True, "dedup_key": "recovered"}]
    complete = {
        "source_hashes": probe["input_sha256"],
        "count": 1,
        "valid": 1,
        "unique": 1,
        "change": "remove exactly one leading diff --git header",
    }
    (tmp_path / "results.json").write_text(json.dumps(rows))
    (tmp_path / "complete.json").write_text(json.dumps(complete))
    replay = module.apply_header_replay(probe, tmp_path)
    assert replay["by_parent"]["a"][1]["valid"]
    assert not probe["by_parent"]["a"][1]["valid"]
    complete["source_hashes"] = {"proposals.json": "different"}
    (tmp_path / "complete.json").write_text(json.dumps(complete))
    with pytest.raises(ValueError, match="identity"):
        module.apply_header_replay(probe, tmp_path)
