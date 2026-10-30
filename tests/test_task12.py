from pathlib import Path

from selfplay_opsd.config import load_config


def test_corrected_study_preserves_scientific_settings_and_binds_new_data():
    old = load_config(Path("configs/task10-online.yaml"))
    source = load_config(Path("configs/task12-source.yaml"))
    corrected = load_config(Path("configs/task12-corrected.yaml"))
    old_science = old.scientific_settings(common=True)
    new_science = source.scientific_settings(common=True)
    old_science.pop("study_id")
    new_science.pop("study_id")
    assert old_science == new_science
    assert source.study_id == corrected.study_id != old.study_id
    assert source.data_root == old.data_root == Path("data/protocol-v2")
    assert corrected.data_root == Path("data/protocol-v3")
    assert source.checkpoint_every == corrected.checkpoint_every == old.checkpoint_every


def test_telemetry_reuse_preserves_provenance_and_checks_dependencies(tmp_path, monkeypatch):
    import importlib.util
    import json

    import pytest

    from selfplay_opsd.storage import atomic_json, file_hash

    spec = importlib.util.spec_from_file_location("task12_remote", "scripts/hpc_remote.py")
    remote = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(remote)
    monkeypatch.setattr(remote, "workspace", lambda _: tmp_path)
    source = tmp_path / "runs/readiness/old/wandb-verification.json"
    atomic_json(
        source,
        {
            "ready": True,
            "cloud_verified": True,
            "sequence": 2,
            "entity": "entity",
            "telemetry_identity": "telemetry",
        },
    )
    atomic_json(
        tmp_path / "runs/old/study_frozen.json",
        {"telemetry_identity": "telemetry", "runtime_identity": {"trainer_sha256": "trainer"}},
    )
    request = {
        "source_study": "old",
        "study_id": "new",
        "entity": "entity",
        "telemetry_identity": "telemetry",
        "trainer_sha256": "trainer",
    }
    result = remote.reuse_telemetry(request)
    assert result["provenance"]["sha256"] == file_hash(source)
    assert json.loads(Path(result["evidence"]).read_text())["reuse_provenance"]["source"] == str(
        source
    )
    assert remote.reuse_telemetry(request) == result
    for field in ("entity", "telemetry_identity", "trainer_sha256"):
        with pytest.raises(ValueError, match="dependencies differ"):
            remote.reuse_telemetry(request | {field: "different"})
    with pytest.raises(ValueError, match="Distinct safe"):
        remote.reuse_telemetry(request | {"study_id": "../bad"})
    atomic_json(tmp_path / "runs/new/study_frozen.json", {"ready": True})
    with pytest.raises(ValueError, match="Target study is frozen"):
        remote.reuse_telemetry(request)
