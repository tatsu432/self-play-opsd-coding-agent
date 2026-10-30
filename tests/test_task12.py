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


def test_study_scoped_backend_proof_keeps_original_unchanged(tmp_path, monkeypatch):
    import importlib.util

    from selfplay_opsd.backend_check import check_backend
    from selfplay_opsd.readiness import backend_evidence_path
    from selfplay_opsd.storage import atomic_json, read_json

    cfg = load_config(
        study_id="new-study",
        hardware_profile="cpu_debug",
        data_root=tmp_path / "unused",
        output_root=tmp_path,
    )
    legacy = tmp_path / "compatibility/backend-v2/cpu_debug.json"
    atomic_json(legacy, {"historic": True})
    scoped = tmp_path / "compatibility/backend-v2/studies/new-study"
    assert backend_evidence_path(cfg, "cpu_debug") == legacy
    check_backend(cfg, scoped / "cpu_debug")
    assert read_json(legacy) == {"historic": True}
    assert backend_evidence_path(cfg, "cpu_debug") == scoped / "cpu_debug.json"
    assert read_json(scoped / "cpu_debug.json")["ready"] is True
    spec = importlib.util.spec_from_file_location("task12_remote_paths", "scripts/hpc_remote.py")
    remote = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(remote)
    monkeypatch.setattr(remote, "paths", lambda _: {"runs": tmp_path})
    payload = {
        "study_id": "new-study",
        "operation": "backend-check",
        "profile": "cpu_debug",
        "backend_identity": {"schema_version": 2},
        "output": "/runs/compatibility/backend-v2/studies/new-study/cpu_debug",
    }
    assert remote.artifact_dir(payload) == scoped / "cpu_debug"


def test_allocation_reuse_requires_matching_image_device_and_verifier(tmp_path, monkeypatch):
    import importlib.util
    import json

    import pytest

    from selfplay_opsd.storage import atomic_json, file_hash

    spec = importlib.util.spec_from_file_location(
        "task12_allocation_remote", "scripts/hpc_remote.py"
    )
    remote = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(remote)
    monkeypatch.setattr(remote, "workspace", lambda _: tmp_path)
    source = tmp_path / "runs/readiness/old/allocation-single_gpu_80gb.json"
    atomic_json(
        source,
        {
            "ready": True,
            "ownership_verified": True,
            "image_sha256": "image",
            "allocation_code_hash": "code",
            "gpu_type": "h200",
            "gpu_count": 1,
        },
    )
    request = {
        "source_study": "old",
        "study_id": "new",
        "profile": "single_gpu_80gb",
        "trainer_sha256": "image",
        "allocation_code_hash": "code",
        "gpu_type": "h200",
        "gpu_count": 1,
    }
    result = remote.reuse_allocation(request)
    assert result["provenance"]["sha256"] == file_hash(source)
    assert json.loads(Path(result["evidence"]).read_text())["ownership_verified"]
    assert remote.reuse_allocation(request) == result
    for field in ("trainer_sha256", "allocation_code_hash", "gpu_type", "gpu_count"):
        with pytest.raises(ValueError, match="dependencies differ"):
            remote.reuse_allocation(request | {field: "changed"})
    atomic_json(tmp_path / "runs/new/study_frozen.json", {"ready": True})
    with pytest.raises(ValueError, match="Target study is frozen"):
        remote.reuse_allocation(request)
