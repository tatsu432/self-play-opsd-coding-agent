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
    pilot_cfg = cfg.model_copy(deep=True)
    pilot_cfg.study_id += "-pilot-h200"
    monkeypatch.setenv("SELFPLAY_MAIN_STUDY", cfg.study_id)
    assert backend_evidence_path(pilot_cfg, "cpu_debug") == scoped / "cpu_debug.json"
    monkeypatch.delenv("SELFPLAY_MAIN_STUDY")
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


def test_evidence_reuse_rejects_inspection_only_accounts():
    import pytest

    from selfplay_opsd.hpc import HpcError, reuse_allocation_command, reuse_telemetry_command

    with pytest.raises(HpcError, match="authorized cpsc4880"):
        reuse_allocation_command("old", alias="yale-hpc-cpsc4710")
    with pytest.raises(HpcError, match="authorized cpsc4880"):
        reuse_telemetry_command("old", alias="yale-hpc-sds6190")


def test_only_explicit_empty_failed_preflight_pilot_can_be_superseded(tmp_path, monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "task12_supersede_remote", "scripts/hpc_remote.py"
    )
    remote = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(remote)
    common = {
        "operation": "pilot",
        "study_id": "study",
        "method": "D",
        "seed": 2026,
        "scientific_hash": "science",
        "manifests_hash": "manifest",
        "runtime_identity": {"code": "same"},
        "image_sha256": "image",
        "worker_sha256": "worker",
        "gpu_type": "h200",
        "profile": "single_gpu_80gb",
    }
    old = common | {"commit": "old"}
    prior = {"ledger": {"specification": old}, "attempts": [{"scheduler": {"state": "FAILED"}}]}
    owner = {"run_id": "old-run"}
    request = common | {
        "commit": "new",
        "storage_root": str(tmp_path),
        "supersede_empty_pilot": "old-run",
    }
    monkeypatch.setattr(
        remote,
        "logs",
        lambda _: {
            "tail": "Pilot preflight blocked: Backend acceptance proof stale or missing single-process gradient comparison"
        },
    )
    artifact = tmp_path / "pilot"
    assert remote.can_supersede_empty_pilot(request, prior, owner, artifact)
    assert not remote.can_supersede_empty_pilot(
        request | {"supersede_empty_pilot": None}, prior, owner, artifact
    )
    assert not remote.can_supersede_empty_pilot(
        request | {"scientific_hash": "changed"}, prior, owner, artifact
    )
    missing_identity = dict(request)
    missing_identity.pop("runtime_identity")
    assert not remote.can_supersede_empty_pilot(missing_identity, prior, owner, artifact)
    assert not remote.can_supersede_empty_pilot(request | {"commit": "old"}, prior, owner, artifact)
    prior["attempts"][0]["scheduler"]["state"] = "RUNNING"
    assert not remote.can_supersede_empty_pilot(request, prior, owner, artifact)
    prior["attempts"][0]["scheduler"]["state"] = "FAILED"
    artifact.mkdir()
    assert not remote.can_supersede_empty_pilot(request, prior, owner, artifact)
    artifact.rmdir()
    monkeypatch.setattr(remote, "logs", lambda _: {"tail": "other failure"})
    assert not remote.can_supersede_empty_pilot(request, prior, owner, artifact)


def test_pilot_preflight_uses_main_study_scoped_backend_proof(tmp_path, monkeypatch):
    from selfplay_opsd import preflight, readiness
    from selfplay_opsd.storage import atomic_json

    cfg = load_config(
        study_id="new-study",
        hardware_profile="single_gpu_80gb",
        data_root=tmp_path / "missing-data",
        output_root=tmp_path,
    )
    cfg.study_id += "-pilot-h200"
    monkeypatch.setenv("SELFPLAY_MAIN_STUDY", "new-study")
    monkeypatch.setenv("SELFPLAY_GPU_TYPE", "h200")
    monkeypatch.setattr(readiness, "backend_identity", lambda _: {"identity": "expected"})
    monkeypatch.setattr(readiness, "backend_environment", lambda **_: {"kernel": "test"})
    proof = tmp_path / "compatibility/backend-v2/studies/new-study/single_gpu_80gb.json"
    atomic_json(
        proof,
        {
            "ready": True,
            "backend_identity": {"identity": "expected"},
            "objective_probes": list("ABCD"),
            "model_revision": cfg.model.revision,
            "gpu_count": 1,
            "gpu_type": "h200",
            "environment": {"kernel": "test"},
        },
    )
    result = preflight.doctor(cfg)
    assert result["checks"]["backend_acceptance"] == {"valid": True, "proof": str(proof)}
    assert not any("Backend acceptance" in item for item in result["blockers"])
