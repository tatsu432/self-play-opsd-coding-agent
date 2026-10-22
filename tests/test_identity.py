"""Identity uses a validated repository without reopening an unrelated data path."""

import pytest

from selfplay_opsd.config import load_config
from selfplay_opsd.core import ExperimentError, digest
from selfplay_opsd.data import TaskRepository
from selfplay_opsd.readiness import backend_identity, execution_identity
from selfplay_opsd.storage import atomic_json, file_hash


def prepared_repository(root):
    atomic_json(root / "private/parents.json", [])
    atomic_json(root / "manifest.json", {"parents_hash": file_hash(root / "private/parents.json")})
    return TaskRepository(root)


def test_identity_reuses_validated_repository(tmp_path, monkeypatch):
    repository = prepared_repository(tmp_path / "data")
    cfg = load_config(
        study_id="identity-test", hardware_profile="cpu_debug", data_root=repository.root
    )
    monkeypatch.setattr(
        "selfplay_opsd.readiness.TaskRepository",
        lambda _: pytest.fail("Validated repository must not be reopened"),
    )
    assert execution_identity(cfg, repository=repository)["manifest_hash"] == digest(
        repository.manifest
    )
    cfg.data_root = tmp_path / "different"
    with pytest.raises(ExperimentError, match="Repository root"):
        execution_identity(cfg, repository=repository)


def test_identity_keeps_filesystem_validation_strict(tmp_path):
    cfg = load_config(study_id="identity-test", hardware_profile="cpu_debug", data_root=tmp_path)
    with pytest.raises(FileNotFoundError):
        execution_identity(cfg)
    prepared_repository(tmp_path)
    (tmp_path / "private/parents.json").write_text("corrupt")
    with pytest.raises(ExperimentError, match="checksum"):
        execution_identity(cfg)


def test_backend_identity_excludes_worker_data_and_hardware_count(tmp_path, monkeypatch):
    cfg = load_config(study_id="backend-identity", hardware_profile="cpu_debug")
    identity = backend_identity(cfg)
    cfg.data_root = tmp_path / "absent"
    cfg.sandbox.memory_mib *= 2
    monkeypatch.setenv("SELFPLAY_WORKER_SHA256", "a" * 64)
    cfg.hardware = load_config(study_id="other", hardware_profile="single_gpu_80gb").hardware
    assert backend_identity(cfg) == identity
    monkeypatch.setenv("SELFPLAY_IMAGE_DIGEST", "b" * 64)
    assert backend_identity(cfg) != identity
    monkeypatch.delenv("SELFPLAY_IMAGE_DIGEST")
    cfg.model.rank *= 2
    assert backend_identity(cfg) != identity
