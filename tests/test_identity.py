"""Identity uses a validated repository without reopening an unrelated data path."""

import pytest

from selfplay_opsd.config import load_config
from selfplay_opsd.core import ExperimentError, digest
from selfplay_opsd.data import TaskRepository
from selfplay_opsd.readiness import execution_identity
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
