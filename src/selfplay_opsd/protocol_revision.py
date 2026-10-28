"""Immutable prompt-only revisions; never reconstruct or select benchmark tasks."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from selfplay_opsd.core import ExperimentError
from selfplay_opsd.data import TaskRepository
from selfplay_opsd.evaluation import finalize_data, validate_data
from selfplay_opsd.protocol import PROMPT_HASH, VERSION
from selfplay_opsd.storage import atomic_json, file_hash


def revise_protocol(source: Path, destination: Path, policy, sandbox, limits) -> dict:
    original = TaskRepository(source)
    if not original.manifest.get("ready") or destination.exists():
        raise ExperimentError("Revision requires finalized source and nonexistent destination")
    staging = destination.with_name(destination.name + ".staging")
    if staging.exists():
        raise ExperimentError(f"Interrupted revision exists: {staging}; inspect before retrying")
    staging.mkdir(parents=True, exist_ok=False)
    validate_data(original, sandbox, execute=False, output=staging / "source-audit.json")
    preserved = [
        "private/parents.json",
        "coverage.json",
        *["private/" + name for name in original.manifest["task_hashes"]],
    ]
    if (source / "inventory.json").exists():
        preserved.append("inventory.json")
    before = {name: file_hash(source / name) for name in preserved}
    # Copy all provenance, notices and inventory, excluding stale derived evidence.
    shutil.copytree(source, staging, dirs_exist_ok=True)
    for name in ("validation_report.json", "manifest.json"):
        (staging / name).unlink(missing_ok=True)
    for path in (staging / "private").glob("anchors-*.json"):
        path.unlink()
    lineage = {
        "source_manifest_sha256": file_hash(source / "manifest.json"),
        "source_manifest_hash": original.hash,
        "source_protocol": original.manifest.get("protocol"),
        "source_prompt_hash": original.manifest.get("prompt_hash"),
        "active_protocol": VERSION,
        "active_prompt_hash": PROMPT_HASH,
        "preserved_files": before,
        "task_protocol_metadata": "historical-creation-identity",
    }
    manifest = {
        k: v
        for k, v in original.manifest.items()
        if k not in {"anchor_hashes", "admitted_tasks", "prompt_hash"}
    }
    atomic_json(
        staging / "manifest.json",
        manifest
        | {
            "ready": False,
            "protocol": VERSION,
            "protocol_revision": lineage,
        },
    )
    try:
        finalize_data(TaskRepository(staging), policy, sandbox, limits)
        revised = TaskRepository(staging)
        validate_data(revised, sandbox, output=staging / "validation_report.json")
        after = {name: file_hash(staging / name) for name in preserved}
        if before != after or before != {name: file_hash(source / name) for name in preserved}:
            raise ExperimentError("Protocol revision changed frozen content")
        if file_hash(source / "manifest.json") != lineage["source_manifest_sha256"]:
            raise ExperimentError("Source manifest changed during revision")
        report = lineage | {"ready": True, "revised_manifest_hash": revised.hash}
        atomic_json(staging / "revision_report.json", report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise ExperimentError("Revision destination appeared during staging")
        os.rename(staging, destination)
        return report
    except Exception as exc:
        atomic_json(staging / "revision_failure.json", {"ready": False, "error": str(exc)})
        raise
