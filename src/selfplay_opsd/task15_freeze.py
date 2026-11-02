"""Task 15's narrow scientific freeze; no historical D-pilot prerequisite."""

from __future__ import annotations

from pathlib import Path

from selfplay_opsd.core import ExperimentError, read_json
from selfplay_opsd.storage import atomic_json, file_hash, scientific_code_hash

REQUIRED_EVIDENCE = {"runtime", "backend", "numerical", "resume", "stabilization"}


def freeze(config, repository, decision_path: Path, output: Path, commit: str):
    decision = read_json(decision_path)
    if config.online_supply is None or decision.get("ready") is not True:
        raise ExperimentError("Task 15 stabilization has not selected a valid runnable route")
    if not decision.get("causal_decisions") or not decision.get("resource_plan"):
        raise ExperimentError(
            "Freeze requires causal decisions and measured six-arm resource planning"
        )
    evidence = []
    for item in decision["evidence"]:
        path = Path(item["path"])
        report = read_json(path)
        kind = item["kind"]
        passed = (
            report.get("ready") is True
            or report.get("available") is True
            or report.get("passed") is True
        )
        if kind == "stabilization":
            passed = report.get("episodes", 0) >= report.get("target_episodes", float("inf"))
        if not passed:
            raise ExperimentError(f"Incomplete Task 15 evidence: {kind}")
        evidence.append({"kind": kind, "path": str(path), "sha256": file_hash(path)})
    if not REQUIRED_EVIDENCE <= {e["kind"] for e in evidence}:
        raise ExperimentError("Task 15 freeze evidence is incomplete")
    from selfplay_opsd.protocol import PROMPT_HASH

    common = {}
    for seed in (2026, 2027, 2028):
        cfg = config.model_copy(update={"seed": seed}, deep=True)
        common[str(seed)] = cfg.scientific_settings(common=True)
    record = {
        "status": "frozen",
        "study_id": config.study_id,
        "execution_commit": commit,
        "scientific_code_hash": scientific_code_hash(),
        "manifest_hash": repository.hash,
        "prompt_hash": PROMPT_HASH,
        "common_configs": common,
        "evidence": evidence,
        "decision_sha256": file_hash(decision_path),
        "causal_decisions": decision["causal_decisions"],
        "resource_plan": decision["resource_plan"],
        "primary_endpoint": "final_common_exposure",
        "secondary_endpoint": None,
        "seeds": [2026, 2027, 2028],
        "analysis": {
            "bootstrap_resamples": 2000,
            "analysis_seed": 2026,
            "task_uncertainty": "paired_family_parent_clustered",
            "seed_uncertainty": "separate_paired_seed_differences",
        },
    }
    if output.exists() and read_json(output) != record:
        raise ExperimentError("Cannot overwrite an existing scientific freeze")
    atomic_json(output, record)
    return record


def check_freeze(config, repository, path: Path | None = None):
    from selfplay_opsd.protocol import PROMPT_HASH

    record = read_json(path or config.output_root / config.study_id / "freeze.json")
    if (
        record.get("status") != "frozen"
        or record.get("primary_endpoint") != "final_common_exposure"
        or record.get("common_configs", {}).get(str(config.seed))
        != config.scientific_settings(common=True)
        or record.get("scientific_code_hash") != scientific_code_hash()
        or record.get("manifest_hash") != repository.hash
        or record.get("prompt_hash") != PROMPT_HASH
        or not REQUIRED_EVIDENCE <= {e["kind"] for e in record["evidence"]}
    ):
        raise ExperimentError("Task 15 freeze differs from active science/data/prompts")
    for evidence in record["evidence"]:
        if file_hash(Path(evidence["path"])) != evidence["sha256"]:
            raise ExperimentError("Task 15 freeze evidence changed")
    return record
