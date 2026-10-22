"""Stable identities, private/public boundaries and explicit experiment failures."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def canonical(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def keyed_seed(seed: int, *keys: object) -> int:
    return int(digest([seed, *keys])[:15], 16)


class Status(StrEnum):
    OK = "ok"
    CANDIDATE_FAILURE = "candidate_verification_failure"
    STUDENT_FAILURE = "student_failure"
    BUDGET = "agent_budget_timeout"
    INFRA = "sandbox_infrastructure_error"
    DEPENDENCY = "missing_dependency"
    ORACLE = "data_oracle_failure"


class ExperimentError(RuntimeError):
    def __init__(self, message: str, status: Status = Status.INFRA):
        super().__init__(message)
        self.status = status


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    schema_version: int = 1


class ParentProgram(Record):
    parent_id: str
    source: str
    specification: str
    entry_point: str
    dataset: str
    provenance: dict[str, Any]
    family_id: str = ""


class Case(Record):
    """Arguments/results use the safe tagged JSON codec, never pickle."""

    args: Any
    expected: Any
    comparison: str = "equal"
    atol: float = 0.0


class TaskSpec(Record):
    task_id: str
    parent_id: str
    split: str
    specification: str
    initial_source: str
    entry_point: str
    public_tests: str
    protocol_version: str = "repair-json-v1"
    source_hash: str
    mutation_seed: int
    public_seconds: float = 10.0
    verifier_seconds: float = 30.0

    def student_files(self) -> dict[str, str]:
        return {
            "README.md": self.specification,
            "src/solution.py": self.initial_source,
            "tests/test_public.py": self.public_tests,
        }


class OracleSpec(Record):
    task_id: str
    correct_source: str = Field(repr=False)
    inverse_patch: str = Field(repr=False)
    public_cases: list[Case] = Field(repr=False)
    hidden_cases: list[Case] = Field(repr=False)
    provenance: dict[str, Any]


class TokenTurn(Record):
    """Prompt and action IDs are immutable at sampling time, including sampled EOS."""

    prompt_ids: list[int]
    target_ids: list[int]
    old_logps: list[float]
    history: list[dict[str, str]]
    raw_text: str
    action: str
    parse_status: str
    teacher_logps: list[float] = []
    base_logps: list[float] = []


class TokenTargetMap(Record):
    target_ids: list[int]
    student_positions: list[int]
    teacher_positions: list[int]


class EpisodeRecord(Record):
    episode_id: str
    task_id: str
    seed: int
    policy_version: str
    turns: list[TokenTurn]
    token_target_maps: list[TokenTargetMap] = []
    reward: float
    public_pass: bool
    hidden_pass: bool
    termination: str
    status: Status
    verifier_status: Status = Status.OK
    semantic_failures: int = 0
    runtime_failures: int = 0
    final_source: str
    final_source_hash: str
    test_calls: int
    seconds: float
    timings: dict[str, float] = {}


class ProposalRecord(Record):
    proposal_id: str
    parent_id: str
    slot: int
    seed: int
    policy_version: str
    turn: TokenTurn
    valid: bool
    reason: str
    task_id: str | None = None
    dedup_key: str | None = None
    changed_nodes: int = 0
    changed_lines: int = 0
    bug_family: str = "unknown"


class IterationRecord(Record):
    iteration: int
    snapshot: str
    proposer_snapshot: str
    attempted: int
    valid: int
    episodes: int
    student_steps: int
    proposer_steps: int
    metrics: dict[str, Any]


class EvaluationRecord(Record):
    checkpoint: str
    iteration: int
    endpoint: str = "primary"
    split: str
    task_id: str
    parent_id: str
    success: bool | None
    status: Status
    episode_path: str | None
    metrics: dict[str, Any]


class CheckpointMetadata(Record):
    runtime_identity: dict[str, Any] | None = None
    iteration: int
    scientific_hash: str
    manifests_hash: str
    parameter_names: dict[str, list[str]]
    student_version: str
    proposer_version: str
    logical_run_id: str
    segment_id: str
    prompt_hash: str
    code_hash: str
    scientific_code_hash: str
    world_size: int
    artifact_kind: str = "fully_resumable"
    best_development: tuple[float, int] | None = None
    resolved_config: dict[str, Any]
    reference_hash: str


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())
