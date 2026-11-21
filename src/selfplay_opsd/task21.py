"""Task 21 controls and bounded PDB feasibility and difficulty studies."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, model_validator

from selfplay_opsd.core import Case, OracleSpec, ParentProgram, TaskSpec, digest, read_json
from selfplay_opsd.data import TaskRepository
from selfplay_opsd.pdb_verifier import PROTOCOL, PdbEnvironment, PdbOracle
from selfplay_opsd.storage import file_hash
from selfplay_opsd.task19_rollout import Repair, repair
from selfplay_opsd.task20 import Task20Control, Task20Study

STUDY = "repair-selfplay-2026-task21-pdb-v1"


class Task21Control(Task20Control):
    study: str = STUDY
    unique_tasks: int = Field(default=4, ge=4, le=8)

    @model_validator(mode="after")
    def validate_study(self):
        if (
            not self.study.startswith("repair-selfplay-2026-task21-")
            or self.seed not in {2026, 2027, 2028}
            or self.precision != "float32"
            or not self.cached_rollout
            or self.episode.edit_protocol != "replace_text"
            or not self.qualification_hash
        ):
            raise ValueError("Task 21 requires its own identity and the qualified native FP32 path")
        if self.endpoint is not None and (
            not self.ready
            or not self.solver_gate_hash
            or not self.shared_gate_hash
            or not 512 <= self.endpoint <= 1024
            or self.endpoint % self.group_size
            or not self.dev_milestones
            or any(m % self.group_size or not 0 < m < self.endpoint for m in self.dev_milestones)
            or tuple(sorted(set(self.dev_milestones))) != self.dev_milestones
        ):
            raise ValueError("Freeze Task 21 evidence and the matched endpoint")
        return self


class PdbRepository(TaskRepository):
    def __init__(self, root: Path):
        self.root = root
        self.manifest = read_json(root / "manifest.json")
        if self.manifest.get("benchmark") != PROTOCOL or self.manifest.get(
            "bundle_sha256"
        ) != file_hash(root / "pdb.json"):
            raise ValueError("PDB bundle SHA-256 verification failed")
        self.hash = digest(self.manifest)
        payload = read_json(root / "pdb.json")
        identity = payload.pop("hash")
        if identity != digest(payload) or self.manifest["hash"] != identity:
            raise ValueError("PDB bundle content identity differs")
        self._programs = {
            key: ParentProgram.model_validate(value) for key, value in payload["parents"].items()
        }
        self._partitions = {
            split: [
                (TaskSpec.model_validate(row["task"]), PdbOracle.model_validate(row["oracle"]))
                for row in rows
            ]
            for split, rows in payload["partitions"].items()
        }
        self._oracles = {
            task.parent_id: oracle for rows in self._partitions.values() for task, oracle in rows
        }

    def parents(self, split: str) -> list[ParentProgram]:
        if split == "train":
            rows = (
                self._partitions.get("solver_train")
                or self._partitions.get("screen")
                or self._partitions.get("feasibility", [])
            )
            return [self._programs[key] for key in sorted({task.parent_id for task, _ in rows})]
        return [self._programs[task.parent_id] for task, _ in self._partitions.get(split, [])]

    def tasks(self, split: str) -> list[tuple[TaskSpec, OracleSpec]]:
        return [(task, oracle) for task, oracle in self._partitions.get(split, [])]

    def parent_oracle(self, parent_id: str) -> PdbOracle:
        return self._oracles[parent_id]

    def parent_cases(self, parent_id: str) -> tuple[list[Case], list[Case]]:
        oracle = self.parent_oracle(parent_id)
        return oracle.public_cases, oracle.hidden_cases


class Task21Study(Task20Study):
    cfg: Task21Control
    repository: PdbRepository

    def repair_episode(self, task: TaskSpec, oracle: OracleSpec, seed: int) -> Repair:
        return repair(
            self.policy, self.sandbox, self.cfg, task, oracle, seed, environment=PdbEnvironment
        )
