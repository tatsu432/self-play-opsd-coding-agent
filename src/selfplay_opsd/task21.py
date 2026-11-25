"""Task 21 controls and bounded PDB feasibility and difficulty studies."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from selfplay_opsd.core import Case, OracleSpec, ParentProgram, TaskSpec, digest, read_json
from selfplay_opsd.data import TaskRepository
from selfplay_opsd.objectives import difficulty
from selfplay_opsd.pdb_proposer import proposal_task, propose_pdb
from selfplay_opsd.pdb_verifier import PROTOCOL, PdbEnvironment, PdbOracle, verify_pdb
from selfplay_opsd.protocol import teacher_messages
from selfplay_opsd.storage import file_hash
from selfplay_opsd.task19_protocol import STUDENT_TOOLS, student_history
from selfplay_opsd.task19_rollout import Proposal, Repair, repair
from selfplay_opsd.task19_train import Grads, SegmentStop, add_gradient, norm
from selfplay_opsd.task20 import Task20Control, Task20Study
from selfplay_opsd.task22_contract import PREFIX

STUDY = "repair-selfplay-2026-task21-pdb-v1"


class Task21Control(Task20Control):
    study: str = STUDY
    unique_tasks: Literal[4, 8] = 4
    proposer_design: Literal["constrained", "flexible"] = "constrained"
    main_mode: Literal["shared", "fixed"] = "shared"
    calibrated_ungated_scale: float | None = Field(default=None, gt=0)
    alignment_gate_hash: str | None = None

    @property
    def ready(self) -> bool:
        return (
            self.calibration_hash is not None
            and self.qualification_hash is not None
            and self.sd_scale is not None
            and (self.main_mode == "fixed" or self.proposer_scale is not None)
        )

    @model_validator(mode="after")
    def validate_study(self):
        if (
            not self.study.startswith(("repair-selfplay-2026-task21-", PREFIX))
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
            or self.endpoint % (4 * self.group_size)
            or len(self.dev_milestones) != 3
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
        self.provenance = payload["provenance"]
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
        self._tasks = {
            task.parent_id: task for rows in self._partitions.values() for task, _ in rows
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
            extra = self._partitions.get("proposer_train", [])
            return [
                self._programs[key]
                for key in sorted({task.parent_id for task, _ in [*rows, *extra]})
            ]
        return [self._programs[task.parent_id] for task, _ in self._partitions.get(split, [])]

    def tasks(self, split: str) -> list[tuple[TaskSpec, OracleSpec]]:
        return [(task, oracle) for task, oracle in self._partitions.get(split, [])]

    def parent_task(self, parent_id: str) -> TaskSpec:
        return self._tasks[parent_id]

    def partition_names(self) -> list[str]:
        return sorted(self._partitions)

    def parent_oracle(self, parent_id: str) -> PdbOracle:
        return self._oracles[parent_id]

    def parent_cases(self, parent_id: str) -> tuple[list[Case], list[Case]]:
        oracle = self.parent_oracle(parent_id)
        return oracle.public_cases, oracle.hidden_cases


class Task21Study(Task20Study):
    cfg: Task21Control
    repository: PdbRepository
    allow_fixed_auxiliary = True

    @property
    def identity(self) -> dict:
        return super().identity | {"proposer_design": self.cfg.proposer_design}

    def loss_scales(self) -> tuple[float, float]:
        sd, proposer = super().loss_scales()
        if self.arm == "A" and self.cfg.calibrated_ungated_scale is not None:
            sd = self.cfg.calibrated_ungated_scale
        return sd, proposer

    def repair_episode(self, task: TaskSpec, oracle: OracleSpec, seed: int) -> Repair:
        return repair(
            self.policy, self.sandbox, self.cfg, task, oracle, seed, environment=PdbEnvironment
        )

    def sampled_record(self, episode: Repair) -> dict:
        return super().sampled_record(episode) | {"repair": episode.model_dump(mode="json")}

    def collect(self, maximum_tasks: int) -> tuple[list[list[Proposal]], dict, dict]:
        if self.mode == "fixed":
            return super().collect(maximum_tasks)
        cap = min(maximum_tasks, self.cfg.unique_tasks)
        groups: list[list[Proposal]] = []
        tasks: dict[str, tuple[TaskSpec, PdbOracle]] = {}
        timing = {"proposer_generation": 0.0, "proposal_validation": 0.0}
        for _ in range(4):
            if len(tasks) >= cap:
                break
            parent = self.parent(self.proposal_groups)
            template = self.repository.parent_task(parent.parent_id)
            oracle = self.repository.parent_oracle(parent.parent_id)
            rows = []
            for slot in range(4):
                started = time.monotonic()
                row, source = propose_pdb(
                    self.policy,
                    self.cfg,
                    parent,
                    oracle,
                    self.proposal_groups,
                    slot,
                    flexible=self.cfg.proposer_design == "flexible",
                )
                timing["proposer_generation"] += time.monotonic() - started
                rows.append(row)
                if source is None or row.mutation_key is None:
                    continue
                key = row.mutation_key
                row.duplicate = key in self.seen or key in tasks
                started = time.monotonic()
                try:
                    if key not in tasks:
                        if not verify_pdb(
                            self.sandbox, oracle.correct_source, template, oracle
                        ).passed:
                            raise RuntimeError(
                                "Verified PDB parent failed under the current worker"
                            )
                        task, new_oracle = proposal_task(
                            self.sandbox, template, oracle, row, source
                        )
                        history = student_history(task, self.cfg.episode)
                        student = self.policy.encode(history, STUDENT_TOOLS, True)
                        teacher = self.policy.encode(
                            teacher_messages(history, new_oracle), STUDENT_TOOLS, True
                        )
                        if len(teacher) - len(student) > self.cfg.episode.privilege_reserve:
                            raise ValueError("PDB proposal exceeds the reserved teacher context")
                        if len(tasks) == cap:
                            row.turn.result = "unscored_capacity"
                            continue
                        tasks[key] = task, new_oracle
                    row.task = tasks[key][0].task_id
                    row.turn.result = "accepted"
                except ValueError as exc:
                    row.invalid_reason = str(exc)[:300]
                finally:
                    timing["proposal_validation"] += time.monotonic() - started
            groups.append(rows)
            self.proposal_groups += 1
        self.seen.update(tasks)
        return groups, {task.task_id: (task, oracle) for task, oracle in tasks.values()}, timing

    def proposer_gradient(self, groups: list[list[Proposal]]) -> tuple[Grads, float]:
        scored = [
            [row for row in group if row.turn.result != "unscored_capacity"] for group in groups
        ]
        return super().proposer_gradient([group for group in scored if len(group) > 1])

    def calibrate(self) -> dict:
        if self.iteration:
            raise ValueError("Task 21 calibration requires initial weights")
        # Candidate rollouts are shared across the three gradients, with no optimizer step.
        groups, tasks, timing = self.collect(self.cfg.unique_tasks)
        if not tasks:
            return {"ready": False, "reason": "no verified proposals", "timing": timing}
        rolls, teacher_seconds = self.rollouts(tasks, distill=True)
        rl, ungated, gated = self.zeros(), self.zeros(), self.zeros()
        scored = time.monotonic()
        for episodes in rolls.values():
            rl_gradient, ungated_gradient, _ = self.student_gradient(
                episodes, gated=False, use_sd=True
            )
            _, gated_gradient, _ = self.student_gradient(episodes, gated=True, use_sd=True)
            for total, gradient in (
                (rl, rl_gradient),
                (ungated, ungated_gradient),
                (gated, gated_gradient),
            ):
                add_gradient(total, gradient, 1 / len(rolls))
        for group in groups:
            for row in group:
                if row.task is not None:
                    row.reward = difficulty(float(ep.success) for ep in rolls[row.task])
        proposer, _ = self.proposer_gradient(groups)
        r, a, b, p = map(norm, (rl, ungated, gated, proposer))
        required = (r, a, b, p) if self.mode == "shared" else (r, a, b)
        ready = all(math.isfinite(n) and n > 1e-12 for n in required)
        report: dict = {
            "ready": ready,
            "rl_norm": r,
            "ungated_norm": a,
            "gated_norm": b,
            "proposer_norm": p,
            "mode": self.mode,
            "teacher_seconds": teacher_seconds,
            "backward_seconds": time.monotonic() - scored,
            "timing": timing,
            "tasks": len(rolls),
        }
        if ready:
            report.update(
                sd_scale=0.3 * r / b,
                calibrated_ungated_scale=0.3 * r / a,
                proposer_scale=0.5 * r / p if self.mode == "shared" else None,
            )
        return report

    def solver(self, marker: Path, publish, should_stop: Callable[[], bool]) -> None:
        if self.mode != "fixed" or not self.fixed:
            raise ValueError("Task 21 solver-only stage requires a frozen curriculum")
        checks = ("solver_train", "solver_probe")
        initial = {}
        if not self.iteration:
            for split in checks:
                initial[split] = self.sample(
                    split, self.cfg.probe_repeats, should_stop, block="paired"
                )
            if self.repository.tasks("dev_monitor"):
                self.evaluate("dev_monitor", should_stop)
        while (
            self.iteration < self.cfg.solver_batches
            and self.effective_updates < self.cfg.solver_updates
        ):
            if should_stop():
                raise SegmentStop("Stopped between solver batches")
            publish(self.update(4, []), self.iteration)
        observed = {}
        for split in checks:
            before = initial.get(split) or read_json(
                self.output / "samples" / self.initial / split / "paired" / "summary.json"
            )
            after = self.sample(split, self.cfg.probe_repeats, should_stop, block="paired")
            old = {(r["task_id"], r["slot"]): r["success"] for r in before["rows"]}
            new = {(r["task_id"], r["slot"]): r["success"] for r in after["rows"]}
            if old.keys() != new.keys():
                raise ValueError("Task 21 solver pairing differs across checkpoints")
            observed[split] = {
                "count": len(old),
                "before": sum(old.values()),
                "after": sum(new.values()),
                "gains": sum(new[k] and not old[k] for k in old),
                "losses": sum(old[k] and not new[k] for k in old),
            }
        dev = (
            self.evaluate("dev_monitor", should_stop)
            if self.repository.tasks("dev_monitor")
            else {}
        )
        rows = [
            read_json(path)["metrics"]
            for path in sorted((self.output / "iterations").glob("*.json"))
            if path.stem != "000000"
        ]
        effective = [row for row in rows if row["rl"]["norm"] > 0]
        self.finish(
            marker,
            {
                "learnable": (
                    len(effective) > 1
                    and all(
                        math.isfinite(row["rl"]["norm"])
                        and row["shared"]["optimizer_step"]
                        and row["shared"]["update_norm"] > 0
                        and row["rl"]["support_exits"] == 0
                        for row in effective
                    )
                    and observed["solver_train"]["gains"] > observed["solver_train"]["losses"]
                ),
                "paired": observed,
                "dev": {key: dev[key] for key in ("count", "verified") if key in dev},
            },
        )

    def shared(
        self, operation: str, marker: Path, publish, should_stop: Callable[[], bool]
    ) -> None:
        if operation != "shared-pilot":
            super().shared(operation, marker, publish, should_stop)
            return
        if self.mode != "shared":
            raise ValueError("Task 21 proposer pilot requires the shared policy")
        if self.repository.tasks("dev_monitor") and not self.iteration:
            self.evaluate("dev_monitor", should_stop)
        while self.iteration < 4:
            if should_stop():
                raise SegmentStop("Stopped between shared-pilot batches")
            publish(self.update(self.cfg.unique_tasks, []), self.iteration)
        if self.repository.tasks("dev_monitor"):
            self.evaluate("dev_monitor", should_stop)
        parents = set()
        valid_proposals = 0
        unscored = 0
        for path in sorted((self.output / "batches").glob("*.json")):
            for group in read_json(path)["proposals"]:
                valid = [row for row in group if row["task"] is not None]
                skipped = sum(row["turn"]["result"] == "unscored_capacity" for row in group)
                valid_proposals += len(valid) + skipped
                unscored += skipped
                if (
                    len({row["task"] for row in valid}) > 1
                    and len({row["reward"] for row in valid}) > 1
                ):
                    parents.add(valid[0]["parent"])
        self.finish(
            marker,
            {
                "viable": self.effective_updates >= 3 and len(parents) >= 2,
                "valid_reward_parents": len(parents),
                "valid_proposals": valid_proposals,
                "valid_unscored_capacity": unscored,
            },
        )

    def final_evaluate(self, checkpoint: int | None, should_stop: Callable[[], bool]) -> None:
        if checkpoint is None or not (self.output / "complete-main.json").is_file():
            raise ValueError("Final evaluation requires a frozen completed training endpoint")
        self.restore(self.checkpoints.root / f"{checkpoint:06d}")
        if self.episodes not in {0, self.cfg.endpoint}:
            raise ValueError("Task 21 final evaluation is reserved for baseline and endpoint")
        result = {}
        for split in (
            "primary_final",
            "cross_source",
            "cross_source_bcb_mutants",
            "harder_transfer",
        ):
            if self.repository.tasks(split):
                row = self.evaluate(split, should_stop)
                result[split] = {"count": row["count"], "verified": row["verified"]}
        self.finish(self.output / f"complete-evaluate-{checkpoint:06d}.json", result)
