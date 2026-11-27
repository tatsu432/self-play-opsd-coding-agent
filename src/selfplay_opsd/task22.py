"""PDB reference supervision and paired, frozen-snapshot alignment diagnostics."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import asdict
from functools import cached_property
from pathlib import Path
from typing import Any, cast

import torch
from pydantic import Field, model_validator

from selfplay_opsd.core import ExperimentError, digest, read_json
from selfplay_opsd.objectives import alignment, difficulty
from selfplay_opsd.pdb_verifier import PdbEnvironment, PdbOracle, verify_pdb
from selfplay_opsd.policy import HFPolicy
from selfplay_opsd.storage import CheckpointManager, atomic_json, file_hash, restore_rng
from selfplay_opsd.task19_policy import visibility
from selfplay_opsd.task19_protocol import STUDENT_TOOLS, parse_call, render_call, student_history
from selfplay_opsd.task19_rollout import Proposal, Repair
from selfplay_opsd.task19_train import Grads, SegmentStop, exact_reference, norm
from selfplay_opsd.task21 import PdbRepository, Task21Control, Task21Study
from selfplay_opsd.task22_contract import (
    B_CHECKPOINTS,
    BASE_SCIENCE,
    BASE_SCIENCE_2027,
    BASE_STUDY,
    EVAL_SPLITS,
    EXTENSIONS,
    INITIAL,
    INITIAL_2027,
    MATCHED_DATA,
    PRIMARY,
    QUALIFY,
    REPLICATION,
    STUDY,
    common_control,
    reference_path,
)

Reference = tuple[list[int], list[int], list[bool]]


class Task22Control(Task21Control):
    study: str = STUDY
    reference_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    parent_marker_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def validate_task22(self):
        if self.study in EXTENSIONS:
            endpoint = EXTENSIONS[self.study]
            points = (128, 256, 640) if endpoint == 768 else (640, 768, 896)
            if (
                self.endpoint != endpoint
                or self.dev_milestones != points
                or not self.parent_marker_sha256
                or self.reference_sha256
                or self.alignment_gate_hash
            ):
                raise ValueError("Task 22 extension requires a frozen Task 21 checkpoint lineage")
        elif (
            self.study not in {STUDY, REPLICATION}
            or self.endpoint != 512
            or self.dev_milestones != (128, 256, 384)
            or not self.reference_sha256
            or self.parent_marker_sha256
        ):
            raise ValueError("Task 22 requires the frozen matched alignment control")
        if self.main_mode != "shared" or self.seed not in (
            {2026, 2027} if self.study == REPLICATION else {2026}
        ):
            raise ValueError("Task 22 requires a matched shared study seed")
        return self


def reference_candidates(repository: PdbRepository) -> list[dict]:
    if repository.hash != MATCHED_DATA:
        raise ValueError("Task 22 requires the frozen Task 21 PDB data")
    source_splits = ("feasibility", "solver_train", "proposer_train")
    excluded = [
        (task, oracle)
        for split in repository.partition_names()
        if split not in source_splits
        for task, oracle in repository.tasks(split)
    ]
    parents = {task.parent_id for task, _ in excluded}
    families = {oracle.provenance["family_id"] for _, oracle in excluded}
    curriculum = {parent.parent_id for parent in repository.parents("train")}
    rows = []
    for split in source_splits:
        for task, oracle in sorted(repository.tasks(split), key=lambda row: row[0].task_id):
            family = oracle.provenance["family_id"]
            if task.parent_id in parents or family in families:
                continue
            parents.add(task.parent_id)
            families.add(family)
            rows.append(
                {
                    "split": split,
                    "task_id": task.task_id,
                    "parent_id": task.parent_id,
                    "family_id": family,
                    "curriculum_overlap": task.parent_id in curriculum,
                    "provenance": oracle.provenance,
                }
            )
    return rows


def reference_sidecar(repository: PdbRepository) -> dict:
    return {"version": 1, "data": repository.hash, "candidates": reference_candidates(repository)}


def paired_rewards(
    rl: Grads, sd: Grads, reference: Grads, scale: float, successes: list[bool]
) -> dict:
    combined = [a.add(b, alpha=scale) for a, b in zip(rl, sd, strict=True)]
    scores = {
        arm: asdict(alignment(candidate, reference))
        for arm, candidate in (("C", rl), ("D", combined))
    }
    for row in scores.values():
        if row["cosine"] is None:
            row["reward"] = 0.0
    interaction = alignment(rl, sd)
    return scores | {
        "difficulty": difficulty(map(float, successes)),
        "successes": sum(successes),
        "rollouts": len(successes),
        "rl_norm": interaction.candidate_norm,
        "opsd_norm": interaction.reference_norm,
        "weighted_opsd_norm": scale * interaction.reference_norm,
        "rl_opsd_cosine": interaction.cosine,
    }


class Task22Study(Task21Study):
    cfg: Task22Control
    source_commit: str
    run_id: str

    @property
    def identity(self) -> dict:
        expected = INITIAL_2027 if self.cfg.seed == 2027 else INITIAL
        if self.initial != expected:
            raise ValueError("Task 22 initialization differs from the Task 21 matched adapter")
        out = super().identity
        if self.cfg.parent_marker_sha256:
            return out | {"parent_marker": self.cfg.parent_marker_sha256}
        return out | {"reference": self.cfg.reference_sha256}

    @property
    def qualification_root(self) -> Path:
        return self.output.parent / f"{QUALIFY}-seed2026"

    def prepare_references(self) -> dict:
        assert self.cfg.reference_sha256
        sidecar = reference_path(self.repository.root, self.cfg.reference_sha256)
        candidates = reference_sidecar(self.repository)
        if file_hash(sidecar) != self.cfg.reference_sha256 or read_json(sidecar) != candidates:
            raise ValueError("Task 22 reference sidecar differs from the frozen PDB selection")
        rows: list[dict] = []
        excluded = []
        for row in candidates["candidates"]:
            if row["curriculum_overlap"] and len(rows) >= 8:
                break
            task, oracle = next(
                pair
                for pair in self.repository.tasks(row["split"])
                if pair[0].task_id == row["task_id"]
            )
            if not isinstance(oracle, PdbOracle):
                raise ValueError("Missing PDB oracle")
            try:
                old, new = exact_reference(task.initial_source, oracle.correct_source)
                call = render_call(
                    "replace_text",
                    {
                        "path": "src/solution.py",
                        "old_text": old,
                        "new_text": new,
                    },
                )
                _, args = parse_call(call, STUDENT_TOOLS)
                env = PdbEnvironment(task, oracle, self.sandbox, cast(HFPolicy, self.policy))
                source = env.replace_source(args["old_text"], args["new_text"])
                if source != oracle.correct_source:
                    raise ValueError("Canonical edit differs from verified source")
                prompt = self.policy.encode(
                    student_history(task, self.cfg.episode), STUDENT_TOOLS, True
                )
                targets = self.policy.tokenizer.encode(
                    "</think>\n\n" + call, add_special_tokens=False
                )
                kinds, _ = visibility(targets, self.policy.markers, self.policy.eos, True)
                mask = [kind == "tool" for kind in kinds]
                limits = self.cfg.episode
                if (
                    not any(mask)
                    or len(targets) > limits.max_turn_generated_tokens
                    or len(prompt) + len(targets) > limits.max_student_sequence_tokens
                ):
                    raise ValueError(
                        "Canonical native action exceeds training context or lacks tool tokens"
                    )
                original = verify_pdb(self.sandbox, task.initial_source, task, oracle)
                repaired = verify_pdb(self.sandbox, source, task, oracle)
                if not original.semantic_failures or not repaired.passed:
                    raise ValueError(
                        "PDB verifier did not confirm both original bug and canonical repair"
                    )
            except ValueError as exc:
                excluded.append({"task_id": row["task_id"], "reason": str(exc)})
                continue
            rows.append(
                row
                | {
                    "action": call,
                    "prompt": prompt,
                    "targets": targets,
                    "mask": mask,
                    "prompt_tokens": len(prompt),
                    "target_tokens": len(targets),
                    "action_tokens": sum(mask),
                    "original_semantic_failures": original.semantic_failures,
                    "repaired_verified": repaired.passed,
                }
            )
        if len(rows) < 8:
            raise ValueError("Fewer than eight isolated PDB reference examples qualified")
        return {
            "data": self.repository.hash,
            "reference": self.cfg.reference_sha256,
            "execution": self.execution,
            "worker": self.sandbox.identity(),
            "rows": rows,
            "excluded": excluded,
        }

    @staticmethod
    def tokenized(records: dict) -> list[Reference]:
        return [(row["prompt"], row["targets"], row["mask"]) for row in records["rows"]]

    @cached_property
    def reference_cache(self) -> list[Reference]:
        gate = self.qualification_root / f"complete-{QUALIFY}.json"
        cache = self.qualification_root / "references.json"
        if not self.cfg.alignment_gate_hash or file_hash(gate) != self.cfg.alignment_gate_hash:
            raise ValueError("Task 22 alignment qualification SHA-256 differs")
        report = read_json(gate)["result"]
        records = read_json(cache)
        if (
            not report["ready"]
            or file_hash(cache) != report["tokenized_sha256"]
            or records["reference"] != self.cfg.reference_sha256
            or records["data"] != self.repository.hash
            or records["execution"] != self.execution
            or records["worker"] != self.sandbox.identity()
        ):
            raise ValueError("Task 22 tokenized references differ from qualification")
        return self.tokenized(records)

    def references(self) -> list[Reference]:
        return self.reference_cache

    def reference_gradient(self, references: list[Reference]) -> Grads:
        gradient = super().reference_gradient(references)
        if not math.isfinite(norm(gradient)):
            raise FloatingPointError("Nonfinite PDB reference gradient")
        return gradient

    def candidate_reward(
        self, rl: Grads, sd: Grads, reference: Grads, scale: float, successes: list[bool]
    ) -> tuple[float, dict]:
        rows = paired_rewards(rl, sd, reference, scale, successes)
        recorded = rows[self.arm]
        return recorded["reward"], recorded | {"paired": rows}

    def evaluate(self, split: str, should_stop=lambda: False) -> dict:
        if self.episodes != 0 or split != "dev_monitor":
            return super().evaluate(split, should_stop)
        output = self.output / "evaluation/0000/dev_monitor/summary.json"
        if output.is_file():
            record = read_json(output)
            if record["identity"] != self.identity or record["snapshot"] != self.snapshot:
                raise ValueError("Reused Task 22 baseline identity differs")
            return record
        root = self.output.parent.parent / BASE_STUDY / f"B-seed{self.cfg.seed}"
        previous = root / "evaluation/0000/dev_monitor/summary.json"
        if not previous.is_file():
            return super().evaluate(split, should_stop)
        record = read_json(previous)
        original = record["identity"]
        tasks = self.repository.tasks(split)
        if (
            record["snapshot"] != self.snapshot
            or original["data"] != self.repository.hash
            or original["science"] != (BASE_SCIENCE_2027 if self.cfg.seed == 2027 else BASE_SCIENCE)
            or original["initial"] != self.initial
            or original["worker"] != self.sandbox.identity()
            or record["split"] != split
            or record["episodes"] != 0
            or record["count"] != len(tasks)
            or record["count"] != len(record["tasks"])
            or {row["task_id"]: row["parent_id"] for row in record["tasks"]}
            != {task.task_id: task.parent_id for task, _ in tasks}
            or record["verified"] != sum(row["success"] for row in record["tasks"])
        ):
            raise ValueError("Task 21 development baseline differs from Task 22")
        record = record | {
            "identity": self.identity,
            "recorded_at": time.time(),
            "reused_from": {"identity": original, "sha256": file_hash(previous)},
        }
        atomic_json(output, record)
        return record

    def import_checkpoint(self, checkpoint: int, exposure: int) -> None:
        """Fork a matched Task 21 checkpoint with new metadata and original full state."""
        root = self.output.parent.parent / BASE_STUDY / f"{self.arm}-seed2026"
        source = root / "checkpoints" / f"{checkpoint:06d}"
        state, record = CheckpointManager.load(source)
        validate_parent(self, record, self.arm)
        if record["episodes"] != exposure:
            raise ValueError("Parent checkpoint exposure differs")
        marker = root / "complete-main.json"
        if self.cfg.parent_marker_sha256 and (
            file_hash(marker) != self.cfg.parent_marker_sha256
            or read_json(marker)["checkpoint_sha256"] != file_hash(source / "complete.json")
        ):
            raise ValueError("Task 21 continuation does not match its completed parent")
        batch = record["batch"]
        if batch is not None and file_hash(root / "batches" / batch["path"]) != batch["sha256"]:
            raise ValueError("Parent committed trajectory checksum differs")
        self.policy.restore(state["policy"])
        if self.cfg.parent_marker_sha256:
            expected = state["policy"]["optimizer"]
            restored = self.policy.optimizer.state_dict()
            if (
                restored["param_groups"] != expected["param_groups"]
                or restored["state"].keys() != expected["state"].keys()
                or any(
                    restored["state"][key].keys() != values.keys()
                    or any(
                        not torch.equal(restored["state"][key][name].detach().cpu(), value)
                        if isinstance(value, torch.Tensor)
                        else restored["state"][key][name] != value
                        for name, value in values.items()
                    )
                    for key, values in expected["state"].items()
                )
            ):
                raise ValueError("Restored Task 21 Adam state differs from its verified parent")
        restore_rng(state["rng"])
        self.snapshot = self.policy.version()
        if self.snapshot != record["snapshot"]:
            raise ValueError("Forked checkpoint does not restore the original adapter")
        self.iteration, self.episodes = record["iteration"], record["episodes"]
        self.tokens, self.proposal_groups = state["tokens"], state["proposal_groups"]
        self.seen, self.empty_batches = state["seen"], state["empty_batches"]
        self.protected = {*state["protected"], self.iteration}
        self.effective_updates = record["metrics"]["learning"]["effective_rl_updates"]
        lineage = {
            "identity": record["identity"],
            "snapshot": self.snapshot,
            "episodes": exposure,
            "checkpoint_iteration": checkpoint,
            "checkpoint_sha256": file_hash(source / "complete.json"),
            "parent_marker_sha256": file_hash(marker),
            "parent_files": read_json(source / "complete.json")["files"],
            "optimizer_step_count": max(
                (
                    int(item["step"].item())
                    for item in state["policy"]["optimizer"]["state"].values()
                ),
                default=0,
            ),
        }
        atomic_json(self.output / "parent-checkpoint.json", lineage)
        self.commit(
            {
                "learning": {"effective_rl_updates": self.effective_updates},
                "student_tokens": self.tokens,
                "parent_checkpoint": lineage,
            },
            None,
        )

    def evaluation(
        self, stage: str, checkpoint: int, exposure: int, should_stop: Callable[[], bool]
    ) -> None:
        allowed = (
            {self.cfg.endpoint} if self.cfg.parent_marker_sha256 else {256, 384, self.cfg.endpoint}
        )
        if exposure not in allowed:
            raise ValueError("Task 22 evaluation requires a prespecified endpoint or fallback")
        if self.arm == "B" and not self.cfg.parent_marker_sha256:
            existing = self.checkpoints.root / f"{checkpoint:06d}"
            if not existing.is_dir():
                self.import_checkpoint(checkpoint, exposure)
        if exposure == self.cfg.endpoint and not (self.output / "complete-main.json").is_file():
            raise ValueError("Task 22 final evaluation requires a completed main run")
        if stage == "evaluate" and exposure == self.cfg.endpoint:
            self.final_evaluate(checkpoint, should_stop)
            return
        path = self.checkpoints.root / f"{checkpoint:06d}"
        record = CheckpointManager.verify(path)
        if record["episodes"] != exposure or record["identity"] != self.identity:
            raise ValueError("Task 22 evaluation checkpoint and declared exposure differ")
        self.restore(path)
        scores = {}
        splits = ("primary_final",) if stage == PRIMARY else EVAL_SPLITS
        for split in splits:
            if self.repository.tasks(split):
                row = self.evaluate(split, should_stop)
                scores[split] = {key: row[key] for key in ("count", "verified")}
        self.finish(self.output / f"complete-{stage}-{checkpoint:06d}.json", scores)


def validate_parent(study: Task22Study, record: dict, arm: str) -> None:
    expected = {
        "study": BASE_STUDY,
        "data": MATCHED_DATA,
        "arm": arm,
        "initial": INITIAL,
        "worker": study.sandbox.identity(),
        "science": BASE_SCIENCE,
    }
    identity = record["identity"]
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError("Archived PDB checkpoint is outside the matched Task 21 study")
    if any(
        identity["execution"][key] != study.execution[key]
        for key in (
            "model",
            "model_revision",
            "tokenizer_revision",
            "image",
            "worker_image",
            "packages",
        )
    ):
        raise ValueError("Archived PDB behavior and current runtime are incompatible")


def archived_batch(
    study: Task22Study, previous: int
) -> tuple[list[list[Proposal]], dict, dict] | None:
    """Pair a retained Task 21 B checkpoint with exactly its subsequent committed rollouts."""
    root = study.output.parent.parent / BASE_STUDY / "B-seed2026"
    checkpoint = root / "checkpoints" / f"{previous:06d}"
    if not checkpoint.is_dir():
        return None
    snapshot = CheckpointManager.verify(checkpoint)
    validate_parent(study, snapshot, "B")
    record = read_json(root / "iterations" / f"{previous + 1:06d}.json")
    info = record["batch"]
    if info["path"] != f"{previous + 1:06d}.json" or record["identity"] != snapshot["identity"]:
        raise ValueError("Archived batch is outside the paired checkpoint ledger")
    path = root / "batches" / info["path"]
    if not path.is_file():
        return None
    if file_hash(path) != info["sha256"]:
        raise ValueError("Archived batch checksum differs")
    result = read_json(path)
    if result["policy"] != snapshot["snapshot"]:
        raise ValueError("Archived behavior snapshot differs")
    groups = [[Proposal.model_validate(row) for row in group] for group in result["proposals"]]
    repairs = {
        key: [Repair.model_validate(row) for row in rows] for key, rows in result["repairs"].items()
    }
    training_parents = {row.parent_id for row in study.parents}
    if (
        set(repairs) != {row.task for group in groups for row in group if row.task is not None}
        or any(row.parent not in training_parents for group in groups for row in group)
        or any(
            len(turn.logps) != len(turn.ids)
            for group in groups
            for row in group
            for turn in [row.turn]
        )
        or any(
            len(rows) != 4
            or any(row.task_id != key or row.parent_id not in training_parents for row in rows)
            for key, rows in repairs.items()
        )
        or any(
            len(turn.ids) != len(turn.logps)
            or (
                turn.ids
                and (
                    not turn.teacher_prompt
                    or len(turn.ids) != len(turn.teacher_logps or [])
                    or len(turn.ids) != len(turn.frozen_student_logps or [])
                )
            )
            for rows in repairs.values()
            for row in rows
            for turn in row.turns
        )
    ):
        raise ValueError(
            "Archived batch lacks matching original behavior and detached teacher records"
        )
    state, _ = CheckpointManager.load(checkpoint)
    study.policy.restore(state["policy"])
    if study.policy.version() != snapshot["snapshot"]:
        raise ValueError("Archived adapter version differs from sampled policy")
    return (
        groups,
        repairs,
        {
            "source": BASE_STUDY,
            "episodes": snapshot["episodes"],
            "snapshot": snapshot["snapshot"],
            "checkpoint_iteration": previous,
            "checkpoint_sha256": file_hash(checkpoint / "complete.json"),
            "batch_sha256": info["sha256"],
        },
    )


def diagnose(
    study: Task22Study, references: list[Reference], groups: list[list[Proposal]], repairs: dict
) -> dict:
    started = time.monotonic()
    ref = study.reference_gradient(references)
    reference_seconds = time.monotonic() - started
    scale = study.cfg.sd_scale
    assert scale is not None
    rows = {}
    for task, episodes in repairs.items():
        rl, sd, _ = study.student_gradient(episodes, gated=True, use_sd=True)
        rows[task] = paired_rewards(rl, sd, ref, scale, [ep.success for ep in episodes])
    comparisons: dict[str, dict[str, Any]] = {}
    for method in ("difficulty", "C", "D"):
        scores = {
            task: value[method] if method == "difficulty" else value[method]["reward"]
            for task, value in rows.items()
        }

        proposals = [
            [
                row.model_copy(update={"reward": scores[row.task]}) if row.task is not None else row
                for row in group
            ]
            for group in groups
        ]
        valid_scores = [
            {row.task: scores[row.task] for row in group if row.task is not None}
            for group in groups
        ]
        proposer, _ = study.proposer_gradient(proposals)
        comparisons[method] = {
            "proposer_norm": norm(proposer),
            "within_parent_distinct": sum(len(set(scores.values())) > 1 for scores in valid_scores),
            "within_parent_scores": list(valid_scores),
        }
        if not math.isfinite(comparisons[method]["proposer_norm"]):
            raise FloatingPointError("Nonfinite proposer gradient in qualification")
    return {
        "reference_norm": norm(ref),
        "reference_seconds": reference_seconds,
        "tasks": rows,
        "comparison": comparisons,
        "diagnostic_seconds": time.monotonic() - started,
        "gpu_peak_bytes": torch.cuda.max_memory_allocated(),
    }


def qualified(batches: list[dict]) -> bool:
    return len(batches) >= 2 and all(
        any(
            row["comparison"][arm]["within_parent_distinct"] > 0
            and row["comparison"][arm]["proposer_norm"] > 0
            for row in batches
        )
        for arm in ("C", "D")
    )


def qualify(study: Task22Study, should_stop: Callable[[], bool], publish) -> None:
    """Run bounded diagnostics without taking an optimizer step."""
    from fcntl import LOCK_EX, LOCK_NB, flock

    root = study.output
    with (root / ".run.lock").open("a+") as lock:
        flock(lock, LOCK_EX | LOCK_NB)
        marker = root / f"complete-{QUALIFY}.json"
        progress = root / "qualification-progress.json"
        refs = root / "references.json"
        if marker.exists():
            if read_json(marker)["identity"] != study.identity:
                raise ValueError("Completed qualification identity differs")
            return
        pointer = study.checkpoints.root / "latest.json"
        if pointer.exists():
            study.restore(study.checkpoints.root / read_json(pointer)["path"])
        else:
            study.commit({}, None)
        initial = study.latest
        assert initial is not None
        saved = read_json(progress) if progress.exists() else None
        if saved is None:
            atomic_json(refs, study.prepare_references())
            saved = {"identity": study.identity, "reference_sha256": file_hash(refs), "batches": []}
            atomic_json(progress, saved)
        elif saved["identity"] != study.identity or saved["reference_sha256"] != file_hash(refs):
            raise ValueError("Qualification progress or reference cache differs")
        references = study.tokenized(read_json(refs))
        batches = saved["batches"]
        for position, previous in enumerate(B_CHECKPOINTS):
            if qualified(batches):
                break
            if position < len(batches):
                continue
            if should_stop():
                raise SegmentStop("Stopped between alignment qualification batches")
            started = time.monotonic()
            try:
                archived = archived_batch(study, previous)
            except (OSError, KeyError, ValueError, ExperimentError) as exc:
                archived = None
                replay_failure = str(exc)
            else:
                replay_failure = "Archived checkpoint or trajectory unavailable"
            if archived is None:
                study.restore(initial)
                study.proposal_groups = 4 * position
                groups, tasks, timing = study.collect(4)
                repairs, timing["teacher_scoring"] = study.rollouts(tasks, distill=True)
                source = {
                    "source": "fresh",
                    "snapshot": study.snapshot,
                    "timing": timing,
                    "replay_unavailable": replay_failure,
                }
            else:
                groups, repairs, source = archived
            batches.append(
                diagnose(study, references, groups, repairs)
                | {
                    "source": source,
                    "elapsed_seconds": time.monotonic() - started,
                }
            )
            atomic_json(progress, saved)
            publish(
                {
                    "qualification": {
                        "batches": len(batches),
                        "reference_norm": batches[-1]["reference_norm"],
                        "seconds": batches[-1]["elapsed_seconds"],
                    }
                },
                len(batches),
            )
        study.restore(initial)
        study.finish(
            marker,
            {
                "ready": qualified(batches),
                "batches": batches,
                "tokenized_sha256": file_hash(refs),
                "common_control": digest(common_control(study.cfg.model_dump(mode="json"))),
                "source_commit": study.source_commit,
                "run_id": study.run_id,
            },
        )
