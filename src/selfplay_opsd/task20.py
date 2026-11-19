"""Task 20 control and bounded solver and shared-policy experiments."""

from __future__ import annotations

import copy
import fcntl
import math
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from selfplay_opsd.config import EpisodeConfig
from selfplay_opsd.core import digest, keyed_seed, read_json
from selfplay_opsd.data import TaskRepository
from selfplay_opsd.objectives import alignment, difficulty
from selfplay_opsd.sandbox import SandboxBackend
from selfplay_opsd.storage import atomic_json, file_hash, restore_rng, rng_state
from selfplay_opsd.task19 import Arm, Task19Control
from selfplay_opsd.task19_policy import STUDENT_SAMPLING, SharedPolicy
from selfplay_opsd.task19_rollout import Proposal, Repair, repair
from selfplay_opsd.task19_train import Grads, SegmentStop, Study, add_gradient, joint_step, norm

Mode = Literal["fixed", "shared", "proposer_frozen"]
Operation = Literal["screen", "solver", "shared-pilot", "calibrate", "main", "evaluate"]


class Task20Control(Task19Control):
    study: str = "repair-selfplay-2026-task20-v1"
    precision: Literal["float32", "bfloat16"] = "float32"
    episode: EpisodeConfig = EpisodeConfig(
        edit_protocol="replace_text",
        max_generated_tokens=12288,
        max_turn_generated_tokens=4096,
        max_student_sequence_tokens=20480,
        max_teacher_sequence_tokens=24576,
        privilege_reserve=4096,
    )
    endpoint: int | None = Field(default=None, ge=32)
    dev_milestones: tuple[int, ...] = ()
    group_size: Literal[4, 8] = 4
    solver_batches: int = Field(default=12, ge=1, le=12)
    solver_updates: int = Field(default=8, ge=1, le=12)
    shared_pilot_episodes: int = Field(default=64, ge=8)
    shared_pilot_mode: Literal["shared", "proposer_frozen"] = "shared"
    probe_repeats: int = Field(default=4, ge=2, le=8)
    solver_gate_hash: str | None = None
    shared_gate_hash: str | None = None
    alignment_gate_hash: str | None = None

    @model_validator(mode="after")
    def validate_study(self):
        if not self.study.startswith("repair-selfplay-2026-task20-"):
            raise ValueError("Task 20 requires a separate study namespace")
        if self.episode.edit_protocol != "replace_text" or self.precision != "float32":
            raise ValueError("Task 20 requires cached FP32 unless separately qualified")
        if not self.cached_rollout and not self.qualification_hash:
            raise ValueError("Uncached execution requires separate qualification")
        if self.solver_updates > self.solver_batches:
            raise ValueError("Solver update target exceeds maximum batches")
        if self.shared_pilot_episodes % self.group_size:
            raise ValueError("Shared pilot must stop on a group boundary")
        if self.endpoint is not None:
            if (
                not self.ready
                or not self.solver_gate_hash
                or not self.shared_gate_hash
                or self.endpoint < self.shared_pilot_episodes
            ):
                raise ValueError(
                    "Main requires calibrated scales and a shared-pilot-sized endpoint"
                )
            if (
                not self.dev_milestones
                or tuple(sorted(set(self.dev_milestones))) != self.dev_milestones
            ):
                raise ValueError("Freeze an increasing development schedule")
            if (
                any(
                    not 0 < step < self.endpoint or step % self.group_size
                    for step in self.dev_milestones
                )
                or self.endpoint % self.group_size
            ):
                raise ValueError("Milestones and endpoint must be reachable group boundaries")
        elif self.dev_milestones:
            raise ValueError("Development milestones require a frozen endpoint")
        return self

    @property
    def science(self) -> str:
        return digest(
            {
                "objective": super().science,
                "stopping": [self.endpoint, self.dev_milestones],
            }
        )

    def milestones(self) -> list[int]:
        return sorted(
            {
                0,
                self.shared_pilot_episodes,
                *self.dev_milestones,
                *([self.endpoint] if self.endpoint is not None else []),
            }
        )


class Task20Study(Study):
    cfg: Task20Control

    def __init__(
        self,
        cfg: Task20Control,
        arm: Arm,
        policy: SharedPolicy,
        sandbox: SandboxBackend,
        repository: TaskRepository,
        output: Path,
        execution: dict,
        *,
        mode: Mode,
    ):
        if mode == "fixed" and arm != "control":
            raise ValueError("The fixed curriculum is reserved for solver-only Dr. GRPO")
        super().__init__(cfg, arm, policy, sandbox, repository, output, execution)
        self.mode = mode
        self.student_group_size = cfg.group_size
        self.effective_updates = 0
        self.fixed = sorted(repository.tasks("solver_train"), key=lambda item: item[0].task_id)

    @property
    def identity(self) -> dict:
        return super().identity | {"curriculum": self.mode}

    def loss_scales(self) -> tuple[float, float]:
        strength = 0.0
        proposer = 0.0
        if self.arm != "control":
            if self.cfg.sd_scale is None:
                raise ValueError("OPSD requires a frozen training-only coefficient")
            strength = self.cfg.sd_scale
        if self.mode == "shared":
            if self.cfg.proposer_scale is None:
                raise ValueError("Shared updates require a frozen proposer coefficient")
            proposer = self.cfg.proposer_scale
        return strength, proposer

    def collect(self, maximum_tasks: int) -> tuple[list[list[Proposal]], dict, dict]:
        if self.mode != "fixed":
            return super().collect(maximum_tasks)
        if not self.fixed:
            raise ValueError("No frozen solver curriculum")
        count = min(4, maximum_tasks, len(self.fixed))
        order = list(self.fixed)
        random.Random(keyed_seed(self.cfg.seed, "solver-order", self.iteration)).shuffle(order)
        return [], {task.task_id: (task, oracle) for task, oracle in order[:count]}, {}

    def proposer_gradient(self, groups: list[list[Proposal]]) -> tuple[Grads, float]:
        return super().proposer_gradient(groups) if self.mode == "shared" else (self.zeros(), 0.0)

    def update_diagnostics(
        self, rl: Grads, sd: Grads, proposer: Grads, rolls: dict[str, list[Repair]], signals: list
    ) -> dict:
        self.effective_updates += norm(rl) > 0
        size = self.student_group_size
        rewards = [sum(row.success for row in group) for group in rolls.values()]
        hist = [rewards.count(n) for n in range(size + 1)]
        mixed_tokens = sum(
            len(turn.ids)
            for reward, group in zip(rewards, rolls.values(), strict=True)
            if 0 < reward < size
            for row in group
            for turn in row.turns
        )
        gates = [value for signal in signals for value in signal["gate"]]
        return {
            "learning": {
                "effective_rl_updates": self.effective_updates,
                "mixed_reward_fraction": sum(hist[1:-1]) / len(rewards) if rewards else None,
                "mixed_group_tokens": mixed_tokens,
                "reward_histogram": {str(n): count for n, count in enumerate(hist)},
                "rl_proposer_cosine": alignment(rl, proposer).cosine,
                "opsd_proposer_cosine": alignment(sd, proposer).cosine,
                "gates": {
                    str(n): sum(min(4, int(value * 5)) == n for value in gates) for n in range(5)
                },
                "distinct_parents": len(
                    {row.parent_id for group in rolls.values() for row in group}
                ),
            }
        }

    def restore(self, path: Path) -> None:
        super().restore(path)
        record = read_json(self.output / "iterations" / f"{self.iteration:06d}.json")
        self.effective_updates = (
            record["metrics"].get("learning", {}).get("effective_rl_updates", 0)
        )

    def sample(
        self,
        split: str,
        repeats: int,
        should_stop: Callable[[], bool],
        *,
        block: str,
        extended: set[str] | None = None,
    ) -> dict:
        root = self.output / "samples" / self.snapshot / split / block
        tasks = self.repository.tasks(split)
        if not tasks:
            raise ValueError(f"No tasks in {split}")
        rows = []
        for task, oracle in tasks:
            count = 8 if extended is not None and task.task_id in extended else repeats
            for slot in range(count):
                if should_stop():
                    raise SegmentStop("Stopped between sampled episodes")
                seed = keyed_seed(self.cfg.seed, "task20", split, task.task_id, block, slot)
                identity = self.identity | {"snapshot": self.snapshot, "seed": seed}
                path = root / f"{task.task_id}-{slot}.json"
                if path.is_file():
                    row = read_json(path)
                    if row["identity"] != identity:
                        raise ValueError("Sample cache identity differs")
                else:
                    episode = repair(self.policy, self.sandbox, self.cfg, task, oracle, seed)
                    row = {"identity": identity, "metrics": episode.metrics()}
                    atomic_json(path, row)
                rows.append(
                    row["metrics"]
                    | {"task_id": task.task_id, "parent_id": task.parent_id, "slot": slot}
                )
        result = {
            "data": self.repository.hash,
            "snapshot": self.snapshot,
            "rows": rows,
            "verified": sum(row["success"] for row in rows),
            "count": len(rows),
        }
        atomic_json(root / "summary.json", result)
        return result

    def one_step(self, should_stop: Callable[[], bool]) -> dict:
        """Check two identical restored-state updates on the same authentic frozen rollouts."""
        path = self.output / "one-step.json"
        if path.is_file():
            saved = read_json(path)
            if saved["identity"] != self.identity:
                raise ValueError("One-step diagnostic identity differs")
            return saved
        curriculum = {task.task_id: (task, oracle) for task, oracle in self.fixed}
        group = None
        group_count = 0
        for task_id, task in curriculum.items():
            if should_stop():
                raise SegmentStop("Stopped before the next one-step diagnostic group")
            rolls, _ = self.rollouts({task_id: task}, distill=False)
            group_count += 1
            candidate = rolls[task_id]
            if 0 < sum(row.success for row in candidate) < len(candidate):
                group = candidate
                break
        report: dict = {
            "identity": self.identity,
            "groups": group_count,
            "mixed_groups": int(group is not None),
        }
        if group is not None:
            gradient, _, _ = self.student_gradient(group, gated=False, use_sd=False)
            state = copy.deepcopy(self.policy.state())

            rng = rng_state()

            def direction() -> tuple[float, int]:
                score = 0.0
                exits = 0
                mean_reward = sum(item.success for item in group) / len(group)
                for row in group:
                    advantage = float(row.success) - mean_reward
                    for turn in row.turns:
                        if turn.ids:
                            current, _ = self.policy.score(
                                turn.prompt,
                                turn.ids,
                                STUDENT_SAMPLING,
                                exact_prefix=not self.cfg.cached_rollout,
                            )
                            deltas = [
                                new - old
                                for new, old in zip(current.tolist(), turn.logps, strict=True)
                            ]
                            exits += sum(not math.isfinite(value) for value in deltas)
                            score += advantage * math.fsum(
                                value for value in deltas if math.isfinite(value)
                            )
                return score, exits

            results = []
            baseline_score, baseline_exits = direction()
            for _ in range(2):
                try:
                    step = joint_step(self.policy, gradient, self.zeros(), 0, self.cfg.grad_clip)
                    score, exits = direction()
                    results.append(
                        step
                        | {
                            "weighted_logprob_change": score - baseline_score,
                            "support_exits": exits + baseline_exits,
                        }
                    )
                finally:
                    self.policy.restore(copy.deepcopy(state))
                    restore_rng(rng)
            if self.policy.version() != self.snapshot:
                raise ValueError("Restored solver diagnostic changed the adapter")
            if not all(
                math.isclose(
                    float(results[0][key]), float(results[1][key]), rel_tol=1e-4, abs_tol=1e-7
                )
                for key in ("update_norm", "weighted_logprob_change", "support_exits")
            ):
                raise ValueError("Restored solver steps failed numerical consistency")
            report["step"] = results[0]
        atomic_json(path, report)
        return report

    def run(
        self,
        operation: Operation,
        publish: Callable[[dict, int], None],
        should_stop: Callable[[], bool],
        *,
        checkpoint: int | None = None,
    ) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output / ".run.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            pointer = self.checkpoints.root / "latest.json"
            if operation == "evaluate":
                self.final_evaluate(checkpoint, should_stop)
                return
            marker = self.output / f"complete-{operation}.json"
            if marker.is_file():
                if read_json(marker)["identity"] != self.identity:
                    raise ValueError("Completed stage identity differs")
                return
            if pointer.is_file():
                self.restore(self.checkpoints.root / read_json(pointer)["path"])
            else:
                self.commit({}, None)
            if operation == "screen":
                probe = self.sample("screen", 4, should_stop, block="difficulty")
                ordered = sorted(
                    {row["task_id"] for row in probe["rows"]},
                    key=lambda task: keyed_seed(self.cfg.seed, "group-probe", task),
                )
                result = self.sample(
                    "screen", 4, should_stop, block="difficulty", extended=set(ordered[:8])
                )
                self.finish(
                    marker, {"screen": result["count"], "group8_tasks": min(8, len(ordered))}
                )
            elif operation == "calibrate":
                self.finish(marker, self.calibrate())
            elif operation == "solver":
                self.solver(marker, publish, should_stop)
            else:
                self.shared(operation, marker, publish, should_stop)

    def finish(self, marker: Path, result: dict) -> None:
        assert self.latest is not None
        self.protected.add(self.iteration)
        atomic_json(
            marker,
            {
                "identity": self.identity,
                "episodes": self.episodes,
                "effective_rl_updates": self.effective_updates,
                "snapshot": self.snapshot,
                "checkpoint": self.latest.name,
                "checkpoint_sha256": file_hash(self.latest / "complete.json"),
                "result": result,
            },
        )

    def solver(self, marker: Path, publish, should_stop) -> None:
        if self.mode != "fixed" or not self.fixed:
            raise ValueError("Solver pilot requires a frozen curriculum")
        blocks = ("paired-0", "paired-1")
        splits = ("solver_train", "solver_probe")
        if self.iteration == 0:
            for split in splits:
                for block in blocks:
                    self.sample(split, self.cfg.probe_repeats, should_stop, block=block)
            self.one_step(should_stop)
            self.evaluate("dev_monitor", should_stop)
        while (
            self.iteration < self.cfg.solver_batches
            and self.effective_updates < self.cfg.solver_updates
        ):
            if should_stop():
                raise SegmentStop("Stopped between solver batches")
            publish(self.update(4, []), self.iteration)
        result: dict[str, dict] = {}
        for split in splits:
            result[split] = {}
            for block in blocks:
                after = self.sample(split, self.cfg.probe_repeats, should_stop, block=block)
                before = read_json(
                    self.output / "samples" / self.initial / split / block / "summary.json"
                )
                result[split][block] = {
                    "before": before["verified"],
                    "after": after["verified"],
                    "count": after["count"],
                }
        step = read_json(self.output / "one-step.json").get("step", {})
        result["gate"] = {
            "learnable": self.effective_updates > 1
            and step.get("update_norm", 0) > 0
            and step.get("weighted_logprob_change", 0) > 0
            and step.get("support_exits") == 0
            and all(pair["after"] > pair["before"] for pair in result["solver_train"].values()),
        }
        result["development"] = {
            "initial": read_json(self.output / "evaluation/0000/dev_monitor/summary.json")[
                "verified"
            ],
            "endpoint": self.evaluate("dev_monitor", should_stop)["verified"],
        }
        self.finish(marker, result)

    def shared(self, operation: str, marker: Path, publish, should_stop) -> None:
        if self.mode == "fixed":
            raise ValueError("Shared training requires a proposer mode")
        limit = self.cfg.endpoint if operation == "main" else self.cfg.shared_pilot_episodes
        if limit is None or self.episodes > limit:
            raise ValueError("Missing or exceeded the frozen stage endpoint")
        points = sorted({limit, *(n for n in self.cfg.milestones() if n <= limit)})
        while True:
            if should_stop():
                raise SegmentStop("Stopped between shared batches")
            if self.episodes in points:
                result = self.evaluate("dev_monitor", should_stop)
                publish(
                    {"evaluation": {"dev_success": result["verified"] / result["count"]}},
                    self.iteration,
                )
            if self.episodes == limit:
                self.finish(marker, {})
                return
            if self.empty_batches >= 4:
                raise RuntimeError("Four proposal refills produced no verified tasks")
            stop = next(point for point in points if point > self.episodes)
            references = self.references() if self.arm in {"C", "D"} else []
            publish(
                self.update((stop - self.episodes) // self.student_group_size, references),
                self.iteration,
            )

    def calibrate(self) -> dict:
        if self.mode != "shared" or self.iteration != 0:
            raise ValueError("Calibration requires fresh shared-policy weights")
        groups, tasks, _ = self.collect(4)
        if not tasks:
            return {"ready": False, "reason": "no verified proposals"}
        rolls, teacher_seconds = self.rollouts(tasks, distill=True)
        rl, sd = self.zeros(), self.zeros()
        opsd_started = time.monotonic()
        for rows in rolls.values():
            a, b, _ = self.student_gradient(rows, gated=False, use_sd=True)
            add_gradient(rl, a, 1 / len(rolls))
            add_gradient(sd, b, 1 / len(rolls))
        opsd_seconds = time.monotonic() - opsd_started
        rl_started = time.monotonic()
        for rows in rolls.values():
            self.student_gradient(rows, gated=False, use_sd=False)
        rl_seconds = time.monotonic() - rl_started
        for group in groups:
            for proposal in group:
                if proposal.task is not None:
                    proposal.reward = difficulty(float(row.success) for row in rolls[proposal.task])
        proposer, _ = self.proposer_gradient(groups)
        r, s, p = norm(rl), norm(sd), norm(proposer)
        return {
            "ready": min(r, s, p) > 1e-12,
            "rl_norm": r,
            "opsd_norm": s,
            "proposer_norm": p,
            "sd_scale": 0.3 * r / s if min(r, s) > 1e-12 else None,
            "proposer_scale": 0.5 * r / p if min(r, p) > 1e-12 else None,
            "teacher_seconds": teacher_seconds,
            "rl_backward_seconds": rl_seconds,
            "rl_and_opsd_backward_seconds": opsd_seconds,
            "extra_opsd_seconds_per_episode": (teacher_seconds + max(0, opsd_seconds - rl_seconds))
            / sum(map(len, rolls.values())),
        }

    def final_evaluate(self, checkpoint: int | None, should_stop: Callable[[], bool]) -> None:
        if checkpoint is None or not (self.output / "complete-main.json").is_file():
            raise ValueError(
                "Final evaluation requires a completed main run and selected checkpoint"
            )
        self.restore(self.checkpoints.root / f"{checkpoint:06d}")
        if self.episodes not in self.cfg.milestones():
            raise ValueError("Final evaluation was requested for an undeclared milestone")
        for split in ("final_id", "external"):
            if self.repository.tasks(split):
                self.evaluate(split, should_stop)
        self.finish(self.output / f"complete-evaluate-{checkpoint:06d}.json", {})
