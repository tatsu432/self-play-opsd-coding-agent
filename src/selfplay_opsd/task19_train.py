"""Frozen-snapshot objectives and the transactional Task 19 training loop."""

from __future__ import annotations

import ast
import fcntl
import math
import os
import random
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import torch
from torch import Tensor

from selfplay_opsd.core import OracleSpec, ParentProgram, TaskSpec, digest, keyed_seed, read_json
from selfplay_opsd.data import TaskRepository, build_task
from selfplay_opsd.objectives import alignment, difficulty
from selfplay_opsd.protocol import teacher_messages
from selfplay_opsd.sandbox import SandboxBackend
from selfplay_opsd.storage import CheckpointManager, atomic_json, file_hash, restore_rng, rng_state
from selfplay_opsd.task19 import (
    GROUP,
    PILOT,
    Arm,
    Task19Control,
    centered_rewards,
    clipped_sum,
    opsd_sum,
)
from selfplay_opsd.task19_policy import (
    PROPOSER_SAMPLING,
    STUDENT_SAMPLING,
    SharedPolicy,
    visibility,
)
from selfplay_opsd.task19_protocol import STUDENT_TOOLS, parse_call, render_call, student_history
from selfplay_opsd.task19_rollout import Proposal, Repair, privilege, propose, repair

Grads = list[Tensor]


def add_gradient(target: Grads, source: Grads, scale: float = 1.0) -> None:
    for total, value in zip(target, source, strict=True):
        total.add_(value, alpha=scale)


def norm(values: Grads) -> float:
    return math.sqrt(
        sum(float(torch.linalg.vector_norm(value.double()).square().item()) for value in values)
    )


def joint_step(
    policy: SharedPolicy, student: Grads, proposer: Grads, alpha: float, clip: float
) -> dict:
    params = policy.parameters
    combined = [g.add(p, alpha=alpha) for g, p in zip(student, proposer, strict=True)]
    joint_norm = norm(combined)
    if not math.isfinite(joint_norm):
        raise FloatingPointError("Nonfinite shared joint gradient")
    before = [p.detach().clone() for p in params]
    policy.optimizer.zero_grad(set_to_none=True)
    if joint_norm:
        for parameter, gradient in zip(params, combined, strict=True):
            parameter.grad = gradient
        try:
            torch.nn.utils.clip_grad_norm_(params, clip, error_if_nonfinite=True)
            policy.optimizer.step()
        finally:
            policy.optimizer.zero_grad(set_to_none=True)
    displacement = norm([p.detach() - old for p, old in zip(params, before, strict=True)])
    if not math.isfinite(displacement):
        raise FloatingPointError("Nonfinite shared adapter displacement")
    score = alignment(student, proposer)
    return {
        "student_gradient_norm": score.candidate_norm,
        "proposer_gradient_norm": score.reference_norm,
        "weighted_proposer_gradient_norm": alpha * score.reference_norm,
        "student_proposer_cosine": score.cosine,
        "joint_gradient_norm": joint_norm,
        "joint_clip": joint_norm > clip,
        "update_norm": displacement,
        "optimizer_step": bool(joint_norm),
    }


def exact_reference(source: str, correct: str) -> tuple[str, str]:
    """Find a compact replacement containing the differing span and unique context."""
    before, after = source.splitlines(keepends=True), correct.splitlines(keepends=True)
    left = 0
    while left < min(len(before), len(after)) and before[left] == after[left]:
        left += 1
    right = 0
    while right < min(len(before), len(after)) - left and before[-1 - right] == after[-1 - right]:
        right += 1
    while True:
        end = len(before) - right
        old = "".join(before[left:end])
        if old and source.find(old, source.find(old) + 1) == -1:
            return old, "".join(after[left : len(after) - right])
        if left:
            left -= 1
        elif right:
            right -= 1
        else:
            raise ValueError("No unique native reference span")


class Study:
    student_group_size = GROUP

    def repair_episode(self, task: TaskSpec, oracle: OracleSpec, seed: int) -> Repair:
        return repair(self.policy, self.sandbox, self.cfg, task, oracle, seed)

    def __init__(
        self,
        cfg: Task19Control,
        arm: Arm,
        policy: SharedPolicy,
        sandbox: SandboxBackend,
        repository: TaskRepository,
        output: Path,
        execution: dict,
    ):
        self.cfg, self.arm, self.policy = cfg, arm, policy
        self.sandbox, self.repository, self.output = sandbox, repository, output
        self.execution = execution
        self.parents = sorted(repository.parents("train"), key=lambda parent: parent.parent_id)
        if not self.parents:
            raise ValueError("Missing training parents")
        self.iteration = self.episodes = self.tokens = self.proposal_groups = self.empty_batches = 0
        self.seen: set[str] = set()
        self.latest: Path | None = None
        self.protected = {0}
        self.checkpoints = CheckpointManager(output)
        self.snapshot = policy.version()
        self.initial = self.snapshot

    @property
    def identity(self) -> dict:
        return {
            "study": self.cfg.study,
            "science": self.cfg.science,
            "arm": self.arm,
            "data": self.repository.hash,
            "execution": self.execution,
            "initial": self.initial,
            "worker": self.sandbox.identity(),
        }

    def zeros(self) -> Grads:
        return [torch.zeros_like(p) for p in self.policy.parameters]

    def accumulate(self, out: Grads, loss: Tensor, *, retain_graph: bool = False) -> None:
        grads = torch.autograd.grad(
            loss, self.policy.parameters, allow_unused=True, retain_graph=retain_graph
        )
        for total, grad in zip(out, grads, strict=True):
            if grad is not None:
                total.add_(grad.detach())

    def parent(self, group: int) -> ParentProgram:
        cycle, position = divmod(group, len(self.parents))
        order = list(self.parents)
        random.Random(keyed_seed(self.cfg.seed, "parent-order", cycle)).shuffle(order)
        return order[position]

    def collect(self, maximum_tasks: int) -> tuple[list[list[Proposal]], dict, dict]:
        tasks: dict[str, tuple[TaskSpec, OracleSpec]] = {}
        groups: list[list[Proposal]] = []
        timing = {"proposer_generation": 0.0, "proposal_validation": 0.0}
        for _ in range(16):
            remaining = maximum_tasks - len(tasks)
            if remaining <= 0 or len(tasks) >= 4:
                break
            parent = self.parent(self.proposal_groups)
            rows = []
            for slot in range(min(GROUP, remaining)):
                started = time.monotonic()
                row, edit = propose(self.policy, self.cfg, parent, self.proposal_groups, slot)
                timing["proposer_generation"] += time.monotonic() - started
                if edit is not None:
                    started = time.monotonic()
                    key = digest([parent.parent_id, edit.dedup_key])
                    row.duplicate = key in tasks or key in self.seen
                    try:
                        if key not in tasks:
                            # Verification is deterministic for this dataset and worker.
                            cache = self.output / "validation" / f"{key}.json"
                            if cache.is_file():
                                saved = read_json(cache)
                                if (
                                    saved["data"] != self.repository.hash
                                    or saved["worker"] != self.sandbox.identity()
                                ):
                                    raise RuntimeError(
                                        "Cached proposal verification identity differs"
                                    )
                                task = TaskSpec.model_validate(saved["task"])
                                oracle = OracleSpec.model_validate(saved["oracle"])
                            else:
                                task, oracle = build_task(
                                    parent,
                                    edit,
                                    *self.repository.parent_cases(parent.parent_id),
                                    "train",
                                    keyed_seed(self.cfg.seed, "mutation", key),
                                    self.sandbox,
                                )
                                atomic_json(
                                    cache,
                                    {
                                        "data": self.repository.hash,
                                        "worker": self.sandbox.identity(),
                                        "task": task.model_dump(mode="json"),
                                        "oracle": oracle.model_dump(mode="json"),
                                    },
                                )
                            history = student_history(task, self.cfg.episode)
                            prefix = self.policy.encode(history, STUDENT_TOOLS, True)
                            teacher = self.policy.encode(
                                teacher_messages(history, oracle), STUDENT_TOOLS, True
                            )
                            if len(teacher) - len(prefix) > self.cfg.episode.privilege_reserve:
                                raise ValueError("Proposed task exceeds reserved privilege tokens")
                            tasks[key] = task, oracle
                        row.task = tasks[key][0].task_id
                        row.turn.result = "accepted"
                    except ValueError as exc:
                        row.invalid_reason = str(exc)[:300]
                    timing["proposal_validation"] += time.monotonic() - started
                rows.append(row)
            groups.append(rows)
            self.proposal_groups += 1
        self.seen.update(tasks)
        return groups, {task.task_id: (task, oracle) for task, oracle in tasks.values()}, timing

    def rollouts(self, tasks: dict, *, distill: bool) -> tuple[dict[str, list[Repair]], float]:
        rolls, teacher_seconds = {}, 0.0
        for task_id, (task, oracle) in tasks.items():
            rows = []
            for slot in range(self.student_group_size):
                seed = keyed_seed(self.cfg.seed, "student", self.iteration, task_id, slot)
                row = self.repair_episode(task, oracle, seed)
                if distill:
                    started = time.monotonic()
                    privilege(self.policy, self.cfg, oracle, row)
                    teacher_seconds += time.monotonic() - started
                rows.append(row)
            rolls[task_id] = rows
        return rolls, teacher_seconds

    def student_gradient(
        self, rolls: list[Repair], *, gated: bool, use_sd: bool
    ) -> tuple[Grads, Grads, dict]:
        if len(rolls) != self.student_group_size:
            raise ValueError("Incomplete student reward group")
        advantages = centered_rewards([float(row.success) for row in rolls]).tolist()
        rl, sd = self.zeros(), self.zeros()
        token_scale = 1 / (self.student_group_size * self.cfg.episode.max_generated_tokens)
        metrics: dict = {
            "rl_loss": 0.0,
            "sd_loss": 0.0,
            "ratio": [],
            "clipped": 0,
            "gap": [],
            "gate": [],
            "reasoning_signal": [],
            "tool_signal": [],
        }
        for roll, advantage in zip(rolls, advantages, strict=True):
            for turn in roll.turns:
                if not turn.ids or (advantage == 0 and not use_sd):
                    continue
                current, raw = self.policy.score(
                    turn.prompt,
                    turn.ids,
                    STUDENT_SAMPLING,
                    gradients=True,
                    exact_prefix=not self.cfg.cached_rollout,
                    raw_also=use_sd,
                )
                if advantage:
                    behavior = current.new_tensor(turn.logps)
                    ratios = (current.detach() - behavior).exp()
                    metrics["ratio"].extend(ratios.tolist())
                    metrics["clipped"] += int(
                        (
                            (ratios > 1 + self.cfg.clip)
                            if advantage > 0
                            else (ratios < 1 - self.cfg.clip)
                        ).sum()
                    )
                    loss = clipped_sum(current, behavior, advantage, self.cfg.clip) * token_scale
                    metrics["rl_loss"] += float(loss.detach())
                    self.accumulate(rl, loss, retain_graph=use_sd)
                if use_sd:
                    if (
                        raw is None
                        or turn.teacher_logps is None
                        or turn.frozen_student_logps is None
                    ):
                        raise ValueError("Missing detached frozen OPSD scores")
                    gap = raw.new_tensor(turn.teacher_logps) - raw.new_tensor(
                        turn.frozen_student_logps
                    )
                    gate = gap.sigmoid() if gated else torch.ones_like(gap)
                    loss = opsd_sum(raw, gap, gated=gated) * token_scale
                    metrics["sd_loss"] += float(loss.detach())
                    metrics["gap"].extend(gap.tolist())
                    metrics["gate"].extend(gate.tolist())
                    for kind, value in zip(
                        turn.token_kinds, (gate * gap.clamp(-5, 5)).tolist(), strict=True
                    ):
                        if kind in {"reasoning", "tool"}:
                            metrics[f"{kind}_signal"].append(value)
                    self.accumulate(sd, loss)
        return rl, sd, metrics

    def proposer_gradient(self, groups: list[list[Proposal]]) -> tuple[Grads, float]:
        total, objective = self.zeros(), 0.0
        for group in groups:
            advantages = centered_rewards([row.reward for row in group]).tolist()
            scale = 1 / (len(groups) * len(group) * self.cfg.proposal_tokens)
            for row, advantage in zip(group, advantages, strict=True):
                if row.turn.ids and advantage:
                    score = self.policy.score(
                        row.turn.prompt,
                        row.turn.ids,
                        PROPOSER_SAMPLING,
                        gradients=True,
                        exact_prefix=not self.cfg.cached_rollout,
                    )[0]
                    loss = (
                        clipped_sum(
                            score, score.new_tensor(row.turn.logps), advantage, self.cfg.clip
                        )
                        * scale
                    )
                    objective += float(loss.detach())
                    self.accumulate(total, loss)
        return total, objective

    def references(self) -> list[tuple[list[int], list[int], list[bool]]]:
        tasks = {
            task.task_id: (task, oracle) for task, oracle in self.repository.tasks("reference")
        }
        anchors = self.repository.anchors("reference")
        expected = {anchor["task_id"] for anchor in anchors}
        if not expected or not expected <= tasks.keys() or len(expected) != len(anchors):
            raise ValueError("Training reference anchors do not match the reference partition")
        records = []
        for task_id in sorted(expected):
            task, oracle = tasks[task_id]
            old, new = exact_reference(task.initial_source, oracle.correct_source)
            from selfplay_opsd.mutations import replace_text

            call = render_call(
                "replace_text",
                {
                    "path": "src/solution.py",
                    "old_text": old,
                    "new_text": new,
                },
            )
            _, args = parse_call(call, STUDENT_TOOLS)
            source = replace_text(task.initial_source, args["old_text"], args["new_text"])
            if (
                source != oracle.correct_source
                or not self.sandbox.verify(
                    source,
                    task.entry_point,
                    oracle.public_cases + oracle.hidden_cases,
                    task.verifier_seconds,
                ).passed
            ):
                raise ValueError("Native training reference did not restore verified source")
            prompt = self.policy.encode(
                student_history(task, self.cfg.episode), STUDENT_TOOLS, True
            )
            ids = self.policy.tokenizer.encode("</think>\n\n" + call, add_special_tokens=False)
            kinds, _ = visibility(ids, self.policy.markers, self.policy.eos, True)
            records.append((prompt, ids, [kind == "tool" for kind in kinds]))
        return records

    def reference_gradient(
        self, references: list[tuple[list[int], list[int], list[bool]]]
    ) -> Grads:
        out = self.zeros()
        for prompt, targets, tool in references:
            score = self.policy.score(prompt, targets, None, gradients=True)[0]
            mask = torch.tensor(tool, device=score.device)
            self.accumulate(out, -score[mask].mean() / len(references))
        if norm(out) <= 1e-12:
            raise ValueError("Training-reference gradient is undefined")
        return out

    def loss_scales(self) -> tuple[float, float]:
        lambda_sd, alpha = self.cfg.sd_scale, self.cfg.proposer_scale
        if lambda_sd is None or alpha is None:
            raise RuntimeError("Main optimization requires executed calibration")
        return lambda_sd, alpha

    def update_diagnostics(
        self, rl: Grads, sd: Grads, proposer: Grads, rolls: dict[str, list[Repair]], signals: list
    ) -> dict:
        return {}

    def candidate_reward(
        self, rl: Grads, sd: Grads, reference: Grads, scale: float, successes: list[bool]
    ) -> tuple[float, dict]:
        candidate = rl
        if self.arm == "D":
            candidate = [a.add(b, alpha=scale) for a, b in zip(rl, sd, strict=True)]
        result = alignment(candidate, reference)
        return (
            result.reward if result.cosine is not None else 0.0,
            {
                "cosine": result.cosine,
                "reference_norm": result.reference_norm,
                "candidate_norm": result.candidate_norm,
                "near_zero": result.near_zero,
            },
        )

    def update(self, maximum_tasks: int, references: list) -> dict:
        started = time.monotonic()
        groups, tasks, timing = self.collect(maximum_tasks)
        use_sd = self.arm != "control"
        rolls, timing["teacher_scoring"] = self.rollouts(tasks, distill=use_sd)
        alignment_arm = self.arm in {"C", "D"}
        started_gradient = time.monotonic()
        reference = self.reference_gradient(references) if alignment_arm and tasks else None
        timing["reference_gradient"] = time.monotonic() - started_gradient
        rl, sd = self.zeros(), self.zeros()
        lambda_sd, alpha = self.loss_scales()
        rewards: dict[str, tuple[float, dict | None]] = {}
        task_metrics = []
        alignment_seconds = 0.0
        for task_id, episodes in rolls.items():
            task_rl, task_sd, signals = self.student_gradient(
                episodes, gated=self.arm != "A", use_sd=use_sd
            )
            add_gradient(rl, task_rl, 1 / len(rolls))
            add_gradient(sd, task_sd, 1 / len(rolls))
            started_alignment = time.monotonic()
            if reference is not None:
                rewards[task_id] = self.candidate_reward(
                    task_rl, task_sd, reference, lambda_sd, [roll.success for roll in episodes]
                )
            else:
                rewards[task_id] = difficulty(float(roll.success) for roll in episodes), None
            alignment_seconds += time.monotonic() - started_alignment
            task_metrics.append(signals)
        timing["student_and_alignment_backward"] = (
            time.monotonic() - started_gradient - timing["reference_gradient"]
        )
        timing["alignment_reductions"] = alignment_seconds
        for group in groups:
            for row in group:
                if row.task is not None:
                    row.reward, row.alignment = rewards[row.task]
        started_proposer = time.monotonic()
        proposer, proposer_loss = self.proposer_gradient(groups)
        timing["proposer_backward"] = time.monotonic() - started_proposer
        interaction = alignment(rl, sd)
        student = [a.add(b, alpha=lambda_sd) for a, b in zip(rl, sd, strict=True)]
        count = sum(len(episodes) for episodes in rolls.values())
        samples = [row.metrics() for episodes in rolls.values() for row in episodes]
        self.episodes += count
        self.tokens += sum(row["tokens"] for row in samples)
        self.empty_batches = 0 if count else self.empty_batches + 1
        self.iteration += 1
        batch = self.output / "batches" / f"{self.iteration:06d}.json"
        atomic_json(
            batch,
            {
                "policy": self.snapshot,
                "proposals": [[row.model_dump(mode="json") for row in group] for group in groups],
                "repairs": {
                    key: [row.model_dump(mode="json") for row in rows]
                    for key, rows in rolls.items()
                },
            },
        )
        started_step = time.monotonic()
        shared = joint_step(self.policy, student, proposer, alpha, self.cfg.grad_clip)
        timing["optimizer"] = time.monotonic() - started_step
        self.snapshot = self.policy.version()
        distinct = sum(len({r.reward for r in group if r.task is not None}) > 1 for group in groups)
        valid = [row for group in groups for row in group if row.task is not None]
        outcomes = Counter(sum(row.success for row in episodes) for episodes in rolls.values())
        ratios = [value for signals in task_metrics for value in signals["ratio"]]
        metrics = {
            "iteration": self.iteration,
            "attempted_episodes": self.episodes,
            "student_tokens": self.tokens,
            "batch_episodes": count,
            "student": {
                "verified": sum(row["success"] for row in samples),
                "public_pass": sum(row["public_pass"] for row in samples),
                "hidden_pass": sum(row["hidden_pass"] for row in samples),
                "mixed_groups": sum(outcomes[n] for n in range(1, self.student_group_size)),
                "all_fail_groups": outcomes[0],
                "all_pass_groups": outcomes[self.student_group_size],
                **{
                    field: sum(row[field] for row in samples)
                    for field in (
                        "reasoning_tokens",
                        "tool_tokens",
                        "other_sampled_tokens",
                        "accepted_edits",
                        "rejected_edits",
                        "invalid_calls",
                        "truncated_thinking",
                        "reads",
                        "test_calls",
                        "turns",
                    )
                },
                "failure_categories": dict(Counter(row["failure"] for row in samples)),
            },
            "proposer": {
                "groups": len(groups),
                "attempted": sum(map(len, groups)),
                "generated_tokens": sum(len(row.turn.ids) for group in groups for row in group),
                "valid": len(valid),
                "unique": len(rolls),
                "duplicates": sum(row.duplicate for row in valid),
                "distinct_reward_groups": distinct,
                "loss": proposer_loss,
                "families": dict(Counter(row.family for row in valid)),
                "mean_reward": average([row.reward for group in groups for row in group]),
                "mean_solve_rate": average(
                    [sum(row.success for row in group) / len(group) for group in rolls.values()]
                ),
                "reward_variance": average(
                    [
                        (row.reward - sum(x.reward for x in group) / len(group)) ** 2
                        for group in groups
                        for row in group
                    ]
                ),
                "mean_alignment": average(
                    [
                        row.alignment["cosine"]
                        for row in valid
                        if row.alignment is not None and row.alignment["cosine"] is not None
                    ]
                ),
                "near_zero_alignments": sum(
                    bool(row.alignment and row.alignment["near_zero"]) for row in valid
                ),
            },
            "rl": {
                "loss": average([item["rl_loss"] for item in task_metrics]),
                "norm": interaction.candidate_norm,
                "clip_fraction": sum(item["clipped"] for item in task_metrics)
                / max(1, len(ratios)),
                "ratio_min": min(ratios, default=0),
                "ratio_max": max(ratios, default=0),
                "behavior_drift": average([-math.log(ratio) for ratio in ratios if ratio > 0]),
                "support_exits": ratios.count(0.0),
            },
            "opsd": {
                "loss": average([item["sd_loss"] for item in task_metrics]),
                "gradient_norm": interaction.reference_norm,
                "rl_cosine": interaction.cosine,
                **{
                    f"mean_{field}": average(
                        [value for item in task_metrics for value in item[field]]
                    )
                    for field in ("gap", "gate", "reasoning_signal", "tool_signal")
                },
                "negative_gap_fraction": average(
                    [float(value < 0) for item in task_metrics for value in item["gap"]]
                ),
            },
            "shared": shared,
            "timing": timing
            | {
                "total": time.monotonic() - started,
                "student_generation": sum(row["timings"]["generation"] for row in samples),
                "tools": sum(row["timings"]["tool"] for row in samples),
                "verifier": sum(row["timings"]["verifier"] for row in samples),
            },
            "gpu_peak_bytes": torch.cuda.max_memory_allocated(),
            "worker_retries": self.sandbox.infrastructure_retries,
        }
        metrics.update(self.update_diagnostics(rl, sd, proposer, rolls, task_metrics))
        self.commit(metrics, batch)
        return metrics

    def commit(self, metrics: dict, batch: Path | None) -> None:
        if self.episodes in self.cfg.milestones():
            self.protected.add(self.iteration)
        record = {
            "identity": self.identity,
            "scientific_hash": self.cfg.science,
            "manifests_hash": self.repository.hash,
            "total_episodes": self.episodes,
            "recorded_at": time.time(),
            "allocation": os.getenv("SLURM_JOB_ID"),
            "snapshot": self.snapshot,
            "episodes": self.episodes,
            "iteration": self.iteration,
            "metrics": metrics,
            "batch": {"path": batch.name, "sha256": file_hash(batch)} if batch else None,
        }
        state = {
            "policy": self.policy.state(),
            "rng": rng_state(),
            "tokens": self.tokens,
            "proposal_groups": self.proposal_groups,
            "seen": self.seen,
            "protected": self.protected,
            "empty_batches": self.empty_batches,
        }
        self.latest = self.checkpoints.save(self.iteration, state, record)
        atomic_json(self.output / "iterations" / f"{self.iteration:06d}.json", record)
        self.checkpoints.prune(self.protected)

    def restore(self, path: Path) -> None:
        state, record = self.checkpoints.load(path)
        if record["identity"] != self.identity:
            raise ValueError("Task 19 checkpoint scientific identity differs")
        if record["batch"] is not None:
            entry = record["batch"]
            if file_hash(self.output / "batches" / entry["path"]) != entry["sha256"]:
                raise ValueError("Committed trajectory checksum differs")
        self.policy.restore(state["policy"])
        restore_rng(state["rng"])
        self.snapshot = self.policy.version()
        if self.snapshot != record["snapshot"]:
            raise ValueError("Restored shared adapter signature differs")
        self.iteration, self.episodes = record["iteration"], record["episodes"]
        self.tokens, self.proposal_groups = state["tokens"], state["proposal_groups"]
        self.seen, self.protected = state["seen"], state["protected"]
        self.empty_batches = state["empty_batches"]
        self.latest = path
        atomic_json(self.output / "iterations" / f"{self.iteration:06d}.json", record)

    def evaluate(self, split: str, should_stop: Callable[[], bool] = lambda: False) -> dict:
        path = self.output / "evaluation" / f"{self.episodes:04d}" / split
        tasks = self.repository.tasks(split)
        if not tasks:
            raise ValueError(f"No frozen evaluation tasks for {split}")
        rows = []
        for task, oracle in tasks:
            if should_stop():
                raise SegmentStop("Evaluation stopped between verified task records")
            location = path / "tasks" / f"{task.task_id}.json"
            seed = keyed_seed(self.cfg.seed, "evaluation", split, task.task_id)
            identity = {
                "task": task.task_id,
                "data": self.repository.hash,
                "snapshot": self.snapshot,
                "seed": seed,
                "science": self.cfg.science,
            }
            if location.is_file():
                row = read_json(location)
                if row["identity"] != identity:
                    raise ValueError("Existing evaluation has a different frozen identity")
            else:
                roll = self.repair_episode(task, oracle, seed)
                row = {
                    "identity": identity,
                    "metrics": roll.metrics()
                    | {
                        "source_lines": len(task.initial_source.splitlines()),
                        "specification_chars": len(task.specification),
                        "ast_nodes": sum(1 for _ in ast.walk(ast.parse(task.initial_source))),
                        "mutation_family": oracle.provenance.get("mutation_family", "unavailable"),
                    },
                    "repair": roll.model_dump(mode="json"),
                }
                atomic_json(location, row)
            rows.append(row)
        summary = {
            "split": split,
            "identity": self.identity,
            "recorded_at": time.time(),
            "snapshot": self.snapshot,
            "episodes": self.episodes,
            "count": len(rows),
            "verified": sum(row["metrics"]["success"] for row in rows),
            "tasks": [
                row["metrics"]
                | {
                    "task_id": row["identity"]["task"],
                    "parent_id": row["repair"]["parent_id"],
                }
                for row in rows
            ],
        }
        atomic_json(path / "summary.json", summary)
        return summary

    def train(self, *, pilot: bool, publish, should_stop: Callable[[], bool]) -> None:
        limit = PILOT if pilot else self.cfg.endpoint
        if not self.cfg.ready or limit is None:
            raise ValueError("Training requires the frozen Task 19 control")
        with (self.output / ".run.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            pointer = self.checkpoints.root / "latest.json"
            if pointer.is_file():
                self.restore(self.checkpoints.root / read_json(pointer)["path"])
            else:
                self.commit({}, None)
            if self.episodes > limit:
                raise ValueError("Run already crossed this requested stopping exposure")
            references = (
                self.references() if self.arm in {"C", "D"} and self.episodes < limit else []
            )
            milestones = [v for v in self.cfg.milestones() if v <= limit]
            while not should_stop():
                if self.episodes in milestones:
                    summary_path = (
                        self.output
                        / "evaluation"
                        / f"{self.episodes:04d}"
                        / "dev_monitor"
                        / "summary.json"
                    )
                    if not summary_path.is_file():
                        summary = self.evaluate("dev_monitor", should_stop)
                        publish(
                            {"development/repair_rate": summary["verified"] / summary["count"]},
                            self.iteration,
                        )
                if self.episodes == limit:
                    if not pilot:
                        for split in ("final_id", "external"):
                            summary = self.evaluate(split, should_stop)
                            publish(
                                {
                                    f"evaluation/{split}/repair_rate": summary["verified"]
                                    / summary["count"]
                                },
                                self.iteration,
                            )
                    atomic_json(
                        self.output / f"complete-{limit}.json",
                        {
                            "identity": self.identity,
                            "checkpoint": self.latest.name if self.latest else None,
                            "checkpoint_sha256": file_hash(self.latest / "complete.json")
                            if self.latest
                            else None,
                            "episodes": self.episodes,
                            "snapshot": self.snapshot,
                        },
                    )
                    return
                if should_stop():
                    break
                if self.empty_batches >= 4:
                    raise RuntimeError("Four bounded proposal refills produced no verified tasks")
                next_stop = min(value for value in milestones if value > self.episodes)
                metrics = self.update((next_stop - self.episodes) // GROUP, references)
                publish(metrics, self.iteration)
        raise SegmentStop("Task 19 segment reached its safe stop; resume its logical run")


class SegmentStop(RuntimeError):
    pass


def average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None
