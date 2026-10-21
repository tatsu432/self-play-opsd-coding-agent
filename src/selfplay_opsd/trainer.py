"""Transactional finite-horizon self-play. Framework details stay in the policy."""

from __future__ import annotations

import dataclasses
import os
import random
import signal
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any

import torch

from selfplay_opsd.config import RunConfig
from selfplay_opsd.core import (
    CheckpointMetadata,
    EpisodeRecord,
    ExperimentError,
    IterationRecord,
    OracleSpec,
    ProposalRecord,
    Status,
    TaskSpec,
    TokenTurn,
    digest,
    keyed_seed,
    read_json,
)
from selfplay_opsd.data import TaskRepository, build_task
from selfplay_opsd.environment import check_admission, score_teacher
from selfplay_opsd.mutations import strict_json_object, validate_mutation
from selfplay_opsd.objectives import advantages, alignment, difficulty, sd_coefficients
from selfplay_opsd.observability import HealthMonitor, WandbSink, distribution, system_metrics
from selfplay_opsd.policy import HFPolicy, LossExample
from selfplay_opsd.protocol import PROMPT_HASH, proposer_messages
from selfplay_opsd.sandbox import SandboxBackend
from selfplay_opsd.storage import (
    CheckpointManager,
    EventStore,
    atomic_json,
    code_hash,
    provenance,
    restore_rng,
    rng_state,
    scientific_code_hash,
)


def student_examples(
    episodes: list[EpisodeRecord], config: RunConfig, *, tasks: int = 1, kind: str = "combined"
) -> tuple[list[LossExample], dict]:
    adv = advantages(torch.tensor([e.reward for e in episodes])).tolist()
    examples, gaps, gates, coefficients = [], [], [], []
    for episode, advantage in zip(episodes, adv, strict=True):
        count = sum(len(t.target_ids) for t in episode.turns)
        for turn in episode.turns:
            if not turn.target_ids:
                continue
            gap, gate, coefficient = sd_coefficients(
                torch.tensor(turn.teacher_logps),
                torch.tensor(turn.old_logps),
                config.gate != "none",
                config.objective.gate_beta,
                config.objective.gap_clip,
            )
            lam = config.objective.sd_lambda if config.method not in {"GRPO", "GRPO_KL"} else 0.0
            values = (coefficient * lam).tolist()
            examples.append(
                LossExample(
                    turn.prompt_ids,
                    turn.target_ids,
                    turn.old_logps,
                    advantage,
                    values,
                    1 / (max(1, count) * len(episodes) * tasks),
                    kind,
                    turn.base_logps if config.method == "GRPO_KL" else None,
                    config.objective.fixed_base_kl if kind in {"combined", "rl"} else 0.0,
                )
            )
            gaps.extend(gap.tolist())
            gates.extend(gate.tolist())
            coefficients.extend(coefficient.tolist())
    return examples, {
        "gap": distribution(gaps),
        "gate": distribution(gates),
        "signed_coefficient": distribution(coefficients),
        "gap_clip_fraction": sum(abs(g) > config.objective.gap_clip for g in gaps) / len(gaps)
        if gaps
        else None,
        "gate_saturation_fraction": sum(g < 0.01 or g > 0.99 for g in gates) / len(gates)
        if gates
        else None,
        "positive_gap_fraction": sum(g > 0 for g in gaps) / len(gaps) if gaps else None,
    }


def reference_examples(anchors: list[dict]) -> list[LossExample]:
    if not anchors:
        raise ExperimentError("No frozen canonical reference anchors", Status.ORACLE)
    return [
        LossExample(
            a["edit"]["prompt"],
            a["edit"]["targets"],
            [],
            0.0,
            [],
            1 / (len(anchors) * len(a["edit"]["targets"])),
            "ce",
        )
        for a in anchors
    ]


def proposer_examples(
    proposals: list[ProposalRecord], rewards: list[float], config: RunConfig
) -> list[LossExample]:
    expected = config.parents_per_iteration * config.proposals_per_parent
    if len(proposals) != expected or len(rewards) != expected:
        raise ValueError("Incomplete attempted-proposal batch")
    output = []
    for start in range(0, len(proposals), config.proposals_per_parent):
        group = proposals[start : start + config.proposals_per_parent]
        if len({p.parent_id for p in group}) != 1:
            raise ValueError("Proposer advantage group crossed parent prompts")
        adv = advantages(
            torch.tensor(rewards[start : start + config.proposals_per_parent])
        ).tolist()
        for proposal, advantage in zip(group, adv, strict=True):
            turn = proposal.turn
            output.append(
                LossExample(
                    turn.prompt_ids,
                    turn.target_ids,
                    turn.old_logps,
                    advantage,
                    [0.0] * len(turn.target_ids),
                    1 / (max(1, len(turn.target_ids)) * expected),
                    "rl",
                )
            )
    return output


class SelfPlayTrainer:
    def __init__(
        self,
        config: RunConfig,
        policy: HFPolicy,
        repository: TaskRepository,
        sandbox: SandboxBackend,
        *,
        resume: Path | None = None,
        development_enabled: bool = True,
    ):
        self.config, self.policy, self.repository, self.sandbox = (
            config,
            policy,
            repository,
            sandbox,
        )
        self.iteration, self.student_steps, self.proposer_steps = 0, 0, 0
        self.history: dict[str, set[str]] = {}
        self.best: tuple[float, int] | None = None
        self.development_enabled = development_enabled
        self.stop_requested = False
        self.session_started = time.monotonic()
        self.elapsed = 0.0
        self.allocated_gpu_hours = 0.0
        self.total_tokens = 0
        self.total_episodes = 0
        self.health = HealthMonitor()
        self._phase_totals: dict[str, float] = defaultdict(float)
        self._detail_times: dict[str, float] = defaultdict(float)
        self.root = config.run_dir
        self.checkpoints = CheckpointManager(self.root) if policy.is_coordinator else None
        if resume:
            state, metadata = CheckpointManager.load(resume)
            if (
                metadata["scientific_hash"] != config.scientific_hash
                or metadata["manifests_hash"] != repository.hash
                or metadata["prompt_hash"] != PROMPT_HASH
                or metadata.get("scientific_code_hash") != scientific_code_hash()
            ):
                raise ExperimentError(
                    "Resume scientific/model/data/prompt compatibility check failed"
                )
            policy.load_state_dict(state["policy"])
            restore_rng(state["rng"])
            self.iteration = state["iteration"]
            self.student_steps, self.proposer_steps = (
                state["student_steps"],
                state["proposer_steps"],
            )
            self.history = {k: set(v) for k, v in state["history"].items()}
            self.total_tokens = state.get("total_tokens", 0)
            self.total_episodes = state.get("total_episodes", 0)
            self.best = tuple(state["best"]) if state["best"] is not None else None
            self.elapsed, self.allocated_gpu_hours = state["elapsed"], state["allocated_gpu_hours"]
        elif self.policy.broadcast(
            bool(list((self.root / "events").glob("*.jsonl")))
            if self.policy.is_coordinator
            else None
        ):
            raise ExperimentError(
                "Logical run already exists; resume explicitly or use a new study ID"
            )
        identity = {
            "study_id": config.study_id,
            "logical_run_id": digest([config.study_id, config.method, config.seed])[:24],
            "config": config.method,
            "seed": config.seed,
        }
        self.store = (
            EventStore(self.root, identity, rollback_to=self.iteration if resume else None)
            if policy.is_coordinator
            else None
        )
        self.sink = WandbSink(config, self.store) if self.store else None
        if self.store:
            self.store.iteration = self.iteration
            atomic_json(self.root / "resolved_config.json", config.resolved())
            atomic_json(self.root / "provenance.json", provenance())
            atomic_json(self.root / "manifests/data.json", repository.manifest)
            self.store.emit("run_config", config.resolved(), committed=True)
        if not resume:
            initialization = {
                "student": policy.version("student"),
                "proposer": policy.version("proposer"),
                "model_revision": config.model.revision,
                "seed": config.seed,
            }

            def check_initialization():
                path = config.output_root / config.study_id / "initial_adapters.json"
                if path.exists() and read_json(path) != initialization:
                    raise ExperimentError("A-D initial adapter tensors/model/seed differ")
                atomic_json(path, initialization)

            policy.coordinator_call(check_initialization)
        self.parents = repository.parents("train")
        self.reference_anchors = repository.anchors() if config.candidate_objective else []
        if resume and "reference_anchors" in state:
            self.reference_anchors = state["reference_anchors"]
            if digest(self.reference_anchors) != metadata["reference_hash"]:
                raise ExperimentError("Resumed canonical reference trajectories changed")
        if len(self.parents) < config.parents_per_iteration:
            raise ExperimentError(
                "Insufficient training parents for fixed parent batch", Status.ORACLE
            )

    @contextmanager
    def phase(self, name: str):
        self.policy.barrier()
        start = time.monotonic()
        if self.store:
            self.store.health(name)
        try:
            yield
        finally:
            self.policy.barrier()
            seconds = time.monotonic() - start
            self._phase_totals[name] += seconds
            self.elapsed += seconds
            self.allocated_gpu_hours += seconds * self.config.charged_gpu_count / 3600
            if self.store:
                self.store.emit(
                    "phase_time",
                    {
                        "seconds": seconds,
                        "allocated_gpu_hours": seconds * self.config.charged_gpu_count / 3600,
                        "allocated_cpu_hours": seconds * self.config.hardware.cpu_workers / 3600,
                    },
                    committed=True,
                    phase=name,
                )

    def checkpoint(self) -> Path | None:
        with self.phase("checkpoint"):
            state: dict[str, Any] = {
                "policy": self.policy.state_dict(),
                "reference_anchors": self.reference_anchors,
                "rng": rng_state(),
                "iteration": self.iteration,
                "student_steps": self.student_steps,
                "proposer_steps": self.proposer_steps,
                "history": {k: sorted(v) for k, v in self.history.items()},
                "best": self.best,
                "total_tokens": self.total_tokens,
                "total_episodes": self.total_episodes,
                "elapsed": self.elapsed,
                "allocated_gpu_hours": self.allocated_gpu_hours,
            }
            metadata = {
                "schema_version": 1,
                "iteration": self.iteration,
                "scientific_hash": self.config.scientific_hash,
                "manifests_hash": self.repository.hash,
                "prompt_hash": PROMPT_HASH,
                "code_hash": code_hash(),
                "scientific_code_hash": scientific_code_hash(),
                "best_development": self.best,
                "resolved_config": self.config.resolved(),
                "reference_hash": digest(self.reference_anchors),
                "student_version": self.policy.version("student"),
                "proposer_version": self.policy.version("proposer"),
                "world_size": self.policy.world_size,
                "parameter_names": state["policy"]["parameter_names"],
                "logical_run_id": self.store.identity["logical_run_id"] if self.store else None,
                "segment_id": self.store.segment if self.store else None,
                "artifact_kind": "fully_resumable",
            }
            path = (
                self.checkpoints.save(
                    self.iteration,
                    state,
                    CheckpointMetadata.model_validate(metadata).model_dump(mode="json"),
                )
                if self.checkpoints
                else None
            )
            if self.checkpoints:
                if path is not None and self.sink is not None:
                    self.sink.checkpoint(path, self.iteration)
                self.checkpoints.prune({0, self.iteration, self.best[1] if self.best else 0})
                for pending in (self.root / "state/pending").glob("*.json"):
                    if (
                        pending.name.split("-", 1)[0].isdigit()
                        and int(pending.name.split("-", 1)[0]) <= self.iteration
                    ):
                        pending.unlink()
        return path

    def proposals(
        self, k: int
    ) -> tuple[list[ProposalRecord], dict[str, tuple[TaskSpec, OracleSpec]]]:
        schedule = random.Random(keyed_seed(self.config.seed, "parents", k))
        parents = schedule.sample(
            sorted(self.parents, key=lambda p: p.parent_id), self.config.parents_per_iteration
        )
        proposals, tasks = [], {}
        pending: dict[str, set[str]] = {}
        with self.policy.active("proposer"):
            version = self.policy.version("proposer")
            for parent in parents:
                messages = proposer_messages(parent)
                prompt = self.policy.encode_history(messages)
                for slot in range(self.config.proposals_per_parent):
                    seed = keyed_seed(self.config.seed, "proposer", k, parent.parent_id, slot)
                    generation_started = time.monotonic()
                    output = self.policy.generate(
                        prompt,
                        min(
                            self.config.proposer_max_tokens,
                            self.config.episode.max_student_sequence_tokens - len(prompt),
                        ),
                        seed,
                    )
                    self._detail_times["propose"] += time.monotonic() - generation_started
                    validation_started = time.monotonic()
                    valid, reason, task_id, key, nodes, lines, family = (
                        False,
                        "invalid_json",
                        None,
                        None,
                        0,
                        0,
                        "unknown",
                    )
                    try:
                        value = strict_json_object(output.text)
                        if (
                            not isinstance(value, dict)
                            or set(value) != {"patch"}
                            or not isinstance(value["patch"], str)
                        ):
                            raise ValueError("One patch field required")
                        mutation = validate_mutation(parent.source, value["patch"])
                        key = mutation.dedup_key
                        if key in self.history.get(parent.parent_id, set()) or key in pending.get(
                            parent.parent_id, set()
                        ):
                            raise ValueError("duplicate")
                        public, hidden = self.repository.parent_cases(parent.parent_id)
                        result = self.policy.coordinator_call(
                            partial(
                                build_task,
                                parent,
                                mutation,
                                public,
                                hidden,
                                "train",
                                seed,
                                self.sandbox,
                            )
                        )
                        task, oracle = result
                        check_admission(self.policy, task, oracle, self.config.episode)
                        task_id, valid, reason = task.task_id, True, "valid"
                        nodes, lines, family = (
                            mutation.changed_nodes,
                            mutation.changed_lines,
                            mutation.family,
                        )
                        tasks[task_id] = task, oracle
                        pending.setdefault(parent.parent_id, set()).add(key)
                    except (ValueError, SyntaxError) as exc:
                        reason = str(exc)[:300]
                    self._detail_times["validate"] += time.monotonic() - validation_started
                    turn = TokenTurn(
                        prompt_ids=prompt,
                        target_ids=output.ids,
                        old_logps=output.logps,
                        history=messages,
                        raw_text=output.text,
                        action="propose_patch",
                        parse_status="valid" if valid else "invalid",
                    )
                    proposals.append(
                        ProposalRecord(
                            proposal_id=digest([k, parent.parent_id, slot, seed]),
                            parent_id=parent.parent_id,
                            slot=slot,
                            seed=seed,
                            policy_version=version,
                            turn=turn,
                            valid=valid,
                            reason=reason,
                            task_id=task_id,
                            dedup_key=key,
                            changed_nodes=nodes,
                            changed_lines=lines,
                            bug_family=family,
                        )
                    )
        return proposals, tasks

    def run_iteration(self, k: int) -> IterationRecord:
        cfg = self.config
        self._phase_totals.clear()
        self._detail_times.clear()
        if self.store:
            self.store.iteration = k
        snapshot, proposer_snapshot = (
            self.policy.version("student"),
            self.policy.version("proposer"),
        )
        cache = self.root / "state/pending" / f"{k:06d}-{snapshot}-{proposer_snapshot}.json"
        with self.phase("propose_validate"):
            if cache.exists():
                stage = read_json(cache)
                if stage["scientific_hash"] != cfg.scientific_hash:
                    raise ExperimentError("Pending batch scientific hash mismatch")
                proposals = [ProposalRecord.model_validate(p) for p in stage["proposals"]]
                tasks = {
                    key: (TaskSpec.model_validate(v[0]), OracleSpec.model_validate(v[1]))
                    for key, v in stage["tasks"].items()
                }
            else:
                proposals, tasks = self.proposals(k)
                stage = {
                    "scientific_hash": cfg.scientific_hash,
                    "proposals": [p.model_dump(mode="json") for p in proposals],
                    "tasks": {
                        key: [t.model_dump(mode="json"), o.model_dump(mode="json")]
                        for key, (t, o) in tasks.items()
                    },
                    "episodes": {},
                }
                if self.store:
                    atomic_json(cache, stage)
        # Canonical JSON sorts mapping keys. Restore scientific candidate order
        # from proposal slots, never from serialized mapping iteration order.
        tasks = {
            p.task_id: tasks[p.task_id] for p in proposals if p.valid and p.task_id is not None
        }
        groups: dict[str, list[EpisodeRecord]] = {}
        mapping_metrics = []
        with self.policy.active("student"), self.phase("rollout_reward"):
            from selfplay_opsd.rollout import episode_wave

            jobs = []
            positions = []
            for task_id, (task, oracle) in tasks.items():
                if task_id in stage["episodes"]:
                    groups[task_id] = [
                        EpisodeRecord.model_validate(e) for e in stage["episodes"][task_id]
                    ]
                else:
                    groups[task_id] = []
                    for index in range(cfg.episodes_per_task):
                        jobs.append(
                            (task, oracle, keyed_seed(cfg.seed, "student", k, task_id, index))
                        )
                        positions.append((task_id, index))
            finished: dict[str, dict[int, EpisodeRecord]] = {}

            def completed(position, episode):
                task_id, index = positions[position]
                finished.setdefault(task_id, {})[index] = episode
                if len(finished[task_id]) == cfg.episodes_per_task:
                    stage["episodes"][task_id] = [
                        finished[task_id][i].model_dump(mode="json")
                        for i in range(cfg.episodes_per_task)
                    ]
                    if self.store:
                        atomic_json(cache, stage)

            episodes = episode_wave(
                self.policy,
                self.sandbox,
                jobs,
                cfg.episode,
                concurrency=min(cfg.hardware.rollout_concurrency, cfg.hardware.cpu_workers),
                completed=completed,
            )
            for (task_id, _), episode in zip(positions, episodes, strict=True):
                groups[task_id].append(episode)
                self._detail_times["rollout"] += episode.timings.get(
                    "generate", 0.0
                ) + episode.timings.get("public_tests", 0.0)
                self._detail_times["reward_eval"] += episode.timings.get("reward_eval", 0.0)
        with self.phase("old_student_teacher_score"):
            for task_id in groups:
                scored = []
                for episode in groups[task_id]:
                    enriched, mappings, metrics = score_teacher(
                        self.policy, episode, tasks[task_id][1], cfg.episode
                    )
                    scored.append(enriched)
                    mapping_metrics.append(metrics)
                    self._detail_times["old_student_score"] += metrics["old_student_score_seconds"]
                    self._detail_times["teacher_score"] += metrics["teacher_score_seconds"]
                groups[task_id] = scored
        rewards = [-1.0] * len(proposals)
        alignments = []
        with self.policy.active("student"), self.phase("alignment"):
            reference = None
            if cfg.candidate_objective and tasks:
                reference_started = time.monotonic()
                reference, ref_metrics = self.policy.gradients(
                    reference_examples(self.reference_anchors)
                )
                self._detail_times["reference_gradient"] += time.monotonic() - reference_started
            for index, proposal in enumerate(proposals):
                if not proposal.valid:
                    continue
                assert proposal.task_id is not None
                episodes = groups[proposal.task_id]
                if cfg.candidate_objective:
                    examples, _ = student_examples(
                        episodes, cfg, kind="rl" if cfg.method == "C" else "combined"
                    )
                    candidate_started = time.monotonic()
                    gradients, _ = self.policy.gradients(examples)
                    self._detail_times["candidate_gradient"] += time.monotonic() - candidate_started
                    assert reference is not None
                    score = alignment(
                        gradients, reference, cfg.objective.near_zero, self.policy.distributed
                    )
                    rewards[index] = score.reward
                    alignments.append(dataclasses.asdict(score) | {"task_id": proposal.task_id})
                else:
                    rewards[index] = difficulty(e.reward for e in episodes)
        all_examples, distill = [], []
        for episodes in groups.values():
            examples, metrics = student_examples(episodes, cfg, tasks=len(groups))
            all_examples.extend(examples)
            distill.append(metrics)
        interaction: dict[str, Any] = {"sampled": False, "ratio": None, "cosine": None}
        if all_examples and (k % 20 == 0 or cfg.study_id.endswith("-pilot")):
            with self.policy.active("student"), self.phase("gradient_diagnostic"):
                rl_grad, rl_info = self.policy.gradients(
                    [dataclasses.replace(e, kind="rl") for e in all_examples]
                )
                sd_grad, sd_info = self.policy.gradients(
                    [dataclasses.replace(e, kind="sd") for e in all_examples]
                )
                interaction = {
                    "sampled": True,
                    "rl_norm": rl_info["grad_norm"],
                    "lambda_sd_norm": sd_info["grad_norm"],
                    "ratio": sd_info["grad_norm"] / rl_info["grad_norm"]
                    if rl_info["grad_norm"] > 0
                    else None,
                    "cosine": alignment(
                        rl_grad, sd_grad, cfg.objective.near_zero, self.policy.distributed
                    ).cosine,
                }
                del rl_grad, sd_grad
        with self.policy.active("student"), self.phase("student_update"):
            _, student_metrics = self.policy.gradients(all_examples, step=True)
        with self.policy.active("proposer"), self.phase("proposer_update"):
            _, proposer_metrics = self.policy.gradients(
                proposer_examples(proposals, rewards, cfg), step=True
            )
        self.student_steps += student_metrics["stepped"]
        self.proposer_steps += proposer_metrics["stepped"]
        for proposal in proposals:
            if proposal.valid:
                assert proposal.dedup_key is not None
                self.history.setdefault(proposal.parent_id, set()).add(proposal.dedup_key)
        episodes = [e for group in groups.values() for e in group]
        self.total_tokens += sum(len(t.target_ids) for e in episodes for t in e.turns)
        self.total_episodes += len(episodes)
        success_mix = Counter(int(sum(e.reward for e in group)) for group in groups.values())
        counts = Counter(p.reason for p in proposals if not p.valid)
        gaps = [
            float(x)
            for group in groups.values()
            for episode in group
            for turn in episode.turns
            for x in (torch.tensor(turn.teacher_logps) - torch.tensor(turn.old_logps))
        ]
        gates = (
            torch.sigmoid(torch.tensor(gaps) * cfg.objective.gate_beta).tolist()
            if cfg.gate != "none"
            else [1.0] * len(gaps)
        )
        metrics = {
            "train": {
                "outer_iteration": k,
                "student_steps": self.student_steps,
                "proposer_steps": self.proposer_steps,
            },
            "proposer": {
                "attempted_count": len(proposals),
                "valid_count": len(tasks),
                "valid_fraction": len(tasks) / len(proposals),
                "rewards": distribution(rewards),
                "invalid_reasons": dict(counts),
                "distinct_parent_count": len({p.parent_id for p in proposals}),
                "generated_tokens": sum(len(p.turn.target_ids) for p in proposals),
                "patch_changed_lines": distribution(
                    [p.changed_lines for p in proposals if p.valid]
                ),
                "patch_changed_nodes": distribution(
                    [p.changed_nodes for p in proposals if p.valid]
                ),
                "bug_families": dict(Counter(p.bug_family for p in proposals if p.valid)),
                "infrastructure_retries": self.sandbox.infrastructure_retries,
                "duplicate_count": sum("duplicate" in p.reason for p in proposals),
                **proposer_metrics,
            },
            "student": {
                "repair_success_fraction": sum(e.reward for e in episodes) / len(episodes)
                if episodes
                else None,
                "public_pass_fraction": sum(e.public_pass for e in episodes) / len(episodes)
                if episodes
                else None,
                "hidden_full_pass_fraction": sum(e.hidden_pass for e in episodes) / len(episodes)
                if episodes
                else None,
                "sampled_token_surprisal": distribution(
                    [-v for e in episodes for t in e.turns for v in t.old_logps]
                ),
                "generated_tokens_per_episode": distribution(
                    [sum(len(t.target_ids) for t in e.turns) for e in episodes]
                ),
                "invalid_action_fraction": sum(
                    t.parse_status != "valid" for e in episodes for t in e.turns
                )
                / max(1, sum(len(e.turns) for e in episodes)),
                "success_numerator": int(sum(e.reward for e in episodes)),
                "success_denominator": len(episodes),
                "success_group_mix": dict(success_mix),
                "mixed_group_fraction": sum(
                    0 < count < cfg.episodes_per_task for count in success_mix.elements()
                )
                / max(1, len(groups)),
                "cumulative_generated_tokens": self.total_tokens,
                "cumulative_episodes": self.total_episodes,
                "gradient_interaction": interaction,
                "tokens": sum(len(t.target_ids) for e in episodes for t in e.turns),
                "turns": distribution([len(e.turns) for e in episodes]),
                "test_calls": sum(e.test_calls for e in episodes),
                "public_pass_hidden_fail": sum(
                    e.public_pass and not e.hidden_pass for e in episodes
                ),
                "invalid_actions": sum(
                    t.parse_status != "valid" for e in episodes for t in e.turns
                ),
                "termination": dict(Counter(e.termination for e in episodes)),
                "pre_update_logprob_abs_max": max(
                    (m["pre_update_logprob_abs_max"] for m in mapping_metrics), default=0
                ),
                **student_metrics,
            },
            "distill": {
                "gating_enabled": cfg.gate != "none",
                "mapped_tokens": sum(m["mapped_tokens"] for m in mapping_metrics),
                "mapping_error_count": 0,
                "gap": distribution(gaps),
                "gate": distribution(gates),
                "gap_clip_fraction": sum(abs(g) > cfg.objective.gap_clip for g in gaps) / len(gaps)
                if gaps
                else None,
                "gate_saturation_fraction": sum(g < 0.01 or g > 0.99 for g in gates) / len(gates)
                if gates
                else None,
            },
            "alignment": {
                "available": bool(cfg.candidate_objective),
                "scored_count": len(alignments),
                "scores": alignments,
                "cosine": distribution(
                    [a["cosine"] for a in alignments if a["cosine"] is not None]
                ),
                "candidate_norm": distribution([a["candidate_norm"] for a in alignments]),
                "reference_norm": alignments[0]["reference_norm"] if alignments else None,
                "exact_zero_fraction": sum(a["exact_zero"] for a in alignments) / len(alignments)
                if alignments
                else None,
                "near_zero_fraction": sum(a["near_zero"] for a in alignments) / len(alignments)
                if alignments
                else None,
                "negative_alignment_fraction": sum(
                    a["cosine"] is not None and a["cosine"] < 0 for a in alignments
                )
                / len(alignments)
                if alignments
                else None,
            },
            "perf": {
                name + "_seconds": self._detail_times.get(name, self._phase_totals.get(name))
                for name in (
                    "propose",
                    "validate",
                    "rollout",
                    "reward_eval",
                    "old_student_score",
                    "teacher_score",
                    "reference_gradient",
                    "candidate_gradient",
                    "student_update",
                    "proposer_update",
                    "checkpoint",
                    "development_eval",
                )
            },
            "compute": {
                "active_seconds": self.elapsed,
                "allocated_gpu_hours": self.allocated_gpu_hours,
            },
        }
        self.iteration = k
        record = IterationRecord(
            iteration=k,
            snapshot=snapshot,
            proposer_snapshot=proposer_snapshot,
            attempted=len(proposals),
            valid=len(tasks),
            episodes=len(episodes),
            student_steps=student_metrics["stepped"],
            proposer_steps=proposer_metrics["stepped"],
            metrics=metrics,
        )
        if self.store:
            proposal_path = self.store.shard(
                k,
                "proposals",
                [
                    p.model_dump(mode="json") | {"reward": r}
                    for p, r in zip(proposals, rewards, strict=True)
                ],
            )
            episode_path = self.store.shard(
                k, "episodes", [e.model_dump(mode="json") for e in episodes]
            )
            self.store.emit(
                "iteration_commit",
                record.model_dump(mode="json")
                | {"proposals_path": proposal_path, "episodes_path": episode_path},
                committed=True,
            )
            self.store.emit("system", system_metrics(self.root), committed=True)
            for warning in self.health.inspect(metrics):
                self.store.emit("health_warning", {"reason": warning}, committed=True)
            assert self.sink is not None
            self.sink.log(metrics)
            if k % cfg.logging.sample_every == 0:
                chosen = sorted(
                    zip(proposals, rewards, strict=True),
                    key=lambda item: digest([cfg.seed, item[0].proposal_id]),
                )[: cfg.logging.table_rows]
                self.sink.table(
                    "examples/proposals",
                    ["id", "valid", "reason", "reward", "patch_json"],
                    [
                        [p.proposal_id, p.valid, p.reason, reward, p.turn.raw_text[:4000]]
                        for p, reward in chosen
                    ],
                )
                chosen_episodes = sorted(episodes, key=lambda e: digest([cfg.seed, e.episode_id]))[
                    : cfg.logging.table_rows
                ]
                self.sink.table(
                    "examples/episodes",
                    ["id", "reward", "termination", "actions", "local_shard"],
                    [
                        [
                            e.episode_id,
                            e.reward,
                            e.termination,
                            "\n".join(t.raw_text[:1000] for t in e.turns)[:4000],
                            episode_path,
                        ]
                        for e in chosen_episodes
                    ],
                )
        with self.phase("adapter_sync"):
            self.policy.sync_adapter("student")
            self.policy.sync_adapter("proposer")
            self.policy.prune_rollout_exports()
        return record

    def train(self):
        def request_stop(signum, frame):
            self.stop_requested = True

        handlers = {}
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
            handlers[sig] = signal.signal(sig, request_stop)
        try:
            if self.iteration == 0:
                self.checkpoint()
                self.evaluate_development()
            for k in range(self.iteration + 1, self.config.iterations + 1):
                self.run_iteration(k)
                if k % self.config.evaluate_every == 0:
                    self.evaluate_development()
                stop_file = Path(
                    os.environ.get("SELFPLAY_STOP_FILE", str(self.root / "state/stop_requested"))
                )
                self.stop_requested = self.stop_requested or stop_file.exists()
                exhausted = (
                    self.config.max_active_hours is not None
                    and self.elapsed / 3600 >= self.config.max_active_hours
                ) or (
                    self.config.max_gpu_hours is not None
                    and self.allocated_gpu_hours >= self.config.max_gpu_hours
                )
                if (
                    k % self.config.checkpoint_every == 0
                    or self.stop_requested
                    or exhausted
                    or k == self.config.iterations
                ):
                    self.checkpoint()
                if self.stop_requested or exhausted:
                    break
            if self.store:
                self.store.health(
                    "stopped" if self.iteration < self.config.iterations else "complete"
                )
        except BaseException as exc:
            if self.store:
                self.store.emit(
                    "error",
                    {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "resume_from": "latest verified checkpoint; pending optimizer updates are not resumable",
                    },
                    committed=True,
                )
                self.store.health("failed", error=str(exc))
            raise
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            if self.store:
                duration = time.monotonic() - self.session_started
                self.store.emit(
                    "resource",
                    {
                        "session_seconds": duration,
                        "allocated_gpu_hours": duration * self.config.charged_gpu_count / 3600,
                        "allocated_cpu_hours": duration * self.config.hardware.cpu_workers / 3600,
                        "includes_idle_and_abandoned_work": True,
                    },
                    committed=True,
                )
                self.store.materialize()
                assert self.sink is not None
                self.sink.finish()

    def evaluate_development(self):
        if not self.development_enabled:
            return
        from selfplay_opsd.evaluation import evaluate_tasks

        with self.phase("development_eval"), self.policy.active("student"):
            records = evaluate_tasks(
                self.policy,
                self.sandbox,
                self.repository,
                "dev_monitor",
                self.config.episode,
                self.iteration,
                self.config.seed,
                self.store,
            )
        successes: dict[str, list[bool]] = {record.parent_id: [] for record in records}
        for record in records:
            if record.success is None:
                raise ExperimentError("Incomplete development evaluation")
            successes[record.parent_id].append(record.success)
        score = sum(sum(outcomes) / len(outcomes) for outcomes in successes.values()) / max(
            1, len(successes)
        )
        if self.best is None or score > self.best[0]:
            self.best = (score, self.iteration)
            self.checkpoint()
        if self.store:
            assert self.sink is not None
            self.sink.log(
                {
                    "train/outer_iteration": self.iteration,
                    "compute/allocated_gpu_hours": self.allocated_gpu_hours,
                    "eval_by_iteration/dev_parent_macro": score,
                    "eval_by_gpu_hours/dev_parent_macro": score,
                }
            )
