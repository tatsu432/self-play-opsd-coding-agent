"""Isolated common-state update probes; never modify main-run artifacts."""

from __future__ import annotations

import copy
import dataclasses
import time
from functools import partial
from pathlib import Path

import torch

from selfplay_opsd.analysis import diagnostic_statistics, write_csv
from selfplay_opsd.config import RunConfig
from selfplay_opsd.core import ExperimentError, digest, keyed_seed
from selfplay_opsd.data import TaskRepository, build_task
from selfplay_opsd.environment import run_episode, score_teacher
from selfplay_opsd.objectives import alignment
from selfplay_opsd.policy import HFPolicy, LossExample
from selfplay_opsd.sandbox import SandboxBackend
from selfplay_opsd.storage import atomic_json, restore_rng, rng_state
from selfplay_opsd.trainer import reference_examples, student_examples


def ce_value(policy: HFPolicy, examples: list[LossExample]) -> float:
    return sum(float(-policy.score(x.prompt, x.targets).sum()) * x.token_weight for x in examples)


def common_gating(policy: HFPolicy, episodes: list, config: RunConfig) -> dict:
    versions = {role: policy.version(role) for role in policy.optimizers}
    a, b = config.model_copy(deep=True), config.model_copy(deep=True)
    a.method, b.method = "A", "B"
    examples_a, metrics_a = student_examples(episodes, a)
    examples_b, metrics_b = student_examples(episodes, b)
    ga, ma = policy.gradients(examples_a)
    gb, mb = policy.gradients(examples_b)
    result = {
        "gated_ungated_cosine": dataclasses.asdict(
            alignment(ga, gb, distributed=policy.distributed)
        ),
        "ungated": metrics_a,
        "gated": metrics_b,
        "ungated_gradient_norm": ma["grad_norm"],
        "gated_gradient_norm": mb["grad_norm"],
    }
    if any(policy.version(role) != version for role, version in versions.items()):
        raise ExperimentError("Scoring mutated adapter parameters")
    return result


def update_probe(
    policy: HFPolicy,
    episodes: list,
    config: RunConfig,
    reference: list[LossExample],
    diagnostic: list[LossExample],
) -> dict:
    state, random_state = copy.deepcopy(policy.state_dict()), rng_state()
    initial_hash = policy.version("student")
    optimizer_hash = _optimizer_digest(state["optimizers"])
    try:
        reference_gradient, _ = policy.gradients(reference)
        c_examples, _ = student_examples(episodes, config, kind="rl")
        d_examples, _ = student_examples(episodes, config)
        gc, mc = policy.gradients(c_examples)
        gd, md = policy.gradients(d_examples)
        c = alignment(gc, reference_gradient, config.objective.near_zero, policy.distributed)
        d = alignment(gd, reference_gradient, config.objective.near_zero, policy.distributed)
        del gc, gd, reference_gradient
        before_ref, before_diag = ce_value(policy, reference), ce_value(policy, diagnostic)
        _, update = policy.gradients(d_examples, step=True)
        after_ref, after_diag = ce_value(policy, reference), ce_value(policy, diagnostic)
        return {
            "c": dataclasses.asdict(c),
            "d": dataclasses.asdict(d),
            "rl_gradient_norm": mc["grad_norm"],
            "combined_gradient_norm": md["grad_norm"],
            "reference_ce_decrease": before_ref - after_ref,
            "diagnostic_ce_decrease": before_diag - after_diag,
            "update": update,
            "initial_hash": initial_hash,
        }
    finally:
        policy.load_state_dict(state)
        restore_rng(random_state)
        if (
            policy.version("student") != initial_hash
            or _optimizer_digest(policy.state_dict()["optimizers"]) != optimizer_hash
        ):
            raise ExperimentError("Counterfactual optimizer/parameter restoration failed")


def _optimizer_digest(state) -> str:
    if isinstance(state, torch.Tensor):
        value = state.detach()
        if hasattr(value, "to_local"):
            value = value.to_local()
        return digest(
            [
                list(value.shape),
                str(value.dtype),
                value.cpu().contiguous().reshape(-1).view(torch.uint8).tolist(),
            ]
        )
    if isinstance(state, dict):
        return digest({str(k): _optimizer_digest(v) for k, v in state.items()})
    if isinstance(state, (list, tuple)):
        return digest([_optimizer_digest(x) for x in state])
    return digest(state)


def run_diagnostics(
    policy: HFPolicy,
    sandbox: SandboxBackend,
    repository: TaskRepository,
    config: RunConfig,
    output: Path,
    *,
    candidates: int = 16,
) -> dict:
    started = time.monotonic()
    if output.exists() and (output / "diagnostics.json").exists():
        raise ExperimentError("Diagnostic output already exists")
    output.mkdir(parents=True, exist_ok=True)
    initial_state, initial_rng = copy.deepcopy(policy.state_dict()), rng_state()
    parents = sorted(repository.parents("train"), key=lambda p: p.parent_id)
    pool_path = output.parent / "candidate_pool.json"
    frozen_proposer_version = policy.version("proposer")
    if pool_path.exists():
        from selfplay_opsd.core import OracleSpec, TaskSpec
        from selfplay_opsd.storage import read_json

        saved_pool = read_json(pool_path)
        if saved_pool["manifest_hash"] != repository.hash:
            raise ExperimentError("Frozen diagnostic pool belongs to a different dataset")
        frozen_proposer_version = saved_pool["proposer_version"]
        pool = [
            (TaskSpec.model_validate(r["task"]), OracleSpec.model_validate(r["oracle"]))
            for r in saved_pool["tasks"]
        ]
    else:
        # Build once from one frozen proposer snapshot, then persist and reuse the
        # exact pool at every checkpoint. No development/test parents are involved.

        from selfplay_opsd.mutations import strict_json_object, validate_mutation
        from selfplay_opsd.protocol import proposer_messages

        pool, history = [], set()
        with policy.active("proposer"):
            for attempt in range(config.diagnostic_attempts):
                parent = parents[attempt % len(parents)]
                seed = keyed_seed(config.seed, "diagnostic_proposal", parent.parent_id, attempt)
                prompt = policy.encode_history(proposer_messages(parent))
                generation = policy.generate(prompt, config.proposer_max_tokens, seed)
                try:
                    value = strict_json_object(generation.text)
                    if not isinstance(value, dict) or set(value) != {"patch"}:
                        continue
                    mutation = validate_mutation(parent.source, value["patch"])
                    key = (parent.parent_id, mutation.dedup_key)
                    if key in history:
                        continue
                    public, hidden = repository.parent_cases(parent.parent_id)
                    task, oracle = policy.coordinator_call(
                        partial(
                            build_task,
                            parent,
                            mutation,
                            public,
                            hidden,
                            "diagnostic_training",
                            seed,
                            sandbox,
                        )
                    )
                except (ValueError, SyntaxError):
                    continue
                history.add(key)
                pool.append((task, oracle))
                if len(pool) == candidates:
                    break
        if policy.is_coordinator:
            atomic_json(
                pool_path,
                {
                    "schema_version": 1,
                    "manifest_hash": repository.hash,
                    "proposer_version": frozen_proposer_version,
                    "tasks": [
                        {"task": t.model_dump(mode="json"), "oracle": o.model_dump(mode="json")}
                        for t, o in pool
                    ],
                },
            )
        policy.barrier()

    results, gating = [], []
    with policy.active("student"):
        reference = reference_examples(repository.anchors())
        diagnostic = reference_examples(repository.anchors("dev_diagnostic"))
        for index, (task, oracle) in enumerate(pool):
            policy.load_state_dict(initial_state)
            restore_rng(initial_rng)
            episodes = []
            for i in range(config.episodes_per_task):
                episode = run_episode(
                    policy,
                    sandbox,
                    task,
                    oracle,
                    config.episode,
                    keyed_seed(config.seed, "diagnostic_episode", task.task_id, i),
                )
                episodes.append(score_teacher(policy, episode, oracle, config.episode)[0])
            if policy.is_coordinator:
                atomic_json(
                    output / f"episodes-{index:03d}.json",
                    [e.model_dump(mode="json") for e in episodes],
                )
            results.append(
                {
                    "task_id": task.task_id,
                    **update_probe(policy, episodes, config, reference, diagnostic),
                }
            )
            gating.append({"task_id": task.task_id, **common_gating(policy, episodes, config)})
    policy.load_state_dict(initial_state)
    restore_rng(initial_rng)
    stats = {
        mode: diagnostic_statistics(
            [r[mode]["reward"] for r in results], [r["diagnostic_ce_decrease"] for r in results]
        )
        for mode in ("c", "d")
    }
    report = {
        "results": results,
        "common_gating": gating,
        "statistics": stats,
        "requested": candidates,
        "realized": len(pool),
        "candidate_pool_hash": digest([{"task_id": t.task_id} for t, _ in pool]),
        "diagnostic_only": True,
        "proposer_version": frozen_proposer_version,
        "greedy_diagnostic_success": {
            "status": "unrun",
            "reason": "Default bounded probe uses separate diagnostic CE",
        },
        "compute": {
            "seconds": time.monotonic() - started,
            "allocated_gpu_hours": (time.monotonic() - started) * config.charged_gpu_count / 3600,
            "category": "diagnostics",
        },
        "student_version": policy.version("student"),
    }
    if not policy.is_coordinator:
        return report
    atomic_json(output / "diagnostics.json", report)
    write_csv(output / "diagnostic_utility.csv", [{"mode": k, **v} for k, v in stats.items()])
    if results:
        from selfplay_opsd.analysis import _figure

        for mode in ("c", "d"):
            _figure(
                output / f"{mode}-score-vs-improvement",
                [r[mode]["reward"] for r in results],
                [r["diagnostic_ce_decrease"] for r in results],
                "Alignment reward",
                "Separate diagnostic-development CE decrease",
                scatter=True,
            )
    return report
