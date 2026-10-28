"""Replay saved initial-snapshot tokens on the allocated production backend."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from selfplay_opsd.config import RunConfig
from selfplay_opsd.core import ExperimentError
from selfplay_opsd.policy import GenerationRequest
from selfplay_opsd.storage import CheckpointManager, atomic_json, file_hash, read_json


def score_saved(policy, episodes: list[dict], clip: float) -> dict:
    if any(e["policy_version"] != policy.version("student") for e in episodes):
        raise ExperimentError("Saved episode snapshot differs from student")
    requests = [
        (turn["prompt_ids"], turn["target_ids"], turn["old_logps"])
        for episode in episodes
        for turn in episode["turns"]
        if turn["target_ids"]
    ]
    differences = []
    with policy.active("student"):
        start = time.monotonic()
        for prompt, targets, old in requests:
            scores = policy.score(prompt, targets).detach().float().cpu()
            delta = scores - torch.tensor(old, dtype=torch.float32)
            if not torch.isfinite(delta).all():
                raise FloatingPointError("Nonfinite rollout/scoring difference")
            differences.extend(delta.tolist())
        seconds = time.monotonic() - start
    values = np.asarray(differences, dtype=float)
    absolute = np.abs(values)
    outside = (values < np.log(1 - clip)) | (values > np.log(1 + clip))
    return {
        "tokens": len(values),
        "scoring_seconds": seconds,
        "absolute_error_mean": float(absolute.mean()) if len(values) else None,
        "absolute_error_p99": float(np.quantile(absolute, 0.99)) if len(values) else None,
        "absolute_error_max": float(absolute.max()) if len(values) else None,
        "absolute_error_above_005": int((absolute > 0.05).sum()),
        "absolute_error_above_02": int((absolute > 0.2).sum()),
        "outside_grpo_clip": int(outside.sum()),
    }


def benchmark_scoring(policy, config: RunConfig, episodes: list[dict]) -> list[dict]:
    requests = [
        (turn["prompt_ids"], turn["target_ids"])
        for episode in episodes
        for turn in episode["turns"]
        if turn["target_ids"]
    ]
    original = config.hardware.microbatch_episodes
    reference = None
    results = []
    try:
        with policy.active("student"):
            for size in (1, 2, 4):
                config.hardware.microbatch_episodes = size
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                start = time.monotonic()
                scored = [s.detach().float().cpu() for s in policy.score_many(requests)]
                seconds = time.monotonic() - start
                if reference is None:
                    reference = scored
                difference = max(
                    (float((a - b).abs().max()) for a, b in zip(scored, reference, strict=True)),
                    default=0.0,
                )
                results.append(
                    {
                        "microbatch": size,
                        "seconds": seconds,
                        "max_abs_difference_from_one": difference,
                        "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated()
                        if torch.cuda.is_available()
                        else None,
                    }
                )
    finally:
        config.hardware.microbatch_episodes = original
    return results


def run(config: RunConfig, output: Path) -> dict:
    from selfplay_opsd.cli import make_policy

    source = config.output_root / "repair-selfplay-2026-efficiency-v1-pilot-h200/D-seed2026"
    shards = list((source / "trajectories").glob("*/000001-episodes.json"))
    if len(shards) != 1:
        raise ExperimentError("Expected exactly one original iteration-one episode shard")
    episodes = read_json(shards[0])
    proposals_path = shards[0].with_name("000001-proposals.json")
    proposals = read_json(proposals_path)
    if not episodes or len(proposals) != 16:
        raise ExperimentError("Incomplete fixed pilot workload")
    policy = make_policy(config)
    try:
        initial_scoring = score_saved(policy, episodes, config.objective.clip)
        scoring_batches = benchmark_scoring(policy, config, episodes)
        with policy.active("proposer"):
            if any(p["policy_version"] != policy.version("proposer") for p in proposals):
                raise ExperimentError("Saved proposal snapshot differs from fresh proposer")
            batch = [
                GenerationRequest(
                    p["turn"]["prompt_ids"],
                    min(
                        config.proposer_max_tokens,
                        config.episode.max_student_sequence_tokens - len(p["turn"]["prompt_ids"]),
                    ),
                    p["seed"],
                )
                for p in proposals
            ]
            start = time.monotonic()
            policy.prepare_rollout()
            rollout_startup_seconds = time.monotonic() - start
            start = time.monotonic()
            sequential = [policy.generate(r.prompt, r.max_tokens, r.seed) for r in batch]
            sequential_seconds = time.monotonic() - start
            start = time.monotonic()
            batched = policy.generate_many(batch)
            batched_seconds = time.monotonic() - start
            # Scoring sleeps vLLM on the production backend; next preparation wakes it.
            policy.score(batch[0].prompt, sequential[0].ids)
            policy.prepare_rollout()
            after_wake = policy.generate(batch[0].prompt, batch[0].max_tokens, batch[0].seed)
        matching_logprob_errors = [
            abs(x - y)
            for a, b in zip(sequential, batched, strict=True)
            if a.ids == b.ids
            for x, y in zip(a.logps, b.logps, strict=True)
        ]
        result = {
            "source_episode_sha256": file_hash(shards[0]),
            "source_proposal_sha256": file_hash(proposals_path),
            "student_version": episodes[0]["policy_version"],
            **initial_scoring,
            "scoring_microbatches": scoring_batches,
            "proposer_requests": len(batch),
            "rollout_startup_seconds": rollout_startup_seconds,
            "sequential_seconds": sequential_seconds,
            "batched_seconds": batched_seconds,
            "identical_output_ids": sum(
                a.ids == b.ids for a, b in zip(sequential, batched, strict=True)
            ),
            "matching_output_logprob_abs_max": max(matching_logprob_errors, default=0.0),
            "identical_ids_after_sleep_wake": sequential[0].ids == after_wake.ids,
            "sequential_tokens": sum(len(item.ids) for item in sequential),
            "batched_tokens": sum(len(item.ids) for item in batched),
            "note": "Fixed saved-snapshot replay; not a full-iteration benchmark.",
        }
        later = list((source / "trajectories").glob("*/000010-episodes.json"))
        checkpoint = source / "checkpoints/000009"
        if len(later) == 1 and checkpoint.is_dir():
            from selfplay_opsd.data import TaskRepository
            from selfplay_opsd.protocol import PROMPT_HASH

            state, metadata = CheckpointManager.load(checkpoint)
            if (
                metadata.get("iteration") != 9
                or metadata.get("artifact_kind") != "fully_resumable"
                or metadata.get("prompt_hash") != PROMPT_HASH
                or metadata.get("manifests_hash") != TaskRepository(config.data_root).hash
            ):
                raise ExperimentError("Saved iteration-nine checkpoint identity differs")
            later_episodes = read_json(later[0])
            if any(e["policy_version"] != metadata["student_version"] for e in later_episodes):
                raise ExperimentError("Saved iteration-ten episodes differ from checkpoint")
            policy.load_state_dict(state["policy"])
            result["later_snapshot"] = score_saved(policy, later_episodes, config.objective.clip)
            result["later_snapshot"]["source_episode_sha256"] = file_hash(later[0])
            result["later_snapshot"]["checkpoint_marker_sha256"] = file_hash(
                checkpoint / "complete.json"
            )
            with policy.active("student"):
                prompt = later_episodes[0]["turns"][0]["prompt_ids"]
                output_after_update = policy.generate(prompt, 32, config.seed)
                rescored = policy.score(prompt, output_after_update.ids).detach().float().cpu()
                errors = (
                    rescored - torch.tensor(output_after_update.logps, dtype=torch.float32)
                ).abs()
                policy.prepare_rollout()
                after_update_wake = policy.generate(prompt, 32, config.seed)
            result["later_snapshot"]["post_sync_logprob_abs_max"] = (
                float(errors.max()) if len(errors) else None
            )
            result["later_snapshot"]["post_sync_wake_ids_match"] = (
                output_after_update.ids == after_update_wake.ids
            )
        if policy.is_coordinator:
            atomic_json(output / "rollout-parity.json", result)
        return result
    finally:
        policy.close()
