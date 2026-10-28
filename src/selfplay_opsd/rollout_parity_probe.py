"""Replay saved initial-snapshot tokens on the allocated production backend."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from selfplay_opsd.config import RunConfig
from selfplay_opsd.core import ExperimentError
from selfplay_opsd.policy import GenerationRequest
from selfplay_opsd.storage import atomic_json, file_hash, read_json


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
        if any(e["policy_version"] != policy.version("student") for e in episodes):
            raise ExperimentError("Saved episode snapshot differs from fresh initial student")
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
            scoring_seconds = time.monotonic() - start
        values = np.asarray(differences, dtype=float)
        absolute = np.abs(values)
        clip = config.objective.clip
        outside = (values < np.log(1 - clip)) | (values > np.log(1 + clip))
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
            sequential = [policy.generate(r.prompt, r.max_tokens, r.seed) for r in batch]
            sequential_seconds = time.monotonic() - start
            start = time.monotonic()
            batched = policy.generate_many(batch)
            batched_seconds = time.monotonic() - start
        result = {
            "source_episode_sha256": file_hash(shards[0]),
            "source_proposal_sha256": file_hash(proposals_path),
            "student_version": episodes[0]["policy_version"],
            "tokens": len(values),
            "scoring_seconds": scoring_seconds,
            "absolute_error_mean": float(absolute.mean()) if len(values) else None,
            "absolute_error_p99": float(np.quantile(absolute, 0.99)) if len(values) else None,
            "absolute_error_max": float(absolute.max()) if len(values) else None,
            "absolute_error_above_005": int((absolute > 0.05).sum()),
            "absolute_error_above_02": int((absolute > 0.2).sum()),
            "outside_grpo_clip": int(outside.sum()),
            "proposer_requests": len(batch),
            "sequential_seconds": sequential_seconds,
            "batched_seconds": batched_seconds,
            "identical_output_ids": sum(
                a.ids == b.ids for a, b in zip(sequential, batched, strict=True)
            ),
            "sequential_tokens": sum(len(item.ids) for item in sequential),
            "batched_tokens": sum(len(item.ids) for item in batched),
            "note": "Fixed initial-snapshot replay; not a full-iteration benchmark.",
        }
        if policy.is_coordinator:
            atomic_json(output / "rollout-parity.json", result)
        return result
    finally:
        policy.close()
