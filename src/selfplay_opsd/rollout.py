"""Bounded tool/execution overlap with ordered, frozen-policy episode results."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import cast

from selfplay_opsd.core import EpisodeRecord, ExperimentError, Status
from selfplay_opsd.environment import run_episode
from selfplay_opsd.policy import HFPolicy


def episode_wave(policy, sandbox, jobs, limits, *, concurrency=4, completed=None):
    """Only rank zero runs tool threads; collective inference setup occurs first.

    Model generation is serialized behind one lock (including debug RNG state),
    while independent public/final sandbox calls overlap. Semantic seeds and
    canonical return order never depend on completion timing.
    """
    if not jobs:
        return []
    snapshot = policy.version("student")
    policy.prepare_rollout()
    results, error = None, None
    if policy.is_coordinator:
        lock = threading.Lock()

        class FrozenClient:
            is_coordinator = True
            tokenizer = policy.tokenizer
            encode_history = HFPolicy.encode_history

            def version(self, role):
                return snapshot

            def broadcast(self, value):
                return value

            def generate(self, prompt, max_tokens, seed, *, greedy=False):
                with lock:
                    return policy.generate_unicast(prompt, max_tokens, seed, greedy=greedy)

        client = cast(HFPolicy, FrozenClient())
        ordered: list[EpisodeRecord | None] = [None] * len(jobs)
        try:
            with ThreadPoolExecutor(
                max_workers=concurrency, thread_name_prefix="repair-tools"
            ) as pool:
                futures = {
                    pool.submit(run_episode, client, sandbox, task, oracle, limits, seed): index
                    for index, (task, oracle, seed) in enumerate(jobs)
                }
                for future in as_completed(futures):
                    index = futures[future]
                    ordered[index] = future.result()
                    if completed:
                        completed(index, ordered[index])
            results = ordered
        except Exception as exc:
            error = (
                str(exc),
                exc.status.value if isinstance(exc, ExperimentError) else Status.INFRA.value,
            )
    results, error = policy.broadcast((results, error))
    if error:
        raise ExperimentError(error[0], Status(error[1]))
    return results
