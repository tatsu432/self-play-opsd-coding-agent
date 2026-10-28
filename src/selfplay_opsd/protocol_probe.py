"""One fixed training-parent protocol check, with no optimizer calls or refills."""

from __future__ import annotations

import json
from collections import Counter

from selfplay_opsd.core import ExperimentError, keyed_seed
from selfplay_opsd.environment import run_episode
from selfplay_opsd.protocol import PROMPT_HASH
from selfplay_opsd.storage import atomic_json
from selfplay_opsd.trainer import SelfPlayTrainer

BATCHES = 2
ATTEMPTS = 32


def run_probe(config, policy, repository, sandbox) -> dict:
    if (config.parents_per_iteration, config.proposals_per_parent, config.episodes_per_task) != (
        4,
        4,
        4,
    ):
        raise ExperimentError("Protocol probe requires the original mathematical batches")
    config = config.model_copy(
        update={"method": "B", "study_id": config.study_id + "-protocol-probe"}, deep=True
    )
    trainer = SelfPlayTrainer(config, policy, repository, sandbox, development_enabled=False)
    root = trainer.root
    declaration = {
        "batches": BATCHES,
        "attempts": ATTEMPTS,
        "max_episodes": 128,
        "optimizer_updates": 0,
        "prompt_hash": PROMPT_HASH,
        "runtime_identity": trainer.runtime_identity,
        "training_parents_only": True,
    }
    if policy.is_coordinator:
        atomic_json(root / "declaration.json", declaration)
    sampled = []

    def sampled_output(record):
        sampled.append(record)
        if policy.is_coordinator:
            atomic_json(root / "sampled_outputs.json", sampled)

    trainer.proposal_observer = sampled_output
    proposals, episodes = [], []
    try:
        for k in range(1, BATCHES + 1):
            with trainer.phase("protocol_proposals"):
                batch, tasks = trainer.proposals(k)
            proposals.extend(batch)
            if policy.is_coordinator:
                atomic_json(root / "proposals.json", [p.model_dump(mode="json") for p in proposals])
            for proposal in batch:
                if not proposal.valid:
                    continue
                assert proposal.task_id is not None and proposal.dedup_key is not None
                task, oracle = tasks[proposal.task_id]
                trainer.history.setdefault(proposal.parent_id, set()).add(proposal.dedup_key)
                for index in range(config.episodes_per_task):
                    with policy.active("student"), trainer.phase("protocol_episode"):
                        episode = run_episode(
                            policy,
                            sandbox,
                            task,
                            oracle,
                            config.episode,
                            keyed_seed(config.seed, "student", k, task.task_id, index),
                        )
                    episodes.append(episode)
                    if policy.is_coordinator:
                        atomic_json(
                            root / "episodes.json", [e.model_dump(mode="json") for e in episodes]
                        )
        meaningful = False
        for episode in episodes:
            read_seen = False
            for turn in episode.turns:
                if turn.parse_status == "valid":
                    if read_seen and turn.action in {"apply_patch", "run_tests"}:
                        meaningful = True
                    read_seen |= (
                        turn.action == "read_file"
                        and json.loads(turn.raw_text).get("path") == "src/solution.py"
                    )
        report = declaration | {
            "attempted": len(proposals),
            "valid": sum(p.valid for p in proposals),
            "rejections": dict(Counter(p.reason for p in proposals if not p.valid)),
            "episodes": len(episodes),
            "tokens": sum(len(t.target_ids) for e in episodes for t in e.turns),
            "invalid_actions": sum(t.parse_status != "valid" for e in episodes for t in e.turns),
            "source_read_then_action": meaningful,
            "ready": len(proposals) == ATTEMPTS
            and any(p.valid for p in proposals)
            and bool(episodes)
            and meaningful,
        }
        if policy.is_coordinator:
            atomic_json(root / "protocol_report.json", report)
        if not report["ready"]:
            raise ExperimentError(
                "Fixed protocol probe failed; inspect outputs before further experiments"
            )
        return report
    finally:
        if trainer.sink:
            trainer.sink.finish()
