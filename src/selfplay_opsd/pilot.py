"""Training-parent-only reference proxies for hardware/difficulty calibration."""

from __future__ import annotations

from functools import partial

from selfplay_opsd.core import ExperimentError, keyed_seed
from selfplay_opsd.data import build_task
from selfplay_opsd.environment import canonical_trajectory
from selfplay_opsd.mutations import fixed_mutation
from selfplay_opsd.storage import atomic_json, read_json


def training_reference_proxies(trainer) -> list[dict]:
    path = trainer.root / "manifests/pilot_reference_anchors.json"
    if path.exists():
        return read_json(path)
    target = len(trainer.reference_anchors)
    if target == 0:
        raise ExperimentError("No main-study reference anchor count available for pilot profiling")
    anchors = []
    parents = sorted(
        trainer.parents,
        key=lambda p: keyed_seed(trainer.config.seed, "pilot_reference_parent", p.parent_id),
    )
    for parent in parents:
        public, hidden = trainer.repository.parent_cases(parent.parent_id)
        for attempt in range(20):
            seed = keyed_seed(
                trainer.config.seed, "pilot_reference_mutant", parent.parent_id, attempt
            )
            try:
                mutation = fixed_mutation(parent.source, seed)
                task, oracle = trainer.policy.coordinator_call(
                    partial(
                        build_task,
                        parent,
                        mutation,
                        public,
                        hidden,
                        "pilot_reference",
                        seed,
                        trainer.sandbox,
                    )
                )
                anchor = canonical_trajectory(
                    trainer.policy, trainer.sandbox, task, oracle, trainer.config.episode
                )
            except (ValueError, SyntaxError):
                continue
            anchors.append(anchor)
            break
        if len(anchors) == target:
            break
    if len(anchors) != target:
        raise ExperimentError(
            "Training-only pilot could not match the main reference workload within fixed attempts"
        )
    if trainer.store:
        atomic_json(path, anchors)
    trainer.policy.barrier()
    return anchors
