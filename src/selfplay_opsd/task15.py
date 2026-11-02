"""Task 15: bounded online generation, explicit task replay, fresh student episodes.

No rejection sampling or constrained token decoding: every attempted proposal is
retained with its actual behavior probabilities. Only task admission is deduped;
repeated valid samples share a measured difficulty reward at this snapshot.
"""

from __future__ import annotations

import math
import random
from collections import Counter

from selfplay_opsd.core import (
    ExperimentError,
    OracleSpec,
    ProposalRecord,
    TaskSpec,
    digest,
    keyed_seed,
)
from selfplay_opsd.environment import check_admission
from selfplay_opsd.mutations import strict_json_object
from selfplay_opsd.structured_mutations import sites
from selfplay_opsd.trainer import SelfPlayTrainer


class OnlineExposureTrainer(SelfPlayTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.config.online_supply is None:
            raise ValueError("OnlineExposureTrainer requires an explicit supply policy")
        self.supply_state.setdefault("proposal_groups", 0)
        self.supply_state.setdefault("proposal_attempts", 0)
        self.supply_state.setdefault("reservoir", {})
        self.supply_state.setdefault("fallback_groups", 0)
        self.supply_state.setdefault("task_groups", 0)
        self.supply_state.setdefault("fallback_parent_uses", {})
        self.supply_state.setdefault("evaluated_milestones", [])
        self._verification_cache: dict = {}
        all_parents = len(self.parents)
        self.parents = [parent for parent in self.parents if sites(parent.source)]
        if not self.parents:
            raise ExperimentError("No training parent has a legal structured mutation site")
        self.supply_state["eligible_parents"] = len(self.parents)
        self.supply_state["parents_without_legal_sites"] = all_parents - len(self.parents)

    def proposer_prompt(self, parent):
        messages = super().proposer_prompt(parent)
        messages = [dict(message) for message in messages]
        messages[0]["content"] += (
            "\nPreviously used valid mutations remain admissible. Repeated samples "
            "are verified tasks, not invalid actions. Prefer diverse sites and operators."
        )
        return messages

    def proposal_parents(self, k):
        parents = sorted(self.parents, key=lambda p: p.parent_id)
        epoch, index = divmod(k - 1, len(parents))
        random.Random(keyed_seed(self.config.seed, "task15-parents", epoch)).shuffle(parents)
        return [parents[index]]

    def reject_duplicate(self, parent_id, key, pending):
        return False

    def verified_task(self, parent, mutation, seed):
        # Canonical task identity does not depend on the proposal attempt seed.
        # Dataset/runtime/scientific identity is checked at checkpoint restoration.
        key = digest([self.repository.hash, parent.parent_id, mutation.dedup_key])
        reservoir = self.supply_state["reservoir"]
        cached = self._verification_cache.get(key) or reservoir.get(key)
        if cached:
            if "rejection" in cached:
                raise ValueError(cached["rejection"])
            task = TaskSpec.model_validate(cached["task"])
            oracle = OracleSpec.model_validate(cached["oracle"])
            check_admission(self.policy, task, oracle, self.config.episode)
            return task, oracle
        try:
            task, oracle = super().verified_task(
                parent, mutation, keyed_seed(self.config.seed, "task15-task", key)
            )
        except (ValueError, SyntaxError) as exc:
            # Only deterministic semantic/admission rejection; infrastructure errors propagate.
            self._verification_cache[key] = {"rejection": str(exc)}
            raise
        value = {
            "task": task.model_dump(mode="json"),
            "oracle": oracle.model_dump(mode="json"),
            "admitted_group": self.supply_state["proposal_groups"],
        }
        self._verification_cache[key] = value
        supply = self.config.online_supply
        assert supply is not None
        count = sum(v["task"]["parent_id"] == parent.parent_id for v in reservoir.values())
        if count < supply.reservoir_per_parent:
            reservoir[key] = value
        return task, oracle

    def proposals(self, k):
        supply = self.config.online_supply
        assert supply is not None
        state = self.supply_state
        # Clear the transient cache at transaction boundaries. Retained entries are
        # part of the checksummed reservoir; no cached student trajectories exist.
        self._verification_cache = {}
        proposals: list[ProposalRecord] = []
        tasks: dict[str, tuple[TaskSpec, OracleSpec]] = {}
        remaining = math.ceil(
            (supply.target_episodes - self.total_episodes) / self.config.episodes_per_task
        )
        target = min(supply.tasks_per_update, remaining)
        # Do not cross an evaluation boundary by a whole accumulated update.
        for milestone in supply.evaluation_episodes:
            if milestone > self.total_episodes:
                target = min(
                    target,
                    math.ceil((milestone - self.total_episodes) / self.config.episodes_per_task),
                )
                break
        replays = 0
        while len(tasks) < target and len(proposals) < supply.max_proposals_per_update:
            if state["proposal_attempts"] >= supply.max_total_proposals:
                break
            state["proposal_groups"] += 1
            group, supplied = super().proposals(state["proposal_groups"])
            proposals.extend(group)
            state["proposal_attempts"] += len(group)
            for proposal in group:
                if proposal.valid and proposal.dedup_key in self.history.get(
                    proposal.parent_id, set()
                ):
                    replays += 1
            tasks.update(supplied)
        if not proposals:
            raise ExperimentError("Frozen online proposal budget exhausted")
        online_task_ids = set(tasks)
        fallback = None
        # At most one fallback; cumulative cap includes this update, never future credit.
        allowed = (state["fallback_groups"] + 1) / (state["task_groups"] + len(tasks) + 1)
        if len(tasks) < target and allowed <= supply.max_fallback_fraction:
            candidates = [
                v for v in state["reservoir"].values() if v["task"]["task_id"] not in tasks
            ]
            uses = state["fallback_parent_uses"]
            candidates.sort(
                key=lambda v: (
                    uses.get(v["task"]["parent_id"], 0),
                    digest([self.config.seed, k, v["task"]["task_id"]]),
                )
            )
            if candidates:
                selected = candidates[0]
                task = TaskSpec.model_validate(selected["task"])
                oracle = OracleSpec.model_validate(selected["oracle"])
                check_admission(self.policy, task, oracle, self.config.episode)
                tasks[task.task_id] = task, oracle
                fallback = task.task_id
                state["fallback_groups"] += 1
                uses[task.parent_id] = uses.get(task.parent_id, 0) + 1
        state["task_groups"] += len(tasks)
        edits = [(p.parent_id, strict_json_object(p.turn.raw_text)) for p in proposals if p.valid]
        ages = [
            state["proposal_groups"] - value.get("admitted_group", state["proposal_groups"])
            for value in state["reservoir"].values()
        ]
        state["metrics"] = {
            "cumulative_proposal_attempts": state["proposal_attempts"],
            "cumulative_task_groups": state["task_groups"],
            "cumulative_fallback_groups": state["fallback_groups"],
            "fallback_fraction": state["fallback_groups"] / max(1, state["task_groups"]),
            "online_tasks": len(online_task_ids),
            "fallback_tasks": int(fallback is not None),
            "historical_replay_samples": replays,
            "within_update_repeated_samples": sum(p.valid for p in proposals)
            - len(online_task_ids),
            "reservoir_tasks": len(state["reservoir"]),
            "parent_coverage": len({v["task"]["parent_id"] for v in state["reservoir"].values()}),
            "eligible_parents": len(self.parents),
            "parents_without_legal_sites": state.get("parents_without_legal_sites", 0),
            "distinct_parent_sites": len({(parent, edit["site"]) for parent, edit in edits}),
            "operator_counts": dict(Counter(edit["operation"] for _, edit in edits)),
            "reservoir_age_groups_max": max(ages, default=0),
            "reservoir_age_groups_mean": sum(ages) / max(1, len(ages)),
            "task_sources": {
                task_id: "fallback" if task_id == fallback else "online" for task_id in tasks
            },
        }
        return proposals, tasks
