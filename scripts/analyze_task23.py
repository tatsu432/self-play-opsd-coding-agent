"""Rebuild the Task 23 research packet from private, verified PDB evidence."""

from __future__ import annotations

import argparse
import statistics
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
from analyze_task19 import allocations_before, interval, paired
from analyze_task22 import CONTRASTS, METHODS, PRIMARY, evidence
from scipy.stats import spearmanr

from selfplay_opsd.core import digest, keyed_seed, read_json
from selfplay_opsd.storage import atomic_json, file_hash
from selfplay_opsd.task21 import PdbRepository

SEEDS = (2026, 2027)
SPLITS = (PRIMARY, "harder_transfer", "cross_source", "cross_source_bcb_mutants")
EXPECTED = ((21, 22, 26, 28, 33, 25), (20, 21, 21, 28, 24, 31))
MILESTONES = (0, 64, 128, 256, 384, 512)
WANDB_PROJECT = "https://wandb.ai/tatsu432s/self-play-opsd-coding-agent/runs/"
WANDB = {
    "task21_endpoints": "task21-final-aggregates",
    "task21_curves": "e3684c8e0122b912f97c238a",
    "task22_2026": "task22-8b6b398847d1",
    "task22_2027": "task22-seed2027-4b1ee552aa6d",
}


def historical_manifest(task21: Path, task22: Path) -> dict:
    matched = read_json(task21 / "matched-analysis-manifest.json")["runs"]
    specs = {
        (row["seed"], row["label"].split("-")[0].title()): row
        for row in matched
        if row["stage"] == "main"
    }
    result = {}
    for seed in SEEDS:
        task22_prefix = "" if seed == 2026 else "seed2027-"
        alignment = read_json(task22 / f"{task22_prefix}analysis-manifest.json")
        for method in METHODS:
            task21_method = "control" if method in {"Initial", "Control"} else method
            if method in {"C", "D"}:
                train_ids = alignment[f"{method}-{seed}"]["training_run_ids"]
                receipts = [
                    task22 / f"{task22_prefix}{method}-{split}-verified.json"
                    for split in ("primary", "transfer")
                ]
                training_receipt = task22 / f"{task22_prefix}{method}-training-verified.json"
                config = f"configs/task22-{task22_prefix}main.json"
                baselines = []
            else:
                original = "Control" if method == "Initial" else method
                historical = specs[seed, original]
                train_ids = [] if method == "Initial" else historical["controller_runs"]
                type_ = "base" if method == "Initial" else "endpoint"
                receipts = [task21 / f"final-{seed}-{task21_method}-{type_}-verified.json"]
                baselines = (
                    [task21 / f"final-{seed}-{m}-base-verified.json" for m in ("A", "B")]
                    if method == "Initial"
                    else []
                )
                training_receipt = None
                config = f"configs/task21-main{'-2027' if seed == 2027 else ''}.json"
            result[f"{seed}-{method}"] = {
                "seed": seed,
                "method": method,
                "config": config,
                "train_ids": train_ids,
                "training_receipt": str(training_receipt) if training_receipt else None,
                "receipts": list(map(str, receipts)),
                "duplicate_initial_receipts": list(map(str, baselines)),
                "declared_training_sha": (
                    historical["checkpoint_sha256"]
                    if train_ids and method not in {"C", "D"}
                    else None
                ),
                "historical_training_identity": (
                    historical["identity"] if train_ids and method not in {"C", "D"} else None
                ),
            }
    return result


def load_allocations(paths: list[Path]) -> dict[str, dict]:
    allocations = {}
    for path in paths:
        for row in read_json(path)["reservations"].values():
            job = str(row["job_id"])
            if row["charged"] is None or row["allocated_gpu_seconds"] is None:
                raise ValueError(f"Unreconciled allocation {job}")
            if job in allocations and allocations[job] != row:
                raise ValueError(f"Conflicting allocation {job}")
            allocations[job] = row
    return allocations


def outcomes_for(receipts: list[str], artifacts: Path, repository: PdbRepository) -> tuple:
    outcomes = {}
    records = []
    for receipt in receipts:
        record, root = evidence(Path(receipt), artifacts)
        records.append(record)
        if record["identity"]["data"] != repository.hash:
            raise ValueError("Evaluation uses a different PDB bundle")
        for split, result in record["result"].items():
            summary = read_json(
                root / "evaluation" / f"{record['episodes']:04d}" / split / "summary.json"
            )
            rows = summary["tasks"]
            expected = {task.task_id: task.parent_id for task, _ in repository.tasks(split)}
            if (
                summary["identity"] != record["identity"]
                or summary["snapshot"] != record["snapshot"]
                or summary["split"] != split
                or summary["episodes"] != record["episodes"]
                or len(rows) != result["count"]
                or len(rows) != len(expected)
                or {row["task_id"]: row["parent_id"] for row in rows} != expected
                or sum(row["success"] for row in rows) != result["verified"]
            ):
                raise ValueError(f"Evaluation record differs from its frozen receipt for {split}")
            prior = outcomes.setdefault(split, (rows, root))
            if prior[0] != rows:
                raise ValueError(f"Reused evaluation differs for {split}")
    if set(outcomes) != set(SPLITS):
        raise ValueError("One or more frozen evaluation splits are missing")
    return outcomes, records


def paired_across_seeds(
    by_seed: dict, families: dict, contrast: tuple[str, str], split: str
) -> dict:
    new, old = contrast
    observations = {}
    by_task = defaultdict(list)
    for seed in SEEDS:
        right = by_seed[seed][new][split][0]
        left = by_seed[seed][old][split][0]
        prior = {row["task_id"]: row for row in left}
        observations[str(seed)] = paired(right, left, families) | {
            "gains": sum(row["success"] and not prior[row["task_id"]]["success"] for row in right),
            "losses": sum(prior[row["task_id"]]["success"] and not row["success"] for row in right),
        }
        for row in right:
            before = prior[row["task_id"]]
            by_task[row["task_id"], row["parent_id"]].append(
                100 * (int(row["success"]) - int(before["success"]))
            )
    groups = defaultdict(list)
    for (_, parent), values in by_task.items():
        if len(values) != len(SEEDS):
            raise ValueError("Pooled contrast must use the same tasks across seeds")
        groups[families[parent]].append(statistics.mean(values))
    return {
        "seed": observations,
        "difference_pp": statistics.mean(map(statistics.mean, by_task.values())),
        "ci95_pp": interval(groups),
        "tasks": len(by_task),
        "families": len(groups),
    }


def ratio(num: float, denom: float) -> float | None:
    return num / denom if denom else None


def median(values) -> float | None:
    finite = [value for value in values if value is not None]
    return statistics.median(finite) if finite else None


def training_history(
    root: Path, marker: dict, allocations: list[dict], repository: PdbRepository
) -> tuple:
    iterations = [read_json(path) for path in sorted((root / "iterations").glob("*.json"))]
    if (
        not iterations
        or iterations[0]["episodes"] != 0
        or iterations[-1]["episodes"] != 512
        or iterations[-1]["snapshot"] != marker["snapshot"]
    ):
        raise ValueError("Training history differs from the verified 512-episode endpoint")
    curves, batches, paired_groups, episodes = [], [], [], []
    task_hist = Counter()
    rolling = deque(maxlen=64)
    steps = proposer_tokens = 0
    task_set = set()
    for iteration, record in enumerate(iterations):
        if record["identity"] != marker["identity"] or record["iteration"] != iteration:
            raise ValueError("Committed training identity or iteration sequence differs")
        metrics = record["metrics"]
        n = record["episodes"]
        if batch := record["batch"]:
            path = root / "batches" / batch["path"]
            if file_hash(path) != batch["sha256"]:
                raise ValueError(f"Committed batch checksum differs at iteration {iteration}")
            contents = read_json(path)
            groups = list(contents["repairs"].values())
            repairs = [ep for group in groups for ep in group]
            if len(repairs) != metrics["batch_episodes"] or n != len(episodes) + len(repairs):
                raise ValueError("Committed batch episode counts differ")
            if any(len(group) != 4 for group in groups):
                raise ValueError("Study requires four attempted rollouts per task")
            # The recorded proposal slot order precedes the task-ID-sorted JSON mapping.
            ordered = dict.fromkeys(
                proposal["task"]
                for group in contents["proposals"]
                for proposal in group
                if proposal["task"] in contents["repairs"]
            )
            if set(ordered) != set(contents["repairs"]):
                raise ValueError("Proposal slots do not cover the committed repair groups")
            ordered_repairs = [ep for task in ordered for ep in contents["repairs"][task]]
            rewards = [int(ep["success"]) for ep in ordered_repairs]
            outcomes = Counter(sum(ep["success"] for ep in group) for group in groups)
            student = metrics["student"]
            if sum(rewards) != student["verified"] or (
                outcomes[0],
                sum(outcomes[i] for i in (1, 2, 3)),
                outcomes[4],
            ) != (student["all_fail_groups"], student["mixed_groups"], student["all_pass_groups"]):
                raise ValueError("Committed rollout rewards disagree with training telemetry")
            episodes.extend(rewards)
            rolling.extend(rewards)
            task_set.update(ordered)
            props = [row for group in contents["proposals"] for row in group]
            prop = metrics["proposer"]
            generated = sum(len(proposal["turn"]["ids"]) for proposal in props)
            if prop.get("generated_tokens", generated) != generated:
                raise ValueError("Proposer-generated token accounting differs")
            proposer_tokens += generated
            counts = Counter(ep["failure"] for ep in repairs)
            statuses = Counter(ep["verifier_status"] for ep in repairs)
            malformed = sum(
                any(turn["result"] == "invalid" for turn in ep["turns"]) for ep in repairs
            )
            task_hist.update(counts)
            step = metrics["shared"]
            steps += int(step["optimizer_step"])
            rl, opsd = metrics["rl"]["norm"], metrics["opsd"]
            candidate_groups = []
            for proposals in contents["proposals"]:
                candidates = {
                    row["task"]: row["alignment"]["paired"]
                    for row in proposals
                    if row["task"] and row["alignment"] and "paired" in row["alignment"]
                }
                if candidates:
                    candidate_groups.append(list(candidates.values()))
            paired_groups.extend(candidate_groups)
            batch_row = {
                "episode": n,
                "iteration": iteration,
                "attempted": len(repairs),
                "verified": sum(rewards),
                "online_rate": ratio(sum(rewards), len(repairs)),
                "rolling_rate": ratio(sum(rolling), len(rolling)),
                "cumulative_rate": ratio(sum(episodes), n),
                "valid_rollouts": sum(
                    ep["verifier_status"] in {"ok", "student_failure"}
                    and any(turn["ids"] for turn in ep["turns"])
                    for ep in repairs
                ),
                "malformed_tool_episodes": malformed,
                "verifier_statuses": dict(statuses),
                "student_failures": dict(counts),
                "student_group_counts": {str(i): outcomes[i] for i in range(5)},
                "mixed_fraction": ratio(sum(outcomes[i] for i in (1, 2, 3)), len(groups)),
                "all_failed_fraction": ratio(outcomes[0], len(groups)),
                "all_successful_fraction": ratio(outcomes[4], len(groups)),
                "proposal_attempts": prop["attempted"],
                "proposal_valid": prop["valid"],
                "proposal_invalid": sum(
                    row["task"] is None and row["turn"]["result"] != "unscored_capacity"
                    for row in props
                ),
                "proposal_unscored_capacity": sum(
                    row["turn"]["result"] == "unscored_capacity" for row in props
                ),
                "proposal_duplicates": prop["duplicates"],
                "duplicate_proposal_rate": ratio(prop["duplicates"], prop["attempted"]),
                "proposal_difficulty_counts": {
                    str(successes): count for successes, count in sorted(outcomes.items())
                },
                "valid_proposal_rate": ratio(prop["valid"], prop["attempted"]),
                "unique_scored_tasks": len(ordered),
                "cumulative_unique_tasks": len(task_set),
                "proposal_distinct_reward_groups": prop["distinct_reward_groups"],
                "proposal_groups": prop["groups"],
                "proposal_group_diversity": ratio(prop["distinct_reward_groups"], prop["groups"]),
                "proposal_mean_reward": prop["mean_reward"],
                "proposal_mean_solve_rate": prop["mean_solve_rate"],
                "proposer_gradient_norm": step["proposer_gradient_norm"],
                "proposer_families": prop["families"],
                "near_zero_alignments": prop["near_zero_alignments"],
                "paired_candidates": sum(map(len, candidate_groups)),
                "alignment_c_cosine_quantiles": quantiles(
                    row["C"]["cosine"] for group in candidate_groups for row in group
                ),
                "alignment_d_cosine_quantiles": quantiles(
                    row["D"]["cosine"] for group in candidate_groups for row in group
                ),
                "rl_norm": rl,
                "opsd_norm": opsd["gradient_norm"],
                "weighted_ratio": ratio(marker["sd_scale"] * opsd["gradient_norm"], rl),
                "rl_opsd_cosine": opsd["rl_cosine"],
                "mean_gap": opsd["mean_gap"],
                "mean_gate": opsd["mean_gate"],
                "negative_gap_fraction": opsd["negative_gap_fraction"],
                "gate_histogram": metrics["learning"]["gates"],
                "joint_gradient_norm": step["joint_gradient_norm"],
                "joint_clip": step["joint_clip"],
                "rl_token_clip_fraction": metrics["rl"]["clip_fraction"],
                "update_norm": step["update_norm"],
                "proposer_tokens": generated,
                "student_tokens": metrics["student_tokens"],
                "optimizer_steps": steps,
            }
            batches.append(batch_row)
        location = root / "evaluation" / f"{n:04d}" / "dev_monitor" / "summary.json"
        if location.is_file():
            row = read_json(location)
            expected = {task.task_id: task.parent_id for task, _ in repository.tasks("dev_monitor")}
            if (
                row["identity"] != marker["identity"]
                or row["snapshot"] != record["snapshot"]
                or row["episodes"] != n
                or row["count"] != len(row["tasks"])
                or row["count"] != len(expected)
                or {task["task_id"]: task["parent_id"] for task in row["tasks"]} != expected
                or sum(task["success"] for task in row["tasks"]) != row["verified"]
            ):
                raise ValueError("Development evaluation differs from the committed checkpoint")
            curves.append(
                {
                    "episodes": n,
                    "verified": row["verified"],
                    "count": row["count"],
                    "rate": row["verified"] / row["count"],
                    "student_tokens": metrics.get("student_tokens", 0),
                    "proposer_tokens": proposer_tokens,
                    "steps": steps,
                    "h200_hours": allocations_before(allocations, row["recorded_at"]) if n else 0,
                }
            )
    if len(episodes) != 512 or not set(MILESTONES) <= {row["episodes"] for row in curves}:
        raise ValueError("A required historical training milestone is missing")
    group_counts = Counter()
    for batch in batches:
        group_counts.update(batch["student_group_counts"])
    summary = {
        "attempted": len(episodes),
        "verified": sum(episodes),
        "cumulative_rate": statistics.mean(episodes),
        "first_128_rate": statistics.mean(episodes[:128]),
        "last_128_rate": statistics.mean(episodes[-128:]),
        "optimizer_steps": steps,
        "student_tokens": batches[-1]["student_tokens"],
        "proposer_tokens": proposer_tokens,
        "valid_rollouts": sum(row["valid_rollouts"] for row in batches),
        "malformed_tool_episodes": sum(row["malformed_tool_episodes"] for row in batches),
        "student_failures": dict(task_hist),
        "group_counts": dict(group_counts),
        "mixed_fraction": ratio(
            sum(group_counts[str(i)] for i in (1, 2, 3)), sum(group_counts.values())
        ),
        "median_weighted_ratio": median(row["weighted_ratio"] for row in batches),
        "median_rl_opsd_cosine": median(row["rl_opsd_cosine"] for row in batches),
    }
    return curves, batches, summary, paired_groups


def bootstrap_rate(rows: list[dict], families: dict) -> list[float]:
    clusters = defaultdict(list)
    for row in rows:
        clusters[families[row["parent_id"]]].append(100 * int(row["success"]))
    return interval(clusters)


def quantiles(values) -> list[float] | None:
    finite = [value for value in values if value is not None]
    return np.quantile(finite, [0, 0.25, 0.5, 0.75, 1]).tolist() if finite else None


def alignment_statistics(groups: list[list[dict]]) -> dict:
    by_outcome = {
        "all_failed": lambda row: row["successes"] == 0,
        "mixed": lambda row: 0 < row["successes"] < row["rollouts"],
        "all_successful": lambda row: row["successes"] == row["rollouts"],
        "all": lambda _: True,
    }
    all_rows = [row for group in groups for row in group]
    rank = []
    distinct = Counter()
    for group in groups:
        rewards = {arm: [row[arm]["reward"] for row in group] for arm in ("C", "D")}
        distinct.update(arm for arm, values in rewards.items() if len(set(values)) > 1)
        if all(len(set(values)) > 1 for values in rewards.values()):
            rank.append(float(spearmanr(rewards["C"], rewards["D"]).statistic))
    return {
        "group_count": len(groups),
        "distinct_reward_groups": {arm: distinct[arm] for arm in ("C", "D")},
        "within_group_spearman_quantiles": quantiles(rank),
        "by_outcome": {
            name: {
                "count": len(rows),
                "weighted_opsd_norm_quantiles": quantiles(
                    row["weighted_opsd_norm"] for row in rows
                ),
                "arms": {
                    arm: {
                        "near_zero": sum(row[arm]["near_zero"] for row in rows),
                        "norm_quantiles": quantiles(row[arm]["candidate_norm"] for row in rows),
                        "cosine_quantiles": quantiles(row[arm]["cosine"] for row in rows),
                        "reward_quantiles": quantiles(row[arm]["reward"] for row in rows),
                        "positive_cosines": sum((row[arm]["cosine"] or 0) > 0 for row in rows),
                        "undefined_cosines": sum(row[arm]["cosine"] is None for row in rows),
                    }
                    for arm in ("C", "D")
                },
            }
            for name, include in by_outcome.items()
            if (rows := [row for row in all_rows if include(row)])
        },
    }


def evaluation_stats(
    rows: list[dict], root: Path, split: str, exposure: int, families: dict
) -> dict:
    terminations = Counter(row["termination"] for row in rows)
    truncated = [row for row in rows if row["failure"] == "truncation"]
    path = root / "evaluation" / f"{exposure:04d}" / split / "tasks"
    raw = [read_json(path / f"{row['task_id']}.json") for row in rows] if split == PRIMARY else []
    return {
        "verified": sum(row["success"] for row in rows),
        "count": len(rows),
        "families": len({families[row["parent_id"]] for row in rows}),
        "ci95_pp": bootstrap_rate(rows, families),
        "failures": dict(Counter(row["failure"] for row in rows)),
        "terminations": dict(terminations),
        "truncation_reasons": dict(Counter(row["termination"] for row in truncated)),
        "truncated_with_public_pass_and_edit": sum(
            row["public_pass"] and row["accepted_edits"] > 0 for row in truncated
        ),
        "verified_at_a_limit": sum(
            row["success"]
            and row["termination"]
            in {"turn_limit", "turn_tokens", "episode_tokens", "context_limit"}
            for row in rows
        ),
        "verifier_statuses": dict(Counter(item["repair"]["verifier_status"] for item in raw)),
        "per_episode_distributions": {
            key: quantiles(row[key] for row in rows)
            for key in (
                "tokens",
                "turns",
                "accepted_edits",
                "rejected_edits",
                "test_calls",
                "invalid_calls",
            )
        },
        "mean_seconds": statistics.mean(sum(row["timings"].values()) for row in rows),
        "median_tokens_success": median(row["tokens"] for row in rows if row["success"]),
        "median_tokens_failure": median(row["tokens"] for row in rows if not row["success"]),
        "median_turns_success": median(row["turns"] for row in rows if row["success"]),
        "median_turns_failure": median(row["turns"] for row in rows if not row["success"]),
    }


def read_followup(private: Path, kind: str, registry: dict, frozen: dict | None) -> dict:
    cells = {}
    for seed in SEEDS:
        for method in METHODS:
            name = f"{seed}-{method}"
            root = private / "evaluations" / f"{kind}-{name}"
            marker = root / "complete.json"
            if not marker.is_file():
                continue
            manifest = read_json(root / "manifest.json")
            record = read_json(marker)
            control = registry[name]
            identity = record["identity"]
            if (
                manifest["identity"] != identity
                or manifest["identity_sha256"] != digest(identity)
                or manifest["method"] != method
                or manifest["seed"] != seed
                or identity["kind"] != kind
                or identity["checkpoint_identity"] != control["identity"]
                or identity["checkpoint_sha256"] != control["checkpoint_sha256"]
                or identity["snapshot"] != control["snapshot"]
            ):
                raise ValueError("New evaluation identity differs from the audited checkpoint")
            budget = control["evaluation_config"]
            if kind == "relaxed":
                expected = {PRIMARY: set(identity["tasks"])}
                budget = budget | {"max_turn_generated_tokens": 6144}
            else:
                assert frozen is not None
                if identity["ood_sha256"] != file_hash(private / "ood-validation/complete.json"):
                    raise ValueError("New evaluation uses another transfer freeze")
                expected = {
                    split: {row["task"]["task_id"] for row in rows}
                    for split, rows in frozen["cohorts"].items()
                    if rows
                }
            if identity["episode"] != budget or set(record["splits"]) != set(expected):
                raise ValueError("New evaluation budget or split differs")
            for split, rows in record["splits"].items():
                if (
                    len(rows) != len(expected[split])
                    or {r["task_id"] for r in rows} != expected[split]
                ):
                    raise ValueError("New evaluation has missing or duplicated tasks")
                for row in rows:
                    if row["identity"] != {
                        "task": row["task_id"],
                        "seed": keyed_seed(seed, "evaluation", split, row["task_id"]),
                        "evaluation": digest(identity),
                    }:
                        raise ValueError("New evaluation task has a different identity")
                    if (root / split / "tasks" / f"{row['task_id']}.json").is_file() and read_json(
                        root / split / "tasks" / f"{row['task_id']}.json"
                    ) != row:
                        raise ValueError(
                            "New evaluation task file differs from its completion marker"
                        )
            cells[name] = record["splits"]
    return cells


def followup_results(private: Path, registry: dict, original: dict, families: dict) -> dict:
    report: dict = {}
    frozen_path = private / "ood-validation/complete.json"
    frozen = read_json(frozen_path) if frozen_path.is_file() else None
    ledger = private / "followup-accounting.json"
    allocations = read_json(ledger) if ledger.is_file() else []
    if len({entry["job"] for entry in allocations}) != len(allocations):
        raise ValueError("Follow-up costs include a duplicate physical allocation")
    report["allocated_h200_seconds"] = sum(row["gpu_seconds"] for row in allocations)
    report["reconciled_jobs"] = len(allocations)
    if frozen:
        report["validated_ood_cohorts"] = {
            split: {
                "count": len(rows),
                "composition": dict(
                    Counter(row["oracle"]["provenance"]["bug_subtype"] for row in rows)
                ),
            }
            for split, rows in frozen["cohorts"].items()
        }
    for kind in ("relaxed", "ood"):
        available = read_followup(private, kind, registry, frozen)
        report[f"{kind}_complete_cells"] = len(available)
        if len(available) != len(registry):
            continue
        by_seed: dict = defaultdict(dict)
        follows: dict = defaultdict(dict)
        for name, splits in available.items():
            seed, method = registry[name]["seed"], registry[name]["method"]
            for split, rows in splits.items():
                by_seed[seed].setdefault(method, {})[split] = (rows, None)
                split_families = {row["parent_id"]: row["family"] for row in rows}
                if kind == "relaxed" and any(
                    families[parent] != family for parent, family in split_families.items()
                ):
                    raise ValueError("Relaxed primary families differ from the original")
                follows[split][name] = {
                    "verified": sum(row["success"] for row in rows),
                    "count": len(rows),
                    "failures": dict(Counter(row["failure"] for row in rows)),
                    "ci95_pp": bootstrap_rate(rows, split_families),
                    "median_tokens": median(row["tokens"] for row in rows),
                    "median_turns": median(row["turns"] for row in rows),
                    "mean_episode_seconds": statistics.mean(
                        sum(row["timings"].values()) for row in rows
                    ),
                    "allocated_h200_seconds": sum(
                        entry["gpu_seconds"]
                        for entry in allocations
                        if entry["cell"] == f"{kind}-{name}"
                    ),
                }
                if kind == "relaxed":
                    ids = {row["task_id"] for row in rows}
                    old = [row for row in original[seed][method][split][0] if row["task_id"] in ids]
                    prior = {row["task_id"]: row["success"] for row in old}
                    follows[split][name] |= {
                        "baseline_verified": sum(prior.values()),
                        "gains": sum(row["success"] and not prior[row["task_id"]] for row in rows),
                        "losses": sum(prior[row["task_id"]] and not row["success"] for row in rows),
                        "paired_change": paired(rows, old, families),
                        "baseline_truncation": sum(row["failure"] == "truncation" for row in old),
                        "baseline_median_tokens": median(row["tokens"] for row in old),
                        "baseline_median_turns": median(row["turns"] for row in old),
                        "baseline_mean_episode_seconds": statistics.mean(
                            sum(row["timings"].values()) for row in old
                        ),
                    }
        comparisons = (
            CONTRASTS
            if kind == "relaxed"
            else tuple(
                (new, old)
                for old in ("Initial", "Control", "B")
                for new in METHODS
                if METHODS.index(new) > METHODS.index(old)
            )
        )
        sections = {}
        for split, cells in follows.items():
            task_families = {}
            expected = None
            for methods in by_seed.values():
                for outcomes in methods.values():
                    rows = outcomes[split][0]
                    mapping = {row["task_id"]: (row["parent_id"], row["family"]) for row in rows}
                    if expected is not None and expected != mapping:
                        raise ValueError("New evaluation cohort differs across cells")
                    expected = mapping
                    task_families.update((row["parent_id"], row["family"]) for row in rows)
            sections[split] = {
                "cells": cells,
                "contrasts": {
                    f"{new} − {old}": paired_across_seeds(by_seed, task_families, (new, old), split)
                    for new, old in comparisons
                },
            }
        report[kind] = sections[PRIMARY] if kind == "relaxed" else sections
    return report


def build(args) -> tuple[dict, dict]:
    repository = PdbRepository(args.data)
    families = {
        task.parent_id: oracle.provenance["family_id"]
        for split in repository.partition_names()
        for task, oracle in repository.tasks(split)
    }
    manifest = (
        read_json(args.manifest) if args.manifest else historical_manifest(args.task21, args.task22)
    )
    allocations = load_allocations(
        [
            args.task21 / "budget-matched-final.json",
            args.task22 / "budget-final.json",
            args.task22 / "seed2027-budget-final.json",
        ]
    )
    source_file = args.private / "source-ledgers.json"
    source_records = read_json(source_file) if source_file.is_file() else {}
    by_seed: dict = defaultdict(dict)
    results = {}
    registry = {}
    matching: dict = defaultdict(set)
    for label, spec in manifest.items():
        seed, method = spec["seed"], spec["method"]
        if label != f"{seed}-{method}":
            raise ValueError("Manifest label differs from the cell identity")
        exposure = 0 if method == "Initial" else 512
        outcomes, receipts = outcomes_for(spec["receipts"], args.artifacts, repository)
        first = receipts[0]
        identity = first["identity"]
        matching[seed].add(
            (
                identity["initial"],
                identity["data"],
                identity["worker"],
                identity["execution"]["model_revision"],
                identity["execution"]["tokenizer_revision"],
                identity["execution"]["device"],
            )
        )
        for receipt in receipts:
            if (
                receipt["episodes"] != exposure
                or receipt["identity"] != identity
                or receipt["snapshot"] != first["snapshot"]
                or receipt["checkpoint_sha256"] != first["checkpoint_sha256"]
            ):
                raise ValueError("Evaluation receipts do not refer to the same checkpoint")
        baseline_receipts = []
        for path in spec["duplicate_initial_receipts"]:
            other, record = outcomes_for([path], args.artifacts, repository)
            if record[0]["snapshot"] != first["snapshot"] or any(
                {row["task_id"]: row["success"] for row in outcomes[split][0]}
                != {row["task_id"]: row["success"] for row in other[split][0]}
                for split in SPLITS
            ):
                raise ValueError("Repeated initial evaluations disagree")
            baseline_receipts.extend(record)
        config = read_json(Path(spec["config"]))
        if config["seed"] != seed or config["study"] != identity["study"]:
            raise ValueError("Frozen control does not match the run")
        if config["episode"] != read_json(Path("configs/task21-main.json"))["episode"]:
            raise ValueError("The original inference settings differ")
        train_ids = set(spec["train_ids"])
        train_alloc = [row for row in allocations.values() if row["run_id"] in train_ids]
        if {row["run_id"] for row in train_alloc} != train_ids:
            raise ValueError("Missing training allocation")
        eval_ids = {row["run_id"] for row in receipts + baseline_receipts}
        eval_alloc = [row for row in allocations.values() if row["run_id"] in eval_ids]
        if {row["run_id"] for row in eval_alloc} != eval_ids:
            raise ValueError("Missing evaluation allocation")
        curves, batches, training, groups = [], [], None, []
        if method != "Initial":
            if len(train_ids) != 1:
                raise ValueError("Expected one logical training run")
            root = args.artifacts / next(iter(train_ids))
            if spec["training_receipt"]:
                marker, verified_root = evidence(Path(spec["training_receipt"]), args.artifacts)
                if verified_root != root:
                    raise ValueError("Training marker belongs to another run")
            else:
                marker = read_json(root / "complete-main.json")
                if (
                    marker["identity"] != spec["historical_training_identity"]
                    or marker["checkpoint_sha256"] != spec["declared_training_sha"]
                ):
                    raise ValueError("Task 21 marker differs from the verified manifest")
            if any(
                marker[key] != first[key]
                for key in ("identity", "episodes", "checkpoint_sha256", "snapshot")
            ):
                raise ValueError("Final evaluation used a different training endpoint")
            curves, batches, training, groups = training_history(
                root, marker | {"sd_scale": config["sd_scale"]}, train_alloc, repository
            )
        by_seed[seed][method] = outcomes
        training_seconds = sum(row["allocated_gpu_seconds"] for row in train_alloc)
        eval_seconds = sum(row["allocated_gpu_seconds"] for row in eval_alloc)
        results[label] = {
            "seed": seed,
            "method": method,
            "training": training,
            "curves": curves,
            "batches": batches,
            "alignment": alignment_statistics(groups) if groups else None,
            "training_h200_seconds": training_seconds,
            "evaluation_h200_seconds": eval_seconds,
            "scores": {
                split: evaluation_stats(rows, root, split, exposure, families)
                for split, (rows, root) in outcomes.items()
            },
        }
        source_id = next(iter(train_ids), None)
        if method == "Initial":
            source_id = manifest[f"{seed}-Control"]["train_ids"][0]
        source = source_records.get(source_id, {})
        if source and source["task20_control_sha256"] != file_hash(Path(spec["config"])):
            raise ValueError("Source controller and frozen control checksums differ")
        registry[label] = {
            **spec,
            "identity": identity,
            "snapshot": first["snapshot"],
            "checkpoint": first["checkpoint"],
            "checkpoint_sha256": first["checkpoint_sha256"],
            "configuration_sha256": file_hash(Path(spec["config"])),
            "source_commit": source.get("commit") or spec.get("source_commit"),
            "source_ledger": source or None,
            "evaluation_config": config["episode"],
            "evaluation_receipts": receipts + baseline_receipts,
            "training": training,
            "training_allocations": train_alloc,
            "evaluation_allocations": eval_alloc,
            "attempted_training_episodes": exposure,
            "completed_training_episodes": training["attempted"] if training else 0,
            "training_h200_seconds": training_seconds,
            "end_to_end_h200_seconds": training_seconds + eval_seconds,
            "wandb_aggregate": WANDB_PROJECT
            + (
                WANDB["task21_endpoints"]
                if method in {"Initial", "Control", "A", "B"}
                else WANDB[f"task22_{seed}"]
            ),
            "status": "verified_complete",
        }
    if any(len(matching[seed]) != 1 or set(by_seed[seed]) != set(METHODS) for seed in SEEDS):
        raise ValueError("Incomplete or unmatched configurations")
    if (
        tuple(
            tuple(results[f"{seed}-{method}"]["scores"][PRIMARY]["verified"] for method in METHODS)
            for seed in SEEDS
        )
        != EXPECTED
    ):
        raise ValueError("Primary outcomes differ from the verified study")
    total = sum(row["allocated_gpu_seconds"] for row in allocations.values())
    if total != 1_161_107:
        raise ValueError("Historical H200-seconds differ from reconciled Tasks 21 and 22")
    contrasts = {
        f"{new} − {old}": {
            split: paired_across_seeds(by_seed, families, (new, old), split) for split in SPLITS
        }
        for new, old in CONTRASTS
    }
    primary = {}
    means = {
        method: statistics.mean(
            100
            * (value := results[f"{seed}-{method}"]["scores"][PRIMARY])["verified"]
            / value["count"]
            for seed in SEEDS
        )
        for method in METHODS
    }
    for method in METHODS:
        cells = [results[f"{seed}-{method}"]["scores"][PRIMARY] for seed in SEEDS]
        values = [100 * cell["verified"] / cell["count"] for cell in cells]
        primary[method] = {
            "mean_percent": means[method],
            "sample_sd_pp": statistics.stdev(values),
            "vs_initial_pp": means[method] - means["Initial"],
            "vs_control_pp": means[method] - means["Control"],
        }
    followup = followup_results(args.private, registry, by_seed, families)
    return {
        "historical_h200_seconds": total,
        "historical_job_count": len(allocations),
        "cells": results,
        "primary": primary,
        "contrasts": contrasts,
        "followup": followup,
    }, {
        "cells": registry,
        "historical_allocations": allocations,
        "wandb_runs": {key: WANDB_PROJECT + value for key, value in WANDB.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task21", type=Path, default=Path("runs/task21"))
    parser.add_argument("--task22", type=Path, default=Path("runs/task22"))
    parser.add_argument("--data", type=Path, default=Path("data/task21/pdb-solver-v1"))
    parser.add_argument("--artifacts", type=Path, default=Path("runs/hpc/yale-hpc-cpsc4880"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private", type=Path, default=Path("runs/task23"))
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()
    args.private.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    report, registry = build(args)
    atomic_json(args.private / "experiment-registry.json", registry)
    atomic_json(args.output / "statistics.json", report)
    if not args.data_only:
        from plot_task23 import export

        export(args.output, report)


if __name__ == "__main__":
    main()
