"""Read-only retrospective analysis of immutable Task 15 artifacts.

No candidate execution, model inference, training, or telemetry writes occur here.
Private identifiers and sources must stay in the ignored analysis directory.
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from selfplay_opsd.config import RunConfig
from selfplay_opsd.core import digest
from selfplay_opsd.data import TaskRepository
from selfplay_opsd.mutations import fixed_mutation
from selfplay_opsd.storage import CheckpointManager, active_events, atomic_json, file_hash
from selfplay_opsd.structured_mutations import sites

SPLITS = ("reference", "dev_monitor", "dev_diagnostic", "final_id", "external")
FROZEN_STUDY = "repair-selfplay-2026-task15-ab-v2"


class Inputs:
    """Hash every consumed file, including missing referenced dependencies."""

    def __init__(self) -> None:
        self.files: dict[str, str] = {}
        self.missing: set[str] = set()

    def track(self, path: Path) -> bool:
        path = path.resolve()
        if not path.is_file():
            self.missing.add(str(path))
            return False
        self.files[str(path)] = file_hash(path)
        return True

    def read(self, path: Path) -> Any:
        if not self.track(path):
            raise FileNotFoundError(path)
        return json.loads(path.read_text())

    def events(self, root: Path) -> list[dict]:
        for path in sorted((root / "events").glob("*.jsonl")):
            self.track(path)
        return active_events(root)


def relative_artifact(root: Path, name: str) -> Path:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("Artifact path must be relative and remain within the run")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("Artifact symlink escapes run")
    return resolved


def complete_evaluations(events: list[dict], expected: dict[str, dict[str, str]]) -> list[dict]:
    """Keep attempts distinct; never silently shrink or merge denominators."""
    pending: dict[tuple, dict] = {}
    complete = []
    seen_commits: set[int] = set()
    for event in events:
        row, kind = event["payload"], event["kind"]
        if kind == "iteration_commit":
            iteration = row["iteration"]
            if iteration in seen_commits:
                raise ValueError("Conflicting active iteration lineage")
            seen_commits.add(iteration)
        split = row.get("split")
        if split not in expected or not kind.startswith("evaluation"):
            continue
        key = (
            event["segment_id"],
            event["completed_iteration"],
            split,
            row.get("checkpoint"),
            row.get("endpoint", "primary"),
        )
        if kind == "evaluation_start":
            if key in pending:
                raise ValueError("Duplicate evaluation start")
            ids = row.get("expected_task_ids")
            if ids is not None and (len(ids) != len(set(ids)) or set(ids) != set(expected[split])):
                raise ValueError("Evaluation start task set mismatch")
            pending[key] = {"rows": [], "start": event["wall_timestamp"]}
        elif kind == "evaluation":
            if key not in pending:
                raise ValueError("Evaluation record lacks matching start")
            pending[key]["rows"].append(row)
        elif kind == "evaluation_complete":
            attempt = pending.pop(key, None)
            if attempt is None:
                raise ValueError("Evaluation completion lacks matching start")
            records = attempt["rows"]
            actual = {r["task_id"]: r["parent_id"] for r in records}
            if len(actual) != len(records):
                raise ValueError("Duplicate evaluation task")
            if (
                actual != expected[split]
                or len(records) != row["expected"]
                or len(records) != row["completed"]
                or any(r["success"] is None for r in records)
            ):
                raise ValueError("Incomplete evaluation or mismatched task/parent denominator")
            complete.append(
                {
                    "iteration": key[1],
                    "split": split,
                    "checkpoint": key[3],
                    "endpoint": key[4],
                    "records": records,
                    "timestamp": event["wall_timestamp"],
                    "segment_id": key[0],
                }
            )
        elif kind == "evaluation_incomplete":
            raise ValueError("Active incomplete evaluation; reconcile lineage before analysis")
    if pending:
        raise ValueError("Unfinished active evaluation attempt")
    return complete


def allocated_hours(allocations: list[dict], timestamp: float | None = None) -> float:
    ids = set()
    intervals = []
    for item in allocations:
        start, end, gpus = item["start"], item["end"], item["gpus"]
        if end < start or gpus <= 0 or item["job_id"] in ids:
            raise ValueError("Invalid or duplicate allocation")
        ids.add(item["job_id"])
        intervals.append((start, end))
    ordered = sorted(intervals)
    if any(b[0] < a[1] for a, b in zip(ordered, ordered[1:], strict=False)):
        raise ValueError("Overlapping allocations for one logical arm")
    return sum(
        max(0, min(timestamp, a["end"]) - a["start"]) * a["gpus"] / 3600
        if timestamp is not None
        else (a["end"] - a["start"]) * a["gpus"] / 3600
        for a in allocations
    )


def parent_values(records: list[dict]) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in records:
        if row["success"] is None:
            raise ValueError("Missing success is not a failure")
        values[row["parent_id"]].append(float(row["success"]))
    return {parent: float(np.mean(v)) for parent, v in values.items()}


def paired_statistics(
    first: list[dict],
    second: list[dict],
    families: dict[str, str],
    seed: int = 2026,
    samples: int = 2000,
) -> dict:
    a, b = {r["task_id"]: r for r in first}, {r["task_id"]: r for r in second}
    if len(a) != len(first) or len(b) != len(second) or not a or a.keys() != b.keys():
        raise ValueError("Paired task sets differ or contain duplicates")
    if any(a[k]["parent_id"] != b[k]["parent_id"] for k in a):
        raise ValueError("Paired parent mismatch")
    pa, pb = parent_values(first), parent_values(second)
    clusters: dict[str, list[float]] = defaultdict(list)
    for parent in sorted(pa):
        clusters[families[parent]].append(pb[parent] - pa[parent])
    vals = [clusters[key] for key in sorted(clusters)]
    rng = np.random.default_rng(seed)
    draws = [
        float(np.mean([v for i in rng.integers(len(vals), size=len(vals)) for v in vals[i]]))
        for _ in range(samples)
    ]
    overlap = Counter((bool(a[k]["success"]), bool(b[k]["success"])) for k in a)
    return {
        "difference_parent_macro": float(np.mean([pb[p] - pa[p] for p in pa])),
        "family_bootstrap_95_interval": np.quantile(draws, [0.025, 0.975]).tolist(),
        "first_only": overlap[True, False],
        "second_only": overlap[False, True],
        "both": overlap[True, True],
        "neither": overlap[False, False],
        "tasks": len(a),
        "parents": len(pa),
        "families": len(vals),
    }


def select_checkpoint(curves: list[dict]) -> dict:
    if not curves:
        raise ValueError("No development checkpoints")
    return min(curves, key=lambda r: (-r["parent_macro"], r["episodes"], r["iteration"]))


def selection_stability(
    curves: list[dict], families: dict[str, str], seed: int = 2026, samples: int = 2000
) -> dict:
    ordered = sorted(curves, key=lambda r: (r["episodes"], r["iteration"]))
    parent_scores = [parent_values(r["records"]) for r in ordered]
    parents = sorted(parent_scores[0])
    if any(set(v) != set(parents) for v in parent_scores):
        raise ValueError("Development parent sets differ")
    clusters: dict[str, list[str]] = defaultdict(list)
    for p in parents:
        clusters[families[p]].append(p)
    keys = sorted(clusters)
    rng = np.random.default_rng(seed)
    selected: Counter[int] = Counter()
    apparent: list[float] = []
    out_of_bag: list[float] = []
    for _ in range(samples):
        draw = rng.integers(len(keys), size=len(keys))
        train = [p for i in draw for p in clusters[keys[i]]]
        scores = [float(np.mean([v[p] for p in train])) for v in parent_scores]
        index = int(np.argmax(scores))
        selected[ordered[index]["iteration"]] += 1
        apparent.append(scores[index] - scores[-1])
        held = [p for i, key in enumerate(keys) if i not in draw for p in clusters[key]]
        if held:
            out_of_bag.append(
                float(np.mean([parent_scores[index][p] - parent_scores[-1][p] for p in held]))
            )
    return {
        "replicates": samples,
        "selection_frequency": {str(k): v / samples for k, v in sorted(selected.items())},
        "selected_nonfinal_fraction": 1 - selected[ordered[-1]["iteration"]] / samples,
        "apparent_gain_95_interval": np.quantile(apparent, [0.025, 0.975]).tolist(),
        "out_of_bag_gain_mean": float(np.mean(out_of_bag)) if out_of_bag else None,
        "out_of_bag_gain_95_interval": np.quantile(out_of_bag, [0.025, 0.975]).tolist()
        if out_of_bag
        else None,
        "caveat": "resampling selection stability, not an independent holdout or bias-free CI",
    }


def failure_category(row: dict) -> str:
    """Terminal descriptive attribution, not a causal explanation."""
    if row["success"] is True:
        return "success"
    if row.get("infrastructure_failure"):
        return "infrastructure"
    if row.get("success") is None:
        return "unknown"
    if row.get("public_pass") is True and row.get("hidden_pass") is False:
        return "public_pass_hidden_fail"
    if row.get("termination") in {"turn_limit", "token_limit", "context_limit"}:
        return "budget_exhausted"
    if row.get("valid_edits") == 0:
        if row.get("rejected_patches", 0):
            return "no_edit_patch_rejected"
        if row.get("parse_errors", 0):
            return "no_edit_parse_failure"
        return "no_edit_other"
    if (row.get("valid_edits") or 0) > 0 or row.get("source_changed") is True:
        return (
            "edited_runtime_failure"
            if row.get("runtime_failures", 0)
            else "edited_semantic_failure"
        )
    return "unknown"


def episode_diagnostics(record: dict, episode: dict | None, initial_hash: str) -> dict:
    metrics = record["metrics"]
    row = {k: record[k] for k in ("task_id", "parent_id", "checkpoint", "success", "status")}
    row.update(
        {
            k: metrics.get(k)
            for k in (
                "turns",
                "tokens",
                "test_calls",
                "public_pass",
                "hidden_pass",
                "termination",
                "runtime_failures",
                "semantic_failures",
            )
        }
    )
    row["infrastructure_failure"] = (
        record["status"] == "sandbox_infrastructure_error"
        or metrics.get("verifier_status") == "sandbox_infrastructure_error"
    )
    row["source_changed"] = (
        metrics.get("final_source_hash") != initial_hash
        if metrics.get("final_source_hash")
        else None
    )
    row["trajectory_available"] = episode is not None
    for name in (
        "valid_actions",
        "invalid_actions",
        "parse_errors",
        "valid_edits",
        "rejected_patches",
        "malformed_hunks",
        "context_errors",
        "path_errors",
        "syntax_errors",
        "unknown_tool_errors",
    ):
        row[name] = None
    if episode is not None:
        if episode["task_id"] != record["task_id"] or bool(episode["reward"]) != record["success"]:
            raise ValueError("Trajectory task/outcome mismatch")
        if (
            re.fullmatch(r"[0-9a-f]{64}", record["checkpoint"])
            and episode["policy_version"] != record["checkpoint"]
        ):
            raise ValueError("Trajectory checkpoint mismatch")
        turns = episode["turns"]
        row.update(
            valid_actions=0,
            invalid_actions=0,
            parse_errors=0,
            valid_edits=0,
            rejected_patches=0,
            malformed_hunks=0,
            context_errors=0,
            path_errors=0,
            syntax_errors=0,
            unknown_tool_errors=0,
        )
        actions: Counter[str] = Counter()
        for i, turn in enumerate(turns):
            action, status = turn["action"], turn["parse_status"]
            actions[f"{action}_{status}"] += 1
            row["valid_actions" if status == "valid" else "invalid_actions"] += 1
            row["parse_errors"] += status == "invalid"
            row["valid_edits"] += action == "apply_patch" and status == "valid"
            row["rejected_patches"] += action == "apply_patch" and status == "tool_error"
            if action == "apply_patch" and status == "tool_error":
                # The observation for a turn is in the NEXT turn's history. The
                # final observation is not persisted, so its subtype is unknown.
                error = ""
                if i + 1 < len(turns):
                    observations = [
                        m["content"] for m in turns[i + 1]["history"] if m["role"] == "tool"
                    ]
                    if observations:
                        try:
                            error = json.loads(observations[-1]).get("error", "").lower()
                        except (json.JSONDecodeError, AttributeError):
                            pass
                category = (
                    "syntax_errors"
                    if "syntax" in error
                    else "path_errors"
                    if "path" in error
                    else "context_errors"
                    if "context" in error
                    else "malformed_hunks"
                    if any(x in error for x in ("hunk", "diff", "patch"))
                    else "unknown_tool_errors"
                )
                row[category] += 1
        row["actions"] = dict(actions)
        if len(turns) != row["turns"] or sum(len(t["target_ids"]) for t in turns) != row["tokens"]:
            raise ValueError("Trajectory token/turn denominator mismatch")
    row["at_least_one_applied_patch"] = (
        row["valid_edits"] > 0
        if episode is not None
        else True
        if row["source_changed"] is True
        else None
    )
    row["primary_category"] = failure_category(row)
    return row


def ast_features(source: str) -> dict:
    tree = ast.parse(source)
    nodes = list(ast.walk(tree))
    functions = [n for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    docstrings = [
        ast.get_docstring(n, clean=False) or ""
        for n in nodes
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    stripped = ast.parse(source)
    for node in ast.walk(stripped):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ) and ast.get_docstring(node, clean=False):
            node.body.pop(0)
    return {
        "code_lines_without_docstrings": len(ast.unparse(stripped).splitlines()),
        "docstring_chars": sum(len(s) for s in docstrings),
        "lines": len(source.splitlines()),
        "ast_nodes": len(nodes),
        "functions": len(functions),
        "largest_function_lines": max(
            ((n.end_lineno or n.lineno) - n.lineno + 1 for n in functions), default=0
        ),
        "branches": sum(
            isinstance(
                n, (ast.If, ast.For, ast.While, ast.IfExp, ast.comprehension, ast.ExceptHandler)
            )
            for n in nodes
        ),
    }


def mutation_sites(before: Any, after: Any, path: str = "", line: int | None = None) -> list[dict]:
    """Structural AST changes, retaining ancestor locations for operator nodes."""
    line = getattr(before, "lineno", line)
    if type(before) is not type(after):
        return [
            {
                "path": path,
                "line": line,
                "change": f"{type(before).__name__}->{type(after).__name__}",
            }
        ]
    if isinstance(before, ast.Constant) and before.value != after.value:
        return [{"path": path, "line": line, "change": "constant_value"}]
    differences = []
    if isinstance(before, ast.AST):
        for field in before._fields:
            differences.extend(
                mutation_sites(
                    getattr(before, field), getattr(after, field), f"{path}.{field}", line
                )
            )
    elif isinstance(before, list):
        if len(before) != len(after):
            return [{"path": path, "line": line, "change": "list_length"}]
        for i, (a, b) in enumerate(zip(before, after, strict=True)):
            differences.extend(mutation_sites(a, b, f"{path}[{i}]", line))
    return differences


def dataset_audit(
    repository: TaskRepository, inputs: Inputs, tokenizer: Any = None
) -> tuple[dict, list[dict], dict]:
    root = repository.root
    for name in (
        "manifest.json",
        "coverage.json",
        "private/parents.json",
        "validation_report.json",
    ):
        inputs.track(root / name)
    families, roles = {}, {}
    for item in repository._parents:
        parent = item["parent"]
        pid, family = parent["parent_id"], parent["family_id"]
        if pid in roles:
            raise ValueError("Duplicate parent")
        roles[pid], families[pid] = item["split"], family
    family_roles: dict[str, set[str]] = defaultdict(set)
    for pid, role in roles.items():
        if not role.startswith("excluded"):
            family_roles[families[pid]].add(role)
    if any(len(v) > 1 for v in family_roles.values()):
        raise ValueError("Family crosses frozen partitions")
    features, expected = [], {}
    for split in SPLITS:
        inputs.track(root / "private" / f"tasks-{split}.json")
        tasks = repository.tasks(split)
        expected[split] = {t.task_id: t.parent_id for t, _ in tasks}
        if (
            len(expected[split]) != len(tasks)
            or len(tasks) != repository.manifest["admitted_tasks"][split]
        ):
            raise ValueError("Frozen split count mismatch")
        for task, oracle in tasks:
            if roles[task.parent_id] != split or oracle.task_id != task.task_id:
                raise ValueError("Task/parent/oracle role mismatch")
            mutation = fixed_mutation(oracle.correct_source, task.mutation_seed)
            if mutation.source != task.initial_source:
                raise ValueError("Frozen mutation regeneration differs")
            original, mutant = (
                ast_features(oracle.correct_source),
                ast_features(task.initial_source),
            )
            differences = mutation_sites(
                ast.parse(oracle.correct_source), ast.parse(task.initial_source)
            )
            location = differences[0]["line"] if differences else None
            row = {
                "task_id": task.task_id,
                "parent_id": task.parent_id,
                "family_id": families[task.parent_id],
                "split": split,
                "mutation_type": "+".join(d["change"] for d in differences),
                "mutation_family_legacy": mutation.family,
                "mutation_sites": differences,
                "mutation_line": location,
                "public_cases": len(oracle.public_cases),
                "hidden_cases": len(oracle.hidden_cases),
                "spec_chars": len(task.specification),
                "spec_lines": len(task.specification.splitlines()),
                "spec_has_function_definition": "def " in task.specification,
                "source_tokens": len(
                    tokenizer.encode(oracle.correct_source, add_special_tokens=False)
                )
                if tokenizer
                else None,
                "spec_tokens": len(tokenizer.encode(task.specification, add_special_tokens=False))
                if tokenizer
                else None,
                **original,
                "mutant_ast_nodes": mutant["ast_nodes"],
                "initial_public_failure": None,
                "initial_hidden_failure": None,
            }
            features.append(row)
    validation = inputs.read(root / "validation_report.json")
    if not validation.get("executed") or validation.get("manifest_hash") != repository.hash:
        raise ValueError("No matching executed frozen oracle-validation evidence")
    train = repository.parents("train")
    eligible = [p for p in train if p.parent_id != "Mbpp/590" and sites(p.source)]
    summary = {
        "manifest_hash": repository.hash,
        "counts": repository.manifest["admitted_tasks"],
        "parents_per_task_split": {s: len(set(expected[s].values())) for s in SPLITS},
        "reserved_parent_counts": dict(Counter(roles.values())),
        "training_reserved": len(train),
        "eligible_training_parents": len(eligible),
        "parent_family_disjoint": True,
        "frozen_mutations_reproduced": len(features),
        "prior_executed_oracle_validation": validation,
        "initial_public_hidden_failure": "not recorded separately; requires isolated CPU replay",
        "tokenizer_available": tokenizer is not None,
        "evaluation_mutation_policy": "fixed_mutation; training uses structured_ast",
        "feature_summary": {
            s: {
                k: {
                    "mean": float(np.mean([r[k] for r in features if r["split"] == s])),
                    "median": float(np.median([r[k] for r in features if r["split"] == s])),
                }
                for k in (
                    "lines",
                    "ast_nodes",
                    "branches",
                    "largest_function_lines",
                    "code_lines_without_docstrings",
                    "docstring_chars",
                    "public_cases",
                    "hidden_cases",
                    "spec_chars",
                    "spec_lines",
                )
            }
            for s in SPLITS
        },
    }
    summary["parent_macro_feature_means"] = {}
    for split in SPLITS:
        summary["parent_macro_feature_means"][split] = {}
        for key in summary["feature_summary"][split]:
            by_parent: dict[str, list[float]] = defaultdict(list)
            for row in features:
                if row["split"] == split:
                    by_parent[row["parent_id"]].append(row[key])
            summary["parent_macro_feature_means"][split][key] = float(
                np.mean([np.mean(v) for v in by_parent.values()])
            )
    return summary, features, expected


def proposer_group_information(proposals: list[dict], episodes: list[dict]) -> dict:
    """Reconstruct A/B difficulty discrimination from matched saved rollouts."""
    if len(proposals) % 4:
        raise ValueError("Incomplete four-proposal group")
    rewards_by_task: dict[str, list[float]] = defaultdict(list)
    for episode in episodes:
        rewards_by_task[episode["task_id"]].append(episode["reward"])
    nonzero, distinct = 0, 0
    for start in range(0, len(proposals), 4):
        group = proposals[start : start + 4]
        if len({p["parent_id"] for p in group}) != 1:
            raise ValueError("Proposer group crossed parent identities")
        rewards, valid_rewards = [], []
        for proposal in group:
            if not proposal["valid"]:
                rewards.append(-1.0)
                continue
            outcomes = rewards_by_task.get(proposal["task_id"], [])
            if len(outcomes) != 4 or any(r not in {0.0, 1.0} for r in outcomes):
                raise ValueError("Valid proposal lacks four matched binary student rollouts")
            rate = sum(outcomes) / 4
            value = 4 * rate * (1 - rate)
            rewards.append(value)
            valid_rewards.append(value)
        nonzero += len(set(rewards)) > 1
        distinct += len(set(valid_rewards)) > 1
    return {
        "nonzero_advantage_proposer_groups": nonzero,
        "reconstructed_distinct_reward_groups": distinct,
    }


def training_health(
    events: list[dict],
    allocations: list[dict],
    root: Path,
    inputs: Inputs,
    training_parents: set[str],
    held_out: set[str],
    reference_parents: set[str],
) -> tuple[list[dict], dict, list[str]]:
    rows, requested = [], []
    totals: Counter = Counter()
    seen_tasks: set[str] = set()
    seen_mutations: set[tuple] = set()
    prior_episodes = 0
    for event in events:
        if event["kind"] != "iteration_commit":
            continue
        p = event["payload"]
        m = p["metrics"]
        prop, student, supply = m["proposer"], m["student"], m["supply"]
        episodes = student["success_denominator"]
        mix = {int(k): v for k, v in student["success_group_mix"].items()}
        if sum(4 * n for n in mix.values()) != episodes or any(k not in range(5) for k in mix):
            raise ValueError("Four-rollout group denominator mismatch")
        if student["cumulative_episodes"] != prior_episodes + episodes:
            raise ValueError("Committed exposure discontinuity")
        if set(supply.get("task_sources", {})) & held_out:
            raise ValueError("Held-out task in training supply")
        prior_episodes += episodes
        counts = {
            "proposal_attempts": prop["attempted_count"],
            "valid_proposals": prop["valid_count"],
            "duplicate_proposals": prop["duplicate_count"],
            "groups": sum(mix.values()),
            "mixed_groups": sum(n for k, n in mix.items() if 0 < k < 4),
            "all_failed_groups": mix.get(0, 0),
            "all_successful_groups": mix.get(4, 0),
            "successes": student["success_numerator"],
            "episodes": episodes,
            "tokens": student["tokens"],
            "invalid_actions": student["invalid_actions"],
            "test_calls": student["test_calls"],
            "total_actions": round(student["turns"]["mean"] * student["turns"]["count"]),
            "public_pass_hidden_fail": student["public_pass_hidden_fail"],
            "multiple_valid_proposer_groups": prop["groups_with_multiple_valid"],
            "distinct_reward_proposer_groups": prop["groups_with_distinct_valid_rewards"],
            "attempted_proposer_groups": prop["attempted_count"] // 4,
        }
        totals.update(counts)
        gpu_hours = allocated_hours(allocations, event["wall_timestamp"])
        row = {
            "iteration": p["iteration"],
            "episodes_start": prior_episodes - episodes,
            "episodes_end": prior_episodes,
            "allocated_gpu_hours": gpu_hours,
            **counts,
            "cumulative_episodes": prior_episodes,
            "cumulative_tokens": student["cumulative_generated_tokens"],
            "student_steps": m["train"]["student_steps"],
            "proposer_steps": m["train"]["proposer_steps"],
            "historical_replay_samples": supply.get("historical_replay_samples"),
            "within_update_repeated_samples": supply.get("within_update_repeated_samples"),
            "parent_coverage": supply.get("parent_coverage"),
            "distinct_parent_sites": supply.get("distinct_parent_sites"),
            "operator_counts": supply.get("operator_counts"),
            "fallback_groups": supply.get("cumulative_fallback_groups"),
            "curriculum_unique_mutations": prop.get("curriculum_unique_mutations"),
            "curriculum_parents": prop.get("curriculum_parents"),
            "nonzero_advantage_candidate_count": prop.get("nonzero_advantage_count"),
            "gate_mean": m["distill"].get("gate", {}).get("mean"),
            "gap_mean": m["distill"].get("gap", {}).get("mean"),
            "gradient_interaction": student.get("gradient_interaction"),
            "phase_seconds": m["perf"],
            "termination_counts": student["termination"],
            "training_trajectory_available": False,
            "proposal_trajectory_available": False,
            "nonzero_advantage_proposer_groups": None,
            "valid_actions_by_type": None,
        }
        raw: dict[str, list[dict]] = {}
        for name in ("episodes_path", "proposals_path"):
            path = relative_artifact(root, p[name])
            requested.append(p[name])
            if not inputs.track(path):
                continue
            records = inputs.read(path)
            raw[name] = records
            if name == "episodes_path":
                if (
                    len(records) != episodes
                    or sum(e["reward"] for e in records) != student["success_numerator"]
                ):
                    raise ValueError("Training episode count/outcome mismatch")
                row["training_trajectory_available"] = True
                if any(e["task_id"] in held_out for e in records):
                    raise ValueError("Held-out task in student trajectories")
                row["valid_actions_by_type"] = dict(
                    Counter(
                        f"{t['action']}_{t['parse_status']}" for e in records for t in e["turns"]
                    )
                )
            else:
                if len(records) != prop["attempted_count"]:
                    raise ValueError("Proposal count mismatch")
                if any(
                    v["parent_id"] not in training_parents or v["parent_id"] in reference_parents
                    for v in records
                ):
                    raise ValueError("Nontraining parent in proposer trajectories")
                row["proposal_trajectory_available"] = True
                # A/B difficulty is determined by saved candidate episodes.
                row["new_task_identities"] = len(
                    {v["task_id"] for v in records if v["valid"]} - seen_tasks
                )
                seen_tasks.update(v["task_id"] for v in records if v["valid"])
                keys = {(v["parent_id"], v["dedup_key"]) for v in records if v["valid"]}
                row["new_mutations"] = len(keys - seen_mutations)
                seen_mutations.update(keys)
        if len(raw) == 2:
            information = proposer_group_information(raw["proposals_path"], raw["episodes_path"])
            if (
                information["reconstructed_distinct_reward_groups"]
                != prop["groups_with_distinct_valid_rewards"]
            ):
                raise ValueError("Recorded/reconstructed proposer reward discrimination differs")
            row.update(information)
        # The nonzero counter is per candidate, not per contiguous parent group.
        # Do not infer group counts from it when refill uses multiple parents.
        rows.append(row)
    last = rows[-1]
    hours = allocated_hours(allocations)
    summary = {
        **dict(totals),
        "allocated_gpu_hours": hours,
        "valid_actions": totals["total_actions"] - totals["invalid_actions"],
        "invalid_action_fraction": totals["invalid_actions"] / totals["total_actions"],
        "historical_replay_samples": sum(r["historical_replay_samples"] for r in rows),
        "within_update_repeated_samples": sum(r["within_update_repeated_samples"] for r in rows),
        "student_steps": last["student_steps"],
        "proposer_steps": last["proposer_steps"],
        "episodes_per_student_update": totals["episodes"] / last["student_steps"],
        "tokens_per_success": totals["tokens"] / totals["successes"]
        if totals["successes"]
        else None,
        "mixed_groups_per_gpu_hour": totals["mixed_groups"] / hours,
        "success_fraction": totals["successes"] / totals["episodes"],
        "mixed_fraction": totals["mixed_groups"] / totals["groups"],
        "valid_proposal_fraction": totals["valid_proposals"] / totals["proposal_attempts"],
        "curriculum_unique_mutations": last["curriculum_unique_mutations"],
        "fallback_groups": last["fallback_groups"],
        "unique_training_task_identities": len(seen_tasks)
        if all(r["proposal_trajectory_available"] for r in rows)
        else None,
        "nonzero_advantage_proposer_group_fraction": sum(
            r["nonzero_advantage_proposer_groups"] for r in rows
        )
        / totals["attempted_proposer_groups"]
        if all(r["nonzero_advantage_proposer_groups"] is not None for r in rows)
        else None,
        "distinct_valid_reward_group_fraction": totals["distinct_reward_proposer_groups"]
        / totals["attempted_proposer_groups"],
        "phase_seconds_committed_only": {
            key: sum(
                r["phase_seconds"][key] for r in rows if r["phase_seconds"].get(key) is not None
            )
            if any(r["phase_seconds"].get(key) is not None for r in rows)
            else None
            for key in (
                "propose_seconds",
                "teacher_score_seconds",
                "old_student_score_seconds",
                "student_update_seconds",
                "proposer_update_seconds",
                "rollout_seconds",
                "development_eval_seconds",
            )
        },
        "trajectory_coverage": {
            k: sum(r[k] for r in rows)
            for k in ("training_trajectory_available", "proposal_trajectory_available")
        },
    }
    return rows, summary, requested


def compute_comparisons(curves: list[dict]) -> list[dict]:
    rows = []
    for seed in sorted({r["seed"] for r in curves}):
        arms = {
            method: [r for r in curves if r["seed"] == seed and r["method"] == method]
            for method in ("A", "B")
        }
        for field in ("allocated_gpu_hours", "tokens", "episodes", "student_steps"):
            low = max(min(r[field] for r in arm) for arm in arms.values())
            high = min(max(r[field] for r in arm) for arm in arms.values())
            budgets = sorted(
                {r[field] for arm in arms.values() for r in arm if low <= r[field] <= high}
            )
            for budget in budgets:
                selected = {
                    method: select_checkpoint([r for r in arm if r[field] <= budget])
                    for method, arm in arms.items()
                }
                rows.append(
                    {
                        "seed": seed,
                        "axis": field,
                        "budget": budget,
                        "a_parent_macro": selected["A"]["parent_macro"],
                        "b_parent_macro": selected["B"]["parent_macro"],
                        "b_minus_a": selected["B"]["parent_macro"] - selected["A"]["parent_macro"],
                        "a_observed_budget": selected["A"][field],
                        "b_observed_budget": selected["B"][field],
                        "a_iteration": selected["A"]["iteration"],
                        "b_iteration": selected["B"]["iteration"],
                        "interpretation": "best measured checkpoint available within budget; not an equal-time evaluation",
                    }
                )
        for threshold in (0.25, 0.35, 0.45):
            for method, arm in arms.items():
                reached = sorted(
                    (r for r in arm if r["parent_macro"] >= threshold),
                    key=lambda r: r["allocated_gpu_hours"],
                )
                rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "axis": "threshold",
                        "threshold": threshold,
                        "first_observed_gpu_hours": reached[0]["allocated_gpu_hours"]
                        if reached
                        else None,
                        "censored": not bool(reached),
                    }
                )
    return rows


def initial_mutant_observations(
    features: list[dict], failures: list[dict]
) -> dict[str, set[tuple[bool, bool]]]:
    initial_observations: dict[str, set[tuple[bool, bool]]] = defaultdict(set)
    for row in failures:
        if (
            row["source_changed"] is False
            and not row["infrastructure_failure"]
            and row["public_pass"] is not None
            and row["hidden_pass"] is not None
        ):
            initial_observations[row["task_id"]].add((row["public_pass"], row["hidden_pass"]))
    for feature in features:
        observations = initial_observations.get(feature["task_id"], set())
        if len(observations) > 1:
            raise ValueError("Unchanged mutant has inconsistent frozen verifier outcomes")
        if observations:
            public, hidden = next(iter(observations))
            feature["initial_public_failure"] = not public
            feature["initial_hidden_failure"] = not hidden
    return initial_observations


def run_analysis(
    manifest_path: Path,
    data: Path,
    base: Path,
    output: Path,
    *,
    tokenizer: Any = None,
    bootstrap_samples: int = 2000,
    run_root: Path | None = None,
) -> dict:
    from selfplay_opsd.analysis import write_csv

    inputs = Inputs()
    source_root = Path(__file__).resolve().parents[2]
    for path in (
        Path(__file__),
        source_root / "scripts/analyze_task16.py",
        source_root / "uv.lock",
    ):
        inputs.track(path)
    if tokenizer is not None:
        for path in sorted(Path(tokenizer.name_or_path).glob("*")):
            if path.is_file():
                inputs.track(path)
    manifest = inputs.read(manifest_path)
    repository = TaskRepository(data)
    audit, features, expected = dataset_audit(repository, inputs, tokenizer)
    families = {r["parent"]["parent_id"]: r["parent"]["family_id"] for r in repository._parents}
    initial = {t.task_id: t.source_hash for s in SPLITS for t, _ in repository.tasks(s)}
    task_info = {r["task_id"]: r for r in features}
    held_out = set(expected["final_id"]) | set(expected["external"])
    training_parents = {p.parent_id for p in repository.parents("train")}
    reference_parents = set(expected["reference"].values())
    curves: list[dict] = []
    failures: list[dict] = []
    health: list[dict] = []
    selections: list[dict] = []
    arms: list[dict] = []
    outcomes: dict[tuple, list[dict]] = {}
    requests: list[dict] = []
    common_by_seed: dict[int, dict] = {}
    initials_by_seed: dict[int, dict] = {}
    seen_arms: set[tuple] = set()
    episode_settings = None
    source_roots = [
        (item, run_root / Path(item["run_dir"]).name if run_root else Path(item["run_dir"]))
        for item in manifest["arms"]
    ]
    source_roots.append(({"method": "base", "seed": None}, base))
    for item, root in source_roots:
        method, seed = item["method"], item["seed"]
        arm = (method, seed)
        if arm in seen_arms:
            raise ValueError("Duplicate logical arm")
        seen_arms.add(arm)
        resolved = inputs.read(root / "resolved_config.json")
        cfg = RunConfig.model_validate(
            {k: v for k, v in resolved.items() if k in RunConfig.model_fields}
        )
        if cfg.study_id != FROZEN_STUDY or cfg.model.name != "Qwen/Qwen3-4B-Instruct-2507":
            raise ValueError("Not the frozen Task 15 model/study")
        episode = cfg.episode.model_dump(mode="json")
        if episode_settings is not None and episode_settings != episode:
            raise ValueError("Evaluation action/token budgets differ")
        episode_settings = episode
        events = inputs.events(root)
        evaluations = complete_evaluations(events, expected)
        fetch_paths = [
            r["episode_path"]
            for e in evaluations
            if e["split"] in {"final_id", "external"}
            for r in e["records"]
            if r.get("episode_path")
        ]
        if method != "base":
            if cfg.seed != seed or cfg.method != method:
                raise ValueError("Run manifest/config identity mismatch")
            common = cfg.scientific_settings(common=True)
            init = inputs.read(root / "initial_versions.json")
            if seed in common_by_seed and common_by_seed[seed] != common:
                raise ValueError("Paired scientific settings differ")
            if seed in initials_by_seed and initials_by_seed[seed] != init:
                raise ValueError("Paired initial adapters differ")
            common_by_seed[seed], initials_by_seed[seed] = common, init
            completion = inputs.read(root / "complete.json")
            if (
                completion["scientific_hash"] != cfg.scientific_hash
                or not 4096 <= completion["episodes"] <= 4108
            ):
                raise ValueError("Final completion identity/exposure mismatch")
            commits = {
                e["payload"]["iteration"]: e for e in events if e["kind"] == "iteration_commit"
            }
            rows = []
            for ev in evaluations:
                if ev["split"] != "dev_monitor":
                    continue
                iteration = ev["iteration"]
                if iteration:
                    if iteration not in commits:
                        raise ValueError("Evaluation lacks committed exposure")
                    metrics = commits[iteration]["payload"]["metrics"]
                    exposure = {
                        "episodes": metrics["student"]["cumulative_episodes"],
                        "tokens": metrics["student"]["cumulative_generated_tokens"],
                        "student_steps": metrics["train"]["student_steps"],
                        "proposer_steps": metrics["train"]["proposer_steps"],
                    }
                else:
                    exposure = {"episodes": 0, "tokens": 0, "student_steps": 0, "proposer_steps": 0}
                rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "iteration": iteration,
                        **exposure,
                        "parent_macro": float(np.mean(list(parent_values(ev["records"]).values()))),
                        "task_macro": float(np.mean([r["success"] for r in ev["records"]])),
                        "allocated_gpu_hours": allocated_hours(
                            item["allocations"], ev["timestamp"]
                        ),
                        "records": ev["records"],
                    }
                )
            rows.sort(key=lambda r: r["episodes"])
            if (
                len(rows) != 5
                or len({r["iteration"] for r in rows}) != 5
                or rows[-1]["episodes"] != completion["episodes"]
            ):
                raise ValueError("Expected five measured development milestones")
            selected, final = select_checkpoint(rows), rows[-1]
            checkpoint = root / "checkpoints" / f"{selected['iteration']:06d}"
            verified = False
            if (checkpoint / "complete.json").exists():
                metadata = CheckpointManager.verify(checkpoint)
                if (
                    metadata["scientific_hash"] != cfg.scientific_hash
                    or metadata["manifests_hash"] != repository.hash
                ):
                    raise ValueError("Selected checkpoint identity mismatch")
                for p in checkpoint.iterdir():
                    if p.is_file():
                        inputs.track(p)
                verified = True
            else:
                inputs.track(checkpoint / "complete.json")
            selections.append(
                {
                    "method": method,
                    "seed": seed,
                    "selected_iteration": selected["iteration"],
                    "final_iteration": final["iteration"],
                    "selected_episodes": selected["episodes"],
                    "selected_gpu_hours": selected["allocated_gpu_hours"],
                    "selected_parent_macro": selected["parent_macro"],
                    "final_parent_macro": final["parent_macro"],
                    "selected_task_macro": selected["task_macro"],
                    "selected_is_final": selected is final,
                    "checkpoint_verified_locally": verified,
                    "stability": selection_stability(rows, families, samples=bootstrap_samples),
                }
            )
            curves.extend(rows)
            hrows, summary, paths = training_health(
                events,
                item["allocations"],
                root,
                inputs,
                training_parents,
                held_out,
                reference_parents,
            )
            fetch_paths.extend(paths)
            health.extend({"method": method, "seed": seed, **r} for r in hrows)
            arms.append({"method": method, "seed": seed, **summary})
        for split in ("final_id", "external"):
            candidates = [e for e in evaluations if e["split"] == split]
            if len(candidates) != 1:
                raise ValueError("Expected one complete final evaluation per arm/split")
            records = candidates[0]["records"]
            outcomes[method, seed, split] = records
            for record in records:
                episode_data = None
                if record.get("episode_path"):
                    path = relative_artifact(root, record["episode_path"])
                    if inputs.track(path):
                        contents = inputs.read(path)
                        if len(contents) != 1:
                            raise ValueError("Expected one task episode in evaluation shard")
                        episode_data = contents[0]
                failures.append(
                    {
                        "method": method,
                        "seed": seed,
                        "split": split,
                        **episode_diagnostics(record, episode_data, initial[record["task_id"]]),
                    }
                )
        requests.append(
            {
                "run_id": root.name if method != "base" else "base",
                "method": method,
                "seed": seed,
                "run_dir": str(root),
                "paths": sorted(set(fetch_paths)),
            }
        )
    if {(m, s) for m, s in seen_arms if m != "base"} != {
        (m, s) for m in ("A", "B") for s in (2026, 2027, 2028)
    }:
        raise ValueError("Six complete A/B arms required")
    paired = []
    for split in ("final_id", "external"):
        for seed in (2026, 2027, 2028):
            paired.append(
                {
                    "comparison": "B-minus-A",
                    "seed": seed,
                    "split": split,
                    **paired_statistics(
                        outcomes["A", seed, split],
                        outcomes["B", seed, split],
                        families,
                        samples=bootstrap_samples,
                    ),
                }
            )
            for method in ("A", "B"):
                paired.append(
                    {
                        "comparison": method + "-minus-base",
                        "seed": seed,
                        "split": split,
                        **paired_statistics(
                            outcomes["base", None, split],
                            outcomes[method, seed, split],
                            families,
                            samples=bootstrap_samples,
                        ),
                    }
                )
    # An unchanged final source was freshly verified by the frozen isolated
    # evaluator. Reuse it as evidence about the initial mutant, without replay.
    initial_observations = initial_mutant_observations(features, failures)
    audit["initial_failure_evidence"] = {
        "source": "fresh isolated final verification of source-hash-identical initial mutants",
        "covered_tasks": len(initial_observations),
        "by_split": {
            split: {
                "known": sum(
                    r["initial_public_failure"] is not None for r in features if r["split"] == split
                ),
                "public_fail": sum(
                    r["initial_public_failure"] is True for r in features if r["split"] == split
                ),
                "hidden_fail": sum(
                    r["initial_hidden_failure"] is True for r in features if r["split"] == split
                ),
            }
            for split in SPLITS
        },
    }
    audit["initial_public_hidden_failure"] = (
        "Recovered where unchanged-source evaluation evidence exists; remaining tasks require isolated CPU replay"
    )
    distributions, strata = [], []
    for method, seed in sorted(seen_arms, key=lambda x: (x[0], x[1] or 0)):
        for split in ("final_id", "external"):
            group = [
                r for r in failures if (r["method"], r["seed"], r["split"]) == (method, seed, split)
            ]
            distributions.append(
                {
                    "method": method,
                    "seed": seed,
                    "split": split,
                    "tasks": len(group),
                    "successes": sum(r["success"] for r in group),
                    "categories": dict(Counter(r["primary_category"] for r in group)),
                    "trajectory_available": sum(r["trajectory_available"] for r in group),
                    **{
                        k + "_mean": float(np.mean([r[k] for r in group if r[k] is not None]))
                        if any(r[k] is not None for r in group)
                        else None
                        for k in (
                            "tokens",
                            "turns",
                            "test_calls",
                            "source_changed",
                            "public_pass",
                            "hidden_pass",
                            "invalid_actions",
                            "rejected_patches",
                        )
                    },
                }
            )
            for feature in ("mutation_type", "public_cases", "lines", "branches"):
                bins: dict[str, list[dict]] = defaultdict(list)
                for r in group:
                    val = task_info[r["task_id"]][feature]
                    # Fixed, interpretable strata; no outcome-chosen cutpoints.
                    label = (
                        str(val)
                        if feature in {"mutation_type", "public_cases"}
                        else ("0-9" if val < 10 else "10-19" if val < 20 else "20+")
                        if feature == "lines"
                        else ("0" if val == 0 else "1-2" if val < 3 else "3+")
                    )
                    bins[label].append(r)
                for label, members in sorted(bins.items()):
                    strata.append(
                        {
                            "method": method,
                            "seed": seed,
                            "split": split,
                            "feature": feature,
                            "stratum": label,
                            "tasks": len(members),
                            "successes": sum(r["success"] for r in members),
                            "parent_macro": float(np.mean(list(parent_values(members).values()))),
                        }
                    )
    windows = []
    for method, seed in sorted(seen_arms, key=lambda x: (x[0], x[1] or 0)):
        if method == "base":
            continue
        milestones = sorted(
            r["episodes"] for r in curves if r["method"] == method and r["seed"] == seed
        )
        for low, high in zip(milestones, milestones[1:], strict=False):
            members = [
                r
                for r in health
                if r["method"] == method and r["seed"] == seed and low < r["episodes_end"] <= high
            ]
            if not members:
                continue
            counts = {
                k: sum(r[k] for r in members)
                for k in (
                    "episodes",
                    "successes",
                    "groups",
                    "mixed_groups",
                    "proposal_attempts",
                    "valid_proposals",
                    "total_actions",
                    "invalid_actions",
                    "distinct_reward_proposer_groups",
                    "attempted_proposer_groups",
                )
            }
            windows.append(
                {
                    "method": method,
                    "seed": seed,
                    "measured_low": low,
                    "measured_high": high,
                    "actual_start": members[0]["episodes_start"],
                    "actual_end": members[-1]["episodes_end"],
                    **counts,
                }
            )
    compute = compute_comparisons(curves)
    development_changes = []
    for seed in (2026, 2027, 2028):
        for method in ("A", "B"):
            arm_curves = sorted(
                [r for r in curves if r["seed"] == seed and r["method"] == method],
                key=lambda r: r["episodes"],
            )
            for row in arm_curves[:-1]:
                development_changes.append(
                    {
                        "method": method,
                        "seed": seed,
                        "earlier_iteration": row["iteration"],
                        "earlier_episodes": row["episodes"],
                        **paired_statistics(
                            row["records"],
                            arm_curves[-1]["records"],
                            families,
                            samples=bootstrap_samples,
                        ),
                    }
                )
    development_pairs = []
    for seed in (2026, 2027, 2028):
        a = {r["episodes"]: r for r in curves if r["method"] == "A" and r["seed"] == seed}
        b = {r["episodes"]: r for r in curves if r["method"] == "B" and r["seed"] == seed}
        for episodes in sorted(a.keys() & b.keys()):
            development_pairs.append(
                {
                    "seed": seed,
                    "episodes": episodes,
                    **paired_statistics(
                        a[episodes]["records"],
                        b[episodes]["records"],
                        families,
                        samples=bootstrap_samples,
                    ),
                }
            )
    from scipy.stats import t

    seed_variation = {}
    for split in ("final_id", "external"):
        values = [
            r["difference_parent_macro"]
            for r in paired
            if r["comparison"] == "B-minus-A" and r["split"] == split
        ]
        mean = float(np.mean(values))
        half = float(t.ppf(0.975, 2) * np.std(values, ddof=1) / np.sqrt(3))
        seed_variation[split] = {
            "paired_seed_differences": values,
            "mean": mean,
            "seed_t_95_interval": [mean - half, mean + half],
            "n": 3,
            "caveat": "three seeds; distinct from evaluation-family uncertainty",
        }
    summary = {
        "study_id": FROZEN_STUDY,
        "analysis_kind": "retrospective exploratory",
        "bootstrap_seed": 2026,
        "bootstrap_samples": bootstrap_samples,
        "dataset": audit,
        "episode_settings": episode_settings,
        "arms": arms,
        "evaluation_distributions": distributions,
        "paired": paired,
        "training_seed_variation": seed_variation,
        "checkpoint_selection": selections,
        "gpu_pilot": {
            "required": any(not r["selected_is_final"] for r in selections),
            "new_gpu_hours": 0,
            "reason": "All six selectors choose the final endpoint"
            if all(r["selected_is_final"] for r in selections)
            else "Review development-only evidence before submission; hard ceiling 8 H200-hours",
        },
        "trajectory_coverage": {
            "evaluation_available": sum(r["trajectory_available"] for r in failures),
            "evaluation_expected": len(failures),
        },
        "limitations": [
            "Trained trajectory-dependent action subtypes and causal explanations require missing Yale shards"
            if any(not r["trajectory_available"] for r in failures)
            else "Observational behavior does not establish causation",
            "Initial mutant public/hidden failures recovered on 224 final tasks; remaining ID tasks and other splits require isolated replay",
            "Checkpoint integrity retained historical verification; live re-verification requires Yale",
            "Three training seeds; bootstrap task uncertainty is distinct from seed variability",
            "No new independent holdout; no matched Task 15 GRPO-only or effective-strength control",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("dataset-features", features),
        ("evaluation-failure-taxonomy", failures),
        ("training-health", health),
        ("checkpoint-selection", selections),
        ("compute-efficiency", compute),
        ("transfer-strata", strata),
        ("training-windows", windows),
        ("paired-development", development_pairs),
        ("development-changes", development_changes),
        ("learning-curves", [{k: v for k, v in r.items() if k != "records"} for r in curves]),
    ):
        write_csv(
            output / f"{name}.csv",
            [
                {
                    k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                    for k, v in r.items()
                }
                for r in rows
            ],
        )
    atomic_json(output / "dataset-audit.json", audit)
    atomic_json(
        output / "transfer-analysis.json",
        {"distributions": distributions, "paired": paired, "strata": strata},
    )
    for request in requests:
        atomic_json(output / "fetch" / f"{request['run_id']}.json", request["paths"])
    atomic_json(output / "trajectory-fetch-manifest.json", requests)
    atomic_json(
        output / "input-manifest.json",
        {
            "files": inputs.files,
            "missing": sorted(inputs.missing),
            "input_identity": digest(inputs.files),
        },
    )
    summary["input_identity"] = digest(inputs.files)
    atomic_json(output / "summary.json", summary)
    return summary
