"""CPU-only analysis of committed evidence. Generated Python is parsed, never executed.

Raw outputs and identifiers belong in the ignored private runs/ workspace.
No model loading, oracle execution, training, or cloud writes occur here.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.special import betaln

from selfplay_opsd.core import digest
from selfplay_opsd.mutations import apply_patch, strict_json_object, validate_mutation
from selfplay_opsd.objectives import advantages, sd_coefficients
from selfplay_opsd.scientific_health import regression
from selfplay_opsd.storage import active_events, atomic_json, file_hash, read_json

WINDOWS = ((1, 20), (21, 40), (41, 80), (81, 100), (101, 140), (141, 200))
STAGES = (
    "generation_returned",
    "json",
    "schema",
    "headers",
    "hunks",
    "context",
    "python",
    "constraints",
    "semantic_failure",
    "execution_requirements",
    "novel",
    "admitted",
)
HUNK = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@[^\n]*\n?")


def window(iteration: int) -> str:
    return next((f"{a}-{b}" for a, b in WINDOWS if a <= iteration <= b), "other")


def category(reason: str) -> str:
    if reason == "valid":
        return "accepted"
    if reason == "duplicate":
        return "duplicate"
    if "no semantic failure" in reason:
        return "semantic_invalid"
    if any(s in reason for s in ("admission", "privilege", "Canonical action", "context exceeds")):
        return "admission"
    if any(
        s in reason.lower()
        for s in ("unsafe", "forbidden", "unsupported", "private", "introspection")
    ):
        return "safety_or_unsupported"
    if any(
        s in reason for s in ("AST", "API", "Imports changed", "metadata", "structure", "Deleted")
    ):
        return "structural_constraint"
    return "protocol_or_syntax"


def hunk_errors(patch: str) -> list[str]:
    """Independent lexical checks; context matching still uses the production parser."""
    lines = patch.splitlines(keepends=True)[2:]
    errors = []
    if not lines:
        return ["Empty patch"]
    index = 0
    while index < len(lines):
        match = HUNK.fullmatch(lines[index])
        if not match:
            return [*errors, "Invalid hunk header"]
        old_count, new_count = int(match[2] or 1), int(match[4] or 1)
        removed = added = 0
        index += 1
        while index < len(lines) and not lines[index].startswith("@@ "):
            line = lines[index]
            if not line or line[0] not in " +-":
                errors.append("Invalid hunk line or extra file")
            else:
                removed += line[0] in " -"
                added += line[0] in " +"
            index += 1
        if (removed, added) != (old_count, new_count):
            errors.append("Hunk count mismatch")
    return sorted(set(errors))


def ast_edits(before: str, after: str) -> list[str]:
    """Changed AST fields/sites, not the trainer's coarse whole-program family label."""
    edits = []

    def visit(a, b, path):
        if type(a) is not type(b):
            edits.append(f"{path}:{type(a).__name__}->{type(b).__name__}")
        elif isinstance(a, ast.AST):
            for key in a._fields:
                visit(getattr(a, key), getattr(b, key), f"{path}.{key}")
        elif isinstance(a, list):
            for i, (x, y) in enumerate(zip(a, b, strict=False)):
                visit(x, y, f"{path}[{i}]")
            if len(a) != len(b):
                edits.append(f"{path}:length")
        elif a != b:
            edits.append(f"{path}:value")

    visit(ast.parse(before), ast.parse(after), "Module")
    return edits


def proposal_syntax(row: dict) -> dict:
    status: dict[str, bool | None] = dict.fromkeys(STAGES)
    status.update(generation_returned=True, admitted=row["valid"])
    if row["reason"] == "duplicate":
        status["novel"] = False
    elif row["valid"] or category(row["reason"]) in {"admission", "semantic_invalid"}:
        status["novel"] = True
    result = {"stages": status, "additional_failures": [], "sites": [], "mutation_key": None}
    try:
        value = strict_json_object(row["turn"]["raw_text"])
    except (ValueError, TypeError):
        status["json"] = False
        return result
    status["json"] = True
    status["schema"] = (
        isinstance(value, dict) and set(value) == {"patch"} and isinstance(value["patch"], str)
    )
    if not status["schema"]:
        return result
    patch = value["patch"]
    lines = patch.splitlines(keepends=True)
    status["headers"] = (
        len(lines) >= 3
        and lines[0].rstrip() in {"--- a/src/solution.py", "--- src/solution.py"}
        and lines[1].rstrip() in {"+++ b/src/solution.py", "+++ src/solution.py"}
    )
    if status["headers"]:
        errors = hunk_errors(patch)
        status["hunks"] = not errors
        result["additional_failures"].extend(errors)
    try:
        source = json.loads(row["turn"]["history"][1]["content"])["correct_source"]
    except (ValueError, KeyError, IndexError, TypeError):
        result["additional_failures"].append("parent_source_unavailable")
        return result
    try:
        mutated = apply_patch(source, patch)
    except ValueError as exc:
        result["additional_failures"].append(str(exc))
        if "context mismatch" in str(exc):
            status["context"] = False
        return result
    status["context"] = True
    try:
        ast.parse(mutated)
    except (SyntaxError, ValueError):
        status["python"] = False
        return result
    status["python"] = True
    try:
        mutation = validate_mutation(source, patch)
    except (SyntaxError, ValueError) as exc:
        status["constraints"] = False
        result["additional_failures"].append(str(exc))
        return result
    status["constraints"] = True
    result["sites"] = ast_edits(source, mutated)
    result["mutation_key"] = mutation.dedup_key
    if row["valid"] or category(row["reason"]) == "admission":
        status["semantic_failure"] = status["execution_requirements"] = True
    elif "no semantic failure" in row["reason"]:
        status["semantic_failure"] = False
    # Duplicates were rejected before execution. They are not new semantic failures.
    return result


def compare_advantages(rewards: list[float], valid: list[bool]) -> dict:
    real = advantages(torch.tensor(rewards, dtype=torch.float32))
    binary = advantages(torch.tensor([0.0 if v else -1.0 for v in valid]))
    delta = float((real - binary).norm())
    norm = float(real.norm() * binary.norm())
    return {
        "valid_count": sum(valid),
        "reward_variance": float(np.var(rewards)),
        "valid_reward_variance": float(
            np.var([r for r, v in zip(rewards, valid, strict=True) if v])
        )
        if any(valid)
        else None,
        "nonzero_advantages": int(torch.count_nonzero(real)),
        "exact_validity_equal": bool(torch.equal(real, binary)),
        "l2": delta,
        "cosine": float(torch.dot(real, binary)) / norm if norm else None,
        "real": real.tolist(),
        "validity_only": binary.tolist(),
    }


def mixed_probability(successes: int, observed: int, group: int, prior: float) -> float:
    """Beta-binomial posterior predictive, conditional iid assumption explicit."""
    a, b = successes + prior, observed - successes + prior
    return float(
        1
        - math.exp(betaln(a + group, b) - betaln(a, b))
        - math.exp(betaln(a, b + group) - betaln(a, b))
    )


def group_size_projection(outcomes: Counter, observed: int) -> list[dict]:
    total = sum(outcomes.values())
    return [
        {
            "prior": prior,
            "group_size": size,
            "expected_mixed_fraction": sum(
                n * mixed_probability(int(k), observed, size, prior) for k, n in outcomes.items()
            )
            / total
            if total
            else None,
            "rollout_cost_multiple": size / observed,
        }
        for prior in (0.5, 1.0)
        for size in (4, 8, 16)
    ]


def parent_contrast(parents: list[dict], early: str, late: str) -> dict:
    """Paired parent-macro contrasts, conditional on parents observed in both windows."""
    paired = [
        p for p in parents if p["windows"][early]["attempted"] and p["windows"][late]["attempted"]
    ]
    rng = np.random.default_rng(2026)
    result = {"early": early, "late": late, "paired_parents": len(paired), "metrics": {}}
    for metric in ("accepted", "duplicate", "syntax_constraints_passed"):
        before = np.array(
            [p["windows"][early][metric] / p["windows"][early]["attempted"] for p in paired]
        )
        after = np.array(
            [p["windows"][late][metric] / p["windows"][late]["attempted"] for p in paired]
        )
        delta = after - before
        boot = rng.choice(delta, size=(2000, len(delta))).mean(axis=1) if len(delta) else []
        result["metrics"][metric] = {
            "early_mean": float(before.mean()) if len(before) else None,
            "late_mean": float(after.mean()) if len(after) else None,
            "difference": float(delta.mean()) if len(delta) else None,
            "parent_bootstrap_95_interval": np.quantile(boot, [0.025, 0.975]).tolist()
            if len(boot)
            else None,
        }
    return result


def allocated_hours(jobs: list[dict], timestamp: float | None = None) -> float:
    """Integrate scheduler allocations, including overhead and completed prior attempts."""
    return sum(
        max(
            0,
            min(
                job["elapsed_seconds"],
                (timestamp - job["start_epoch"])
                if timestamp is not None
                else job["elapsed_seconds"],
            ),
        )
        * job["gpus"]
        / 3600
        for job in jobs
    )


def committed_records(root: Path) -> tuple[list[dict], list[dict]]:
    events = active_events(root)
    commits = {}
    for event in events:
        if event["kind"] != "iteration_commit":
            continue
        row = event["payload"]
        iteration = row["iteration"]
        if iteration in commits and row != commits[iteration]:
            raise ValueError(f"Conflicting active commits at iteration {iteration}")
        commits[iteration] = row
    return events, sorted(commits.values(), key=lambda c: c["iteration"])


def load_shard(root: Path, relative: str, hashes: dict) -> list[dict] | None:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Shard path escapes run directory")
    if not path.is_file():
        return None
    hashes[relative] = file_hash(path)
    return read_json(path)


def summarize_groups(rows: list[dict]) -> dict:
    return {
        "groups": len(rows),
        "valid_counts": dict(sorted(Counter(r["valid_count"] for r in rows).items())),
        "nonzero_groups": sum(r["nonzero_advantages"] > 0 for r in rows),
        "valid_reward_discriminating_groups": sum(
            (r["valid_reward_variance"] or 0) > 0 for r in rows
        ),
        "exact_validity_equal": sum(r["exact_validity_equal"] for r in rows),
        "approximately_validity_equal": {
            str(t): sum(r["l2"] <= t for r in rows) for t in (1e-6, 1e-5, 1e-4)
        },
        "max_l2": max((r["l2"] for r in rows), default=None),
        "mean_l2": float(np.mean([r["l2"] for r in rows])) if rows else None,
    }


def funnel(rows: list[dict]) -> list[dict]:
    results = []
    for stage in STAGES:
        passed = sum(r["stages"][stage] is True for r in rows)
        failed = sum(r["stages"][stage] is False for r in rows)
        known = passed + failed
        results.append(
            {
                "stage": stage,
                "passed": passed,
                "failed": failed,
                "unknown": len(rows) - known,
                "pass_fraction_all": passed / len(rows) if rows else None,
                "reject_fraction_all": failed / len(rows) if rows else None,
                "conditional_pass_among_assessed": passed / known if known else None,
            }
        )
    return results


def recorded_pipeline(rows: list[dict]) -> list[dict]:
    """Actual ordered production gates; the diff parser interleaves context and counts."""
    stages = (
        "json",
        "schema",
        "diff_application",
        "python_and_constraints",
        "novelty",
        "semantic_execution",
        "admission",
    )
    failures = Counter()
    for row in rows:
        if row["valid"]:
            continue
        status = row["stages"]
        if status["json"] is False:
            stage = "json"
        elif status["schema"] is False:
            stage = "schema"
        elif status["context"] is not True:
            stage = "diff_application"
        elif status["constraints"] is not True:
            stage = "python_and_constraints"
        elif row["category"] == "duplicate":
            stage = "novelty"
        elif row["category"] == "semantic_invalid":
            stage = "semantic_execution"
        elif row["category"] == "admission":
            stage = "admission"
        else:
            raise ValueError(f"Unclassified recorded rejection: {row['reason']}")
        failures[stage] += 1
    result, reached = [], len(rows)
    for stage in stages:
        failed = failures[stage]
        result.append(
            {
                "stage": stage,
                "reached": reached,
                "rejected": failed,
                "passed": reached - failed,
                "conditional_pass_rate": (reached - failed) / reached if reached else None,
            }
        )
        reached -= failed
    if reached != sum(r["valid"] for r in rows):
        raise ValueError("Recorded funnel does not conserve attempts")
    return result


def episode_metrics(episodes: list[dict], gated: bool, objective: dict) -> dict:
    actions, observations, endings = Counter(), Counter(), Counter()
    strata = defaultdict(
        lambda: Counter(
            tokens=0,
            gap_sum=0.0,
            gate_sum=0.0,
            coefficient_sum=0.0,
            abs_coefficient_sum=0.0,
            positive=0,
            clipped=0,
        )
    )
    lengths, patch_newlines = [], Counter()
    for episode in episodes:
        turns = episode["turns"]
        endings[episode["termination"]] += 1
        for i, turn in enumerate(turns):
            key = f"{turn['action']}:{turn['parse_status']}"
            actions[key] += 1
            lengths.append(len(turn["target_ids"]))
            if turn["action"] == "apply_patch":
                try:
                    patch = json.loads(turn["raw_text"])["patch"]
                    patch_newlines["examined"] += 1
                    patch_newlines["actual_newline"] += "\n" in patch
                    patch_newlines["literal_backslash_n"] += "\\n" in patch
                except (ValueError, KeyError, TypeError):
                    patch_newlines["unavailable"] += 1
            # Only the suffix following this action; cumulative histories are not re-counted.
            if i + 1 < len(turns):
                suffix = turns[i + 1]["history"][len(turn["history"]) :]
                for message in suffix:
                    if message["role"] == "tool":
                        try:
                            observation = json.loads(message["content"])
                            observations[
                                f"{turn['action']}:{observation.get('ok', 'unknown')}"
                            ] += 1
                        except (ValueError, AttributeError):
                            observations["unparseable"] += 1
            old, teacher = turn["old_logps"], turn["teacher_logps"]
            if len(old) != len(teacher) or len(old) != len(turn["target_ids"]):
                raise ValueError("Student teacher/behavior/target token count mismatch")
            if not old:
                continue
            gap, gate, coeff = sd_coefficients(
                torch.tensor(teacher),
                torch.tensor(old),
                gated,
                objective.get("gate_beta", 1),
                objective.get("gap_clip", 5),
            )
            for name in (
                key,
                "successful_episode" if episode["reward"] else "failed_episode",
                "all",
            ):
                strata[name].update(
                    tokens=len(old),
                    gap_sum=float(gap.double().sum()),
                    gate_sum=float(gate.double().sum()),
                    coefficient_sum=float(coeff.double().sum()),
                    abs_coefficient_sum=float(coeff.double().abs().sum()),
                    positive=int((gap > 0).sum()),
                    clipped=int((gap.abs() > objective.get("gap_clip", 5)).sum()),
                )
    return {
        "episodes": len(episodes),
        "actions": dict(actions),
        "observations": dict(observations),
        "termination": dict(endings),
        "patch_newlines": dict(patch_newlines),
        "action_tokens": sum(lengths),
        "mean_action_tokens": float(np.mean(lengths)) if lengths else None,
        "test_calls": sum(e["test_calls"] for e in episodes),
        "unsuccessful_finish_episodes": sum(
            e["termination"] == "finish" and not e["reward"] for e in episodes
        ),
        "opsd": {
            k: {
                "tokens": v["tokens"],
                "mean_gap": v["gap_sum"] / v["tokens"],
                "mean_gate": v["gate_sum"] / v["tokens"],
                "mean_coefficient": v["coefficient_sum"] / v["tokens"],
                "mean_abs_coefficient": v["abs_coefficient_sum"] / v["tokens"],
                "positive_gap_fraction": v["positive"] / v["tokens"],
                "clipped_fraction": v["clipped"] / v["tokens"],
            }
            for k, v in strata.items()
        },
    }


def analyze(root: Path, output: Path, accounting: dict | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    config = read_json(root / "resolved_config.json")
    if config["episodes_per_task"] != 4:
        raise ValueError("This observer targets the corrected four-rollout study")
    events, commits = committed_records(root)
    if any(e.get("study_id", config["study_id"]) != config["study_id"] for e in events):
        raise ValueError("Event study identity mismatch")
    if any(
        e["payload"].get("scientific_hash") != config["scientific_hash"]
        for e in events
        if e["kind"] == "run_config"
    ):
        raise ValueError("Resolved/event scientific identity mismatch")
    hashes = {
        str(p.relative_to(root)): file_hash(p)
        for p in [root / "resolved_config.json", *sorted((root / "events").glob("*.jsonl"))]
    }
    for name in ("provenance.json", "manifests/data.json", "state/telemetry-reconciliation.json"):
        if (root / name).is_file():
            hashes[name] = file_hash(root / name)
    proposal_rows, group_rows, episodes = [], [], []
    history: dict[str, set[str]] = defaultdict(set)
    history_complete = True
    ownership, coverage = Counter(), Counter()
    unique_tasks = set()
    group_size = config["proposals_per_parent"]
    for commit in commits:
        iteration = commit["iteration"]
        proposals = load_shard(root, commit["proposals_path"], hashes)
        students = load_shard(root, commit["episodes_path"], hashes)
        coverage["commits"] += 1
        if proposals is None:
            history_complete = False
        else:
            coverage["proposal_shards"] += 1
            if (
                len(proposals) != commit["attempted"]
                or sum(r["valid"] for r in proposals) != commit["valid"]
            ):
                raise ValueError(f"Proposal/commit mismatch at {iteration}")
            if Counter(r["reason"] for r in proposals if not r["valid"]) != Counter(
                commit["metrics"]["proposer"]["invalid_reasons"]
            ):
                raise ValueError(f"Proposal rejection totals mismatch at {iteration}")
            pending: dict[str, set[str]] = defaultdict(set)
            for start in range(0, len(proposals), group_size):
                group = proposals[start : start + group_size]
                if (
                    len(group) != group_size
                    or len({r["parent_id"] for r in group}) != 1
                    or sorted(r["slot"] for r in group) != list(range(group_size))
                ):
                    raise ValueError("Incomplete/cross-parent proposal group")
                group_rows.append(
                    {
                        "iteration": iteration,
                        "window": window(iteration),
                        "parent_id": group[0]["parent_id"],
                        "distinct_seeds": len({r["seed"] for r in group}),
                        "distinct_texts": len({r["turn"]["raw_text"] for r in group}),
                        "distinct_token_sequences": len(
                            {tuple(r["turn"]["target_ids"]) for r in group}
                        ),
                        "identical_text_pairs": sum(
                            a["turn"]["raw_text"] == b["turn"]["raw_text"]
                            for i, a in enumerate(group)
                            for b in group[i + 1 :]
                        ),
                        "assessed_ast_pairs": sum(
                            bool(a.get("dedup_key") and b.get("dedup_key"))
                            for i, a in enumerate(group)
                            for b in group[i + 1 :]
                        ),
                        "identical_ast_pairs": sum(
                            bool(a.get("dedup_key")) and a.get("dedup_key") == b.get("dedup_key")
                            for i, a in enumerate(group)
                            for b in group[i + 1 :]
                        ),
                        **compare_advantages(
                            [r["reward"] for r in group], [r["valid"] for r in group]
                        ),
                    }
                )
            for row in proposals:
                syntax = proposal_syntax(row)
                parent, key = row["parent_id"], row.get("dedup_key")
                owner = None
                if row["reason"] == "duplicate":
                    owner = (
                        "previous_iteration"
                        if key in history[parent]
                        else "same_iteration"
                        if key in pending[parent]
                        else "unresolved"
                    )
                    ownership[owner] += 1
                    if owner != "unresolved":
                        syntax["stages"]["semantic_failure"] = syntax["stages"][
                            "execution_requirements"
                        ] = True
                if row["valid"]:
                    pending[parent].add(key)
                    unique_tasks.add((parent, key))
                turn = row["turn"]
                proposal_rows.append(
                    {
                        "iteration": iteration,
                        "window": window(iteration),
                        "parent_id": parent,
                        "slot": row["slot"],
                        "proposal_id": row["proposal_id"],
                        "reason": row["reason"],
                        "category": category(row["reason"]),
                        "reward": row["reward"],
                        "valid": row["valid"],
                        "duplicate_owner": owner,
                        "semantic_evidence": "inherited_from_accepted_AST"
                        if owner and owner != "unresolved"
                        else "recorded_or_unknown",
                        "text_hash": digest(turn["raw_text"]),
                        "token_hash": digest(turn["target_ids"]),
                        "tokens": len(turn["target_ids"]),
                        "sampled_surprisal": -sum(turn["old_logps"]) / len(turn["old_logps"])
                        if turn["old_logps"]
                        else None,
                        **syntax,
                    }
                )
            for parent, keys in pending.items():
                history[parent].update(keys)
        if students is not None:
            coverage["episode_shards"] += 1
            if len(students) != commit["episodes"]:
                raise ValueError(f"Episode/commit mismatch at {iteration}")
            grouped = defaultdict(list)
            for episode in students:
                grouped[episode["task_id"]].append(episode)
            observed = Counter()
            for group in grouped.values():
                if len(group) != config["episodes_per_task"]:
                    raise ValueError("Incomplete student task group")
                observed[str(int(sum(e["reward"] for e in group)))] += 1
            if observed != Counter(commit["metrics"]["student"]["success_group_mix"]):
                raise ValueError(f"Student group/commit mismatch at {iteration}")
            episodes.extend(students)
    baseline = regression(root, output / "health")
    outcomes = Counter()
    for commit in commits:
        outcomes.update(commit["metrics"]["student"]["success_group_mix"])
    windows = []
    for a, b in WINDOWS:
        ps = [r for r in proposal_rows if a <= r["iteration"] <= b]
        gs = [r for r in group_rows if a <= r["iteration"] <= b]
        windows.append(
            {
                "window": f"{a}-{b}",
                "raw_proposals": len(ps),
                "categories": dict(Counter(r["category"] for r in ps)),
                "reasons": dict(Counter(r["reason"] for r in ps)),
                "funnel": funnel(ps),
                "recorded_pipeline": recorded_pipeline(ps),
                "groups": summarize_groups(gs),
                "mean_tokens": float(np.mean([r["tokens"] for r in ps])) if ps else None,
                "mean_sampled_surprisal": float(
                    np.mean(
                        [r["sampled_surprisal"] for r in ps if r["sampled_surprisal"] is not None]
                    )
                )
                if ps
                else None,
            }
        )
    parents = []
    for parent in sorted({r["parent_id"] for r in proposal_rows}):
        ps = [r for r in proposal_rows if r["parent_id"] == parent]
        parents.append(
            {
                "parent_id": parent,
                "proposals": len(ps),
                "accepted": sum(r["valid"] for r in ps),
                "duplicate": sum(r["category"] == "duplicate" for r in ps),
                "distinct_texts": len({r["text_hash"] for r in ps}),
                "distinct_tokens": len({r["token_hash"] for r in ps}),
                "distinct_mutations": len({r["mutation_key"] for r in ps if r["mutation_key"]}),
                "distinct_sites": len({s.split(":")[0] for r in ps for s in r["sites"]}),
                "mutation_families": dict(
                    Counter(s.split(":")[-1] for r in ps for s in r["sites"])
                ),
                "windows": {
                    f"{a}-{b}": {
                        "attempted": len(ws := [r for r in ps if a <= r["iteration"] <= b]),
                        "accepted": sum(r["valid"] for r in ws),
                        "duplicate": sum(r["category"] == "duplicate" for r in ws),
                        "syntax_constraints_passed": sum(
                            r["stages"]["constraints"] is True for r in ws
                        ),
                    }
                    for a, b in WINDOWS
                },
            }
        )
    exposures = []
    for evaluation in baseline["evaluations"]:
        k = evaluation["iteration"]
        cs = [c for c in commits if c["iteration"] <= k]
        exposures.append(
            {
                "iteration": k,
                "dev_successes": evaluation["successes"],
                "dev_tasks": evaluation["tasks"],
                "accepted_tasks": sum(c["valid"] for c in cs),
                "unique_tasks_observed": len(
                    {
                        (r["parent_id"], r["mutation_key"])
                        for r in proposal_rows
                        if r["valid"] and r["iteration"] <= k
                    }
                ),
                "mixed_groups": sum(
                    sum(
                        v
                        for n, v in c["metrics"]["student"]["success_group_mix"].items()
                        if 0 < int(n) < config["episodes_per_task"]
                    )
                    for c in cs
                ),
                "student_steps": sum(c["student_steps"] for c in cs),
                "student_tokens": sum(c["metrics"]["student"].get("tokens", 0) for c in cs),
                "event_allocated_gpu_hours": cs[-1]["metrics"]
                .get("compute", {})
                .get("allocated_gpu_hours")
                if cs
                else 0,
            }
        )
        complete_times = [
            e["wall_timestamp"]
            for e in events
            if e["kind"] == "evaluation_complete"
            and e["completed_iteration"] == k
            and e["payload"].get("split") == "dev_monitor"
            and e["payload"].get("completed") == e["payload"].get("expected")
        ]
        jobs = (accounting or {}).get(config["method"], [])
        exposures[-1]["scheduler_allocated_gpu_hours"] = (
            allocated_hours(jobs, max(complete_times)) if jobs and complete_times else None
        )
    report = {
        "schema_version": 1,
        "method": config["method"],
        "study_id": config["study_id"],
        "scientific_hash": config["scientific_hash"],
        "model": config["model"],
        "source_root": str(root),
        "input_sha256": hashes,
        "coverage": dict(coverage) | {"history_complete": history_complete},
        "attempted": sum(c["attempted"] for c in commits),
        "accepted": sum(c["valid"] for c in commits),
        "unique_accepted_observed": len(unique_tasks),
        "categories": dict(Counter(r["category"] for r in proposal_rows)),
        "funnel": funnel(proposal_rows),
        "recorded_pipeline": recorded_pipeline(proposal_rows),
        "analysis_sha256": {
            str(p): file_hash(p)
            for p in (
                Path(__file__),
                Path(__file__).parents[1] / "uv.lock",
                Path(__file__).parents[1] / "src/selfplay_opsd/objectives.py",
                Path(__file__).parents[1] / "src/selfplay_opsd/mutations.py",
                Path(__file__).parents[1] / "src/selfplay_opsd/scientific_health.py",
                Path(__file__).parents[1] / "src/selfplay_opsd/storage.py",
            )
        },
        "groups": summarize_groups(group_rows),
        "reward_discriminating_update_iterations": len(
            {
                r["iteration"]
                for r in group_rows
                if r["l2"] > 1e-5
                and next(c["proposer_steps"] for c in commits if c["iteration"] == r["iteration"])
            }
        ),
        "identical_text_pairs": sum(r["identical_text_pairs"] for r in group_rows),
        "assessed_ast_pairs": sum(r["assessed_ast_pairs"] for r in group_rows),
        "identical_ast_pairs": sum(r["identical_ast_pairs"] for r in group_rows),
        "duplicate_ownership": dict(ownership),
        "outcome_groups": dict(sorted(outcomes.items())),
        "student": episode_metrics(episodes, config["gate"] != "none", config["objective"]),
        "windows": windows,
        "group_size_projection": group_size_projection(outcomes, config["episodes_per_task"]),
        "parent_contrasts": [
            parent_contrast(parents, "1-20", late) for late in ("81-100", "141-200")
        ],
        "scheduler_allocated_gpu_hours": allocated_hours((accounting or {})[config["method"]])
        if accounting and config["method"] in accounting
        else None,
        "exposure": exposures,
        "recorded_optimizer": {
            role: {
                "steps": sum(c.get(role + "_steps", 0) for c in commits),
                "clipped_steps": sum(
                    bool(c.get(role + "_steps", 0))
                    and c["metrics"][role].get("clip_coefficient", 1) < 1
                    for c in commits
                ),
                "max_preclip_grad_norm": max(
                    (c["metrics"][role].get("grad_norm", 0) for c in commits), default=None
                ),
            }
            for role in ("student", "proposer")
        },
        "repeated_text_groups": sum(r["distinct_texts"] < group_size for r in group_rows),
        "repeated_token_groups": sum(
            r["distinct_token_sequences"] < group_size for r in group_rows
        ),
        "repeated_seed_groups": sum(r["distinct_seeds"] < group_size for r in group_rows),
        "limitations": [
            "Generation stop reasons and full policy entropy are not recorded.",
            "Funnel stages overlap; unassessed stages are unknown, not failures.",
            "Duplicate semantic evidence is inherited only from an observed accepted normalized AST.",
            "Posterior projections assume conditional iid rollouts and are prior-sensitive, not measured G=8 outcomes.",
            "Advantage discrimination is not a count of measured nonzero policy gradients.",
        ],
    }
    atomic_json(output / "diagnosis.json", report)
    atomic_json(output / "proposals.json", proposal_rows)
    atomic_json(output / "groups.json", group_rows)
    atomic_json(output / "parents.json", parents)
    health_rows = []
    for commit in commits:
        groups = [r for r in group_rows if r["iteration"] == commit["iteration"]]
        student = commit["metrics"]["student"]
        health_rows.append(
            {
                "iteration": commit["iteration"],
                "proposer_groups_observed": len(groups),
                "proposer_groups_with_2plus_valid": sum(r["valid_count"] >= 2 for r in groups)
                if groups
                else None,
                "proposer_groups_reward_discriminating": sum(r["l2"] > 1e-5 for r in groups)
                if groups
                else None,
                "proposer_mean_advantage_l2_vs_validity": float(np.mean([r["l2"] for r in groups]))
                if groups
                else None,
                "accepted_tasks": commit["valid"],
                "student_mixed_groups": sum(
                    n
                    for k, n in student["success_group_mix"].items()
                    if 0 < int(k) < config["episodes_per_task"]
                ),
                "student_optimizer_step": commit["student_steps"],
                "alignment_neutral_fraction": commit["metrics"]
                .get("alignment", {})
                .get("near_zero_fraction"),
            }
        )
    for name, rows in (
        ("funnel", report["funnel"]),
        ("recorded_pipeline", report["recorded_pipeline"]),
        ("exposure", exposures),
        ("group_size_projection", report["group_size_projection"]),
        ("health", health_rows),
    ):
        if rows:
            with (output / f"{name}.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    return report


def dashboard(reports: dict[str, dict], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fields = (
        ("iteration", "Iteration"),
        ("accepted_tasks", "Accepted unique tasks"),
        ("mixed_groups", "Mixed student groups"),
        ("student_tokens", "Generated student tokens"),
        ("scheduler_allocated_gpu_hours", "Allocated GPU-hours (Slurm)"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for axis, (field, label) in zip(axes.flat, fields, strict=False):
        for name, report in reports.items():
            rows = [r for r in report["exposure"] if r[field] is not None]
            axis.plot(
                [r[field] for r in rows],
                [r["dev_successes"] for r in rows],
                marker="o",
                markersize=3,
                label=name,
            )
        axis.set(xlabel=label, ylabel="Development successes / 41", ylim=(0, 16))
        axis.grid(alpha=0.2)
    axes.flat[0].legend()
    axis = axes.flat[5]
    names = list(reports)
    axis.bar(
        names,
        [
            100 * (1 - r["groups"]["approximately_validity_equal"]["1e-05"] / r["groups"]["groups"])
            for r in reports.values()
        ],
    )
    axis.set(
        ylabel="Groups with advantage L2 difference > 1e-5 (%)",
        title="Actual reward versus validity only",
    )
    for extension in ("png", "pdf"):
        fig.savefig(output / f"learning-efficiency.{extension}", dpi=160)
    plt.close(fig)
    lines = [
        "# Recorded scientific health",
        "",
        "| Method | Accepted / attempts | Mixed / tasks | Reward-discriminating groups | Allocated GPU-hours |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, report in reports.items():
        g = report["groups"]
        mixed = sum(n for k, n in report["outcome_groups"].items() if 0 < int(k) < 4)
        lines.append(
            f"| {name} | {report['accepted']}/{report['attempted']} | {mixed}/{report['accepted']} | {g['valid_reward_discriminating_groups']}/{g['groups']} | {report['scheduler_allocated_gpu_hours']} |"
        )
    lines += [
        "",
        "Raw proposals and action histories remain private. See each method's diagnosis.json for stage missingness, coverage, provenance, OPSD strata, and parent contrasts.",
        "",
        "![Learning efficiency](learning-efficiency.png)",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--accounting",
        type=Path,
        help="Optional JSON: method -> scheduler jobs with start_epoch, elapsed_seconds, gpus",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    names = set()
    reports = {}
    for entry in args.run:
        name, path = entry.split("=", 1)
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in names:
            parser.error("Run names must be unique simple identifiers")
        names.add(name)
        root = Path(path).resolve()
        if output.is_relative_to(root) or root.is_relative_to(output):
            parser.error("Output must be separate from input run directories")
        accounting = read_json(args.accounting) if args.accounting else None
        report = analyze(root, output / name, accounting)
        reports[name] = report
        if args.accounting:
            report["input_sha256"]["accounting:" + str(args.accounting)] = file_hash(
                args.accounting
            )
            atomic_json(output / name / "diagnosis.json", report)
        print(
            json.dumps(
                {
                    "name": name,
                    "attempted": report["attempted"],
                    "accepted": report["accepted"],
                    "coverage": report["coverage"],
                    "groups": report["groups"],
                }
            )
        )

    dashboard(reports, output)


if __name__ == "__main__":
    main()
