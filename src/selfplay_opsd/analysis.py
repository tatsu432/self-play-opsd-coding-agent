"""Reproducible local-only analysis. Missing data is never a successful result."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from selfplay_opsd.core import canonical, digest
from selfplay_opsd.storage import active_events, all_events, atomic_json, read_json


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {k: canonical(v) if isinstance(v, (dict, list)) else v for k, v in row.items()}
            for row in rows
        )


def paired_bootstrap(a: list[dict], b: list[dict], seed: int = 2026, samples: int = 2000) -> dict:
    first, second = {x["task_id"]: x for x in a}, {x["task_id"]: x for x in b}
    if first.keys() != second.keys() or not first:
        return {
            "status": "incomplete_or_unpaired",
            "difference": None,
            "ci_low": None,
            "ci_high": None,
        }
    groups = defaultdict(list)
    discordant = [0, 0]
    for key in sorted(first):
        x, y = first[key], second[key]
        if x["success"] is None or y["success"] is None or x["parent_id"] != y["parent_id"]:
            return {
                "status": "incomplete_or_unpaired",
                "difference": None,
                "ci_low": None,
                "ci_high": None,
            }
        groups[x["parent_id"]].append(int(y["success"]) - int(x["success"]))
        discordant[0] += bool(x["success"] and not y["success"])
        discordant[1] += bool(y["success"] and not x["success"])
    parents = sorted(groups)
    values = np.array([np.mean(groups[p]) for p in parents])
    rng = np.random.default_rng(seed)
    estimates = values[rng.integers(0, len(parents), size=(samples, len(parents)))].mean(axis=1)
    lo, hi = np.quantile(estimates, [0.025, 0.975])
    return {
        "status": "complete",
        "difference": float(values.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "parents": len(parents),
        "tasks": len(first),
        "resamples": samples,
        "seed": seed,
        "a_only": discordant[0],
        "b_only": discordant[1],
        "uncertainty": "evaluation parents; does not estimate training-seed variance",
    }


def diagnostic_statistics(scores: list[float], improvements: list[float]) -> dict:
    if not scores:
        return {"n": 0, "spearman": None, "status": "missing"}
    if len(scores) != len(improvements):
        raise ValueError("Diagnostic lengths differ")
    constant = len(set(scores)) < 2 or len(set(improvements)) < 2
    correlation = None if constant else float(spearmanr(scores, improvements).statistic)
    selected = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[
        : max(1, (len(scores) + 3) // 4)
    ]
    return {
        "n": len(scores),
        "spearman": correlation,
        "status": "undefined_constant" if constant else "complete",
        "top_quartile_ce_decrease": float(np.mean([improvements[i] for i in selected])),
        "negative_transfer_fraction": float(np.mean(np.array(improvements) < 0)),
    }


def _figure(
    path: Path, x, y, xlabel: str, ylabel: str, *, scatter: bool = False, label: str | None = None
):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    if scatter:
        ax.scatter(x, y, s=18, alpha=0.8, label=label)
    else:
        ax.plot(x, y, marker="o", markersize=3, label=label)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.grid(alpha=0.2)
    if label:
        ax.legend()
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(path.with_suffix("." + extension), dpi=200)
    plt.close(fig)


def analyze(study: Path, *, seed: int = 2026) -> dict:
    output = study / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    summaries, curves, evaluations, phase_rows, dev_rows = [], [], [], [], []
    gaps, align_rows, gradient_rows, qualitative, metadata = [], [], [], [], []
    paired_sets = {}
    for root in sorted(p for p in study.iterdir() if p.is_dir() and (p / "events").exists()):
        events, physical = active_events(root), all_events(root)
        config = read_json(root / "resolved_config.json")
        method = config.get("evaluation_label", config["method"])
        commits = [e for e in events if e["kind"] == "iteration_commit"]
        resources = [e["payload"] for e in physical if e["kind"] == "resource"]
        cost = (
            sum(r["allocated_gpu_hours"] for r in resources)
            if resources
            else sum(
                e["payload"].get("allocated_gpu_hours", 0)
                for e in physical
                if e["kind"] == "phase_time"
            )
        )
        cpu_cost = sum(r.get("allocated_cpu_hours", 0) for r in resources)
        manifest = (
            read_json(root / "manifests/data.json")
            if (root / "manifests/data.json").exists()
            else None
        )
        metadata.append(
            {
                "method": method,
                "resolved_config": config,
                "manifest": manifest,
                "provenance": read_json(root / "provenance.json")
                if (root / "provenance.json").exists()
                else None,
                "segments": [e for e in physical if e["kind"] == "segment_start"],
            }
        )
        for event in physical:
            if event["kind"] == "phase_time":
                phase_rows.append(
                    {
                        "method": method,
                        "iteration": event["completed_iteration"],
                        "phase": event["phase"],
                        **event["payload"],
                    }
                )
        for event in events:
            if event["kind"] == "evaluation":
                evaluations.append({"method": method, **event["payload"]})
        for event in commits:
            record, iteration = event["payload"], event["completed_iteration"]
            m = record["metrics"]
            mix = m["student"]["success_group_mix"]
            curves.append(
                {
                    "method": method,
                    "iteration": iteration,
                    "student_tokens": m["student"]["tokens"],
                    "cumulative_tokens": m["student"].get("cumulative_generated_tokens"),
                    "cumulative_episodes": m["student"].get("cumulative_episodes"),
                    "train_success": m["student"]["repair_success_fraction"],
                    "valid_fraction": m["proposer"]["valid_fraction"],
                    "mixed_group_fraction": m["student"]["mixed_group_fraction"],
                    "gap_mean": m["distill"]["gap"]["mean"],
                    "gate_mean": m["distill"]["gate"]["mean"],
                    "allocated_gpu_hours": m["compute"]["allocated_gpu_hours"],
                    **{f"success_group_{k}": mix.get(str(k), mix.get(k, 0)) for k in range(5)},
                }
            )
            gradient_rows.append(
                {
                    "method": method,
                    "iteration": iteration,
                    **m["student"].get("gradient_interaction", {"sampled": False}),
                }
            )
            for score in m["alignment"]["scores"]:
                align_rows.append({"method": method, "iteration": iteration, **score})
            # Bounded mechanism sampling rule is fixed independently of outcomes.
            if iteration % config.get("logging", {}).get("sample_every", 20) == 0 and record.get(
                "episodes_path"
            ):
                for episode in read_json(root / record["episodes_path"]):
                    for turn_index, turn in enumerate(episode["turns"]):
                        teacher, old = (
                            np.asarray(turn["teacher_logps"]),
                            np.asarray(turn["old_logps"]),
                        )
                        delta = teacher - old
                        gate = (
                            np.ones_like(delta)
                            if method == "A"
                            else 1
                            / (
                                1
                                + np.exp(
                                    -config["objective"]["gate_beta"] * np.clip(delta, -700, 700)
                                )
                            )
                        )
                        gaps.append(
                            {
                                "method": method,
                                "iteration": iteration,
                                "task_id": episode["task_id"],
                                "turn": turn_index,
                                "action": turn["action"],
                                "success": episode["reward"],
                                "tokens": len(delta),
                                "gap_mean": float(delta.mean()) if len(delta) else None,
                                "gate_mean": float(gate.mean()) if len(gate) else None,
                            }
                        )
        # Each evaluation attempt has explicit start/completion markers. Select
        # the latest attempt, never a favorable task subset or a highest test score.
        for split in ("final_id", "external", "dev_monitor"):
            endpoints = (
                ("base",)
                if method == "base"
                else (("primary", "secondary") if split != "dev_monitor" else ("primary",))
            )
            for endpoint in endpoints:
                starts = [
                    e
                    for e in events
                    if e["kind"] == "evaluation_start"
                    and e["payload"]["split"] == split
                    and e["payload"].get("endpoint", "primary") == endpoint
                ]
                chosen_starts: tuple[dict | None, ...] = (
                    tuple(starts)
                    if split == "dev_monitor"
                    else ((starts[-1],) if starts else (None,))
                )
                for start in chosen_starts:
                    attempt = []
                    complete = None
                    expected = None
                    if start:
                        expected_ids = start["payload"]["expected_task_ids"]
                        expected = len(expected_ids)
                        for e in events:
                            if (
                                e["segment_id"] != start["segment_id"]
                                or e["event_index"] <= start["event_index"]
                            ):
                                continue
                            p = e["payload"]
                            if (
                                p.get("split") != split
                                or p.get("checkpoint") != start["payload"]["checkpoint"]
                            ):
                                continue
                            if e["kind"] == "evaluation_complete":
                                complete = e
                                break
                            if e["kind"] == "evaluation":
                                attempt.append(p)
                    ready = (
                        bool(complete)
                        and len(attempt) == expected
                        and len({r["task_id"] for r in attempt}) == expected
                        and all(r["success"] is not None for r in attempt)
                    )
                    groups = defaultdict(list)
                    for row in attempt:
                        if row["success"] is not None:
                            groups[row["parent_id"]].append(int(row["success"]))
                    parent_macro = (
                        float(np.mean([np.mean(v) for v in groups.values()])) if ready else None
                    )
                    if split == "dev_monitor":
                        if ready:
                            iteration = attempt[0]["iteration"]
                            before = [
                                r
                                for r in curves
                                if r["method"] == method and r["iteration"] <= iteration
                            ]
                            latest = before[-1] if before else {}
                            dev_rows.append(
                                {
                                    "method": method,
                                    "iteration": iteration,
                                    "parent_macro": parent_macro,
                                    "task_macro": float(np.mean([r["success"] for r in attempt])),
                                    "parents": len(groups),
                                    "tasks": expected,
                                    "allocated_gpu_hours": latest.get("allocated_gpu_hours", 0.0),
                                    "cumulative_tokens": latest.get("cumulative_tokens", 0),
                                    "cumulative_episodes": latest.get("cumulative_episodes", 0),
                                }
                            )
                        continue
                    completed_iterations = max(
                        (e["completed_iteration"] for e in commits), default=0
                    )
                    # A common-budget primary endpoint cannot silently become an
                    # earlier checkpoint merely because its final test was run.
                    if (
                        ready
                        and method != "base"
                        and endpoint == "primary"
                        and attempt[0]["iteration"] != config["iterations"]
                    ):
                        ready, parent_macro = False, None
                    summaries.append(
                        {
                            "method": method,
                            "split": split,
                            "endpoint": endpoint,
                            "status": "complete" if ready else "incomplete_or_unrun",
                            "numerator": sum(
                                int(r["success"]) for r in attempt if r["success"] is not None
                            ),
                            "denominator": expected,
                            "observed_tasks": len(attempt),
                            "parents": len(groups),
                            "task_macro": float(np.mean([r["success"] for r in attempt]))
                            if ready
                            else None,
                            "parent_macro": parent_macro,
                            "checkpoint": attempt[0]["checkpoint"] if ready else None,
                            "completed_iterations": completed_iterations,
                            "consumed_gpu_hours": cost,
                            "consumed_cpu_hours": cpu_cost,
                            "attempted_proposals": sum(r["payload"]["attempted"] for r in commits),
                            "valid_proposals": sum(r["payload"]["valid"] for r in commits),
                            "student_episodes": sum(r["payload"]["episodes"] for r in commits),
                            "generated_student_tokens": sum(
                                r["payload"]["metrics"]["student"]["tokens"] for r in commits
                            ),
                            "errors": sum(e["kind"] == "error" for e in events),
                            "nonfinite_failures": sum(
                                e["kind"] == "error"
                                and e["payload"].get("type") == "FloatingPointError"
                                for e in events
                            ),
                            "incomplete": not ready,
                        }
                    )
                    if ready and endpoint in {"primary", "base"}:
                        paired_sets[method, split] = attempt
                    # Deterministic, outcome-stratified cases; include failures.
                    for success in (False, True):
                        cases = sorted(
                            [r for r in attempt if r["success"] is success],
                            key=lambda r: digest([seed, r["checkpoint"], r["task_id"]]),
                        )[:4]
                        for row in cases:
                            path = root / row["episode_path"]
                            episode = read_json(path)[0]
                            qualitative.append(
                                {
                                    "method": method,
                                    "split": split,
                                    "checkpoint": row["checkpoint"],
                                    "task_id": row["task_id"],
                                    "success": row["success"],
                                    "termination": episode["termination"],
                                    "actions": [t["raw_text"] for t in episode["turns"]],
                                    "final_source": episode["final_source"],
                                    "selection": "first four by seeded hash within split/outcome/checkpoint",
                                }
                            )
    for method in ("base", "A", "B", "C", "D"):
        for split in ("final_id", "external"):
            if not any(
                r["method"] == method
                and r["split"] == split
                and r["endpoint"] in {"primary", "base"}
                for r in summaries
            ):
                summaries.append(
                    {
                        "method": method,
                        "split": split,
                        "endpoint": "primary",
                        "status": "unrun",
                        "task_macro": None,
                        "parent_macro": None,
                    }
                )
    differences = [
        {
            "a": a,
            "b": b,
            "split": split,
            **paired_bootstrap(
                paired_sets.get((a, split), []), paired_sets.get((b, split), []), seed
            ),
        }
        for split in ("final_id", "external")
        for a, b in (("A", "B"), ("B", "C"), ("C", "D"))
    ]
    diagnostic_rows: list[dict] = []
    utility: list[dict] = []
    for path in sorted(study.glob("diagnostics/*/diagnostics.json")):
        report = read_json(path)
        for r in report["results"]:
            diagnostic_rows.append(
                {
                    "checkpoint": path.parent.name,
                    "task_id": r["task_id"],
                    "c_score": r["c"]["reward"],
                    "d_score": r["d"]["reward"],
                    "diagnostic_ce_decrease": r["diagnostic_ce_decrease"],
                    "reference_ce_decrease": r["reference_ce_decrease"],
                }
            )
        utility.extend(
            {"checkpoint": path.parent.name, "mode": mode, **values}
            for mode, values in report["statistics"].items()
        )
    for name, rows in (
        ("summary", summaries),
        ("paired_differences", differences),
        ("learning_curves", curves),
        ("development_curves", dev_rows),
        ("evaluations", evaluations),
        ("diagnostics", diagnostic_rows),
        ("diagnostic_utility", utility),
        ("phase_times", phase_rows),
        ("gap_gate_by_turn", gaps),
        ("alignments", align_rows),
        ("gradient_interactions", gradient_rows),
    ):
        write_csv(output / f"{name}.csv", rows)
    generated = []
    for method in sorted({r["method"] for r in curves + dev_rows}):
        train = [r for r in curves if r["method"] == method]
        dev = [r for r in dev_rows if r["method"] == method]
        for axis in (
            "iteration",
            "allocated_gpu_hours",
            "cumulative_tokens",
            "cumulative_episodes",
        ):
            valid = [r for r in dev if r[axis] is not None]
            if valid:
                name = f"{method}-dev-success-vs-{axis}"
                _figure(
                    output / name,
                    [r[axis] for r in valid],
                    [r["parent_macro"] for r in valid],
                    axis,
                    "Greedy development parent-macro success",
                    label=method,
                )
                generated.append(name)
        for field in ("valid_fraction", "mixed_group_fraction", "gap_mean", "gate_mean"):
            valid = [r for r in train if r[field] is not None]
            if valid:
                name = f"{method}-{field}"
                _figure(
                    output / name,
                    [r["iteration"] for r in valid],
                    [r[field] for r in valid],
                    "Completed outer iteration",
                    field,
                    label=method,
                )
                generated.append(name)
        if train:
            fig, ax = plt.subplots(figsize=(6.5, 4))
            values = np.array(
                [[r[f"success_group_{k}"] for r in train] for k in range(5)], dtype=float
            )
            values /= np.maximum(1, values.sum(axis=0))
            ax.stackplot(
                [r["iteration"] for r in train], *values, labels=[f"k={k}" for k in range(5)]
            )
            ax.set(xlabel="Completed outer iteration", ylabel="Student success-group fraction")
            ax.legend(loc="upper left", fontsize="small")
            fig.tight_layout()
            for ext in ("png", "pdf"):
                fig.savefig(output / f"{method}-curriculum-mix.{ext}", dpi=200)
            plt.close(fig)
        for field in ("ratio", "cosine"):
            rows = [r for r in gradient_rows if r["method"] == method and r.get(field) is not None]
            if rows:
                _figure(
                    output / f"{method}-rl-sd-{field}",
                    [r["iteration"] for r in rows],
                    [r[field] for r in rows],
                    "Completed outer iteration",
                    f"RL/SD {field}",
                )
    if phase_rows:
        totals: dict[str, float] = defaultdict(float)
        for row in phase_rows:
            totals[row["phase"]] += row["seconds"]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(list(totals), list(totals.values()))
        ax.set(xlabel="Consumed seconds including abandoned work", ylabel="Phase")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(output / f"phase-time-breakdown.{ext}", dpi=200)
        plt.close(fig)
    for method in ("C", "D"):
        rows = [r for r in align_rows if r["method"] == method]
        if rows:
            _figure(
                output / f"{method}-alignment-reward",
                [r["iteration"] for r in rows],
                [r["reward"] for r in rows],
                "Completed outer iteration",
                "Candidate alignment reward",
                scatter=True,
            )
    for mode in ("c", "d"):
        if diagnostic_rows:
            _figure(
                output / f"{mode}-counterfactual-improvement",
                [r[f"{mode}_score"] for r in diagnostic_rows],
                [r["diagnostic_ce_decrease"] for r in diagnostic_rows],
                "Alignment reward",
                "Separate diagnostic-development CE decrease",
                scatter=True,
            )
    result: dict[str, Any] = {
        "summary": summaries,
        "paired_differences": differences,
        "analysis_seed": seed,
        "input_hash": digest([summaries, curves, evaluations]),
        "figures": generated,
        "missing_categories": [
            name
            for name, values in (
                ("development_evaluation", dev_rows),
                ("gradient_interaction", [r for r in gradient_rows if r.get("sampled")]),
                ("alignment", align_rows),
                ("counterfactual", diagnostic_rows),
                ("sampled_gap_gate_by_turn", gaps),
            )
            if not values
        ],
        "limitations": [
            "One training seed; intervals measure evaluation-parent uncertainty only",
            "MBPP+/HumanEval+-derived repair, not official generation scores",
            "Separately evolving curricula confound an isolated gating causal interpretation",
            "Missing and incomplete runs are explicitly retained",
            "Compute includes abandoned work; methods are compared only within observed ranges",
        ],
    }
    atomic_json(output / "summary.json", result)
    atomic_json(output / "reproduction_metadata.json", metadata)
    atomic_json(output / "qualitative_cases.json", qualitative)
    report = "# Experiment results\n\n" + "\n".join(
        f"- {r['method']} / {r['split']} / {r['endpoint']}: {r['status']}; parent-macro={r.get('parent_macro')}"
        for r in summaries
    )
    report += (
        "\n\n## Interpretation\n\n"
        + "\n".join(f"- {x}" for x in result["limitations"])
        + "\n\nNo winner is inferred from missing results or noisy point estimates. Missing diagnostics: "
        + ", ".join(result["missing_categories"])
        + "\n"
    )
    (output / "results.md").write_text(report)
    return result
