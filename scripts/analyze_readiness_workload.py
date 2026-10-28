"""Build auditable conditional cost/memory summaries from fetched readiness evidence."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from selfplay_opsd.config import load_config
from selfplay_opsd.core import digest
from selfplay_opsd.readiness import hardware_settings
from selfplay_opsd.storage import active_events, atomic_json, file_hash, read_json
from selfplay_opsd.workload import project_method


def allocation(budget: dict, run_id: str) -> tuple[float, list[str]]:
    rows = [r for r in budget["reservations"].values() if r["run_id"] == run_id]
    if not rows or any(r.get("charged") is None for r in rows):
        raise ValueError(f"Allocation accounting unresolved for {run_id}")
    return 3600 * sum(r["charged"] for r in rows), [r["job_id"] for r in rows]


def memory_summary(events: list[dict], samples: list[dict], fixture: dict) -> dict:
    phase_resources = [e["payload"]["resources"] for e in events if e["kind"] == "phase_time"]
    ordinary = [
        r
        for r in samples
        if not fixture.get("started_at", float("inf"))
        <= r["timestamp"]
        <= fixture.get("ended_at", -float("inf"))
    ]
    processes = defaultdict(lambda: {"gpu_mib": 0, "host_rss_kib": 0})
    for r in ordinary:
        for p in r.get("processes", []):
            row = processes[(p["pid"], p.get("process_name"))]
            try:
                row["gpu_mib"] = max(row["gpu_mib"], float(p["gpu_memory_mib"]))
            except (ValueError, TypeError):
                pass
            row["host_rss_kib"] = max(row["host_rss_kib"], p.get("host_rss_kib") or 0)
    return {
        "sampling": "Nominal 1s plus query latency; sampled peaks are lower bounds",
        "samples": len(samples),
        "sampling_errors": sum("sampling_error" in r for r in samples),
        "whole_device_peak_mib": max(
            (d["used_mib"] for r in ordinary for d in r.get("devices", [])), default=0
        ),
        "whole_device_capacity_mib": sorted(
            {d["total_mib"] for r in ordinary for d in r.get("devices", [])}
        ),
        "actor_plus_descendant_phase_sample_peak_gib": max(
            (r["host_rss_bytes"] + r["descendant_rss_bytes"] for r in phase_resources),
            default=0,
        )
        / 2**30,
        "gpu_process_peaks": [
            {"pid": pid, "name": name, **peaks} for (pid, name), peaks in processes.items()
        ],
        "fixture_interval_excluded": bool(fixture),
    }


def workload_summary(events: list[dict], root: Path) -> dict:
    commits = [e["payload"] for e in events if e["kind"] == "iteration_commit"]
    invalid, groups, termination = Counter(), Counter(), Counter()
    for c in commits:
        m = c["metrics"]
        invalid.update(m["proposer"]["invalid_reasons"])
        groups.update(m["student"]["success_group_mix"])
        termination.update(m["student"]["termination"])
    actions, token_lengths = Counter(), Counter()
    truncated_histories, proposal_limit_hits = 0, 0
    sources, episode_ids = {}, []
    for c in commits:
        ep = root / c["episodes_path"]
        pp = root / c["proposals_path"]
        sources[str(ep)] = file_hash(ep)
        sources[str(pp)] = file_hash(pp)
        proposals = read_json(pp)
        assert len(proposals) == c["attempted"]
        proposal_limit_hits += sum(len(p["turn"]["target_ids"]) == 512 for p in proposals)
        episodes = read_json(ep)
        assert len(episodes) == c["episodes"]
        for episode in episodes:
            episode_ids.append(episode["episode_id"])
            for turn in episode["turns"]:
                actions[f"{turn['action']}:{turn['parse_status']}"] += 1
                token_lengths[len(turn["target_ids"])] += 1
                truncated_histories += any(
                    "[OBSERVATION_TRUNCATED]" in h["content"] for h in turn["history"]
                )
    assert len(episode_ids) == len(set(episode_ids)), "Duplicated active episode exposure"
    phases = Counter()
    for e in events:
        if e["kind"] == "phase_time":
            phases[e["phase"]] += e["payload"]["seconds"]
    return {
        "proposal_rejections": dict(invalid),
        "raw_shard_sha256": sources,
        "unique_episode_ids": len(episode_ids),
        "student_actions": dict(actions),
        "student_turn_token_lengths": dict(token_lengths),
        "histories_with_observation_truncation": truncated_histories,
        "proposer_outputs_at_512_tokens": proposal_limit_hits,
        "success_group_mix": dict(groups),
        "termination": dict(termination),
        "phase_seconds": dict(phases),
        "iterations": [
            {
                **{k: c[k] for k in ("iteration", "attempted", "valid", "episodes")},
                "student": c["metrics"]["student"],
                "proposer": c["metrics"]["proposer"],
                "distill": c["metrics"]["distill"],
                "alignment": c["metrics"]["alignment"],
            }
            for c in commits
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--study-state", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--baseline", required=True, help="RUN_ID:LOCAL_DIRECTORY")
    parser.add_argument(
        "--run", action="append", required=True, help="METHOD:RUN_ID:LOCAL_DIRECTORY"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.settings)
    state = read_json(args.study_state)
    budget = state["budget"]
    baseline_id, baseline_path = args.baseline.split(":", 1)
    baseline_seconds, _ = allocation(budget, baseline_id)
    baseline = read_json(Path(baseline_path) / "evaluation_complete.json")
    runs = {m: (i, Path(p)) for m, i, p in (s.split(":", 2) for s in args.run)}
    if set(runs) != set("ABCD"):
        raise ValueError("All four actual method windows are required")
    pilot = read_json(runs["D"][1] / "pilot_report.json")
    methods = {m: read_json(args.readiness / "method-smokes" / f"{m}.json") for m in "ABC"}
    projections, measured = {}, {}
    for method in "DABC":
        run_id, root = runs[method]
        seconds, jobs = allocation(budget, run_id)
        events = active_events(root)
        fixture = methods.get(method, {}).get("optimizer_mechanics_fixture", {})
        projections[method] = project_method(
            events,
            allocated_seconds=seconds,
            baseline_seconds=baseline_seconds,
            checkpoint_every=cfg.checkpoint_every,
            acceptance_only_seconds=fixture.get("seconds", 0),
            diagnostic_seconds_per_valid=projections.get("D", {}).get(
                "diagnostic_seconds_per_valid_p90"
            ),
        )
        samples = []
        hashes = {}
        for job in jobs:
            path = args.readiness / f"resource-samples-{job}.jsonl"
            samples.extend(json.loads(line) for line in path.read_text().splitlines())
            hashes[str(path)] = file_hash(path)
        measured[method] = {
            "run_id": run_id,
            "jobs": jobs,
            "sample_hashes": hashes,
            "memory": memory_summary(events, samples, fixture),
            "workload": workload_summary(events, root),
            "fixture": fixture,
        }
    plan = {
        "schema_version": 1,
        "runtime_identity": pilot["runtime_identity"],
        "common_config": cfg.scientific_settings(common=True),
        "pilot_hash": digest(pilot),
        "baseline_hash": digest(baseline),
        "method_hashes": {m: digest(r) for m, r in methods.items()},
        "checkpoint_every": cfg.checkpoint_every,
        "hardware": hardware_settings(cfg.hardware.model_dump(mode="json")),
        "gpu_type": "h200",
        "per_method": projections,
        "total_planning_gpu_hours": sum(
            p["planning_projection_200_hours"] for p in projections.values()
        ),
        "measurement_sources": {
            m: {k: v for k, v in r.items() if k != "workload"} for m, r in measured.items()
        },
    }
    atomic_json(args.output / "resource_plan.json", plan)
    atomic_json(args.output / "workload_measurements.json", measured)


if __name__ == "__main__":
    main()
