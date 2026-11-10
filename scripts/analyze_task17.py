"""Audit recovered Task 16 actions without executing candidate programs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from selfplay_opsd.analysis import write_csv
from selfplay_opsd.data import TaskRepository
from selfplay_opsd.storage import atomic_json, file_hash
from selfplay_opsd.task16_analysis import ast_features
from selfplay_opsd.task17_recovery import audit_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/protocol-v3"))
    parser.add_argument(
        "--recovered", type=Path, default=Path("runs/task16/recovered-analysis-verified")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figures", type=Path)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Emit explicitly incomplete coverage when referenced files are absent",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output exists; preserve prior analysis and use a new output directory")
    inputs = {}

    def read(path):
        inputs[str(path.resolve())] = file_hash(path)
        return json.loads(path.read_text())

    manifest = read(args.recovered / "input-manifest.json")
    repository = TaskRepository(args.data)
    tasks = {t.task_id: t for split in ("final_id", "external") for t, _ in repository.tasks(split)}
    parents = {p["parent"]["parent_id"]: p["parent"] for p in repository._parents}
    with (args.recovered / "evaluation-failure-taxonomy.csv").open() as stream:
        taxonomy = list(csv.DictReader(stream))
    inputs[str((args.recovered / "evaluation-failure-taxonomy.csv").resolve())] = file_hash(
        args.recovered / "evaluation-failure-taxonomy.csv"
    )
    feature_path = args.recovered / "dataset-features.csv"
    inputs[str(feature_path.resolve())] = file_hash(feature_path)
    with feature_path.open() as stream:
        features_by_task = {r["task_id"]: r for r in csv.DictReader(stream)}
    lookup = {(r["method"], r["seed"], r["task_id"]): r for r in taxonomy}
    if len(lookup) != len(taxonomy):
        raise ValueError("Ambiguous evaluation checkpoint/task identity")
    paths = [
        Path(p) for p in manifest["files"] if "/trajectories/" in p and "-eval-" in Path(p).name
    ]
    rows, candidates, episodes, seen = [], [], [], set()
    missing = []
    for path in sorted(paths):
        if not path.is_file() and args.allow_missing:
            missing.append(str(path))
            continue
        if file_hash(path) != manifest["files"][str(path)]:
            raise ValueError("Recovered trajectory checksum changed")
        root = Path(str(path).split("/trajectories/")[0])
        resolved = read(root / "resolved_config.json")
        payload = read(path)
        for episode in payload if isinstance(payload, list) else [payload]:
            key = (
                "base" if root.name == "base" else resolved["method"],
                "" if root.name == "base" else str(resolved["seed"]),
                episode["task_id"],
            )
            if key not in lookup:
                continue
            if key in seen:
                raise ValueError("Duplicate evaluation trajectory")
            seen.add(key)
            info = lookup[key]
            if bool(episode["reward"]) != (info["success"].lower() == "true"):
                raise ValueError("Trajectory outcome differs from recovered evaluation table")
            task = tasks[episode["task_id"]]
            actions, recovered = audit_episode(task, episode)
            identity = {k: info[k] for k in ("method", "seed", "split", "checkpoint")}
            identity.update(family_id=parents[task.parent_id]["family_id"])
            historical_features = features_by_task[task.task_id]
            identity.update(
                {
                    name: historical_features[name]
                    for name in (
                        "source_tokens",
                        "spec_tokens",
                        "public_cases",
                        "mutation_type",
                        "mutation_sites",
                    )
                }
            )
            features = ast_features(task.initial_source)
            identity.update(
                source_characters=len(task.initial_source),
                specification_characters=len(task.specification),
                **features,
            )
            rows.extend(identity | r for r in actions)
            candidates.extend(identity | r for r in recovered)
            episodes.append(
                identity
                | {
                    "task_id": task.task_id,
                    "parent_id": task.parent_id,
                    "success": bool(episode["reward"]),
                    "actions": len(actions),
                    "rejected_edits": sum(
                        r["parse_status"] == "tool_error" and r["action"] == "apply_patch"
                        for r in actions
                    ),
                    "recoverable_actions": len(recovered),
                }
            )
    if len(seen) != len(taxonomy) and not args.allow_missing:
        raise ValueError(f"Incomplete evaluation coverage: {len(seen)}/{len(taxonomy)}")
    if len(seen) + len(missing) != len(taxonomy):
        raise ValueError("Coverage differs beyond explicitly missing trajectories")
    grouped = defaultdict(Counter)
    for row in rows:
        group = (row["method"], row["seed"], row["split"])
        grouped[group][row["category"]] += 1
        grouped[group]["recovered"] += row["recovered"]
        grouped[group]["admitted"] += row["admitted"]
    summary = {
        "episodes": len(seen),
        "expected_episodes": len(taxonomy),
        "missing_trajectories": len(missing),
        "actions": len(rows),
        "reconstructed_admitted_edits": len(candidates),
        "semantic_replays": 0,
        "new_gpu_hours": 0,
        "groups": [
            {"method": key[0], "seed": key[1], "split": key[2], **dict(value)}
            for key, value in sorted(grouped.items())
        ],
    }
    args.output.mkdir(parents=True, mode=0o700)
    write_csv(args.output / "actions.csv", rows)
    write_csv(args.output / "episodes.csv", episodes)
    atomic_json(args.output / "replay-candidates.json", candidates)
    atomic_json(args.output / "summary.json", summary)
    for path in [
        args.data / "manifest.json",
        args.data / "private/parents.json",
        *[args.data / f"private/tasks-{split}.json" for split in ("external", "final_id")],
        Path(__file__),
        Path("src/selfplay_opsd/task17_recovery.py"),
        Path("src/selfplay_opsd/mutations.py"),
        Path("src/selfplay_opsd/core.py"),
        Path("src/selfplay_opsd/data.py"),
        Path("src/selfplay_opsd/task16_analysis.py"),
        Path("uv.lock"),
    ]:
        inputs[str(path.resolve())] = file_hash(path)
    atomic_json(
        args.output / "input-manifest.json",
        {"files": inputs, "missing": missing, "task16_input_identity": manifest["input_identity"]},
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = args.figures or args.output / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    categories = [
        "context",
        "hunk_counts",
        "hunk_offsets",
        "hunk_format",
        "diff_headers",
        "path",
        "syntax",
        "api_preservation",
        "import_source_policy",
        "unknown",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for ax, split in zip(axes, ("final_id", "external"), strict=True):
        rs = [
            r
            for r in rows
            if r["split"] == split
            and r["method"] != "base"
            and r["parse_status"] == "tool_error"
            and r["action"] == "apply_patch"
        ]
        counts = Counter(r["category"] for r in rs)
        ax.barh(categories, [counts[c] for c in categories], color="#355C9A")
        ax.set(title=f"{split}: {len(rs)} rejected edits", xlabel="Actions (six trained arms)")
    fig.suptitle("Static replay taxonomy; semantic recovery remains unverified")
    for suffix in ("png", "pdf"):
        fig.savefig(destination / f"rejection-taxonomy.{suffix}", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({k: v for k, v in summary.items() if k != "groups"}))


if __name__ == "__main__":
    main()
