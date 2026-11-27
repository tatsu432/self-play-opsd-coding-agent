"""Publish and verify only the consolidated aggregate evidence on W&B."""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import wandb
from analyze_task22 import METHODS

from selfplay_opsd.core import read_json
from selfplay_opsd.storage import atomic_json, file_hash

ENTITY = "tatsu432s"
PROJECT = "self-play-opsd-coding-agent"
RUN = "task23-final-aggregates"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("statistics", type=Path)
    parser.add_argument("--private", type=Path, default=Path("runs/task23/wandb"))
    args = parser.parse_args()
    stats = read_json(args.statistics)
    args.private.mkdir(parents=True, exist_ok=True)
    endpoints, learning, dynamics, contrasts, alignment, followups = [], [], [], [], [], []
    for _, cell in stats["cells"].items():
        method, seed = cell["method"], cell["seed"]
        for split, row in cell["scores"].items():
            endpoints.append([method, seed, split, row["verified"], row["count"]])
        for row in cell["curves"]:
            learning.append(
                [
                    method,
                    seed,
                    row["episodes"],
                    row["student_tokens"],
                    row["steps"],
                    row["h200_hours"],
                    row["verified"],
                    row["count"],
                ]
            )
        for row in cell["batches"]:
            dynamics.append(
                [
                    method,
                    seed,
                    row["episode"],
                    row["online_rate"],
                    row["rolling_rate"],
                    row["mixed_fraction"],
                    row["valid_proposal_rate"],
                    row["weighted_ratio"],
                ]
            )
        if stats_ := cell["alignment"]:
            for group, row in stats_["by_outcome"].items():
                alignment.append(
                    [
                        method,
                        seed,
                        group,
                        row["count"],
                        row["arms"]["C"]["near_zero"],
                        row["arms"]["D"]["near_zero"],
                        row["arms"]["D"]["positive_cosines"],
                    ]
                )
    for name, splits in stats["contrasts"].items():
        for split, row in splits.items():
            contrasts.append([name, split, row["tasks"], row["difference_pp"], *row["ci95_pp"]])
    secondary = stats.get("followup", {})
    if "relaxed" in secondary:
        for name, row in secondary["relaxed"]["cells"].items():
            followups.append(
                ["relaxed_primary", name, row["count"], row["verified"], row["baseline_verified"]]
            )
    for split, section in secondary.get("ood", {}).items():
        for name, row in section["cells"].items():
            followups.append([split, name, row["count"], row["verified"], None])
    table_spec = {
        "verified_endpoints": (["method", "seed", "split", "verified", "count"], endpoints),
        "fixed_development": (
            [
                "method",
                "seed",
                "episodes",
                "tokens",
                "optimizer_steps",
                "h200_hours",
                "verified",
                "count",
            ],
            learning,
        ),
        "online_dynamics": (
            [
                "method",
                "seed",
                "episodes",
                "raw_success",
                "rolling_success",
                "mixed_groups",
                "valid_proposals",
                "weighted_opsd_to_rl",
            ],
            dynamics,
        ),
        "paired_contrasts": (
            ["contrast", "split", "tasks", "pp", "conditional_low", "conditional_high"],
            contrasts,
        ),
        "paired_alignment": (
            [
                "trajectory",
                "seed",
                "rollout_group",
                "candidates",
                "C_near_zero",
                "D_near_zero",
                "D_positive_cosine",
            ],
            alignment,
        ),
    }
    if followups:
        table_spec["verified_followups"] = (
            ["split", "configuration", "count", "verified", "same_task_baseline"],
            followups,
        )
    checksum = file_hash(args.statistics)
    family = f"{ENTITY}/{PROJECT}/{RUN}"
    api = wandb.Api(timeout=20)
    if not api.viewer:
        raise RuntimeError("W&B credentials unavailable")
    try:
        previous = api.run(family)
    except wandb.errors.CommError:
        previous = None
    previous_history = list(previous.scan_history(keys=["statistics_sha256"])) if previous else []
    if not previous_history or previous_history[-1]["statistics_sha256"] != checksum:
        with wandb.init(
            entity=ENTITY,
            project=PROJECT,
            id=RUN,
            name="Task 23 verified study aggregates",
            group="task23",
            resume="allow",
            dir=str(args.private),
            config={"methods": list(METHODS), "historical_seeds": [2026, 2027]},
            settings=wandb.Settings(init_timeout=45),
        ) as run:
            run.log(
                {
                    name: wandb.Table(columns=columns, data=rows)
                    for name, (columns, rows) in table_spec.items()
                }
                | {"statistics_sha256": checksum}
            )
    for attempt in range(6):
        api = wandb.Api(timeout=20)
        published = api.run(family)
        published_history = list(published.scan_history(keys=["statistics_sha256"]))
        if published_history and published_history[-1]["statistics_sha256"] == checksum:
            break
        if attempt < 5:
            time.sleep(5)
    else:
        raise RuntimeError("W&B server did not report the published statistics digest")
    verified = {}
    with tempfile.TemporaryDirectory(prefix="task23-wandb-") as temp:
        for name, (columns, rows) in table_spec.items():
            artifact = api.artifact(f"{ENTITY}/{PROJECT}/run-{RUN}-{name}:latest")
            downloaded = artifact.get_entry(f"{name}.table.json").download(root=temp)
            table = read_json(Path(downloaded))
            if table["columns"] != columns or table["data"] != rows:
                raise RuntimeError(f"W&B aggregate table contents differ: {name}")
            verified[name] = {"rows": len(rows), "artifact_version": artifact.version}
    atomic_json(
        args.private / "verified.json",
        {
            "url": published.url,
            "statistics_sha256": checksum,
            "tables": verified,
        },
    )
    print(published.url)


if __name__ == "__main__":
    main()
