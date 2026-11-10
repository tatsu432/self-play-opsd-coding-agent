"""Analyze complete Task 17 interface cells from fetched private artifacts."""

import argparse
import json
from pathlib import Path

from selfplay_opsd.analysis import write_csv
from selfplay_opsd.storage import atomic_json, file_hash
from selfplay_opsd.task17_analysis import compare_interfaces, episode_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="JSON with cells [{arm,run_dir,allocated_gpu_hours}]",
    )
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve earlier results; output exists")
    manifest, pool = json.loads(args.manifest.read_text()), json.loads(args.pool.read_text())
    info = {r["task"]["task_id"]: r for r in pool["rows"]}
    families = {r["task"]["parent_id"]: r["family_id"] for r in pool["rows"]}
    cells, rows, hashes = (
        [],
        [],
        {str(args.manifest): file_hash(args.manifest), str(args.pool): file_hash(args.pool)},
    )
    for item in manifest["cells"]:
        root = Path(item["run_dir"])
        for name in ("manifest.json", "complete.json", "episodes.json"):
            hashes[str(root / name)] = file_hash(root / name)
        cell = json.loads((root / "manifest.json").read_text())
        complete = json.loads((root / "complete.json").read_text())
        episodes = json.loads((root / "episodes.json").read_text())
        if (
            cell["arm"] != item["arm"]
            or complete["episodes"] != len(episodes)
            or complete["pool_sha256"] != file_hash(args.pool)
            or cell["pool_sha256"] != file_hash(args.pool)
        ):
            raise ValueError("Cell completion/manifest identity differs")
        cell["records"] = [
            episode_metrics(e)
            | {
                "parent_id": info[e["task_id"]]["task"]["parent_id"],
                "presentation": info[e["task_id"]]["presentation"],
                "family_id": info[e["task_id"]]["family_id"],
                **cell.get("task_features", {}).get(e["task_id"], {}),
            }
            for e in episodes
        ]
        rows.extend(r | {"arm": item["arm"]} for r in cell["records"])
        cells.append(cell)
    result = compare_interfaces(cells, families)
    result["allocated_gpu_hours"] = sum(item["allocated_gpu_hours"] for item in manifest["cells"])
    atomic_json(args.output / "summary.json", result)
    atomic_json(args.output / "input-manifest.json", hashes)
    write_csv(args.output / "episodes.csv", rows)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    for ax, metric, title in zip(
        axes,
        ("applied_edit", "success"),
        ("Tasks receiving an edit", "Verified repairs"),
        strict=True,
    ):
        values = [
            sum(r[metric] for r in c["records"] if r["presentation"] == "original") for c in cells
        ]
        ax.bar([c["arm"] for c in cells], values)
        ax.set(title=title, ylabel="Original tasks (denominator 40)")
    for suffix in ("png", "pdf"):
        fig.savefig(args.output / f"interface-outcomes.{suffix}")
    plt.close(fig)


if __name__ == "__main__":
    main()
