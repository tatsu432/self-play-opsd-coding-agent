"""Plot aggregate action failures from the fetched Task 17 paired analysis."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pool = json.loads(args.pool.read_text())
    originals = {r["task"]["task_id"] for r in pool["rows"] if r["presentation"] == "original"}
    with (args.analysis / "tool-errors.csv").open(newline="") as source:
        rows = [r for r in csv.DictReader(source) if r["task_id"] in originals]
    counts = {arm: Counter(r["category"] for r in rows if r["arm"] == arm) for arm in "ABCD"}
    categories = sorted({category for group in counts.values() for category in group})
    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    positions = range(len(categories))
    for offset, (arm, label) in enumerate(
        (("A", "Base diff"), ("B", "Base text"), ("C", "Trained diff"), ("D", "Trained text"))
    ):
        ax.bar(
            [x + (offset - 1.5) * 0.19 for x in positions],
            [counts[arm][category] for category in categories],
            width=0.19,
            label=label,
        )
    ax.set(
        xticks=list(positions),
        xticklabels=[x.replace("_", " ") for x in categories],
        ylabel="Failed actions across 40 original tasks",
        title="Observed action failure categories",
    )
    ax.tick_params(axis="x", rotation=25)
    ax.legend(ncol=2, fontsize=8)
    args.output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(args.output / f"interface-failures.{extension}")
    plt.close(fig)


if __name__ == "__main__":
    main()
