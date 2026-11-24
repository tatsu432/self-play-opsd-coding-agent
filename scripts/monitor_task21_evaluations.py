"""Publish aggregate task completion counts for the frozen held-out evaluations."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from pathlib import Path

import wandb

from selfplay_opsd.core import digest

REMOTE = r"""
import json
from pathlib import Path
base = Path('/nfs/roberts/project/cpsc4880/cpsc4880_ts2236/selfplay-opsd-cpsc4880_ts2236/runs/repair-selfplay-2026-task21-pdb-prompt-v2')
splits = ('primary_final', 'cross_source', 'cross_source_bcb_mutants', 'harder_transfer')
out = {}
for seed in (2026, 2027):
    for arm in ('control', 'A', 'B'):
        for label, episode in (('base', '0000'), ('trained', '0512')):
            root = base / f'{arm}-seed{seed}' / 'evaluation' / episode
            out[f'seed_{seed}/{arm}/{label}_tasks_completed'] = sum(
                len(list((root / split / 'tasks').glob('*.json'))) for split in splits
            )
print(json.dumps(out))
"""


def main() -> None:
    os.umask(0o077)
    private = Path("runs/task21")
    run_id = digest(["task21-heldout-progress-v1"])[:24]
    run = wandb.init(
        entity="tatsu432s",
        project="self-play-opsd-coding-agent",
        id=run_id,
        resume="allow",
        group="repair-selfplay-2026-task21-pdb-prompt-v2",
        name="task21-heldout-evaluation-progress",
        config={"tasks_per_cell": 82, "source": "task record counts; no test outcomes"},
        dir=str(private),
    )
    control = Path.home() / ".ssh/selfplay-control/yale-hpc-cpsc4880"
    try:
        while True:
            raw = subprocess.check_output(
                [
                    "ssh",
                    "-o",
                    f"ControlPath={control}",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=10",
                    "yale-hpc-cpsc4880",
                    "python3",
                    "-c",
                    shlex.quote(REMOTE),
                ],
                text=True,
            )
            counts = json.loads(raw)
            if any(not isinstance(value, int) or not 0 <= value <= 82 for value in counts.values()):
                raise ValueError("Unexpected held-out task record count")
            run.log(counts | {"tasks_per_cell": 82})
            print(json.dumps(counts), flush=True)
            if (private / "final-evals-complete.json").exists():
                break
            time.sleep(180)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
