"""Freeze the selected Task 15 protocol from completed allocated evidence."""

import argparse
from pathlib import Path

from selfplay_opsd.config import load_config
from selfplay_opsd.data import TaskRepository
from selfplay_opsd.storage import provenance
from selfplay_opsd.task15_freeze import freeze

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--settings", type=Path, required=True)
parser.add_argument("--decision", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
cfg = load_config(args.settings)
freeze(cfg, TaskRepository(cfg.data_root), args.decision, args.output, provenance()["git_commit"])
