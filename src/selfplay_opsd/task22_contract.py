"""Artifact identities shared by the PDB alignment runner and Yale controller."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PREFIX = "repair-selfplay-2026-task22-"
STUDY = PREFIX + "pdb-alignment-v1"
EXTENSIONS = {PREFIX + f"pdb-extension-{n}-v1": n for n in (768, 1024)}
B_FALLBACK = {256: 10, 384: 15}
BASE_STUDY = "repair-selfplay-2026-task21-pdb-prompt-v2"
INITIAL = "9ec1fcbbf0e43250c603ee9fd41bb82a8d0f1c6db88160eaeeb987b5708cb9d1"
MATCHED_DATA = "58af9abb5c6e727312dfbc73b78364a6c8e5380a2760bbfaa7b84c5a6cfc8930"
BASE_SCIENCE = "6a1782ad4cb3d60fe3679d00852c32964da033ff14b96ba07dd2dc07e1b3cc10"
B_CHECKPOINTS = (0, 6, 10, 15)
PRIMARY = "evaluate-primary"
QUALIFY = "align-qualify"
FINAL_STAGES = {"evaluate", PRIMARY}
EVAL_SPLITS = ("primary_final", "cross_source", "cross_source_bcb_mutants", "harder_transfer")
CODE_FILES = (
    "scripts/task21_experiment.py",
    "src/selfplay_opsd/config.py",
    "src/selfplay_opsd/task22.py",
    "src/selfplay_opsd/task22_contract.py",
    "src/selfplay_opsd/task21.py",
    "src/selfplay_opsd/task20.py",
    "src/selfplay_opsd/task19.py",
    "src/selfplay_opsd/task19_train.py",
    "src/selfplay_opsd/task19_rollout.py",
    "src/selfplay_opsd/task19_policy.py",
    "src/selfplay_opsd/task19_protocol.py",
    "src/selfplay_opsd/objectives.py",
    "src/selfplay_opsd/pdb_data.py",
    "src/selfplay_opsd/pdb_proposer.py",
    "src/selfplay_opsd/pdb_verifier.py",
    "src/selfplay_opsd/environment.py",
    "src/selfplay_opsd/sandbox.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity_hash(data: object) -> str:
    return hashlib.sha256(
        json.dumps(
            data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def code_identity(root: Path) -> str:
    return identity_hash({name: sha256(root / name) for name in CODE_FILES})


def common_control(control: dict) -> dict:
    """Qualification and launch controls differ only in artifact bookkeeping."""
    return {
        key: value
        for key, value in control.items()
        if key not in {"study", "alignment_gate_hash", "reference_sha256", "parent_marker_sha256"}
    }


def reference_path(root: Path, checksum: str) -> Path:
    if len(checksum) != 64 or any(ch not in "0123456789abcdef" for ch in checksum):
        raise ValueError("Invalid Task 22 reference SHA-256")
    return root / f"reference-{checksum}.json"
