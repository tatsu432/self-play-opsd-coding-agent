"""Verify and load the previously reconciled corrected B200 diagnostic snapshot."""

from pathlib import Path

from selfplay_opsd.storage import CheckpointManager, file_hash

CHECKPOINT = Path("/runs/repair-selfplay-2026-syntax-fix-v1/B-seed2026/checkpoints/000200")
EXPECTED = {
    "scientific_hash": "069a6e0ae26fd5a98bd2327d8532074f2d97bd659fd286600a59bfaa338d2",
    "manifests_hash": "fd47b9117ffa627531a24b938999d9e063f1e6eee47e4b0b91b1bbf46c4ed714",
    "prompt_hash": "6e35ae63871454564a039a3c0a6b748beeb03ae7aae67f268c5ec556193b9b82",
    "scientific_code_hash": "ec5be92f3e5591413026f9547f01d92baca3c08835ad91e1b1bacd9626ab50a9",
    "iteration": 200,
    "artifact_kind": "fully_resumable",
}


def load_b200(cfg, repo):
    state, metadata = CheckpointManager.load(CHECKPOINT)
    if any(metadata.get(key) != value for key, value in EXPECTED.items()):
        raise ValueError("Historical B200 checkpoint identity differs")
    if repo.hash != EXPECTED["manifests_hash"] or metadata["resolved_config"][
        "model"
    ] != cfg.model.model_dump(mode="json"):
        raise ValueError("Historical B200 model or dataset differs")
    return state, metadata, file_hash(CHECKPOINT / "complete.json")
