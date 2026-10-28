import contextlib
import json

import pytest
import torch
from typer.testing import CliRunner

from selfplay_opsd.config import load_config
from selfplay_opsd.policy import Generation
from selfplay_opsd.rollout_parity_probe import run


def test_rollout_parity_cli_accepts_scheduler_options(tmp_path, monkeypatch):
    from selfplay_opsd.cli import app

    monkeypatch.setattr("selfplay_opsd.rollout_parity_probe.run", lambda cfg, output: {"ok": True})
    result = CliRunner().invoke(
        app,
        [
            "rollout-parity",
            "--settings",
            "configs/task09-diagnostics.yaml",
            "--hardware",
            "single_gpu_80gb",
            "--output",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.exception
    assert json.loads(result.output)["ok"]


def test_replay_counts_real_token_errors_and_keeps_request_order(tmp_path, monkeypatch):
    source = (
        tmp_path / "repair-selfplay-2026-efficiency-v1-pilot-h200/D-seed2026/trajectories/segment"
    )
    source.mkdir(parents=True)
    episode = {
        "policy_version": "snapshot",
        "turns": [{"prompt_ids": [1], "target_ids": [2, 3], "old_logps": [0.0, -0.4]}],
    }
    proposals = [
        {"policy_version": "snapshot", "seed": i, "turn": {"prompt_ids": [1]}} for i in range(16)
    ]
    (source / "000001-episodes.json").write_text(json.dumps([episode]))
    (source / "000001-proposals.json").write_text(json.dumps(proposals))

    class Policy:
        is_coordinator = True

        def version(self, role):
            return "snapshot"

        def active(self, role):
            return contextlib.nullcontext()

        def score(self, prompt, targets):
            return torch.tensor([0.0, 0.0])

        def score_many(self, requests):
            return [self.score(prompt, targets) for prompt, targets in requests]

        def generate(self, prompt, max_tokens, seed):
            return Generation([seed], [-0.1], str(seed))

        def prepare_rollout(self):
            pass

        def generate_many(self, requests):
            return [self.generate(r.prompt, r.max_tokens, r.seed) for r in requests]

        def close(self):
            pass

    monkeypatch.setattr("selfplay_opsd.cli.make_policy", lambda cfg: Policy())
    cfg = load_config(study_id="parity", hardware_profile="cpu_debug", output_root=tmp_path)
    result = run(cfg, tmp_path / "output")
    assert result["tokens"] == 2
    assert result["absolute_error_max"] == pytest.approx(0.4)
    assert result["outside_grpo_clip"] == 1
    assert result["identical_output_ids"] == 16
    assert (tmp_path / "output/rollout-parity.json").exists()
