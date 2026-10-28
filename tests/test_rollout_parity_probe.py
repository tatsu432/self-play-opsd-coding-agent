import contextlib
import json

import pytest
import torch
from typer.testing import CliRunner

from selfplay_opsd.config import load_config
from selfplay_opsd.core import EpisodeRecord, Status, TokenTurn
from selfplay_opsd.policy import Generation, HFPolicy
from selfplay_opsd.rollout_parity_probe import check_zero_candidate, run


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


def test_equal_reward_shortcut_matches_actual_backward(tmp_path):
    cfg = load_config(study_id="zero", hardware_profile="cpu_debug", output_root=tmp_path)
    policy = HFPolicy(cfg)
    prompt, targets = [1, 7, 8], [9, 10]
    old = policy.score(prompt, targets).tolist()
    turn = TokenTurn(
        prompt_ids=prompt,
        target_ids=targets,
        old_logps=old,
        teacher_logps=old,
        history=[],
        raw_text="",
        action="finish",
        parse_status="valid",
    )
    episode = EpisodeRecord(
        episode_id="example",
        task_id="task",
        seed=1,
        policy_version=policy.version("student"),
        turns=[turn],
        reward=0,
        public_pass=False,
        hidden_pass=False,
        termination="finish",
        status=Status.STUDENT_FAILURE,
        final_source="",
        final_source_hash="hash",
        test_calls=0,
        seconds=0,
    )
    result = check_zero_candidate(
        policy,
        cfg,
        [episode.model_dump(mode="json")] * 4,
        [{"edit": {"prompt": prompt, "targets": targets}}],
    )
    assert result["available"] and result["exact_match"]
    assert result["old_alignment"]["reward"] == 0.5
    assert result["old_alignment"]["exact_zero"]
    policy.close()
