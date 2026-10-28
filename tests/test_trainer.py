"""Scripted generation is an explicit test mock; real model scoring is retained."""

import json
import types

import pytest

from selfplay_opsd.config import load_config
from selfplay_opsd.core import ExperimentError, canonical
from selfplay_opsd.diagnostics import _optimizer_digest
from selfplay_opsd.mutations import make_patch
from selfplay_opsd.policy import Generation, HFPolicy
from selfplay_opsd.storage import CheckpointManager, active_events
from selfplay_opsd.trainer import SelfPlayTrainer


@pytest.mark.sandbox
def test_batched_proposer_preserves_canonical_records(tmp_path, repository, sandbox):
    cfg = configuration(tmp_path, "B", repository)
    policy = scripted_policy(cfg)
    trainer = SelfPlayTrainer(cfg, policy, repository, sandbox)
    original, _ = trainer.proposals(1)
    cfg.hardware.proposer_batch_size = 16
    batched, _ = trainer.proposals(1)
    assert [row.model_dump(mode="json") for row in original] == [
        row.model_dump(mode="json") for row in batched
    ]
    trainer.sink.finish()


@pytest.mark.sandbox
def test_c_zero_reward_groups_skip_candidate_backward(tmp_path, repository, sandbox):
    cfg = configuration(tmp_path, "C", repository)
    policy = scripted_policy(cfg)
    original_generate = policy.generate
    original_gradients = policy.gradients
    candidate_calls = []

    def finish_only(self, prompt, max_tokens, seed, greedy=False):
        if self.role == "proposer":
            return original_generate(prompt, max_tokens, seed, greedy=greedy)
        output = canonical({"action": "finish"})
        ids = self.tokenizer.encode(output) + [self.tokenizer.eos_token_id]
        return Generation(ids, self.score(prompt, ids).tolist(), output)

    def count_gradients(examples, step=False):
        if policy.role == "student" and not step and examples and examples[0].kind == "rl":
            candidate_calls.append(1)
        return original_gradients(examples, step=step)

    policy.generate = types.MethodType(finish_only, policy)
    policy.gradients = count_gradients
    trainer = SelfPlayTrainer(cfg, policy, repository, sandbox)
    result = trainer.run_iteration(1)
    assert result.valid > 0 and not candidate_calls
    assert result.metrics["alignment"]["scored_count"] == result.valid
    assert result.metrics["alignment"]["exact_zero_fraction"] == 1
    assert all(score["reward"] == 0.5 for score in result.metrics["alignment"]["scores"])
    trainer.sink.finish()


def scripted_policy(config):
    policy = HFPolicy(config)

    def generate(self, prompt, max_tokens, seed, greedy=False):
        text = self.tokenizer.decode(prompt)
        if self.role == "proposer":
            start = text.index("[user]\n") + len("[user]\n")
            body = json.loads(text[start : text.rindex("\n[assistant]\n")].strip())
            source = body["correct_source"]
            # Within each parent: some invalid, first valid owns duplicates.
            output = (
                canonical({"patch": make_patch(source, source.replace("a + b", "a - b"))})
                if seed % 3
                else "invalid"
            )
        elif '"apply_patch"' in text[text.rfind("[assistant]") :] or "source_hash" in text:
            output = canonical({"action": "finish"})
        else:
            # Include positive and negative advantage episodes in real tensor tests.
            if seed % 2:
                entry = "add_" + text.split("Implement add_")[1].split("(")[0]
                index = int(entry.split("_")[1])
                before = f"def {entry}(a, b):\n    return a - b + {index}\n"
                after = before.replace("a - b", "a + b")
                output = canonical({"action": "apply_patch", "patch": make_patch(before, after)})
            else:
                output = canonical({"action": "finish"})
        ids = self.tokenizer.encode(output) + [self.tokenizer.eos_token_id]
        return Generation(ids, self.score(prompt, ids).tolist(), output)

    policy.generate = types.MethodType(generate, policy)
    return policy


def configuration(tmp_path, method, repository):
    cfg = load_config(
        study_id="test",
        hardware_profile="cpu_debug",
        output_root=tmp_path,
        data_root=repository.root,
        method=method,
        iterations=2,
    )
    cfg.logging.mode = "disabled"
    cfg.hardware.cpu_workers = 3
    return cfg


@pytest.mark.sandbox
@pytest.mark.parametrize("method", ["A", "B", "C"])
def test_full_method_smoke_is_separate_and_rejects_checkpoint_corruption(
    method, tmp_path, repository, sandbox
):
    from selfplay_opsd.method_smoke import run_smoke, smoke_config, verify_smoke

    cfg = configuration(tmp_path, method, repository)
    policy = scripted_policy(smoke_config(cfg, method))
    try:
        run_smoke(cfg, policy, repository, sandbox)
        assert verify_smoke(cfg, method)["ready"]
        assert not cfg.run_dir.exists()
        checkpoint = smoke_config(cfg, method).run_dir / "checkpoints/000003/state.pt"
        checkpoint.write_bytes(b"corrupt")
        with pytest.raises(ExperimentError, match="checksum"):
            verify_smoke(cfg, method)
    finally:
        policy.close()


@pytest.mark.sandbox
@pytest.mark.parametrize("method", ["A", "B", "C", "D"])
def test_complete_transaction_and_checkpoint(method, tmp_path, repository, sandbox, monkeypatch):
    cfg = configuration(tmp_path, method, repository)
    monkeypatch.setattr(
        "selfplay_opsd.readiness.TaskRepository",
        lambda _: pytest.fail("Trainer must reuse its validated repository"),
    )
    policy = scripted_policy(cfg)
    trainer = SelfPlayTrainer(cfg, policy, repository, sandbox)
    trainer.checkpoint()
    result = trainer.run_iteration(1)
    path = trainer.checkpoint()
    assert result.attempted == 16 and 0 < result.valid <= 4
    assert result.episodes == result.valid * 4
    assert result.student_steps == 1 and result.proposer_steps == 1
    assert all(len(keys) == 1 for keys in trainer.history.values())
    if method in {"C", "D"}:
        assert result.metrics["alignment"]["scored_count"] == result.valid
    restored = scripted_policy(cfg)
    resumed = SelfPlayTrainer(cfg, restored, repository, sandbox, resume=path)
    assert resumed.iteration == 1 and resumed.history == trainer.history
    assert _optimizer_digest(restored.state_dict()) == _optimizer_digest(policy.state_dict())
    if method == "A":
        state, metadata = CheckpointManager.load(path)
        metadata["manifests_hash"] = "different-manifest"
        incompatible = CheckpointManager(tmp_path / "incompatible").save(1, state, metadata)
        with pytest.raises(ExperimentError, match="compatibility"):
            SelfPlayTrainer(cfg, restored, repository, sandbox, resume=incompatible)
    trainer.sink.finish()
    resumed.sink.finish()


@pytest.mark.sandbox
def test_interruption_between_updates_replays_from_joint_checkpoint(
    tmp_path, repository, sandbox, monkeypatch
):
    cfg = configuration(tmp_path, "D", repository)
    policy = scripted_policy(cfg)
    trainer = SelfPlayTrainer(cfg, policy, repository, sandbox)
    checkpoint = trainer.checkpoint()
    original = policy.gradients

    def fail(examples, step=False):
        if policy.role == "proposer" and step:
            raise RuntimeError("injected after student update")
        return original(examples, step=step)

    monkeypatch.setattr(policy, "gradients", fail)
    with pytest.raises(RuntimeError, match="injected"):
        trainer.run_iteration(1)
    assert not any(e["kind"] == "iteration_commit" for e in active_events(cfg.run_dir))
    resumed_policy = scripted_policy(cfg)
    resumed = SelfPlayTrainer(cfg, resumed_policy, repository, sandbox, resume=checkpoint)
    resumed.run_iteration(1)
    # Independent uninterrupted run must produce identical joint states.
    baseline_cfg = configuration(tmp_path / "baseline", "D", repository)
    baseline_policy = scripted_policy(baseline_cfg)
    baseline = SelfPlayTrainer(baseline_cfg, baseline_policy, repository, sandbox)
    baseline.run_iteration(1)
    assert _optimizer_digest(resumed_policy.state_dict()) == _optimizer_digest(
        baseline_policy.state_dict()
    )
    commits = [e for e in active_events(cfg.run_dir) if e["kind"] == "iteration_commit"]
    assert len(commits) == 1
    for instance in (trainer, resumed, baseline):
        instance.sink.finish()
