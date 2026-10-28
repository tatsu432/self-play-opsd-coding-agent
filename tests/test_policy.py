import copy
import dataclasses

import pytest
import torch

from selfplay_opsd.config import load_config
from selfplay_opsd.core import EpisodeRecord, OracleSpec, Status, TokenTurn, canonical
from selfplay_opsd.diagnostics import _optimizer_digest, update_probe
from selfplay_opsd.environment import append_context, run_episode, score_teacher
from selfplay_opsd.policy import Generation, HFPolicy, LossExample
from selfplay_opsd.trainer import reference_examples, student_examples


def model():
    return HFPolicy(load_config(study_id="unit", hardware_profile="cpu_debug"))


def test_score_teacher_reports_logprob_error_counts():
    policy = model()
    history = [{"role": "system", "content": "repair"}, {"role": "user", "content": "task"}]
    prompt = policy.encode_history(history)
    targets = [7, 8, 9]
    scores = policy.score(prompt, targets).tolist()
    turn = TokenTurn(
        prompt_ids=prompt,
        target_ids=targets,
        old_logps=[scores[0], scores[1] + 0.1, scores[2] + 0.3],
        history=history,
        raw_text="",
        action="finish",
        parse_status="valid",
    )
    episode = EpisodeRecord(
        episode_id="episode",
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
    oracle = OracleSpec(
        task_id="task",
        correct_source="def f(): pass",
        inverse_patch="",
        public_cases=[],
        hidden_cases=[],
        provenance={},
    )
    _, _, stats = score_teacher(policy, episode, oracle, policy.config.episode)
    assert stats["mapped_tokens"] == 3
    assert stats["pre_update_logprob_abs_sum"] == pytest.approx(0.4, abs=1e-5)
    assert stats["pre_update_logprob_abs_above_005"] == 2
    assert stats["pre_update_logprob_abs_above_02"] == 1


def test_real_generation_reproducibility_and_causal_shift():
    policy = model()
    prompt = policy.encode_history([{"role": "user", "content": "hello"}])
    generation = policy.generate(prompt, 7, 2026)
    assert generation == policy.generate(prompt, 7, 2026)
    ids = torch.tensor([prompt + generation.ids])
    with torch.no_grad():
        logits = policy.model(ids, use_cache=False).logits
        independent = logits[0, len(prompt) - 1 : -1].float().log_softmax(-1)
        values = independent.gather(1, torch.tensor(generation.ids)[:, None]).squeeze(1)
    torch.testing.assert_close(policy.score(prompt, generation.ids), values)
    assert generation.logps == pytest.approx(values.tolist())


def test_adapter_isolation_scoring_and_zero_update():
    policy = model()
    assert policy.version("student") == policy.version("proposer")
    initial = policy.version("proposer")
    prompt, targets = [1, 60, 61], [15, 16, 2]
    old = policy.score(prompt, targets).tolist()
    example = LossExample(prompt, targets, old, 1.0, [0.3] * 3, 1 / 3)
    state = copy.deepcopy(policy.state_dict())
    grads, _ = policy.gradients([example])
    assert any(g.abs().sum() > 0 for g in grads)
    assert _optimizer_digest(state) == _optimizer_digest(policy.state_dict())
    policy.gradients([example], step=True)
    assert policy.version("proposer") == initial
    assert policy.version("student") != initial
    after = policy.version("student")
    no_op = dataclasses.replace(example, advantage=0.0, coefficient=[0.0] * 3)
    _, stats = policy.gradients([no_op], step=True)
    assert stats["stepped"] == 0 and policy.version("student") == after
    # An AdamW optimizer with momentum must also remain unchanged on a zero batch.
    state = _optimizer_digest(policy.state_dict())
    policy.gradients([no_op], step=True)
    assert _optimizer_digest(policy.state_dict()) == state


def test_streaming_matches_independent_global_autograd():
    policy = model()
    parameters = policy.parameters("student")
    x = LossExample([1, 5, 6], [7, 8], [-5.0] * 2, -0.5, [0.1, 0.2], 1 / 4)
    y = LossExample([1, 9], [8], [-5.0], 0.7, [-0.2], 1 / 2)
    streamed, _ = policy.gradients([x, y])
    from selfplay_opsd.objectives import grpo, signed_gap

    total = 0
    for e in (x, y):
        logp = policy.score(e.prompt, e.targets, gradients=True)[None]
        loss = grpo(
            logp, torch.tensor([e.old]), torch.tensor([e.advantage]), torch.ones_like(logp)
        ) + signed_gap(logp, torch.tensor([e.coefficient]), torch.ones_like(logp))
        total = total + loss * len(e.targets) * e.token_weight
    grads = torch.autograd.grad(total, parameters, allow_unused=True)
    for actual, expected, p in zip(streamed, grads, parameters, strict=True):
        expected = torch.zeros_like(p) if expected is None else expected
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_retains_noncanonical_sampled_ids_across_tool_boundary():
    policy = model()
    history = [{"role": "system", "content": "repair"}, {"role": "user", "content": "test"}]
    initial = policy.encode_history(history)
    text = canonical({"action": "run_tests"})
    # A genuinely sampled special token disappears from decoded text. The next
    # history must retain it instead of reconstructing IDs from that text.
    targets = (
        [policy.tokenizer.bos_token_id]
        + policy.tokenizer.encode(text)
        + [policy.tokenizer.eos_token_id]
    )
    context = [{"role": "tool", "content": "pass"}, {"role": "user", "content": "budget"}]
    actual = append_context(policy, history, initial, targets, context)
    assert actual[: len(initial) + len(targets)] == initial + targets
    retokenized = policy.encode_history(
        history + [{"role": "assistant", "content": text}] + context
    )
    assert actual != retokenized
    assert actual[len(initial) + len(targets)] != policy.tokenizer.eos_token_id


def test_scoring_preserves_existing_gradient_buffers():
    policy = model()
    parameters = policy.parameters("student")
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    example = LossExample([1, 5], [6, 7], [-5.0, -5.0], 1.0, [0.0, 0.0], 0.5)
    policy.gradients([example])
    assert all(torch.equal(p.grad, torch.ones_like(p)) for p in parameters)


def test_named_kl_variant_is_explicit_and_base_detached():
    from selfplay_opsd.config import RunConfig

    config = load_config(study_id="kl", hardware_profile="cpu_debug")
    values = config.model_dump()
    values["method"] = "GRPO_KL"
    values["objective"]["fixed_base_kl"] = 0.1
    policy = HFPolicy(RunConfig.model_validate(values))
    reference = policy.score_base([1, 5], [6, 7])
    assert not reference.requires_grad
    example = LossExample(
        [1, 5], [6, 7], reference.tolist(), 1.0, [0.0, 0.0], 0.5, "rl", reference.tolist(), 0.1
    )
    policy.gradients([example], step=True)
    _, metrics = policy.gradients([example])
    assert metrics["kl_loss"] >= -1e-7
    values["method"] = "B"
    with pytest.raises(ValueError):
        RunConfig.model_validate(values)


def test_activation_checkpointing_really_runs_and_preserves_gradients(monkeypatch):
    config = load_config(study_id="checkpointing", hardware_profile="cpu_debug")
    config.hardware.activation_checkpointing = True
    checked = HFPolicy(config)
    layer = checked.model.get_base_model().model.layers[0]
    original = layer._gradient_checkpointing_func
    calls = []

    def record(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(layer, "_gradient_checkpointing_func", record)
    example = LossExample([1, 5, 6], [7, 8], [-5.0, -5.0], 1.0, [0.0, 0.0], 0.5)
    actual, _ = checked.gradients([example])
    plain = model()
    expected, _ = plain.gradients([example])
    assert calls and not checked.model.training
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)


def test_unequal_left_padded_microbatches_match_scores_and_gradients():
    policy = model()
    requests = [([1, 5, 6], [7, 8]), ([1, 9], [8]), ([1], []), ([1, 4, 5, 6], [2, 0, 1])]
    single = policy.score_many(requests)
    examples = [
        LossExample(
            prefix,
            target,
            [-5.0] * len(target),
            0.7,
            [0.2] * len(target),
            1 / (4 * max(1, len(target))),
        )
        for prefix, target in requests
    ]
    expected, _ = policy.gradients(examples)
    policy.config.hardware.microbatch_episodes = 3
    batched = policy.score_many(requests)
    actual, _ = policy.gradients(examples)
    for a, b in zip(batched, single, strict=True):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)


@pytest.mark.sandbox
def test_multiturn_teacher_exact_targets_and_masks(repository, sandbox):
    policy = model()
    task, oracle = repository.tasks("dev_diagnostic")[0]
    actions = [
        canonical({"action": "read_file", "path": "src/solution.py"}),
        canonical({"action": "finish"}),
    ]
    calls = []

    def scripted(prompt, max_tokens, seed, greedy=False):
        text = actions[len(calls)]
        calls.append(text)
        ids = policy.tokenizer.encode(text) + [policy.tokenizer.eos_token_id]
        return Generation(ids, policy.score(prompt, ids).tolist(), text)

    policy.generate = scripted
    episode = run_episode(policy, sandbox, task, oracle, policy.config.episode, 2026)
    scored, mappings, stats = score_teacher(policy, episode, oracle, policy.config.episode)
    assert len(scored.turns) == 2 and len(mappings) == 2
    for turn, mapping in zip(scored.turns, mappings, strict=True):
        assert mapping.target_ids == turn.target_ids
        assert mapping.student_positions[0] == len(turn.prompt_ids) - 1
        assert len(turn.teacher_logps) == len(turn.target_ids)
        assert mapping.teacher_positions[0] > mapping.student_positions[0]
    assert stats["pre_update_logprob_abs_max"] < 1e-5
    examples, _ = student_examples([scored] * 4, policy.config)
    assert sum(len(x.targets) * x.token_weight for x in examples) == pytest.approx(1.0)
    reference = reference_examples(repository.anchors())
    report = update_probe(
        policy,
        [scored] * 4,
        policy.config,
        reference,
        reference_examples(repository.anchors("dev_diagnostic")),
    )
    assert report["rl_gradient_norm"] == 0
    assert report["combined_gradient_norm"] > 0
    assert report["c"]["reward"] == 0.5


def test_bfloat16_base_keeps_identical_independent_adapter_master_weights(monkeypatch):
    import selfplay_opsd.policy as module

    factory = module.Qwen3ForCausalLM
    monkeypatch.setattr(
        module, "Qwen3ForCausalLM", lambda config: factory(config).to(torch.bfloat16)
    )
    policy = HFPolicy(
        load_config(study_id="mixed-precision-init", hardware_profile="cpu_debug"), device="cpu"
    )
    initial = policy.version("student")
    assert initial == policy.version("proposer")
    proposer = dict(policy.named_parameters("proposer"))
    for name, student in policy.named_parameters("student"):
        other = proposer[name.replace(".student.", ".proposer.")]
        assert student.dtype == other.dtype == torch.float32
        assert student.data_ptr() != other.data_ptr()
        assert torch.equal(student, other)
    with torch.no_grad():
        policy.parameters("student")[0].add_(1)
    assert policy.version("proposer") == initial
    assert policy.version("student") != initial
