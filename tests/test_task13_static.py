"""Fixed-task batches must not produce a proposer policy update."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from selfplay_opsd.config import load_config
from selfplay_opsd.trainer import proposer_examples

spec = importlib.util.spec_from_file_location("task13_static", Path("scripts/task13_static.py"))
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_fixed_batches_cycle_and_have_no_proposer_tokens():
    trainer = object.__new__(module.StaticTrainer)
    trainer.policy = SimpleNamespace(version=lambda role: "initial")
    tasks = [SimpleNamespace(task_id=f"task{i}", parent_id=f"parent{i}") for i in range(4)]
    trainer.pool = [(task, object(), f"key{i}") for i, task in enumerate(tasks)]
    first, first_tasks = trainer.proposals(1)
    second, second_tasks = trainer.proposals(2)
    assert list(first_tasks) == ["task0", "task1"]
    assert list(second_tasks) == ["task2", "task3"]
    assert [p.valid for p in first] == [True, False, True, False]
    cfg = load_config(Path("configs/task12-corrected.yaml"), method="GRPO")
    cfg.parents_per_iteration = 2
    cfg.proposals_per_parent = 2
    examples = proposer_examples(first, [0, -1, 1, -1], cfg)
    assert len(examples) == 4
    assert not any(example.targets for example in examples)


def test_static_continuation_checks_checkpoint_identity(monkeypatch, tmp_path):
    from selfplay_opsd.protocol import PROMPT_HASH

    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    spec = importlib.util.spec_from_file_location(
        "static_continue", Path("scripts/task13_static_continue.py")
    )
    assert spec and spec.loader
    continuation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(continuation)
    runtime = {"scientific_code_hash": "code", "allocation_code_hash": "allocation"}
    state = {"iteration": 3}
    metadata = {
        "scientific_hash": "science",
        "manifests_hash": "dataset",
        "prompt_hash": PROMPT_HASH,
        "scientific_code_hash": "code",
        "runtime_identity": runtime,
        "iteration": 3,
    }
    monkeypatch.setattr(continuation.CheckpointManager, "load", lambda _: (state, metadata))
    monkeypatch.setattr(continuation, "execution_identity", lambda *_args, **_kwargs: runtime)
    cfg, repo = SimpleNamespace(scientific_hash="science"), SimpleNamespace(hash="dataset")
    assert continuation.verified_checkpoint(tmp_path, cfg, repo) == state
    metadata["scientific_hash"] = "different"
    import pytest

    with pytest.raises(ValueError, match="identity"):
        continuation.verified_checkpoint(tmp_path, cfg, repo)


def test_static_iteration_updates_student_without_proposer(tmp_path, monkeypatch):
    from selfplay_opsd.core import EpisodeRecord, OracleSpec, Status, TaskSpec, TokenTurn
    from selfplay_opsd.policy import HFPolicy

    cfg = load_config(
        study_id="static-test",
        hardware_profile="cpu_debug",
        output_root=tmp_path,
        data_root=tmp_path,
        method="GRPO",
        iterations=1,
    )
    cfg.logging.mode = "disabled"
    cfg.parents_per_iteration = cfg.proposals_per_parent = 2
    cfg.hardware.cpu_workers = 1
    repo = SimpleNamespace(
        root=tmp_path, hash="fixture", manifest={}, parents=lambda _: [object(), object()]
    )
    sandbox = SimpleNamespace(infrastructure_retries=0)
    policy = HFPolicy(cfg)
    pool = []
    for i in range(2):
        task = TaskSpec(
            task_id=f"task{i}",
            parent_id=f"parent{i}",
            split="train",
            specification="f",
            initial_source="def f(x):\n    return x\n",
            entry_point="f",
            public_tests="",
            source_hash="hash",
            mutation_seed=i,
        )
        oracle = OracleSpec(
            task_id=task.task_id,
            correct_source=task.initial_source,
            inverse_patch="",
            public_cases=[],
            hidden_cases=[],
            provenance={},
        )
        pool.append((task, oracle, f"key{i}"))

    def fake_wave(active_policy, _sandbox, jobs, _limits, *, concurrency, completed):
        episodes = []
        for index, (task, _oracle, seed) in enumerate(jobs):
            reward = float(index % 2)
            prompt = [active_policy.tokenizer.bos_token_id]
            targets = active_policy.tokenizer.encode("yes" if reward else "no")
            old = active_policy.score(prompt, targets).tolist()
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
                episode_id=str(index),
                task_id=task.task_id,
                seed=seed,
                policy_version=active_policy.version("student"),
                turns=[turn],
                reward=reward,
                public_pass=bool(reward),
                hidden_pass=bool(reward),
                termination="finish",
                status=Status.OK,
                final_source=task.initial_source,
                final_source_hash="hash",
                test_calls=0,
                seconds=0,
            )
            completed(index, episode)
            episodes.append(episode)
        return episodes

    metrics = {
        "old_student_score_seconds": 0,
        "teacher_score_seconds": 0,
        "pre_update_logprob_abs_max": 0,
        "pre_update_logprob_abs_sum": 0,
        "pre_update_logprob_abs_above_005": 0,
        "pre_update_logprob_abs_above_02": 0,
        "mapped_tokens": 1,
    }
    monkeypatch.setattr("selfplay_opsd.rollout.episode_wave", fake_wave)
    monkeypatch.setattr(
        "selfplay_opsd.trainer.score_teacher", lambda _p, e, _o, _l: (e, [], metrics)
    )
    try:
        trainer = module.StaticTrainer(cfg, policy, repo, sandbox, pool=pool)
        result = trainer.run_iteration(1)
        assert result.valid == 2 and result.episodes == 8
        assert result.student_steps == 1
        assert result.proposer_steps == 0
        checkpoint = trainer.checkpoint()
        assert checkpoint is not None
        trainer.sink.finish()
        resumed = module.StaticTrainer(cfg, policy, repo, sandbox, pool=pool, resume=checkpoint)
        assert resumed.iteration == 1
        assert resumed.student_steps == 1 and resumed.proposer_steps == 0
        resumed.sink.finish()
    finally:
        policy.close()
