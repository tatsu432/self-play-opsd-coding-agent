"""Scientific independence, bounded supply, actual launch commands and recovery."""

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from selfplay_opsd.config import OnlineSupplyConfig, load_config
from selfplay_opsd.core import ExperimentError, TokenTurn
from selfplay_opsd.task15 import OnlineExposureTrainer
from selfplay_opsd.task15_audit import intervention_examples
from selfplay_opsd.trainer import SelfPlayTrainer, proposer_examples, student_examples


def config():
    return load_config(Path("configs/task15.yaml"), hardware_profile="cpu_debug")


def turn():
    return TokenTurn(
        prompt_ids=[1],
        target_ids=[2, 3],
        old_logps=[-2.0, -2.0],
        frozen_unprivileged_logps=[-1.0, -1.0],
        teacher_logps=[-0.5, -1.5],
        history=[],
        raw_text="",
        action="finish",
        parse_status="valid",
    )


def episodes():
    from selfplay_opsd.core import EpisodeRecord, Status

    return [
        EpisodeRecord(
            episode_id=str(i),
            task_id="task",
            seed=i,
            policy_version="initial",
            turns=[turn()],
            reward=float(i % 2),
            public_pass=False,
            hidden_pass=False,
            termination="finish",
            status=Status.OK,
            final_source="",
            final_source_hash="x",
            test_calls=0,
            seconds=0,
        )
        for i in range(4)
    ]


def test_signed_gap_uses_same_scorer_without_overwriting_behavior():
    cfg = config()
    sample = episodes()
    original = copy.deepcopy(sample)
    examples, _ = student_examples(sample, cfg)
    expected = torch.sigmoid(torch.tensor([0.5, -0.5])) * torch.tensor([0.5, -0.5]) * 0.1
    assert examples[0].coefficient == pytest.approx(expected.tolist())
    assert examples[0].old == [-2.0, -2.0]
    assert sample == original
    assert examples[0].coefficient[1] < 0
    with pytest.raises(ExperimentError, match="Missing same-scorer"):
        student_examples(
            [
                sample[0].model_copy(
                    update={"turns": [turn().model_copy(update={"frozen_unprivileged_logps": []})]}
                )
            ],
            cfg,
        )


def test_factorial_interventions_change_only_the_intended_component():
    sample, cfg = episodes(), config()
    baseline = intervention_examples(sample, cfg, denominator="behavior", gap="behavior")
    denominator = intervention_examples(sample, cfg, denominator="scorer", gap="behavior")
    gap = intervention_examples(sample, cfg, denominator="behavior", gap="scorer")
    assert baseline[0].coefficient == denominator[0].coefficient
    assert baseline[0].old != denominator[0].old
    assert baseline[0].old == gap[0].old
    assert baseline[0].coefficient != gap[0].coefficient
    assert all(
        c == 0 for e in intervention_examples(sample, cfg, no_privilege=True) for c in e.coefficient
    )
    assert any(
        c != 0
        for e in intervention_examples(sample, cfg, gap="behavior", no_privilege=True)
        for c in e.coefficient
    )


def test_independent_episode_normalization_and_signed_gradient():
    cfg = config()
    cfg.method = "A"
    sample = episodes()
    # Every token receives weight 1/(2 tokens * 4 episodes * 2 tasks).
    examples, _ = student_examples(sample, cfg, tasks=2)
    current = torch.tensor([-1.0, -1.0], requires_grad=True)
    loss = sum(-(current * torch.tensor(e.coefficient)).sum() * e.token_weight for e in examples)
    loss.backward()
    assert current.grad.tolist() == pytest.approx([-0.0125, 0.0125])
    uniform = [e.model_copy(update={"reward": 0}) for e in sample]
    examples, _ = student_examples(uniform, cfg)
    assert all(e.advantage == 0 for e in examples)
    assert any(c != 0 for e in examples for c in e.coefficient)


def supply_trainer():
    trainer = object.__new__(OnlineExposureTrainer)
    trainer.config = config()
    trainer.parents = [SimpleNamespace(parent_id=str(i)) for i in range(6)]
    trainer.history = {}
    trainer.total_episodes = 0
    trainer.supply_state = {
        "proposal_groups": 0,
        "proposal_attempts": 0,
        "reservoir": {},
        "fallback_groups": 0,
        "task_groups": 0,
        "fallback_parent_uses": {},
        "evaluated_milestones": [],
    }
    return trainer


def proposal(task_id, index, *, valid=True, parent="p"):
    return SimpleNamespace(
        parent_id=parent,
        valid=valid,
        task_id=task_id,
        dedup_key=task_id,
        bug_family="arithmetic",
        turn=turn().model_copy(update={"raw_text": '{"site":0,"operation":"Sub"}'}),
        slot=index,
    )


def test_refill_retains_complete_groups_and_stops_at_limit(monkeypatch):
    trainer = supply_trainer()
    monkeypatch.setattr(
        SelfPlayTrainer,
        "proposals",
        lambda self, k: ([proposal(None, i, valid=False) for i in range(4)], {}),
    )
    proposals, tasks = trainer.proposals(1)
    assert len(proposals) == 64 and not tasks
    assert trainer.supply_state["proposal_attempts"] == 64
    assert trainer.supply_state["proposal_groups"] == 16


def test_shared_difficulty_keeps_all_sampled_proposer_actions(monkeypatch):
    trainer = supply_trainer()
    trainer.history = {"p": {"t"}}
    monkeypatch.setattr(
        SelfPlayTrainer,
        "proposals",
        lambda self, k: ([proposal("t", i) for i in range(4)], {"t": (object(), object())}),
    )
    proposals, tasks = trainer.proposals(1)
    assert len(tasks) == 1 and len(proposals) == 64
    examples = proposer_examples(proposals, [0.75] * len(proposals), trainer.config)
    assert len(examples) == 64
    assert all(e.advantage == 0 for e in examples)
    assert all(e.old == [-2.0, -2.0] for e in examples)
    assert trainer.supply_state["metrics"]["historical_replay_samples"] == 64
    assert not trainer.reject_duplicate("p", "t", {"p": {"t"}})


def test_milestone_finishes_parent_group_with_at_most_three_tasks_overshoot(monkeypatch):
    trainer = supply_trainer()
    trainer.total_episodes = 252
    monkeypatch.setattr(
        SelfPlayTrainer,
        "proposals",
        lambda self, k: (
            [proposal(str(i), i) for i in range(4)],
            {str(i): (object(), object()) for i in range(4)},
        ),
    )
    proposals, tasks = trainer.proposals(1)
    assert len(proposals) == 4 and len(tasks) == 4
    assert trainer.total_episodes + len(tasks) * 4 - 256 == 12


def test_parent_schedule_is_balanced_and_independent_of_arm():
    first, second = supply_trainer(), supply_trainer()
    second.config.method = "A"
    sequence = [first.proposal_parents(i)[0].parent_id for i in range(1, 13)]
    assert len(set(sequence[:6])) == len(set(sequence[6:])) == 6
    assert sequence == [second.proposal_parents(i)[0].parent_id for i in range(1, 13)]


def test_refill_resume_restores_cursor_and_exact_attempt_budget(monkeypatch):
    first = supply_trainer()
    first.supply_state["proposal_attempts"] = 65532
    first.supply_state["proposal_groups"] = 16383
    restored = supply_trainer()
    restored.supply_state = copy.deepcopy(first.supply_state)
    seen = []

    def generate(self, k):
        seen.append(k)
        return [proposal(None, i, valid=False) for i in range(4)], {}

    monkeypatch.setattr(SelfPlayTrainer, "proposals", generate)
    assert len(restored.proposals(2)[0]) == 4
    assert seen == [16384]
    with pytest.raises(ExperimentError, match="budget exhausted"):
        restored.proposals(3)


def test_config_rejects_invalid_group_and_exposure_rules():
    with pytest.raises(ValueError, match="milestones"):
        OnlineSupplyConfig(evaluation_episodes=[1024, 256, 4096])
    cfg = config().model_dump()
    cfg["method"] = "C"
    with pytest.raises(ValueError, match="paired A/B"):
        type(config()).model_validate(cfg)
    assert config().objective.sd_lambda == 0.1


def script(name):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_actual_remote_command_has_freeze_and_resume():
    job = script("hpc_job")
    env = {
        "SELFPLAY_OPERATION": "task15-online",
        "SELFPLAY_METHOD": "A",
        "SELFPLAY_SEED": "2028",
        "SELFPLAY_SETTINGS": "configs/task15.yaml",
        "SELFPLAY_OUTPUT": "/runs",
        "SELFPLAY_TASK15_FREEZE": "/runs/study/freeze.json",
        "SELFPLAY_RESUME": "/runs/study/A-seed2026/checkpoints/000017",
    }
    args = job.task13_command_args(env)
    assert args[args.index("--freeze") + 1] == env["SELFPLAY_TASK15_FREEZE"]
    assert args[args.index("--resume") + 1] == env["SELFPLAY_RESUME"]
    assert "/workspace/scripts/task15_online.py" in args
    assert args[args.index("--seed") + 1] == "2028"
    env["SELFPLAY_OPERATION"] = "task15-stabilize"
    args = job.task13_command_args(env)
    assert "--stabilization" in args and "--freeze" not in args


def test_audit_runs_real_tiny_model_updates_and_restores_full_state():
    from selfplay_opsd.diagnostics import _optimizer_digest
    from selfplay_opsd.policy import HFPolicy
    from selfplay_opsd.task15_audit import audit_updates

    cfg = config()
    policy = HFPolicy(cfg)
    with policy.active("student"):
        values = policy.score([1], [2, 3]).tolist()
        sample = [
            e.model_copy(
                update={
                    "policy_version": policy.version("student"),
                    "turns": [turn().model_copy(update={"frozen_unprivileged_logps": values})],
                }
            )
            for e in episodes()
        ]
        before = _optimizer_digest(policy.state_dict())
        report = audit_updates(policy, cfg, sample)
        assert report["available"]
        assert report["no_privilege_scorer_repeat"]["max_absolute"] == 0
        assert report["gradient_mode_vs_frozen_scorer"]["max_absolute"] == 0
        assert report["metrics"]["no_privilege:scorer"]["grad_norm"] == 0
        assert report["metrics"]["no_privilege:behavior"]["grad_norm"] > 0
        assert _optimizer_digest(policy.state_dict()) == before
    policy.close()


def test_checkpoint_restores_supply_and_pending_snapshot_state(tmp_path, monkeypatch):
    from selfplay_opsd.core import ParentProgram
    from selfplay_opsd.data import TaskRepository
    from selfplay_opsd.policy import HFPolicy
    from selfplay_opsd.storage import CheckpointManager, atomic_json, file_hash

    cfg = config()
    cfg.output_root = tmp_path / "runs"
    cfg.data_root = tmp_path / "data"
    cfg.logging.mode = "disabled"
    parent = ParentProgram(
        parent_id="fixture/p",
        source="def f(x):\n    return x + 1\n",
        specification="f",
        entry_point="f",
        dataset="fixture",
        family_id="fixture/p",
        provenance={},
    )
    atomic_json(
        cfg.data_root / "private/parents.json",
        [
            {"parent": parent.model_dump(), "split": "train", "public": [], "hidden": []},
            {
                "parent": parent.model_copy(
                    update={"parent_id": "fixture/no-sites", "source": "def f(x):\n    return x\n"}
                ).model_dump(),
                "split": "train",
                "public": [],
                "hidden": [],
            },
        ],
    )
    atomic_json(
        cfg.data_root / "manifest.json",
        {"parents_hash": file_hash(cfg.data_root / "private/parents.json"), "fixture_only": True},
    )
    repo = TaskRepository(cfg.data_root)
    policy = HFPolicy(cfg)
    trainer = OnlineExposureTrainer(cfg, policy, repo, object(), development_enabled=False)
    assert len(trainer.parents) == 1
    assert trainer.supply_state["parents_without_legal_sites"] == 1
    trainer.supply_state.update(
        {
            "proposal_groups": 19,
            "proposal_attempts": 76,
            "task_groups": 8,
            "fallback_groups": 1,
            "evaluated_milestones": [256],
            "milestone_iterations": [1],
        }
    )
    trainer.iteration = 1
    trainer.total_episodes = 256
    checkpoint = trainer.checkpoint()
    state, metadata = CheckpointManager.load(checkpoint)
    assert metadata["total_episodes"] == 256
    assert state["supply_state"] == trainer.supply_state
    trainer.sink.finish()
    restored = OnlineExposureTrainer(
        cfg, policy, repo, object(), resume=checkpoint, development_enabled=False
    )
    assert restored.supply_state == trainer.supply_state
    assert restored.total_episodes == 256
    assert restored.proposal_parents(20)[0].parent_id == "fixture/p"
    restored.sink.finish()
    policy.close()


def test_task15_freeze_requires_completed_evidence_and_guards_final_split(tmp_path):
    from selfplay_opsd.storage import atomic_json
    from selfplay_opsd.task15_freeze import check_freeze, freeze

    cfg = config()
    repo = SimpleNamespace(hash="fixture-data")
    evidence = []
    for kind in ("runtime", "backend", "resume", "numerical", "stabilization"):
        path = tmp_path / f"{kind}.json"
        atomic_json(path, {"ready": True, "episodes": 128, "target_episodes": 128})
        evidence.append({"kind": kind, "path": str(path)})
    decision = tmp_path / "decision.json"
    atomic_json(
        decision,
        {
            "ready": True,
            "evidence": evidence,
            "resource_plan": {"fixture": True},
            "causal_decisions": ["fixture"],
        },
    )
    path = tmp_path / "freeze.json"
    freeze(cfg, repo, decision, path, "a" * 40)
    assert check_freeze(cfg, repo, path)["primary_endpoint"] == "final_common_exposure"
    other_seed = cfg.model_copy(update={"seed": 2028})
    assert check_freeze(other_seed, repo, path)["status"] == "frozen"
    cfg.objective.sd_lambda = 0.01
    with pytest.raises(ExperimentError, match="differs"):
        check_freeze(cfg, repo, path)
    cfg.objective.sd_lambda = 0.1
    (tmp_path / "numerical.json").write_text("{}")
    with pytest.raises(ExperimentError, match="evidence changed"):
        check_freeze(cfg, repo, path)


def test_task15_final_analysis_reports_missing_and_refuses_unpaired_tasks(monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    analysis = script("analyze_task15")
    assert analysis.paired_clustered(None, None, {}) == {"status": "missing"}
    a = [{"task_id": "t", "parent_id": "p", "success": False}]
    b = [{"task_id": "t", "parent_id": "p", "success": True}]
    report = analysis.paired_clustered(a, b, {"p": "family"})
    assert report["b_minus_a_parent_macro"] == 1
    assert report["family_clustered_95_interval"] == [1, 1]
    with pytest.raises(ValueError, match="task sets differ"):
        analysis.paired_clustered(a, [b[0] | {"task_id": "other"}], {"p": "family"})


@pytest.mark.parametrize(
    "scheduler_state,total,phase,reason",
    [
        ("COMPLETED", 4096, "complete", "scientifically_complete"),
        ("TIMEOUT", 128, "rollout_reward", "resumed"),
        ("COMPLETED", 128, "rollout_reward", "not_a_clean_stop"),
        ("TIMEOUT", 128, "failed", "training_reported_failure"),
    ],
)
def test_task15_recovery_stops_at_episode_endpoint_before_iteration_ceiling(
    tmp_path, monkeypatch, scheduler_state, total, phase, reason
):
    import hashlib
    import json

    remote = script("hpc_remote")
    run = tmp_path / "runs/study/A-seed2026"
    checkpoint = run / "checkpoints/000300"
    checkpoint.mkdir(parents=True)
    state = checkpoint / "state.pt"
    state.write_bytes(b"fixture joint state")
    (checkpoint / "metadata.json").write_text(
        json.dumps(
            {
                "iteration": 300,
                "total_episodes": total,
                "scientific_hash": "science",
                "manifests_hash": "data",
            }
        )
    )
    (checkpoint / "complete.json").write_text(
        json.dumps({"files": {"state.pt": hashlib.sha256(state.read_bytes()).hexdigest()}})
    )
    (checkpoint.parent / "latest.json").write_text(json.dumps({"path": "000300", "iteration": 300}))
    (run / "state").mkdir()
    (run / "state/health.json").write_text(json.dumps({"phase": phase}))
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps({"identity": "id", "attempts": [{"job_id": "123"}], "run_dir": str(run)})
    )
    monkeypatch.setattr(remote, "paths", lambda _: {"runs": tmp_path / "runs"})
    monkeypatch.setattr(remote, "ledger_path", lambda _: ledger)
    monkeypatch.setattr(remote, "job_state", lambda _: {"state": scheduler_state})
    submitted = []
    monkeypatch.setattr(
        remote, "submit", lambda payload: submitted.append(payload) or {"submitted": True}
    )
    result = remote.recover(
        {
            "identity": "id",
            "study_id": "study",
            "operation": "task15-online",
            "method": "A",
            "seed": 2026,
            "target_episodes": 4096,
            "target_iterations": 16384,
            "scientific_hash": "science",
            "manifests_hash": "data",
            "explicit_continue": True,
            "max_continuations": 64,
            "time": "06:00:00",
        }
    )
    if reason == "resumed":
        assert result["submitted"]
        assert submitted[0]["resume"] == str(checkpoint)
    else:
        assert result["reason"] == reason
        assert not submitted


@pytest.mark.sandbox
def test_task15_real_transaction_stops_by_exposure_and_resumes(tmp_path, repository, sandbox):
    import types

    from test_trainer import scripted_policy

    from selfplay_opsd.core import canonical
    from selfplay_opsd.policy import Generation
    from selfplay_opsd.storage import CheckpointManager, active_events

    cfg = config()
    cfg.output_root = tmp_path
    cfg.data_root = repository.root
    cfg.sandbox = sandbox.config
    cfg.logging.mode = "disabled"
    cfg.online_supply = cfg.online_supply.model_copy(
        update={
            "target_episodes": 4,
            "evaluation_episodes": [4],
            "max_total_proposals": 16,
            "max_proposals_per_update": 16,
        }
    )
    policy = scripted_policy(cfg)
    original = policy.generate

    def generate(self, prompt, max_tokens, seed, greedy=False):
        if self.role != "proposer":
            return original(prompt, max_tokens, seed, greedy=greedy)
        text = canonical({"site": 1, "operation": "Sub"}) if seed % 3 else "invalid"
        ids = self.tokenizer.encode(text) + [self.tokenizer.eos_token_id]
        return Generation(ids, self.score(prompt, ids).tolist(), text)

    policy.generate = types.MethodType(generate, policy)
    trainer = OnlineExposureTrainer(cfg, policy, repository, sandbox, development_enabled=False)
    trainer.train()
    assert trainer.total_episodes == 4
    assert trainer.iteration < cfg.iterations
    assert trainer.student_steps > 0
    metadata = CheckpointManager.verify(cfg.run_dir / "checkpoints" / f"{trainer.iteration:06d}")
    assert metadata["total_episodes"] == 4
    events = active_events(cfg.run_dir)
    assert len([event for event in events if event["kind"] == "iteration_commit"]) == 1
    restored = OnlineExposureTrainer(
        cfg,
        policy,
        repository,
        sandbox,
        resume=cfg.run_dir / "checkpoints" / f"{trainer.iteration:06d}",
        development_enabled=False,
    )
    restored.train()
    assert restored.total_episodes == 4
    assert (
        len([event for event in active_events(cfg.run_dir) if event["kind"] == "iteration_commit"])
        == 1
    )
    policy.close()


def test_task15_payload_carries_seed_exposure_and_bounded_continuation(monkeypatch):
    from selfplay_opsd import hpc

    class Remote:
        def call(self, name, *args):
            if name == "context":
                return {
                    "storage": {
                        "project": {"project_cpsc4880": "/project/cpsc4880"},
                        "scratch": {"scratch_cpsc4880": "/scratch/cpsc4880"},
                    },
                    "slurm": {
                        "associations": {"exit_code": 0, "stdout": "cpsc4880||normal|normal\n"}
                    },
                }
            assert name == "study"
            return {
                "images": {
                    "trainer": {"path": "/trainer.sif", "sha256": "a" * 64},
                    "worker": {"path": "/worker.sif", "sha256": "b" * 64},
                },
                "reports": {"worker": {}, "freeze": {}},
            }

    monkeypatch.setattr(
        hpc, "snapshot", lambda *a: {"commit": "a" * 40, "manifest_sha256": "c" * 64}
    )
    monkeypatch.setattr(
        "selfplay_opsd.readiness.execution_identity", lambda *a: {"manifest_hash": "d" * 64}
    )
    payload = hpc.run_payload(
        Remote(),
        "yale-hpc-cpsc4880",
        operation="task15-online",
        method="A",
        seed=2028,
        profile="single_gpu_80gb",
        gpu_type="h200",
        account="cpsc4880",
        image="",
        image_sha256="",
        settings=Path("configs/task15.yaml"),
        stop_after_iteration=2,
    )
    assert payload["target_episodes"] == payload["exposure_policy"]["target_episodes"] == 4096
    assert payload["iterations"] == payload["target_iterations"] == 16384
    assert payload["seed"] == payload["common_config"]["seed"] == 2028
    assert payload["stop_after_iteration"] == 2
    assert payload["time"] == payload["continuation_time"] == "06:00:00"
    assert payload["max_continuations"] == 64


def test_fallback_uses_only_realized_budget_and_never_gets_proposer_credit(monkeypatch):
    from selfplay_opsd.core import OracleSpec, TaskSpec

    trainer = supply_trainer()
    trainer.policy = object()
    task = TaskSpec(
        task_id="cached",
        parent_id="p",
        split="train",
        specification="f",
        initial_source="def f(x):\n    return x - 1\n",
        entry_point="f",
        public_tests="",
        source_hash="fixture",
        mutation_seed=1,
    )
    oracle = OracleSpec(
        task_id="cached",
        correct_source="def f(x):\n    return x + 1\n",
        inverse_patch="fixture",
        public_cases=[],
        hidden_cases=[],
        provenance={},
    )
    trainer.supply_state["reservoir"] = {
        "key": {"task": task.model_dump(), "oracle": oracle.model_dump(), "admitted_group": 0}
    }
    monkeypatch.setattr("selfplay_opsd.task15.check_admission", lambda *args: None)
    monkeypatch.setattr(
        SelfPlayTrainer,
        "proposals",
        lambda self, k: ([proposal(None, i, valid=False) for i in range(4)], {}),
    )
    trainer.supply_state["task_groups"] = 2
    _, tasks = trainer.proposals(1)
    assert not tasks  # Cannot borrow exposure from hypothetical future batches.
    assert trainer.supply_state["fallback_groups"] == 0
    trainer.supply_state["task_groups"] = 3
    proposals, tasks = trainer.proposals(2)
    assert set(tasks) == {"cached"}
    assert trainer.supply_state["metrics"]["task_sources"] == {"cached": "fallback"}
    assert trainer.supply_state["metrics"]["fallback_fraction"] == 0.25
    assert all(not p.valid and p.task_id is None for p in proposals)
    examples = proposer_examples(proposals, [-1.0] * len(proposals), trainer.config)
    assert all(e.advantage == 0 for e in examples)
    assert len(examples) == len(proposals) == 64
