"""Protocol revision regressions across actual rendering, episodes and accounting."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from selfplay_opsd import protocol
from selfplay_opsd.config import load_config
from selfplay_opsd.core import Case, OracleSpec, Status, TaskSpec, digest
from selfplay_opsd.environment import run_episode
from selfplay_opsd.evaluation import evaluate_tasks
from selfplay_opsd.mutations import parse_action, strict_json_object, validate_mutation
from selfplay_opsd.policy import Generation, HFPolicy
from selfplay_opsd.sandbox import Verification


def test_protocol_identity_covers_rendered_output_and_tools(monkeypatch):
    original = digest(protocol.protocol_definition())
    assert original == protocol.PROMPT_HASH
    monkeypatch.setattr(protocol, "PROPOSER_OUTPUT", {"patch": "changed output description"})
    assert digest(protocol.protocol_definition()) != original
    monkeypatch.undo()
    monkeypatch.setattr(protocol, "TOOLS", protocol.TOOLS + [{"action": "unexpected"}])
    assert digest(protocol.protocol_definition()) != original


def test_format_example_is_valid_in_both_json_wrappers():
    patch = protocol.PATCH_FORMAT[protocol.PATCH_FORMAT.rindex("--- a/") :]
    source = "def f(x):\n    return x + 1\n"
    for wrapper in ({"patch": patch}, {"action": "apply_patch", "patch": patch}):
        raw = json.dumps(wrapper)
        parsed = parse_action(raw) if "action" in wrapper else strict_json_object(raw)
        assert validate_mutation(source, parsed["patch"]).source == source.replace("+", "-")


def fake_task():
    task = TaskSpec(
        task_id="task",
        parent_id="parent",
        split="dev_monitor",
        specification="Return x.",
        initial_source="def f(x):\n    return x\n",
        entry_point="f",
        public_tests="",
        source_hash="source",
        mutation_seed=0,
    )
    case = Case(args=[1], expected=1)
    oracle = OracleSpec(
        task_id="task",
        correct_source=task.initial_source,
        inverse_patch="",
        public_cases=[case],
        hidden_cases=[case],
        provenance={},
    )
    sandbox = SimpleNamespace(
        verify=lambda *args: Verification(Status.OK, True, [True, True], 0, 0, 0.0)
    )
    return task, oracle, sandbox


@pytest.mark.parametrize("backend", ["hf", "verl"])
def test_evaluation_reaches_every_generation_turn_and_training_stays_sampled(backend):
    cfg = load_config(study_id="decode-test", hardware_profile="cpu_debug")
    policy = HFPolicy(cfg)
    task, oracle, sandbox = fake_task()
    observed = []
    actions = [{"action": "read_file", "path": "src/solution.py"}, {"action": "finish"}]

    def generate(prompt, max_tokens, seed, *, greedy=False):
        text = json.dumps(actions[len(observed) % 2])
        observed.append(greedy)
        ids = policy.tokenizer.encode(text)
        return Generation(ids, [0.0] * len(ids), text)

    policy.generate = generate
    if backend == "verl":
        import types

        from selfplay_opsd.verl_backend import VerlPolicy

        policy._ensure_rollout = lambda: "snapshot"
        policy._adapter_ids = {}
        policy._adapter_paths = {"snapshot": "/fixture"}

        def rpc(request):
            result = generate(
                request["prompt"], request["max_tokens"], request["seed"], greedy=request["greedy"]
            )
            return {"ids": result.ids, "logps": result.logps, "text": result.text}

        policy._rpc = rpc
        policy.generate = types.MethodType(VerlPolicy.generate, policy)
    repository = SimpleNamespace(tasks=lambda split: [(task, oracle)], hash="manifest")
    try:
        records = evaluate_tasks(
            policy, sandbox, repository, "dev_monitor", cfg.episode, 0, 2026, None
        )
        assert records[0].metrics["turns"] == 2
        assert observed == [True, True]
        run_episode(policy, sandbox, task, oracle, cfg.episode, 2026)
        assert observed == [True, True, False, False]
    finally:
        policy.close()


def endpoint():
    spec = importlib.util.spec_from_file_location("task08_remote", "scripts/hpc_remote.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_study_change_imports_all_history_once_and_retains_unknown_reservations(
    tmp_path, monkeypatch
):
    remote = endpoint()
    monkeypatch.setattr(remote, "workspace", lambda _: tmp_path)
    monkeypatch.setattr(remote, "capture", lambda _: {"exit_code": 1})
    state = tmp_path / "state"
    state.mkdir()
    history = {
        "limit": 24,
        "reservations": {
            "old-charge": {"reserved": 12, "charged": 10, "operation": "pilot", "job_id": "1"},
            "uncertain": {"reserved": 12, "charged": None, "operation": "backend-check"},
        },
    }
    for study in ["old", "copied"]:
        (state / f"budget-{study}.json").write_text(json.dumps(history))
    payload = {
        "study_id": "new",
        "operation": "backend-check",
        "run_id": "new",
        "time": "02:00:00",
        "gpu_count": 1,
        "readiness_budget": 24,
    }
    remote.reserve_budget(payload, "new")
    assert remote.refreshed_budget(payload)["committed_gpu_hours"] == 24
    assert remote.refreshed_budget(payload | {"study_id": "another"})["committed_gpu_hours"] == 24
    with pytest.raises(ValueError, match="exhausted"):
        remote.reserve_budget(payload | {"study_id": "old"}, "over")
    changed = copy.deepcopy(history)
    changed["reservations"]["old-charge"]["charged"] = 0
    (state / "budget-old.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="changed after migration"):
        remote.reserve_budget(payload, "legacy-writer")


def test_cross_study_reservations_share_atomic_lock(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    remote = endpoint()
    monkeypatch.setattr(remote, "workspace", lambda _: tmp_path)
    payload = {
        "operation": "backend-check",
        "run_id": "run",
        "time": "13:00:00",
        "gpu_count": 1,
        "readiness_budget": 24,
    }

    def reserve(study):
        try:
            remote.reserve_budget(payload | {"study_id": study}, study)
            return True
        except ValueError:
            return False

    with ThreadPoolExecutor(2) as pool:
        assert sorted(pool.map(reserve, ["old", "revised"])) == [False, True]


def fixture_revision_source(tmp_path):
    from selfplay_opsd.config import SandboxConfig
    from selfplay_opsd.smoke import fixture_repository
    from selfplay_opsd.storage import atomic_json

    cfg = load_config(study_id="revision-fixture", hardware_profile="cpu_debug")
    policy = HFPolicy(cfg)

    def verify(source, entry, cases, seconds):
        passed = "a + b" in source
        return Verification(Status.OK, passed, [passed] * len(cases), 0 if passed else 1, 0, 0.0)

    sandbox = SimpleNamespace(
        infrastructure_retries=0,
        config=SandboxConfig(),
        identity=lambda: "sha256:fixture",
        verify=verify,
    )
    root = tmp_path / "original"
    fixture_repository(root, sandbox, policy)
    atomic_json(root / "coverage.json", {"fixture_only": True})
    return root, policy, sandbox


def test_revision_preserves_every_frozen_record_and_source_manifest(tmp_path):
    from selfplay_opsd.protocol_revision import revise_protocol
    from selfplay_opsd.storage import file_hash

    source, policy, sandbox = fixture_revision_source(tmp_path)
    before = {str(p.relative_to(source)): file_hash(p) for p in source.rglob("*") if p.is_file()}
    destination = tmp_path / "revision"
    try:
        report = revise_protocol(source, destination, policy, sandbox, policy.config.episode)
        assert report["ready"]
        assert before == {
            str(p.relative_to(source)): file_hash(p) for p in source.rglob("*") if p.is_file()
        }
        for name in report["preserved_files"]:
            assert file_hash(destination / name) == before[name]
        manifest = json.loads((destination / "manifest.json").read_text())
        assert manifest["prompt_hash"] == protocol.PROMPT_HASH
        assert manifest["admitted_tasks"] == dict.fromkeys(
            ["reference", "dev_monitor", "dev_diagnostic", "final_id", "external"], 1
        )
        with pytest.raises(Exception, match="nonexistent destination"):
            revise_protocol(source, destination, policy, sandbox, policy.config.episode)
    finally:
        policy.close()


def test_failed_revision_remains_unpublished_and_keeps_failure(tmp_path, monkeypatch):
    from selfplay_opsd.protocol_revision import revise_protocol

    source, policy, sandbox = fixture_revision_source(tmp_path)
    destination = tmp_path / "revision"

    def fail(*args):
        raise ValueError("fixed admission budget exceeded")

    monkeypatch.setattr("selfplay_opsd.protocol_revision.finalize_data", fail)
    try:
        with pytest.raises(ValueError, match="admission budget"):
            revise_protocol(source, destination, policy, sandbox, policy.config.episode)
        assert not destination.exists()
        assert (tmp_path / "revision.staging/revision_failure.json").exists()
    finally:
        policy.close()


def test_explicit_h200_pilot_diagnostics_do_not_depend_on_name(tmp_path):
    # Real tiny scoring/gradients with labelled deterministic fixture rollouts.
    from test_trainer import scripted_policy

    from selfplay_opsd.data import TaskRepository
    from selfplay_opsd.trainer import SelfPlayTrainer

    source, tiny, sandbox = fixture_revision_source(tmp_path)
    tiny.close()
    cfg = load_config(
        study_id="name-pilot-h200",
        hardware_profile="cpu_debug",
        data_root=source,
        output_root=tmp_path,
        method="D",
        gradient_diagnostics=True,
    )
    cfg.logging.mode = "disabled"
    policy = scripted_policy(cfg)
    try:
        trainer = SelfPlayTrainer(
            cfg, policy, TaskRepository(source), sandbox, development_enabled=False
        )
        record = trainer.run_iteration(1)
        interaction = record.metrics["student"]["gradient_interaction"]
        assert interaction["sampled"]
        assert interaction["lambda_sd_norm"] >= 0
        assert trainer.student_steps > 0
    finally:
        policy.close()


@pytest.mark.parametrize(
    "norm,stepped,advantages,mechanics,expected",
    [
        (0.0, 0, 0, True, True),
        (0.0, 0, 0, False, False),
        (0.0, 0, 1, True, False),
        (1.0, 0, 1, True, False),
        (1.0, 1, 1, False, True),
        (float("nan"), 0, 0, True, False),
    ],
)
def test_acceptance_distinguishes_zero_advantages_from_broken_optimizer(
    norm, stepped, advantages, mechanics, expected
):
    from selfplay_opsd.method_smoke import update_path_accepted

    commits = [
        {
            "metrics": {
                "proposer": {
                    "grad_norm": norm,
                    "stepped": stepped,
                    "nonzero_advantage_count": advantages,
                }
            }
        }
    ]
    assert update_path_accepted(commits, "proposer", stepped, mechanics) is expected
    assert not update_path_accepted([], "proposer", 0, True)


def test_revision_cannot_reset_pilot_sublimit(tmp_path, monkeypatch):
    remote = endpoint()
    monkeypatch.setattr(remote, "workspace", lambda _: tmp_path)
    monkeypatch.setattr(remote, "capture", lambda _: {"exit_code": 1})
    state = tmp_path / "state"
    state.mkdir()
    (state / "budget-original.json").write_text(
        json.dumps(
            {
                "limit": 24,
                "reservations": {
                    "charged": {"reserved": 11, "charged": 11, "operation": "pilot"},
                    "unknown": {"reserved": 1, "charged": None, "operation": "pilot"},
                },
            }
        )
    )
    with pytest.raises(ValueError, match="Pilot allocation budget"):
        remote.reserve_budget(
            {
                "study_id": "revised",
                "run_id": "run",
                "gpu_count": 1,
                "time": "01:00:00",
                "readiness_budget": 24,
                "operation": "pilot",
            },
            "new",
        )


def test_worker_reuse_requires_identical_limits_and_retains_provenance(tmp_path, monkeypatch):
    from selfplay_opsd.config import SandboxConfig
    from selfplay_opsd.storage import atomic_json, file_hash

    remote = endpoint()
    monkeypatch.setattr(remote, "workspace", lambda _: tmp_path)
    monkeypatch.setattr(remote, "study_state", lambda _: {"gates": {"worker": True}})
    cfg = SandboxConfig(process_policy="single_process", resource_backend="slurm")
    source = tmp_path / "runs/readiness/original/worker.json"
    atomic_json(
        source,
        {
            "ready": True,
            "checks": {"cleanup": True},
            "worker_setup_id": "",
            "limits": cfg.model_dump(mode="json"),
        },
    )
    payload = {
        "source_study": "original",
        "study_id": "revised",
        "worker_setup_id": "",
        "execution_limits": cfg.execution_limits(),
    }
    result = remote.reuse_worker(payload)
    assert result["provenance"]["sha256"] == file_hash(source)
    assert json.loads(Path(result["evidence"]).read_text())["reuse_provenance"]["source"] == str(
        source
    )
    changed = cfg.execution_limits() | {"memory_mib": 2048}
    with pytest.raises(ValueError, match="limits changed"):
        remote.reuse_worker(payload | {"execution_limits": changed})


def test_syntax_error_in_generated_patch_is_a_tool_failure():
    from selfplay_opsd.mutations import make_patch

    cfg = load_config(study_id="syntax-failure", hardware_profile="cpu_debug")
    policy = HFPolicy(cfg)
    task, oracle, sandbox = fake_task()
    patch = make_patch(task.initial_source, "def f(x):\n    return (\n")
    actions = [
        json.dumps({"action": "apply_patch", "patch": patch}),
        json.dumps({"action": "finish"}),
    ]
    calls = []
    verified_sources = []

    def verify(source, *args):
        verified_sources.append(source)
        return Verification(Status.OK, True, [True, True], 0, 0, 0.0)

    sandbox.verify = verify

    def generate(prompt, max_tokens, seed, *, greedy=False):
        text = actions[len(calls)]
        calls.append(text)
        ids = policy.tokenizer.encode(text)
        return Generation(ids, [0.0] * len(ids), text)

    policy.generate = generate
    try:
        episode = run_episode(policy, sandbox, task, oracle, cfg.episode, 2026)
        assert [turn.parse_status for turn in episode.turns] == ["tool_error", "valid"]
        assert episode.termination == "finish"
        assert episode.final_source == task.initial_source
        assert verified_sources == [task.initial_source]
        assert "error" in json.loads(episode.turns[1].history[-2]["content"])
    finally:
        policy.close()
