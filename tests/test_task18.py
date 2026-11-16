"""Task 18 protocol boundaries, real optimizer restoration, and shared allocation cap."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from selfplay_opsd.config import EpisodeConfig, load_config
from selfplay_opsd.core import ExperimentError, canonical
from selfplay_opsd.environment import append_context
from selfplay_opsd.policy import HFPolicy
from selfplay_opsd.protocol import student_messages
from selfplay_opsd.storage import file_hash
from selfplay_opsd.task18 import STUDY, configure_operation
from selfplay_opsd.task18_numerical import audit_updates, saved_token_report, select_groups
from selfplay_opsd.task18_protocol import (
    NativePolicy,
    Task18EpisodeConfig,
    episode_messages,
    parse_episode_action,
)


def script(name):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def episodes():
    from test_task15 import episodes as make

    return make()


def test_native_validator_keeps_existing_edit_restrictions():
    limits = Task18EpisodeConfig(edit_protocol="replace_text", tool_serialization="qwen_native")
    args = {"path": "src/solution.py", "old_text": "x", "new_text": "y"}
    raw = "<tool_call>" + canonical({"name": "replace_text", "arguments": args}) + "</tool_call>"
    assert parse_episode_action(raw, limits) == {"action": "replace_text", **args}
    for invalid in [
        raw + raw,
        "prose" + raw,
        raw.replace("src/solution.py", "tests/test_public.py"),
        raw.replace('"arguments":', '"extra":1,"arguments":'),
        '<tool_call>{"name":"finish","arguments":{"action":"read_file"}}</tool_call>',
        '<tool_call>{"name":"finish","name":"run_tests","arguments":{}}</tool_call>',
    ]:
        with pytest.raises(ValueError):
            parse_episode_action(invalid, limits)
    assert parse_episode_action('{"action":"finish"}', EpisodeConfig()) == {"action": "finish"}


def test_legacy_prompts_and_scientific_hash_remain_unmodified():
    from selfplay_opsd.core import TaskSpec, digest

    task = TaskSpec(
        task_id="t",
        parent_id="p",
        split="train",
        specification="Return x.",
        initial_source="def f(x):\n return x\n",
        entry_point="f",
        public_tests="",
        source_hash=digest("source"),
        mutation_seed=1,
    )
    for mode in ("unified_diff", "normalized_diff", "replace_text"):
        limits = EpisodeConfig(edit_protocol=mode)
        assert episode_messages(task, limits) == student_messages(task, limits)
    cfg = load_config(
        Path("configs/task15-v2.yaml"), method="B", seed=2026, hardware_profile="single_gpu_80gb"
    )
    assert cfg.scientific_hash == "3fb8aaa585826f21695e89758e1a937fa371e1e83ed341b6ba3d1454d03ab709"


def test_native_pinned_tokenizer_preserves_actual_targets_and_eos():
    from transformers import AutoTokenizer

    cache = (
        Path.home()
        / ".cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"
    )
    if not cache.is_dir():
        pytest.skip("Pinned tokenizer not cached; no download in local tests")
    tokenizer = AutoTokenizer.from_pretrained(cache, local_files_only=True)
    native = NativePolicy(SimpleNamespace(tokenizer=tokenizer))
    history = [
        {"role": "system", "content": "Repair code."},
        {"role": "user", "content": "Inspect source."},
    ]
    prompt = native.encode_history(history)
    assert "<tools>" in tokenizer.decode(prompt)
    targets = tokenizer.encode(
        '<tool_call>{"name":"read_file","arguments":{"path":"src/solution.py"}}</tool_call>',
        add_special_tokens=False,
    ) + [tokenizer.eos_token_id]
    context = [{"role": "tool", "content": '{"ok":true}'}, {"role": "user", "content": "Continue."}]
    actual = append_context(native, history, prompt, targets, context)
    assert actual[: len(prompt) + len(targets)] == prompt + targets
    assert actual[len(prompt) + len(targets)] != tokenizer.eos_token_id
    next_history = history + [
        {"role": "assistant", "content": tokenizer.decode(targets[:-1])},
        *context,
    ]
    # A second turn must remain prefix-consistent as well.
    again = append_context(native, next_history, actual, targets, context)
    assert again[: len(actual) + len(targets)] == actual + targets


def test_numerical_report_and_whole_group_selection():
    rows = episodes()
    report = saved_token_report(rows)
    assert report["tokens"] == sum(len(t.target_ids) for e in rows for t in e.turns)
    assert report["above_02_fraction"] == 1
    assert report["mixed_groups"] == 1
    assert report["clipped_surrogate_branch_changes"] > 0
    assert select_groups(rows, 1) == rows
    rows[0].turns[0] = rows[0].turns[0].model_copy(update={"frozen_unprivileged_logps": []})
    with pytest.raises(ExperimentError, match="mapping"):
        saved_token_report(rows)


def test_real_optimizer_audit_restores_both_adapters_and_optimizers(monkeypatch):
    from selfplay_opsd.diagnostics import _optimizer_digest

    cfg = load_config(study_id="task18-test", hardware_profile="cpu_debug")
    cfg.sd_gap_definition = "same_scorer_v2"
    policy = HFPolicy(cfg)
    rows = episodes()
    rows = [e.model_copy(update={"policy_version": policy.version("student")}) for e in rows]
    before = copy.deepcopy(policy.state_dict())
    result = audit_updates(policy, cfg, rows)
    assert result["training_ready"] is False
    assert result["comparisons"]["repeat"]["update"]["relative_l2"] == 0
    assert _optimizer_digest(before) == _optimizer_digest(policy.state_dict())
    original = policy.gradients

    def fail_after_step(examples, *, step=False):
        result = original(examples, step=step)
        if step:
            raise RuntimeError("injected post-update failure")
        return result

    monkeypatch.setattr(policy, "gradients", fail_after_step)
    with pytest.raises(RuntimeError, match="post-update"):
        audit_updates(policy, cfg, rows)
    assert _optimizer_digest(before) == _optimizer_digest(policy.state_dict())
    policy.close()


def test_training_is_disabled_and_serialization_is_part_of_identity(tmp_path):
    cfg = load_config(Path("configs/task18.yaml"))
    control = Path("configs/task18-control.json")
    with pytest.raises(ValueError, match="Freeze"):
        configure_operation(cfg, "student", "A", control)
    plain = configure_operation(cfg, "numerical", "A", control)
    native = configure_operation(cfg, "native", "A", control)
    assert plain.scientific_hash != native.scientific_hash
    assert native.scientific_settings()["episode"]["tool_serialization"] == "qwen_native"
    assert native.task18_control_sha256 == file_hash(control)
    with pytest.raises(ValueError, match="two paired"):
        cfg.seed = 2028
        configure_operation(cfg, "numerical", "A", control)


def test_task18_budget_is_shared_across_all_operations_and_seeds(tmp_path, monkeypatch):
    remote = script("hpc_remote")
    monkeypatch.setattr(remote, "workspace", lambda p: tmp_path)
    for operation in ("numerical", "normalized", "native", "student"):
        payload = {"operation": "task18-" + operation, "study_id": STUDY + "-objectives-seed2027"}
        assert remote.readiness_limit(payload) == 12
        assert remote.budget_path(payload) == tmp_path / "state/task18-budget.json"
    payload = {
        "operation": "task18-native",
        "study_id": STUDY,
        "readiness_budget": 12,
        "gpu_count": 1,
        "time": "00:30:00",
        "run_id": "one",
    }
    remote.reserve_budget(payload, "first")
    with pytest.raises(ValueError, match="already exists"):
        remote.reserve_budget(payload, "first")
    payload["time"] = "12:00:00"
    with pytest.raises(ValueError, match="exhausted"):
        remote.reserve_budget(payload, "second")
    ledger = json.loads((tmp_path / "state/task18-budget.json").read_text())
    assert ledger["authorization_id"] == "cpsc4880-task18-12h-v1"
    assert ledger["reservations"]["first"]["reserved"] == 0.5


def test_allocated_command_preserves_arm_seed_and_continuation():
    job = script("hpc_job")
    env = {
        "SELFPLAY_OPERATION": "task18-student",
        "SELFPLAY_METHOD": "D",
        "SELFPLAY_SETTINGS": "configs/task18.yaml",
        "SELFPLAY_OUTPUT": "/runs/study/GRPO-seed2027",
        "SELFPLAY_SEED": "2027",
        "SELFPLAY_RESUME": "/runs/study/GRPO-seed2027/checkpoints/000004",
    }
    command = job.task13_command_args(env)
    assert command[0] == "torchrun"
    assert "/workspace/scripts/task18_experiment.py" in command
    assert command[command.index("--arm") + 1] == "D"
    assert command[command.index("--seed") + 1] == "2027"
    assert command[-2:] == ["--resume", env["SELFPLAY_RESUME"]]


@pytest.mark.parametrize(
    "operation,interface,time_limit",
    [
        ("numerical", "replace_text", "00:30:00"),
        ("normalized", "normalized_diff", "00:30:00"),
        ("native", "replace_text", "00:30:00"),
    ],
)
def test_controller_and_runner_bind_same_science(monkeypatch, operation, interface, time_limit):
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
        "selfplay_opsd.readiness.execution_identity",
        lambda cfg: {"manifest_hash": "d" * 64, "interface": cfg.episode.edit_protocol},
    )
    result = hpc.run_payload(
        Remote(),
        "yale-hpc-cpsc4880",
        operation="task18-" + operation,
        method="A",
        profile="single_gpu_80gb",
        gpu_type="h200",
        account="cpsc4880",
        image="",
        image_sha256="",
        settings=Path("configs/task18.yaml"),
        seed=2026,
    )
    cfg = configure_operation(
        load_config(
            Path("configs/task18.yaml"), method="A", hardware_profile="single_gpu_80gb", seed=2026
        ),
        operation,
        "A",
        Path("configs/task18-control.json"),
    )
    assert result["scientific_hash"] == cfg.scientific_hash
    assert result["runtime_identity"]["interface"] == interface
    assert result["readiness_budget"] == 12
    assert result["time"] == time_limit
    assert result["gpu_count"] == 1
    assert result["output"] == f"/runs/{STUDY}/{result['run_id']}"
    remote = script("hpc_remote")
    monkeypatch.setattr(remote, "paths", lambda p: {"runs": Path("/runs")})
    assert remote.artifact_dir(result).name == result["run_id"]


def test_runtime_proof_scope_preserves_model_and_verifier_but_reuses_across_formats():
    from selfplay_opsd.readiness import execution_identity

    base = load_config(Path("configs/task18.yaml"))
    repository = SimpleNamespace(root=base.data_root, hash="immutable-data")
    plain = execution_identity(base, repository=repository)
    for operation in ("numerical", "normalized", "native"):
        cfg = configure_operation(base, operation, "A", Path("configs/task18-control.json"))
        assert execution_identity(cfg, repository=repository) == plain
    cfg = configure_operation(base, "native", "A", Path("configs/task18-control.json"))
    cfg.episode.max_observation_tokens += 1
    assert execution_identity(cfg, repository=repository) != plain


@pytest.mark.parametrize("stage", ["updates", "cached-scoring"])
def test_replay_keeps_original_ungated_objective_and_optimizer_state(tmp_path, monkeypatch, stage):
    monkeypatch.syspath_prepend("scripts")
    runner = script("task18_experiment")
    rows = episodes()
    path = tmp_path / "episodes.json"
    path.write_text(json.dumps([e.model_dump(mode="json") for e in rows]))
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "complete.json").write_text("{}")
    cfg = configure_operation(
        load_config(Path("configs/task18.yaml")),
        "numerical",
        "A",
        Path("configs/task18-control.json"),
    )
    original = cfg.model_copy(deep=True)
    original.method = "A"
    original.objective.sd_lambda = 0.029
    state = {"policy": {"original_optimizers": True}, "rng": {}}
    metadata = {
        "manifests_hash": "data",
        "resolved_config": {
            "model": cfg.model.model_dump(mode="json"),
            "method": "A",
            "objective": original.objective.model_dump(mode="json"),
            "sd_gap_definition": "same_scorer_v2",
        },
    }
    monkeypatch.setattr(runner.CheckpointManager, "load", lambda p: (state, metadata))
    monkeypatch.setattr(runner, "restore_rng", lambda p: None)
    monkeypatch.setattr(runner, "score_teacher", lambda p, e, o, limits: (e, None, None))
    monkeypatch.setattr(runner, "gradient_mode_probe", lambda *a: {"tested": True})
    seen = []

    def audit(policy, config, *args):
        seen.append((config.method, config.objective.sd_lambda, policy.loaded))
        return {"training_ready": False}

    monkeypatch.setattr(runner, "audit_updates", audit)
    monkeypatch.setattr(runner, "cached_scoring_probe", lambda policy, rows: audit(policy, cfg))
    monkeypatch.setattr(runner, "frozen_rollout_replay", lambda *args: {"new_generated_tokens": 4})
    oracle = {
        "task_id": rows[0].task_id,
        "correct_source": "def f(x):\n return x\n",
        "inverse_patch": "",
        "public_cases": [],
        "hidden_cases": [],
        "provenance": {},
    }
    from selfplay_opsd.core import OracleSpec

    # Use a validated oracle without executing its source.
    oracle = OracleSpec.model_validate(oracle)
    pool = {
        "rows": [
            {
                "purpose": "curriculum",
                "task": {"task_id": e.task_id},
                "oracle": oracle.model_dump(mode="json"),
            }
            for e in rows
        ]
    }

    class Policy:
        def load_state_dict(self, value):
            self.loaded = value

        def version(self, role):
            return rows[0].policy_version

    control = {
        "numerical_stage": stage,
        "numerical_sources": [
            {
                "episodes": str(path),
                "episodes_sha256": file_hash(path),
                "checkpoint": str(checkpoint),
                "checkpoint_complete_sha256": file_hash(checkpoint / "complete.json"),
            }
        ],
        "update_relative_l2_screen": 0.01,
        "update_cosine_screen": 0.99,
    }
    runner.replay_numerical(Policy(), cfg, SimpleNamespace(hash="data"), pool, control, tmp_path)
    assert seen == [("A", 0.029, state["policy"])]


def test_verifier_reuse_accepts_only_unchanged_verified_dependencies(tmp_path):
    remote = script("hpc_remote")
    previous, current = "a" * 40, "b" * 40
    for commit in (previous, current):
        for name in remote.TASK18_VERIFIER_FILES:
            path = tmp_path / "checkouts" / commit / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
    actual = {
        "code_hash": "old",
        "scientific_code_hash": "old",
        "prompt_hash": "p",
        "common_hash": "v",
        "manifest_hash": "d",
    }
    expected = actual | {"code_hash": "new", "scientific_code_hash": "new"}
    report = {
        "ready": True,
        "runtime_identity": actual,
        "provenance": {"git_commit": previous},
        "validation": {"ready": True, "executed": True},
        "isolation": {"ready": True},
    }
    payload = {"study_id": STUDY, "commit": current}
    assert remote.task18_verifier_compatible(report, expected, payload, tmp_path)
    assert not remote.task18_verifier_compatible(
        report, expected | {"manifest_hash": "other"}, payload, tmp_path
    )
    assert not remote.task18_verifier_compatible(
        report, expected, payload | {"study_id": "historical"}, tmp_path
    )
    (tmp_path / "checkouts" / current / "src/selfplay_opsd/worker_policy.py").write_text("changed")
    assert not remote.task18_verifier_compatible(report, expected, payload, tmp_path)


def test_interface_dependencies_keep_single_gpu_sequence_and_exclude_training():
    remote = script("hpc_remote")
    dependency = {"run_id": "a" * 24, "job_id": "123"}
    for operation in ("task18-normalized", "task18-native"):
        assert remote.sequential_dependency({"operation": operation, "afterok": dependency}) == [
            "--dependency=afterok:123"
        ]
    with pytest.raises(ValueError, match="sequential"):
        remote.sequential_dependency({"operation": "task18-student", "afterok": dependency})


def test_cached_scoring_uses_saved_ids_and_restores_mode():
    from selfplay_opsd.task18_numerical import cached_scoring_probe

    cfg = load_config(study_id="task18-cached-test", hardware_profile="cpu_debug")
    policy = HFPolicy(cfg)
    rows = [e.model_copy(update={"policy_version": policy.version("student")}) for e in episodes()]
    was_training = policy.model.training
    report = cached_scoring_probe(policy, rows)
    assert report["cached_vs_full"]["quantiles_0_50_90_95_99_100"][-1] < 1e-5
    assert report["full_repeat"]["mean_absolute"] == 0
    assert policy.model.training == was_training
    assert report["tokens"] == sum(len(t.target_ids) for e in rows for t in e.turns)
    policy.close()


def test_frozen_replay_compares_probabilities_only_before_target_divergence():
    from contextlib import nullcontext

    from selfplay_opsd.policy import Generation
    from selfplay_opsd.task18_numerical import frozen_rollout_replay

    rows = episodes()

    class Policy:
        config = SimpleNamespace(model=SimpleNamespace(backend="verl"))

        def version(self, role):
            return rows[0].policy_version

        def active(self, role):
            return nullcontext()

        def generate(self, prompt, max_tokens, seed, *, greedy):
            assert greedy is False
            return Generation([2, 99], [-1.9, -100], "unexecuted")

    report = frozen_rollout_replay(Policy(), rows, 4)
    assert all(r["identical_prefix_tokens"] == 1 for r in report["turns"])
    assert all(
        r["probability_difference_on_identical_prefix"]["max_absolute"]
        if "max_absolute" in r["probability_difference_on_identical_prefix"]
        else r["probability_difference_on_identical_prefix"]["quantiles_0_50_90_95_99_100"][-1]
        < 0.11
        for r in report["turns"]
    )
    with pytest.raises(ValueError, match="1..4"):
        frozen_rollout_replay(Policy(), rows, 5)
