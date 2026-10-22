"""Operational commands; GPU/paid work is always an explicit invocation."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Annotated, Literal

import typer

from selfplay_opsd.config import RunConfig, load_config
from selfplay_opsd.core import ExperimentError, canonical, digest
from selfplay_opsd.data import TaskRepository
from selfplay_opsd.hpc import app as hpc_app
from selfplay_opsd.sandbox import SandboxBackend
from selfplay_opsd.storage import CheckpointManager, EventStore, atomic_json, provenance, read_json

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
app.add_typer(hpc_app, name="hpc")
ConfigOption = Annotated[
    Path, typer.Option("--settings", help="Strict composed YAML configuration")
]


def make_policy(config: RunConfig):
    import torch

    torch.set_num_threads(max(1, config.hardware.cpu_workers // max(1, config.hardware.gpu_count)))
    if config.model.backend == "verl":
        from selfplay_opsd.verl_backend import VerlPolicy

        return VerlPolicy(config)
    from selfplay_opsd.policy import HFPolicy

    return HFPolicy(config)


def checked_preflight(config: RunConfig) -> dict:
    from selfplay_opsd.preflight import doctor as inspect

    if config.model.backend != "verl":
        return inspect(config, isolation=True)
    import torch

    if not torch.cuda.is_available():
        return inspect(config, isolation=True)
    torch.cuda.set_device(int(os.getenv("LOCAL_RANK", "0")))
    if not torch.distributed.is_initialized():
        if "RANK" not in os.environ:
            raise ExperimentError("Use torchrun for production execution")
        torch.distributed.init_process_group("nccl")
    result = [inspect(config, isolation=True) if torch.distributed.get_rank() == 0 else None]
    torch.distributed.broadcast_object_list(
        result, src=0, device=torch.device("cuda", torch.cuda.current_device())
    )
    assert result[0] is not None
    return result[0]


@app.command()
def doctor(
    settings: ConfigOption,
    isolation: bool = typer.Option(False, "--isolation"),
    output: Path | None = None,
):
    """Inspect allocation, dependencies, manifests and optional real isolation."""
    from selfplay_opsd.preflight import doctor as inspect

    report = inspect(load_config(settings), isolation=isolation)
    if output:
        atomic_json(output, report)
    typer.echo(json.dumps(report, indent=2))


@app.command("prepare-data")
def prepare(settings: ConfigOption, allow_reduced: bool = False, inventory_only: bool = False):
    """Download/audit pinned sources, construct frozen tasks, then canonical anchors."""
    from selfplay_opsd.data import download_inventory, prepare_data
    from selfplay_opsd.evaluation import finalize_data
    from selfplay_opsd.policy import HFPolicy

    cfg = load_config(settings)
    if inventory_only:
        download_inventory(cfg.data_root)
        typer.echo((cfg.data_root / "inventory_report.json").read_text())
        return
    sandbox = SandboxBackend(cfg.sandbox)
    sandbox.preflight()
    # A tokenizer-only policy is sufficient for admission and canonical actions.
    from selfplay_opsd.core import OracleSpec, TaskSpec
    from selfplay_opsd.environment import check_admission
    from selfplay_opsd.policy import tokenizer_for
    from selfplay_opsd.protocol import student_messages, teacher_messages

    policy = object.__new__(HFPolicy)
    policy.config, policy.tokenizer = cfg, tokenizer_for(cfg)

    def admit(parent, public, hidden):
        task = TaskSpec(
            task_id="admission",
            parent_id=parent.parent_id,
            split="admission",
            specification=parent.specification,
            initial_source=parent.source,
            entry_point=parent.entry_point,
            public_tests="",
            source_hash=digest(parent.source),
            mutation_seed=0,
        )
        oracle = OracleSpec(
            task_id="admission",
            correct_source=parent.source,
            inverse_patch="",
            public_cases=public,
            hidden_cases=hidden,
            provenance=parent.provenance,
        )
        check_admission(policy, task, oracle, cfg.episode)
        history = student_messages(task, cfg.episode)
        privilege = len(policy.encode_history(teacher_messages(history, oracle))) - len(
            policy.encode_history(history)
        )
        if privilege + cfg.proposer_max_tokens > cfg.episode.privilege_reserve:
            raise ValueError(
                "Parent lacks reserved capacity for a maximum-length inverse proposal patch"
            )

    if not (cfg.data_root / "manifest.json").exists():
        prepare_data(
            cfg.data_root, sandbox, cfg.seed, allow_reduced=allow_reduced, admit_parent=admit
        )
    repository = TaskRepository(cfg.data_root)
    if not repository.manifest.get("ready"):
        finalize_data(repository, policy, sandbox, cfg.episode)
    typer.echo(canonical(read_json(cfg.data_root / "manifest.json")))


@app.command("validate-data")
def validate(
    settings: ConfigOption, execute: bool = typer.Option(True, "--execute/--structural-only")
):
    from selfplay_opsd.evaluation import validate_data

    cfg = load_config(settings)
    sandbox = SandboxBackend(cfg.sandbox)
    if execute:
        sandbox.preflight()
    typer.echo(canonical(validate_data(TaskRepository(cfg.data_root), sandbox, execute=execute)))


@app.command()
def smoke(
    output: Path = Path("runs/local-smoke"),
    image: str = "selfplay-worker:local",
    wandb_mode: Literal["offline", "online", "disabled"] = "disabled",
):
    """Real tiny-model generation/scoring and explicit fixture-only A–D checks."""
    from selfplay_opsd.config import SandboxConfig
    from selfplay_opsd.smoke import run_smoke

    typer.echo(
        canonical(
            run_smoke(output, SandboxBackend(SandboxConfig(image=image)), wandb_mode=wandb_mode)
        )
    )


@app.command()
def train(
    settings: ConfigOption,
    config: str | None = typer.Option(None, "--config"),
    hardware: str | None = None,
    study_id: str | None = None,
    resume: Path | None = None,
):
    """Run a frozen main study, requiring explicit target-hardware readiness."""
    from selfplay_opsd.preflight import check_frozen
    from selfplay_opsd.trainer import SelfPlayTrainer

    kwargs = {"method": config, "study_id": study_id}
    if hardware:
        kwargs["hardware_profile"] = hardware
    cfg = load_config(settings, **kwargs)
    repository = TaskRepository(cfg.data_root)
    check_frozen(cfg, repository)
    report = checked_preflight(cfg)
    if not report["ready_for_pilot"]:
        raise ExperimentError("Preflight blocked launch: " + "; ".join(report["blockers"]))
    if repository.manifest.get("fixture_only"):
        raise ExperimentError("Fixture data is forbidden in main study runs")
    policy = make_policy(cfg)
    try:
        trainer = SelfPlayTrainer(
            cfg, policy, repository, SandboxBackend(cfg.sandbox), resume=resume
        )
        trainer.train()
    finally:
        policy.close()


@app.command()
def pilot(
    settings: ConfigOption,
    iterations: int = 10,
    resume: Path | None = None,
    hardware: str | None = None,
    config: str = "D",
):
    """Training-only target-hardware calibration; no final-test evaluation."""
    from selfplay_opsd.storage import active_events
    from selfplay_opsd.trainer import SelfPlayTrainer

    cfg = load_config(
        settings, method=config, **({"hardware_profile": hardware} if hardware else {})
    )
    cfg.study_id += "-pilot"
    cfg.iterations = iterations
    report = checked_preflight(cfg)
    if not report["ready_for_pilot"]:
        raise ExperimentError("Pilot preflight blocked: " + "; ".join(report["blockers"]))
    if cfg.method not in {"C", "D"}:
        raise ExperimentError("Calibration pilot must include C/D scoring; choose method D")
    policy = make_policy(cfg)
    try:
        trainer = SelfPlayTrainer(
            cfg, policy, TaskRepository(cfg.data_root), SandboxBackend(cfg.sandbox), resume=resume
        )
        from selfplay_opsd.pilot import training_reference_proxies

        with trainer.phase("pilot_reference_preparation"):
            trainer.reference_anchors = training_reference_proxies(trainer)
        # Development data must not influence training-only pilot settings.
        trainer.development_enabled = False
        trainer.train()
        if policy.is_coordinator:
            import numpy as np

            events = active_events(cfg.run_dir)
            phase_rows = [e for e in events if e["kind"] == "phase_time"]
            times: dict[int, float] = {}
            for row in phase_rows:
                times.setdefault(row["completed_iteration"], 0.0)
                times[row["completed_iteration"]] += row["payload"]["seconds"]
            values = [value for key, value in times.items() if key > 0]
            commits = [e["payload"] for e in events if e["kind"] == "iteration_commit"]
            alignment_count = sum(r["metrics"]["alignment"]["scored_count"] for r in commits)
            student_steps = sum(r["student_steps"] for r in commits)
            proposer_steps = sum(r["proposer_steps"] for r in commits)
            if alignment_count == 0 or student_steps == 0 or proposer_steps == 0:
                raise ExperimentError(
                    "Pilot did not exercise alignment and both real updates; inspect curriculum validity/difficulty"
                )
            pilot_report = {
                "iterations": trainer.iteration,
                "includes_alignment": True,
                "target_hardware_verified": True,
                "hardware": cfg.hardware.model_dump(),
                "pilot_config": cfg.model_dump(mode="json"),
                "manifest_hash": trainer.repository.hash,
                "median_iteration_seconds": float(np.median(values)),
                "p90_iteration_seconds": float(np.quantile(values, 0.9)),
                "projection_100_hours": float(np.median(values)) * 100 / 3600,
                "projection_200_hours": float(np.median(values)) * 200 / 3600,
                "phase_times": phase_rows,
                "lambda": cfg.objective.sd_lambda,
                "gradient_interactions": [
                    r["metrics"]["student"]["gradient_interaction"] for r in commits
                ],
                "scored_candidates": alignment_count,
                "reference_policy": "canonical proxies constructed from training parents only",
                "reference_proxy_hash": digest(trainer.reference_anchors),
                "student_steps": student_steps,
                "proposer_steps": proposer_steps,
                "calibration": "retained configured common lambda; automatic rescaling disabled",
                "provenance": provenance(),
            }
            atomic_json(cfg.run_dir / "pilot_report.json", pilot_report)
            typer.echo(canonical(pilot_report))
    finally:
        policy.close()


@app.command("freeze-study")
def freeze(settings: ConfigOption, pilot_report: Annotated[Path, typer.Option("--pilot-report")]):
    from selfplay_opsd.preflight import freeze_study

    typer.echo(canonical(freeze_study(load_config(settings), pilot_report)))


@app.command()
def evaluate(
    settings: ConfigOption,
    split: str = "dev",
    checkpoint: Path | None = None,
    endpoint: str = "primary",
    base: bool = False,
    protocol: str = "frozen",
    config: str | None = None,
    hardware: str | None = None,
):
    """Greedy frozen evaluation, guarded final endpoints and explicit base identity."""
    from selfplay_opsd.evaluation import evaluate_tasks
    from selfplay_opsd.preflight import check_frozen

    cfg = load_config(
        settings, method=config, **({"hardware_profile": hardware} if hardware else {})
    )
    if protocol != "frozen" or endpoint not in {"primary", "secondary"}:
        raise ValueError("Only frozen protocol and primary/secondary endpoints supported")
    split = "dev_monitor" if split == "dev" else split
    repository = TaskRepository(cfg.data_root)
    if split in {"final_id", "external"}:
        check_frozen(cfg, repository)
    if base and checkpoint is not None:
        raise ValueError("Base and checkpoint evaluation are mutually exclusive")
    if not base and checkpoint is None:
        raise ValueError("Supply --checkpoint or --base")
    sandbox = SandboxBackend(cfg.sandbox)
    sandbox.preflight()
    policy = make_policy(cfg)
    try:
        iteration = 0
        if checkpoint:
            state, metadata = CheckpointManager.load(checkpoint)
            if (
                metadata["manifests_hash"] != repository.hash
                or metadata["scientific_hash"] != cfg.scientific_hash
            ):
                raise ExperimentError("Evaluation checkpoint/config/data mismatch")
            if (
                split in {"final_id", "external"}
                and endpoint == "primary"
                and metadata["iteration"] != cfg.iterations
            ):
                raise ExperimentError(
                    "Primary final evaluation requires final common-budget checkpoint"
                )
            if endpoint == "secondary":
                final = cfg.run_dir / "checkpoints" / f"{cfg.iterations:06d}"
                if not final.exists():
                    raise ExperimentError(
                        "Secondary endpoint selection requires the completed common-budget run"
                    )
                final_metadata = CheckpointManager.verify(final)
                selected = final_metadata.get("best_development")
                if selected is None or selected[1] != metadata["iteration"]:
                    raise ExperimentError(
                        "Checkpoint is not the final run's predeclared best-development selection"
                    )
            policy.load_state_dict(state["policy"])
            iteration = metadata["iteration"]
        root = cfg.output_root / cfg.study_id / "base" if base else cfg.run_dir
        store = None
        if policy.is_coordinator:
            store = EventStore(
                root,
                {
                    "study_id": cfg.study_id,
                    "logical_run_id": digest(
                        [cfg.study_id, "base" if base else cfg.method, cfg.seed]
                    )[:24],
                    "config": "base" if base else cfg.method,
                    "seed": cfg.seed,
                },
            )
            store.iteration = iteration
            atomic_json(root / "provenance.json", provenance())
            atomic_json(
                root / "resolved_config.json",
                cfg.resolved() | {"evaluation_label": "base" if base else cfg.method},
            )
        with policy.active("student"):
            if base:
                # Initial PEFT B matrices are zero: verify the adapter is no-op.
                import torch

                if any(
                    torch.count_nonzero(p).item()
                    for n, p in policy.named_parameters("student")
                    if ".lora_B." in n
                ):
                    raise ExperimentError("Base evaluation adapter is not a no-op")
            rows = evaluate_tasks(
                policy,
                sandbox,
                repository,
                split,
                cfg.episode,
                iteration,
                cfg.seed,
                store,
                checkpoint="base" if base else str(checkpoint),
                endpoint="base" if base else endpoint,
            )
        if store:
            store.materialize()
            typer.echo(
                canonical(
                    {
                        "tasks": len(rows),
                        "successes": sum(bool(r.success) for r in rows),
                        "output": str(root),
                    }
                )
            )
    finally:
        policy.close()


@app.command()
def diagnose(
    settings: ConfigOption,
    checkpoint: Annotated[Path, typer.Option("--checkpoint")],
    output: Annotated[Path, typer.Option("--output")],
    config: str | None = typer.Option(None, "--config"),
    hardware: str | None = None,
):
    from selfplay_opsd.diagnostics import run_diagnostics

    cfg = load_config(
        settings, method=config, **({"hardware_profile": hardware} if hardware else {})
    )
    policy = make_policy(cfg)
    sandbox = SandboxBackend(cfg.sandbox)
    sandbox.preflight()
    try:
        state, metadata = CheckpointManager.load(checkpoint)
        repository = TaskRepository(cfg.data_root)
        if metadata["manifests_hash"] != repository.hash:
            raise ExperimentError("Diagnostic dataset differs from checkpoint")
        policy.load_state_dict(state["policy"])
        report = run_diagnostics(policy, sandbox, repository, cfg, output)
        if policy.is_coordinator:
            typer.echo(canonical(report))
    finally:
        policy.close()


@app.command()
def analyze(study: Path, seed: int = 2026):
    from selfplay_opsd.analysis import analyze as aggregate

    result = aggregate(study, seed=seed)
    typer.echo(canonical({"output": str(study / "analysis"), "runs": len(result["summary"])}))


@app.command("backend-check")
def backend_check(
    settings: ConfigOption,
    output: Annotated[Path, typer.Option("--output")],
    hardware: str | None = None,
    reference: Path | None = None,
):
    """Verify two weight reloads, gradients and checkpoint restoration on actual GPUs."""
    from selfplay_opsd.backend_check import check_backend

    cfg = load_config(settings, **({"hardware_profile": hardware} if hardware else {}))
    typer.echo(canonical(check_backend(cfg, output, reference=reference)))


@app.command("wandb-sync")
def wandb_sync(path: Annotated[Path, typer.Option("--path")]):
    """Explicitly upload a chosen offline segment using the pinned SDK CLI."""
    subprocess.run(["wandb", "sync", str(path)], check=True)


@app.command("wandb-export")
def wandb_export(
    run_dir: Annotated[Path, typer.Option("--run-dir")],
    output: Annotated[Path, typer.Option("--output")],
):
    """Export authoritative local events; no network or credentials required."""
    from selfplay_opsd.storage import active_events

    atomic_json(output, active_events(run_dir))
    typer.echo(str(output))


if __name__ == "__main__":
    app()
