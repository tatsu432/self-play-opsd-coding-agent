import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from selfplay_opsd.cli import app
from selfplay_opsd.config import PROFILES, load_config


def launcher():
    spec = importlib.util.spec_from_file_location("launcher", Path("scripts/launch.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("profile", [p for p in PROFILES if p != "cpu_debug"])
@pytest.mark.parametrize("environment", ["aws", "local", "yale"])
def test_all_launcher_profiles(profile, environment, monkeypatch):
    count = PROFILES[profile]["gpu_count"]
    monkeypatch.setenv("SLURM_JOB_ID", "test-fixture")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(str(i) for i in range(count)))
    monkeypatch.setenv("SELFPLAY_WORKER_TOKEN", "test-secret-not-an-actual-credential")
    args = SimpleNamespace(
        hardware=profile,
        environment=environment,
        settings="configs/main.yaml",
        operation="train",
        method="D",
        resume=None,
        image="trainer-fixture.sif",
        iterations=10,
        dry_run=True,
    )
    command = launcher().command(args)
    assert f"--nproc-per-node={count}" in command
    assert command[-1] == profile
    assert not any("test-secret" in part for part in command)
    assert "/var/run/docker.sock" not in str(command)
    assert (
        load_config(Path("configs/main.yaml"), hardware_profile=profile).hardware.gpu_count == count
    )


def test_documented_cli_options():
    runner = CliRunner()
    for name, options in {
        "backend-check": ["--output", "--hardware", "--reference"],
        "freeze-study": ["--pilot-report"],
        "diagnose": ["--output", "--checkpoint"],
        "wandb-sync": ["--path"],
        "wandb-export": ["--run-dir", "--output"],
    }.items():
        result = runner.invoke(app, [name, "--help"])
        assert result.exit_code == 0
        for option in options:
            assert option in result.output
