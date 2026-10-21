import subprocess

import pytest
import torch

from selfplay_opsd.config import SandboxConfig, load_config
from selfplay_opsd.policy import HFPolicy
from selfplay_opsd.sandbox import SandboxBackend
from selfplay_opsd.smoke import fixture_repository


@pytest.fixture(scope="session", autouse=True)
def small_torch_threads():
    torch.set_num_threads(1)


@pytest.fixture(scope="session")
def sandbox():
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", "selfplay-worker:local"], capture_output=True, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        pytest.skip("Docker worker unavailable; build docker/worker.Dockerfile")
    if result.returncode:
        pytest.skip("Docker worker image unavailable")
    return SandboxBackend(SandboxConfig())


@pytest.fixture(scope="session")
def repository(tmp_path_factory, sandbox):
    cfg = load_config(study_id="fixture-build", hardware_profile="cpu_debug")
    policy = HFPolicy(cfg)
    return fixture_repository(tmp_path_factory.mktemp("fixtures"), sandbox, policy)
