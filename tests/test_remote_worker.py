import os
import secrets
import socket
import subprocess
import sys
import time

import httpx
import pytest

from selfplay_opsd.codec import encode
from selfplay_opsd.config import SandboxConfig
from selfplay_opsd.core import Case
from selfplay_opsd.sandbox import SandboxBackend


@pytest.mark.sandbox
def test_authenticated_remote_docker_worker(sandbox, monkeypatch, tmp_path):
    with socket.socket() as bound:
        bound.bind(("127.0.0.1", 0))
        port = bound.getsockname()[1]
    token = secrets.token_urlsafe(32)
    monkeypatch.setenv("SELFPLAY_WORKER_TOKEN", token)
    log = (tmp_path / "worker.log").open("w")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "selfplay_opsd.remote_worker",
            "--port",
            str(port),
            "--concurrency",
            "2",
        ],
        env=dict(os.environ),
        stdout=log,
        stderr=log,
    )
    try:
        url = f"http://127.0.0.1:{port}"
        ready = False
        for _ in range(150):
            if process.poll() is not None:
                raise AssertionError((tmp_path / "worker.log").read_text())
            try:
                response = httpx.get(
                    url + "/capabilities", headers={"Authorization": "Bearer " + token}, timeout=1
                )
                if response.status_code == 200:
                    ready = True
                    break
            except httpx.HTTPError:
                time.sleep(0.1)
        assert ready
        assert httpx.get(url + "/capabilities").status_code == 401
        remote = SandboxBackend(SandboxConfig(backend="remote", remote_url=url))
        assert remote.preflight()["ready"]
        assert remote.verify(
            "def f(x):\n return x + 1\n", "f", [Case(args=encode([1]), expected=encode(2))], 2
        ).passed
    finally:
        process.terminate()
        process.wait(timeout=10)
        log.close()
