"""Real kernel tests run in disposable child interpreters, never the pytest host."""

import platform
import signal
import subprocess
import sys

import pytest

from selfplay_opsd.worker_policy import install


@pytest.mark.skipif(platform.system() == "Linux", reason="Non-Linux fail-closed path")
def test_policy_rejects_unsupported_host():
    with pytest.raises(RuntimeError, match="Linux x86_64"):
        install()


@pytest.mark.skipif(
    platform.system() != "Linux" or platform.machine() != "x86_64",
    reason="Requires real Linux AMD64 seccomp kernel",
)
@pytest.mark.parametrize(
    "case",
    [
        "ordinary",
        "fork",
        "vfork",
        "clone",
        "clone3",
        "thread",
        "exec",
        "unshare",
        "setns",
        "ptrace",
        "process_vm",
        "io_uring",
        "prctl",
        "seccomp",
        "network",
        "x32",
        "compat_abi",
    ],
)
def test_kernel_single_process_policy(case):
    result = subprocess.run(
        [sys.executable, "scripts/worker_feasibility.py", "--case", case],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    expected = -signal.SIGSYS if case in {"x32", "compat_abi"} else 0
    assert result.returncode == expected, result.stderr
