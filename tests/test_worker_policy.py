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


def test_slurm_runtime_requires_explicit_single_process_contract():
    from selfplay_opsd.worker_runtime import RuntimeConfig

    with pytest.raises(ValueError, match="Slurm worker requires"):
        RuntimeConfig.from_values({"backend": "apptainer", "resource_backend": "slurm"})


def test_slurm_worker_command_cannot_request_nested_cgroups_or_mount_trainer():
    from selfplay_opsd.worker_runtime import LocalRuntime, RuntimeConfig

    config = RuntimeConfig(
        backend="apptainer",
        resource_backend="slurm",
        process_policy="single_process",
        image="/worker.sif",
    )
    command = LocalRuntime(config).command("fixture")
    assert "--cpus" not in command and "--memory" not in command and "--pids-limit" not in command
    assert "hostfs,bind-paths,dev" in command
    bindings = command[command.index("--bind") + 1].split(",")
    assert len(bindings) == 3 and all(b.endswith(":ro") for b in bindings)
    assert all("/opt/" in b for b in bindings)


@pytest.mark.parametrize(
    "cpuset,writable,expected", [("2", False, True), ("0-7", False, False), ("2", True, False)]
)
def test_cpu_probe_requires_enforced_cpuset(cpuset, writable, expected, monkeypatch):
    import io
    import os

    from selfplay_opsd.worker_runtime import RuntimeConfig, isolation_probes

    files = {
        "/proc/self/cgroup": "0::/job/worker\n",
        "/proc/self/mountinfo": "1 0 0:42 / /cg rw - cgroup2 cgroup rw\n",
        "/cg/job/worker/cpu.max": "max 100000",
        "/cg/job/worker/cpuset.cpus.effective": cpuset,
    }
    monkeypatch.setattr(
        "builtins.open", lambda filename, *a, **k: io.StringIO(files[str(filename)])
    )
    monkeypatch.setattr(os.path, "isfile", lambda filename: str(filename) in files)
    monkeypatch.setattr(os, "access", lambda *a: writable)
    namespace = {}
    exec(isolation_probes(RuntimeConfig())["cpu_limit"], namespace)
    assert namespace["check"]() is expected
