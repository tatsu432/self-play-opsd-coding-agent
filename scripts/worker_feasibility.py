"""CPU-only trusted fixtures. Never accepts generated source or declares readiness."""

from __future__ import annotations

import argparse
import ctypes
import errno
import json
import os
import platform
import signal
import subprocess
import sys
from pathlib import Path

CASES = (
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
    "resources",
    "tkill",
    "memfd",
)


def fixture(name):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/selfplay_opsd"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    # Load trusted helper modules before the filter, never candidate modules.
    import collections
    import math
    import mmap
    import threading

    from worker_policy import install

    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    install()
    if name == "ordinary":
        assert sum(math.factorial(i) for i in range(6)) == 154
        assert collections.Counter("abcab")["a"] == 2
        print(json.dumps({"ok": True}))
        return
    if name == "resources":
        print(
            json.dumps(
                {
                    "ok": True,
                    "affinity": sorted(os.sched_getaffinity(0)),
                    "cgroup": Path("/proc/self/cgroup").read_text(),
                    "mounts": Path("/proc/self/mountinfo").read_text(),
                    "tmp_bytes": os.statvfs("/tmp").f_blocks * os.statvfs("/tmp").f_frsize,
                    "devices": os.listdir("/dev"),
                    "uid": os.getuid(),
                }
            )
        )
        return
    if name == "thread":
        try:
            t = threading.Thread(target=lambda: None)
            t.start()
            t.join()
        except RuntimeError:
            print(json.dumps({"ok": True}))
            return
        raise AssertionError("thread creation succeeded")
    if name == "compat_abi":
        # i386 getpid through int 0x80: safe even if filtering is broken.
        code = mmap.mmap(-1, 4096, prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC)
        code.write(bytes.fromhex("b814000000cd80c3"))
        call = ctypes.CFUNCTYPE(ctypes.c_long)(ctypes.addressof(ctypes.c_char.from_buffer(code)))
        call()
        raise AssertionError("compatibility ABI was not killed")
    number = {
        "fork": 57,
        "vfork": 58,
        "clone": 56,
        "clone3": 435,
        "exec": 59,
        "unshare": 272,
        "setns": 308,
        "ptrace": 101,
        "process_vm": 310,
        "io_uring": 425,
        "prctl": 157,
        "seccomp": 317,
        "network": 41,
        "x32": 0x40000027,
        "tkill": 200,
        "memfd": 319,
    }[name]
    # Invalid arguments prevent dangerous side effects even without a filter.
    # fork is the sole argument-free creation call: reap its one bounded child.
    if name == "vfork":
        # If a broken filter permits vfork, the native child immediately _exits
        # without touching Python/interpreter state shared with its parent.
        code = mmap.mmap(-1, 4096, prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC)
        code.write(bytes.fromhex("b83a0000000f054885c07509b83c00000031ff0f05c3"))
        call = ctypes.CFUNCTYPE(ctypes.c_long, use_errno=True)(
            ctypes.addressof(ctypes.c_char.from_buffer(code))
        )
        value = call()
        if value > 0:
            os.waitpid(value, 0)
        assert value == -errno.EPERM, (name, value)
        print(json.dumps({"ok": True}))
        return
    value = libc.syscall(number, ctypes.c_ulong(-1), 0, 0, 0, 0, 0)
    if name == "fork" and value == 0:
        os._exit(0)
    if name == "fork" and value > 0:
        os.waitpid(value, 0)
    assert value == -1 and ctypes.get_errno() == errno.EPERM, (name, value, ctypes.get_errno())
    print(json.dumps({"ok": True}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--image")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resource-oom", action="store_true")
    parser.add_argument("--hold-worker", action="store_true")
    args = parser.parse_args()
    if args.hold_worker:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from selfplay_opsd.codec import encode
        from selfplay_opsd.worker_runtime import LocalRuntime, RuntimeConfig, worker_scope

        config = RuntimeConfig(
            backend="apptainer",
            image=args.image,
            runtime="apptainer",
            resource_backend="slurm",
            process_policy="single_process",
        )
        child_env = {
            k: v
            for k, v in os.environ.items()
            if k in {"PATH", "LANG", "LC_ALL", "APPTAINER_CACHEDIR", "APPTAINER_TMPDIR"}
        }
        process = subprocess.Popen(
            LocalRuntime(config).command("termination-witness"),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=child_env,
        )
        assert process.stdin and process.stdout
        process.stdin.write(
            json.dumps(
                {
                    "source": 'import os,time\ndef hold():\n os.write(1,b"holding\\n"); time.sleep(60)\n',
                    "entry_point": "hold",
                    "inputs": [encode([])],
                }
            ).encode()
        )
        process.stdin.close()
        assert process.stdout.readline() == b"holding\n"
        members = {}
        for pid in worker_scope().read_text().split():
            fields = Path("/proc", pid, "stat").read_text().rsplit(")", 1)[1].split()
            members[pid] = fields[19]  # starttime, to distinguish PID reuse
        print(
            json.dumps(
                {
                    "holding": True,
                    "step_id": os.environ["SLURM_STEP_ID"],
                    "job_id": os.environ["SLURM_JOB_ID"],
                    "members": members,
                }
            ),
            flush=True,
        )
        process.wait(timeout=70)
        raise SystemExit("Termination witness was not cancelled")
    if args.resource_oom:
        membership = next(
            line.split(":", 2)[2].strip()
            for line in Path("/proc/self/cgroup").read_text().splitlines()
            if line.startswith("0::")
        )
        group = Path("/sys/fs/cgroup") / membership.lstrip("/")
        groups = [group, *group.parents]
        limits = [
            int((p / "memory.max").read_text())
            for p in groups
            if (p / "memory.max").is_file() and (p / "memory.max").read_text().strip() != "max"
        ]
        limit = min(limits)
        assert 0 < limit <= 1024**3, "No independent worker-sized memory bound"
        before = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, set(range(os.cpu_count())))
        after = sorted(os.sched_getaffinity(0))
        assert len(after) == 1 and after == before, "Affinity escaped the Slurm cpuset"
        print(
            json.dumps(
                {
                    "memory_limit_bytes": limit,
                    "cpuset_before": before,
                    "cpuset_after_expansion": after,
                    "cgroup": membership,
                    "next": "bounded OOM witness",
                }
            ),
            flush=True,
        )
        # At most 1.0625 GiB even if site enforcement is unexpectedly broken.
        # This fixture runs only in a dedicated 1-GiB step of an 8-GiB CPU job.
        witness = bytearray(limit + 64 * 1024**2)
        print(json.dumps({"enforced": False, "allocated": len(witness)}), flush=True)
        raise SystemExit(1)
    if args.case:
        fixture(args.case)
        return
    root = Path(__file__).resolve().parents[1]
    bindings = [
        f"{root / 'src/selfplay_opsd/worker_policy.py'}:/opt/worker_policy.py:ro",
        f"{Path(__file__).resolve()}:/opt/worker_feasibility.py:ro",
    ]
    command = [
        "apptainer",
        "exec",
        "--containall",
        "--cleanenv",
        "--no-home",
        "--no-mount",
        "hostfs,bind-paths",
        "--net",
        "--network",
        "none",
        "--userns",
        "--drop-caps",
        "all",
        "--bind",
        ",".join(bindings),
        args.image,
        "python",
        "-I",
        "/opt/worker_feasibility.py",
    ]
    report = {
        "diagnostic_only": True,
        "ready": False,
        "host": platform.node(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "cases": {},
    }
    for case in CASES:
        try:
            r = subprocess.run(
                command + ["--case", case],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
                start_new_session=True,
            )
            expected_kill = case in {"x32", "compat_abi"}
            report["cases"][case] = {
                "exit_code": r.returncode,
                "stdout": r.stdout[-20000:],
                "stderr": r.stderr[-2000:],
                "passed": (
                    r.returncode in {-signal.SIGSYS, 128 + signal.SIGSYS}
                    if expected_kill
                    else r.returncode == 0
                ),
            }
            if "FATAL:" in r.stderr:
                break  # Unsupported startup is shared by every fixture; do not repeat it.
        except subprocess.TimeoutExpired:
            report["cases"][case] = {"passed": False, "error": "startup_timeout"}
            break
    report["feasible"] = len(report["cases"]) == len(CASES) and all(
        row["passed"] for row in report["cases"].values()
    )
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
