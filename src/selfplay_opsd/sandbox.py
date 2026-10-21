"""Disposable execution. Candidate code never executes in this module/process."""

from __future__ import annotations

import json
import os
import selectors
import signal
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np

from selfplay_opsd.codec import decode
from selfplay_opsd.config import SandboxConfig
from selfplay_opsd.core import Case, ExperimentError, Status, canonical, digest
from selfplay_opsd.mutations import validate_source


@dataclass(frozen=True)
class Execution:
    status: Status
    results: list[dict]
    seconds: float
    detail: str = ""


@dataclass(frozen=True)
class Verification:
    status: Status
    passed: bool
    details: list[bool]
    semantic_failures: int
    runtime_failures: int
    seconds: float


def bounded_process(
    command: list[str], data: bytes, timeout: float, limit: int
) -> tuple[int, bytes, bytes, str]:
    """Drain both pipes concurrently, kill at a bound, and always reap the process."""
    with tempfile.TemporaryFile() as request:
        request.write(data)
        request.seek(0)
        process = subprocess.Popen(
            command,
            stdin=request,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        selector = selectors.DefaultSelector()
        assert process.stdout is not None and process.stderr is not None
        selector.register(process.stdout, selectors.EVENT_READ, 0)
        selector.register(process.stderr, selectors.EVENT_READ, 1)
        chunks: list[bytearray] = [bytearray(), bytearray()]
        deadline, reason = time.monotonic() + timeout, ""
        try:
            while selector.get_map():
                if time.monotonic() > deadline:
                    reason = "timeout"
                    break
                for key, _ in selector.select(
                    timeout=min(0.1, max(0, deadline - time.monotonic()))
                ):
                    chunk = os.read(key.fd, 8192)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    chunks[key.data].extend(chunk)
                    if sum(map(len, chunks)) > limit:
                        reason = "output_limit"
                        break
                if reason:
                    break
        finally:
            selector.close()
            if not reason:
                try:
                    process.wait(timeout=max(0.1, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    reason = "timeout"
            if reason:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    reason = reason or "process_exited"
            process.wait()
            process.stdout.close()
            process.stderr.close()
        return process.returncode, bytes(chunks[0][:limit]), bytes(chunks[1][:limit]), reason


class SandboxBackend:
    def __init__(self, config: SandboxConfig):
        self.config = config
        self.attestation: dict | None = None
        self._image_id: str | None = None
        self.infrastructure_retries = 0

    def identity(self) -> str:
        if self._image_id is not None:
            return self._image_id
        if self.config.backend == "docker":
            result = subprocess.run(
                [self.config.runtime, "image", "inspect", self.config.image, "--format", "{{.Id}}"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode:
                raise ExperimentError(
                    f"Worker image unavailable: {result.stderr.strip()}", Status.DEPENDENCY
                )
            self._image_id = result.stdout.strip()
            return self._image_id
        if self.config.backend == "apptainer":
            from selfplay_opsd.storage import file_hash

            return "sha256:" + file_hash(Path(self.config.image))
        if not self.config.remote_url:
            raise ExperimentError("Remote worker URL is required")
        response = httpx.get(
            self.config.remote_url.rstrip("/") + "/capabilities",
            headers=self._headers(),
            timeout=10,
        )
        response.raise_for_status()
        return response.json()["image_digest"]

    def _headers(self) -> dict:
        token = os.getenv(self.config.remote_token_env)
        if not token:
            raise ExperimentError(
                f"Missing remote worker token in {self.config.remote_token_env}", Status.DEPENDENCY
            )
        return {"Authorization": f"Bearer {token}"}

    def _command(self, name: str) -> list[str]:
        c = self.config
        if c.backend == "docker":
            return [
                c.runtime,
                "run",
                "--rm",
                "-i",
                "--name",
                name,
                "--network=none",
                "--read-only",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                "--user=65534:65534",
                f"--cpus={c.cpus}",
                f"--memory={c.memory_mib}m",
                f"--memory-swap={c.memory_mib}m",
                f"--pids-limit={c.pids}",
                "--ulimit",
                "core=0",
                "--ulimit",
                f"fsize={c.output_bytes}",
                "--ulimit",
                "nofile=64:64",
                "--tmpfs",
                f"/tmp:rw,noexec,nosuid,nodev,size={c.disk_mib}m",
                self.identity(),
            ]
        if c.backend == "apptainer":
            # User/network namespaces and cgroups must actually be permitted by
            # the site. No --nv: execution workers never receive GPU devices.
            return [
                c.runtime,
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
                "--memory",
                str(c.memory_mib * 1024 * 1024),
                "--cpus",
                str(c.cpus),
                "--pids-limit",
                str(c.pids),
                c.image,
                "python",
                "-I",
                "-u",
                "/opt/worker.py",
            ]
        raise ValueError("Remote worker has no local execution command")

    def execute(
        self, source: str, entry_point: str, inputs: list, seconds: float, *, probe: bool = False
    ) -> Execution:
        for attempt in range(self.config.retries + 1):
            result = self._execute_once(source, entry_point, inputs, seconds, probe=probe)
            if result.status != Status.INFRA or attempt == self.config.retries:
                return result
            self.infrastructure_retries += 1
            time.sleep(0.1 * (attempt + 1))
        raise AssertionError("Unreachable retry state")

    def _execute_once(
        self, source: str, entry_point: str, inputs: list, seconds: float, *, probe: bool = False
    ) -> Execution:
        if not probe:
            validate_source(source)
        payload = {"source": source, "entry_point": entry_point, "inputs": inputs}
        raw = canonical(payload).encode()
        if len(raw) > 8 * 1024 * 1024:
            raise ExperimentError("Worker input exceeds admitted task limit", Status.ORACLE)
        if self.config.backend == "remote":
            if not self.config.remote_url:
                raise ExperimentError("Remote URL required")
            try:
                response = httpx.post(
                    self.config.remote_url.rstrip("/") + "/execute",
                    json={"request": payload, "seconds": seconds, "probe": probe},
                    headers=self._headers(),
                    timeout=seconds + 30,
                )
                response.raise_for_status()
                value = response.json()
                return Execution(
                    Status(value["status"]),
                    value["results"],
                    value["seconds"],
                    value.get("detail", ""),
                )
            except httpx.HTTPError as exc:
                return Execution(Status.INFRA, [], 0, str(exc))
        name = "selfplay-" + uuid.uuid4().hex
        started = time.monotonic()
        try:
            # Include startup outside code's calibrated allowance; Docker startup
            # is measured separately in the pilot. Overall host timeout is bounded.
            code, stdout, stderr, reason = bounded_process(
                self._command(name), raw, seconds + 5, self.config.output_bytes
            )
        except OSError as exc:
            return Execution(Status.DEPENDENCY, [], time.monotonic() - started, str(exc))
        finally:
            if self.config.backend == "docker":
                # Killing the client is insufficient: explicitly reap its container.
                subprocess.run(
                    [self.config.runtime, "rm", "--force", name],
                    capture_output=True,
                    timeout=15,
                    check=False,
                )
        elapsed = time.monotonic() - started
        if reason in {"timeout", "output_limit"}:
            return Execution(Status.STUDENT_FAILURE, [], elapsed, reason)
        if (
            (
                code != 0
                and self.config.backend == "docker"
                and any(
                    marker in stderr
                    for marker in (
                        b"Cannot connect",
                        b"Error response from daemon",
                        b"error during connect",
                    )
                )
            )
            or code in {125, 126, 127}
            or (code != 0 and self.config.backend == "apptainer" and b"FATAL" in stderr)
        ):
            return Execution(Status.INFRA, [], elapsed, stderr.decode(errors="replace")[:1000])
        if code != 0:
            return Execution(Status.STUDENT_FAILURE, [], elapsed, "candidate_or_resource_exit")
        try:
            result = json.loads(stdout)
            if result["status"] != "ok":
                return Execution(Status.STUDENT_FAILURE, [], elapsed, result["status"])
            rows = result["results"]
            if len(rows) != len(inputs) or any(
                set(row) not in ({"ok", "value", "seconds"}, {"ok", "error", "seconds"})
                for row in rows
            ):
                raise ValueError("Malformed worker result")
            for row in rows:
                if row["ok"]:
                    decode(row["value"])
            return Execution(Status.OK, rows, elapsed)
        except (ValueError, KeyError, TypeError) as exc:
            return Execution(
                Status.STUDENT_FAILURE, [], elapsed, f"malformed_candidate_output:{exc}"
            )

    def verify(
        self, source: str, entry_point: str, cases: list[Case], seconds: float
    ) -> Verification:
        result = self.execute(source, entry_point, [c.args for c in cases], seconds)
        if result.status in {Status.INFRA, Status.DEPENDENCY, Status.ORACLE}:
            raise ExperimentError(result.detail, result.status)
        if result.status != Status.OK:
            return Verification(
                result.status, False, [False] * len(cases), 0, len(cases), result.seconds
            )
        checks = [
            compare(case, row["value"]) if row["ok"] else False
            for case, row in zip(cases, result.results, strict=True)
        ]
        runtime = sum(not row["ok"] for row in result.results)
        return Verification(
            Status.OK if all(checks) else Status.STUDENT_FAILURE,
            bool(cases) and all(checks),
            checks,
            sum(not c for c in checks) - runtime,
            runtime,
            result.seconds,
        )

    def preflight(self) -> dict:
        from selfplay_opsd.codec import encode

        image = self.identity()
        probes = {
            "no_network": "import socket\ndef check():\n try:\n  socket.create_connection(('169.254.169.254',80),timeout=0.2); return False\n except OSError: return True\n",
            "no_host_credentials": "import os\ndef check():\n return not any(os.path.exists(p) for p in ['/root/.aws/credentials','/root/.ssh','/home/trainer','/var/run/docker.sock']) and not any(k in os.environ for k in ['AWS_ACCESS_KEY_ID','WANDB_API_KEY','HF_TOKEN','SELFPLAY_WORKER_TOKEN'])\n",
            "readonly_root": "def check():\n try:\n  open('/opt/escape','w').write('x'); return False\n except OSError: return True\n",
            "no_gpu": "import os\ndef check():\n return not any(n.startswith('nvidia') for n in os.listdir('/dev'))\n",
            "nonroot": "import os\ndef check():\n return os.geteuid()!=0\n",
            "pids_limit": f"def check():\n try:\n  value=open('/sys/fs/cgroup/pids.max').read().strip()\n  return value!='max' and int(value)<={self.config.pids}\n except OSError: return False\n",
            "memory_limit": f"def check():\n try:\n  value=open('/sys/fs/cgroup/memory.max').read().strip()\n  return value!='max' and int(value)<={self.config.memory_mib * 1024**2}\n except OSError: return False\n",
            "cpu_limit": f"def check():\n try:\n  quota,period=open('/sys/fs/cgroup/cpu.max').read().split()\n  return quota!='max' and int(quota)/int(period)<={self.config.cpus}\n except OSError: return False\n",
            "disk_limit": f"import os\ndef check():\n v=os.statvfs('/tmp'); return v.f_blocks*v.f_frsize<={self.config.disk_mib * 1024**2}\n",
        }
        outcomes = {}
        for name, source in probes.items():
            outcome = self.execute(source, "check", [encode([])], 5, probe=True)
            outcomes[name] = (
                outcome.status == Status.OK
                and bool(outcome.results)
                and outcome.results[0].get("value") is True
            )
        self.attestation = {
            "image_digest": image,
            "config_hash": digest(self.config.model_dump()),
            "checks": outcomes,
            "ready": all(outcomes.values()),
            "timestamp": time.time(),
            "threat_model": "OS isolation; expected outputs remain outside candidate process; hidden inputs are observable while executing",
        }
        if not self.attestation["ready"]:
            raise ExperimentError(f"Sandbox isolation preflight failed: {self.attestation}")
        return self.attestation


def compare(case: Case, encoded_output) -> bool:
    """Trusted comparison over safe native data; never call candidate Python methods."""
    try:
        output, expected, args = decode(encoded_output), decode(case.expected), decode(case.args)
        if case.comparison == "non_none":
            return (output if type(output) is bool else output is not None) == expected
        if case.comparison == "set":
            return set(output) == set(expected)
        if case.comparison == "sum_div":
            return output == expected or output == 0
        if case.comparison == "surface_area":
            from evalplus.eval._special_oracle import _surface_Area

            return output == expected or abs(output - _surface_Area(*args)) <= case.atol
        if case.comparison == "digit_distance":
            from evalplus.eval._special_oracle import _digit_distance_nums

            return output == expected or output == _digit_distance_nums(*args)
        if case.comparison == "polynomial_root":
            return abs(sum(c * output**i for i, c in enumerate(args[0]))) <= case.atol
        if output == expected:
            return True
        floats = isinstance(expected, float) or (
            isinstance(expected, (list, tuple))
            and bool(expected)
            and all(isinstance(x, float) for x in expected)
        )
        atol = case.atol or (1e-6 if floats else 0)
        return bool(
            atol
            and type(output) is type(expected)
            and np.allclose(output, expected, rtol=1e-7, atol=atol)
        )
    except (ValueError, TypeError, OverflowError, ArithmeticError):
        return False
