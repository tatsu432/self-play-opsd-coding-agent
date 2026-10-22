"""Standard-library container supervisor, also used outside the trainer image."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import selectors
import signal
import socket
import subprocess
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from selfplay_opsd.codec import decode, encode


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def supervisor_hash() -> str:
    package = Path(__file__).parent
    return digest(
        {
            name: hashlib.sha256((package / name).read_bytes()).hexdigest()
            for name in ("worker_runtime.py", "remote_worker.py", "codec.py")
        }
    )


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def publish_json(path: Path, value: dict):
    """Readers see the previous report or the complete replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x") as stream:
            stream.write(canonical(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass
class RuntimeConfig:
    backend: str = "docker"
    image: str = "selfplay-worker:local"
    runtime: str = "docker"
    cpus: int = 1
    memory_mib: int = 1024
    pids: int = 32
    output_bytes: int = 4198400
    disk_mib: int = 64
    retries: int = 2
    public_seconds: float = 10.0
    verifier_seconds: float = 30.0
    remote_token_env: str = "SELFPLAY_WORKER_TOKEN"

    @classmethod
    def from_values(cls, values: dict):
        names = {f.name for f in fields(cls)}
        config = cls(**{k: v for k, v in values.items() if k in names})
        if config.backend not in {"docker", "apptainer"}:
            raise ValueError("Worker service requires Docker or Apptainer isolation")
        for name in ("cpus", "memory_mib", "pids", "output_bytes", "disk_mib"):
            if type(getattr(config, name)) is not int or getattr(config, name) <= 0:
                raise ValueError(f"Invalid worker limit: {name}")
        for name in ("public_seconds", "verifier_seconds"):
            if not math.isfinite(getattr(config, name)) or getattr(config, name) <= 0:
                raise ValueError(f"Invalid worker limit: {name}")
        if not 0 <= config.retries <= 5:
            raise ValueError("Invalid retry count")
        return config

    def model_dump(self, **_) -> dict:
        return asdict(self)


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


def isolation_probes(config) -> dict[str, str]:
    return {
        "network_namespace": "import socket,fcntl,struct\ndef check():\n s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)\n try:\n  return all(name=='lo' or not (struct.unpack('H',fcntl.ioctl(s.fileno(),0x8913,struct.pack('256s',name.encode()))[16:18])[0]&1) for _,name in socket.if_nameindex())\n finally: s.close()\n",
        "no_network": "import socket\ndef check():\n try:\n  socket.create_connection(('169.254.169.254',80),timeout=0.2); return False\n except OSError: return True\n",
        "no_host_credentials": "import os\ndef check():\n return not any(os.path.exists(p) for p in ['/root/.aws/credentials','/root/.ssh','/home/trainer','/var/run/docker.sock']) and not any(k in os.environ for k in ['AWS_ACCESS_KEY_ID','WANDB_API_KEY','HF_TOKEN','SELFPLAY_WORKER_TOKEN'])\n",
        "readonly_root": "def check():\n try:\n  open('/opt/escape','w').write('x'); return False\n except OSError: return True\n",
        "no_gpu": "import os\ndef check():\n return not any(n.startswith('nvidia') for n in os.listdir('/dev'))\n",
        "nonroot": "import os\ndef check():\n return os.geteuid()!=0\n",
        "pids_limit": f"def check():\n try:\n  value=open('/sys/fs/cgroup/pids.max').read().strip()\n  return value!='max' and int(value)<={config.pids}\n except OSError: return False\n",
        "memory_limit": f"def check():\n try:\n  value=open('/sys/fs/cgroup/memory.max').read().strip()\n  return value!='max' and int(value)<={config.memory_mib * 1024**2}\n except OSError: return False\n",
        "cpu_limit": f"def check():\n try:\n  quota,period=open('/sys/fs/cgroup/cpu.max').read().split()\n  return quota!='max' and int(quota)/int(period)<={config.cpus}\n except OSError: return False\n",
        "disk_limit": f"import os\ndef check():\n v=os.statvfs('/tmp'); return v.f_blocks*v.f_frsize<={config.disk_mib * 1024**2}\n",
    }


class LocalRuntime:
    def __init__(self, config):
        self.config = config
        self._image_id = None

    def identity(self) -> str:
        if self._image_id is None:
            if self.config.backend == "docker":
                self._image_id = subprocess.run(
                    [
                        self.config.runtime,
                        "image",
                        "inspect",
                        self.config.image,
                        "--format",
                        "{{.Id}}",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=30,
                ).stdout.strip()
            else:
                h = hashlib.sha256()
                with Path(self.config.image).open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        h.update(chunk)
                self._image_id = "sha256:" + h.hexdigest()
        return self._image_id

    def command(self, name: str) -> list[str]:
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
    ) -> dict:
        raw = canonical({"source": source, "entry_point": entry_point, "inputs": inputs}).encode()
        if not math.isfinite(seconds) or not 0 < seconds <= self.config.verifier_seconds:
            raise ValueError("Invalid worker time limit")
        if len(raw) > 8 * 1024 * 1024:
            raise ValueError("Worker request exceeds limit")

        def result(status, results, elapsed, detail=""):
            return {"status": status, "results": results, "seconds": elapsed, "detail": detail}

        name = "selfplay-" + uuid.uuid4().hex
        started = time.monotonic()
        try:
            # Include startup outside code's calibrated allowance; Docker startup
            # is measured separately in the pilot. Overall host timeout is bounded.
            code, stdout, stderr, reason = bounded_process(
                self.command(name), raw, seconds + 5, self.config.output_bytes
            )
        except OSError as exc:
            return result("missing_dependency", [], time.monotonic() - started, str(exc))
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
            return result("student_failure", [], elapsed, reason)
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
            return result(
                "sandbox_infrastructure_error", [], elapsed, stderr.decode(errors="replace")[:1000]
            )
        if code != 0:
            return result("student_failure", [], elapsed, "candidate_or_resource_exit")
        try:
            value = json.loads(stdout)
            if value["status"] != "ok":
                return result("student_failure", [], elapsed, value["status"])
            rows = value["results"]
            if len(rows) != len(inputs) or any(
                set(row) not in ({"ok", "value", "seconds"}, {"ok", "error", "seconds"})
                for row in rows
            ):
                raise ValueError("Malformed worker result")
            for row in rows:
                if row["ok"]:
                    decode(row["value"])
            return result("ok", rows, elapsed)
        except (ValueError, KeyError, TypeError) as exc:
            return result("student_failure", [], elapsed, f"malformed_candidate_output:{exc}")

    def preflight(self) -> dict:
        outcomes, details = {}, {}
        for name, source in isolation_probes(self.config).items():
            result = self.execute(source, "check", [encode([])], 5, probe=True)
            outcomes[name] = (
                result["status"] == "ok"
                and bool(result["results"])
                and result["results"][0].get("value") is True
            )
            if not outcomes[name]:
                details[name] = result.get("detail") or str(result.get("results"))[:1000]
        report = {
            "ready": all(outcomes.values()),
            "host": socket.gethostname(),
            "kernel": platform.release(),
            "uid": os.getuid(),
            "checks": outcomes,
            "details": details,
            "image_digest": self.identity(),
            "config_hash": digest(self.config.model_dump()),
            "supervisor_hash": supervisor_hash(),
            "timestamp": time.time(),
            "limits": self.config.model_dump(),
        }
        return report
