"""Standard-library container supervisor, also used outside the trainer image."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import platform
import re
import selectors
import signal
import socket
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from selfplay_opsd.codec import decode, encode

ALLOCATION_FUNCTIONS = (
    "sha256",
    "write",
    "diagnostic_command",
    "device_access",
    "allocated_gpus",
    "resolve_gpus",
    "verify_container_gpus",
)


def allocation_fingerprint(path: Path) -> str:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        # The controller uses Python 3.12; Yale's standard-library endpoint uses
        # 3.9. Empty PEP 695 fields must not change the same verifier's identity.
        if hasattr(node, "type_params") and not node.type_params:
            node.__dict__["_fields"] = tuple(name for name in node._fields if name != "type_params")
    definitions = {
        n.name: ast.dump(n, include_attributes=False)
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in ALLOCATION_FUNCTIONS
    }
    if set(definitions) != set(ALLOCATION_FUNCTIONS):
        raise ValueError("Allocation verification implementation missing")
    return digest(definitions)


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def supervisor_hash() -> str:
    package = Path(__file__).parent
    return digest(
        {
            name: hashlib.sha256((package / name).read_bytes()).hexdigest()
            for name in (
                "worker_runtime.py",
                "remote_worker.py",
                "codec.py",
                "worker.py",
                "worker_policy.py",
            )
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
    process_policy: str = "pids"
    resource_backend: str = "container"
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
        if config.process_policy not in {
            "pids",
            "single_process",
        } or config.resource_backend not in {"container", "slurm"}:
            raise ValueError("Invalid worker execution contract")
        if config.resource_backend == "slurm" and (
            config.backend != "apptainer" or config.process_policy != "single_process"
        ):
            raise ValueError("Slurm worker requires Apptainer and single-process enforcement")
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
    command: list[str],
    data: bytes,
    timeout: float,
    limit: int,
    cancelled: threading.Event | None = None,
    env: dict | None = None,
    cleanup_scope: Path | None = None,
) -> tuple[int, bytes, bytes, str]:
    """Drain both pipes concurrently, kill at a bound, and always reap the process."""
    baseline = set(cleanup_scope.read_text().split()) if cleanup_scope else set()
    with tempfile.TemporaryFile() as request:
        request.write(data)
        request.seek(0)
        process = subprocess.Popen(
            command,
            stdin=request,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            env=env,
        )
        selector = selectors.DefaultSelector()
        assert process.stdout is not None and process.stderr is not None
        selector.register(process.stdout, selectors.EVENT_READ, 0)
        selector.register(process.stderr, selectors.EVENT_READ, 1)
        chunks: list[bytearray] = [bytearray(), bytearray()]
        deadline, reason = time.monotonic() + timeout, ""
        try:
            while selector.get_map():
                if cancelled is not None and cancelled.is_set():
                    reason = "cancelled"
                    break
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
            while not reason and process.poll() is None:
                if cancelled is not None and cancelled.is_set():
                    reason = "cancelled"
                    break
                if time.monotonic() >= deadline:
                    reason = "timeout"
                    break
                try:
                    process.wait(timeout=min(0.1, max(0.001, deadline - time.monotonic())))
                except subprocess.TimeoutExpired:
                    pass
            if reason:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    reason = reason or "process_exited"
            process.wait()
            process.stdout.close()
            process.stderr.close()
            if cleanup_scope:
                deadline = time.monotonic() + 2
                remaining = set(cleanup_scope.read_text().split()) - baseline
                while remaining and time.monotonic() < deadline:
                    time.sleep(0.05)
                    remaining = set(cleanup_scope.read_text().split()) - baseline
                if remaining:
                    for pid in remaining:
                        try:
                            os.kill(int(pid), signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    deadline = time.monotonic() + 2
                    while remaining and time.monotonic() < deadline:
                        time.sleep(0.05)
                        remaining = set(cleanup_scope.read_text().split()) - baseline
                    if remaining:
                        reason = "cleanup_failure"
        return process.returncode, bytes(chunks[0][:limit]), bytes(chunks[1][:limit]), reason


def worker_scope() -> Path:
    membership = next(
        line.split(":", 2)[2].strip()
        for line in Path("/proc/self/cgroup").read_text().splitlines()
        if line.startswith("0::")
    )
    if not re.search(r"/step_[0-9]+/", membership):
        raise OSError("Worker must run in a dedicated Slurm step")
    path = Path("/sys/fs/cgroup") / membership.lstrip("/") / "cgroup.procs"
    if str(os.getpid()) not in path.read_text().split():
        raise OSError("Worker cgroup membership cannot be verified")
    return path


def cgroup_probe(expression: str) -> str:
    """Resolve membership and mount roots; containers need not mount their cgroup at /."""
    return (
        "import os\n"
        "def check():\n"
        " try:\n"
        "  membership=next(line.strip().split(':',2)[2] for line in open('/proc/self/cgroup') if line.startswith('0::'))\n"
        "  mount=next(line.split(' - ',1)[0].split() for line in open('/proc/self/mountinfo') if ' - cgroup2 ' in line)\n"
        "  root,point=mount[3:5]\n"
        "  if root!='/' and membership!=root and not membership.startswith(root+'/'): return False\n"
        "  relative=membership[len(root):] if root!='/' else membership\n"
        "  current=os.path.normpath(point+'/'+relative.lstrip('/'))\n"
        "  if os.path.commonpath([point,current])!=point: return False\n"
        "  groups=[]\n"
        "  while True:\n"
        "   groups.append(current)\n"
        "   if current==point: break\n"
        "   current=os.path.dirname(current)\n"
        "  def values(name):\n"
        "   return [open(group+'/'+name).read().strip() for group in groups if os.path.isfile(group+'/'+name)]\n"
        "  def cpuset_size(value):\n"
        "   return sum(int(x.split('-')[-1])-int(x.split('-')[0])+1 for x in value.split(',') if x)\n"
        f"  return {expression}\n"
        " except (OSError,ValueError,StopIteration): return False\n"
    )


def isolation_probes(config) -> dict[str, str]:
    probes = {
        "network_namespace": "import socket,fcntl,struct\ndef check():\n s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)\n try:\n  return all(name=='lo' or not (struct.unpack('H',fcntl.ioctl(s.fileno(),0x8913,struct.pack('256s',name.encode()))[16:18])[0]&1) for _,name in socket.if_nameindex())\n finally: s.close()\n",
        "no_network": "import socket\ndef check():\n try:\n  socket.create_connection(('169.254.169.254',80),timeout=0.2); return False\n except OSError: return True\n",
        "no_host_credentials": "import os\ndef check():\n return not any(os.path.exists(p) for p in ['/root/.aws/credentials','/root/.ssh','/home/trainer','/var/run/docker.sock']) and not any(k in os.environ for k in ['AWS_ACCESS_KEY_ID','WANDB_API_KEY','HF_TOKEN','SELFPLAY_WORKER_TOKEN'])\n",
        "readonly_root": "def check():\n try:\n  open('/opt/escape','w').write('x'); return False\n except OSError: return True\n",
        "no_gpu": "import os\ndef check():\n return not any(n.startswith('nvidia') for n in os.listdir('/dev'))\n",
        "nonroot": "import os\ndef check():\n return os.geteuid()!=0\n",
        "pids_limit": cgroup_probe(
            f"any(v!='max' and int(v)<={config.pids} for v in values('pids.max'))"
        ),
        "memory_limit": cgroup_probe(
            f"any(v!='max' and int(v)<={config.memory_mib * 1024**2} for v in values('memory.max'))"
        ),
        "cpu_limit": cgroup_probe(
            f"any(v.split()[0]!='max' and int(v.split()[0])/int(v.split()[1])<={config.cpus} for v in values('cpu.max')) or (any(0<cpuset_size(v)<={config.cpus} for v in values('cpuset.cpus.effective')) and not any(os.access(group+'/cgroup.procs',os.W_OK) for group in groups))"
        ),
        "cgroup_readonly": cgroup_probe(
            "not any(os.access(group+'/cgroup.procs',os.W_OK) for group in groups)"
        ),
        "disk_limit": f"import os\ndef check():\n v=os.statvfs('/tmp'); return v.f_blocks*v.f_frsize<={config.disk_mib * 1024**2}\n",
    }
    if config.process_policy == "single_process":
        probes.pop("pids_limit")
        probes["network_namespace"] = (
            "def check():\n return all(line.split(':')[0].strip()=='lo' for line in open('/proc/net/dev').readlines()[2:])\n"
        )
        probes["single_process"] = (
            "import ctypes,errno,os\ndef check():\n"
            " c=ctypes.CDLL(None,use_errno=True); c.syscall.restype=ctypes.c_long\n"
            " for n in [56,57,435,59,101,157,272,308,310,311,317,425,426,427]:\n"
            "  r=c.syscall(n,ctypes.c_ulong(-1),0,0,0,0,0)\n"
            "  if n==57 and r==0: os._exit(0)\n"
            "  if n==57 and r>0: os.waitpid(r,0)\n"
            "  if r!=-1 or ctypes.get_errno()!=errno.EPERM: return False\n"
            " return True\n"
        )
        probes["readonly_devices"] = (
            "import os\ndef check():\n try:\n  f=open('/dev/selfplay-escape','w'); f.close(); return False\n except OSError: return True\n"
        )
        probes["temporary_storage_enforced"] = (
            "import os,errno\ndef check():\n files=[]\n"
            " try:\n"
            "  for directory in ['/tmp','/var/tmp']:\n"
            "   f=open(directory+'/selfplay-disk-probe','wb',buffering=0); files.append(f)\n"
            f"  for i in range({config.disk_mib + 1}): files[i%2].write(bytes(1024*1024))\n"
            "  return False\n"
            " except OSError as e: return e.errno==errno.ENOSPC\n"
            " finally:\n"
            "  for f in files:\n"
            "   f.close(); os.unlink(f.name)\n"
        )
        probes["ordinary_imports"] = (
            "import math,cmath,collections,itertools,functools,operator,heapq,bisect,statistics,string,re,typing,fractions,decimal,array,copy,enum,numbers\n"
            "def check():\n return math.factorial(5)==120 and fractions.Fraction(2,4)==fractions.Fraction(1,2)\n"
        )
    return probes


class LocalRuntime:
    def __init__(self, config, *, cancelled: threading.Event | None = None):
        self.config = config
        self.cancelled = cancelled
        self._image_id: str | None = None

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
            command = [
                c.runtime,
                "exec",
                "--containall",
                "--cleanenv",
                "--no-home",
                "--no-mount",
                "hostfs,bind-paths,dev" if c.resource_backend == "slurm" else "hostfs,bind-paths",
                "--net",
                "--network",
                "none",
                "--userns",
                "--drop-caps",
                "all",
                "--pwd",
                "/tmp",
            ]
            if c.resource_backend == "slurm":
                package = Path(__file__).resolve().parent
                # Bind only immutable trusted entry scripts, never the checkout,
                # trainer, dataset or a writable host directory. This permits
                # reuse of the pinned stdlib-only SIF when entry code changes.
                command += [
                    "--bind",
                    ",".join(
                        f"{package / filename}:/opt/{filename}:ro"
                        for filename in ("worker.py", "codec.py", "worker_policy.py")
                    ),
                ]
                command += [
                    "--env",
                    "SELFPLAY_PROCESS_POLICY=single_process",
                    "--env",
                    f"SELFPLAY_OUTPUT_BYTES={c.output_bytes}",
                    "--env",
                    f"SELFPLAY_DISK_BYTES={c.disk_mib * 1024**2}",
                ]
            else:
                command += [
                    "--memory",
                    str(c.memory_mib * 1024 * 1024),
                    "--cpus",
                    str(c.cpus),
                    "--pids-limit",
                    str(c.pids),
                ]
            return command + [
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
                self.command(name),
                raw,
                seconds + 5,
                self.config.output_bytes,
                cancelled=self.cancelled,
                env=(
                    {
                        k: v
                        for k, v in os.environ.items()
                        if k in {"PATH", "LANG", "LC_ALL", "APPTAINER_CACHEDIR", "APPTAINER_TMPDIR"}
                    }
                    if self.config.resource_backend == "slurm"
                    else None
                ),
                cleanup_scope=worker_scope() if self.config.resource_backend == "slurm" else None,
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
        if reason in {"timeout", "output_limit", "cancelled"}:
            return result("student_failure", [], elapsed, reason)
        if reason == "cleanup_failure":
            return result("sandbox_infrastructure_error", [], elapsed, reason)
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
        if self.config.resource_backend == "slurm":
            timeout = self.execute(
                "def check():\n while True: pass\n", "check", [encode([])], 0.1, probe=True
            )
            outcomes["timeout_cleanup"] = timeout["detail"] == "timeout"
            if not outcomes["timeout_cleanup"]:
                details["timeout_cleanup"] = timeout["detail"]
            output = self.execute(
                f"import os\ndef check():\n os.write(1, b'x'*{self.config.output_bytes + 1})\n",
                "check",
                [encode([])],
                5,
                probe=True,
            )
            outcomes["output_limit_cleanup"] = output["detail"] == "output_limit"
            if not outcomes["output_limit_cleanup"]:
                details["output_limit_cleanup"] = output["detail"]
            cancelled = threading.Event()
            timer = threading.Timer(0.5, cancelled.set)
            timer.start()
            try:
                cancellation = LocalRuntime(self.config, cancelled=cancelled).execute(
                    "def check():\n while True: pass\n", "check", [encode([])], 5, probe=True
                )
            finally:
                timer.cancel()
            outcomes["cancellation_cleanup"] = cancellation["detail"] == "cancelled"
            if not outcomes["cancellation_cleanup"]:
                details["cancellation_cleanup"] = cancellation["detail"]
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
            "worker_setup_id": os.getenv("SELFPLAY_WORKER_SETUP_ID", ""),
            "timestamp": time.time(),
            "limits": self.config.model_dump(),
            "execution_contract": {
                "process": self.config.process_policy,
                "resources": self.config.resource_backend,
                "candidate_processes": 1
                if self.config.process_policy == "single_process"
                else None,
                "candidate_threads": 1 if self.config.process_policy == "single_process" else None,
            },
        }
        if self.config.process_policy == "single_process":
            report["limits"].pop("pids")
        return report
