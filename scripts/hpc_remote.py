#!/usr/bin/env python3
"""Standard-library Yale control endpoint, sent over SSH on stdin by hpc.py.

The login node only performs metadata queries, small file transfers and Slurm
submission. All model and sandbox work belongs to allocated compute nodes.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from collections import deque
from contextlib import contextmanager
from pathlib import Path

GPU_VRAM_GB = {"h100": 80, "h200": 141, "rtx_pro_6000_blackwell": 96}
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
SHA = re.compile(r"^[0-9a-f]{40}$")
DIGEST = re.compile(r"^[0-9a-f]{64}$")
ACTIVE = {"PENDING", "CONFIGURING", "RUNNING", "COMPLETING", "SUSPENDED", "REQUEUED"}


def run(args: list[str], *, timeout: int = 30, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(args, text=True, capture_output=True, timeout=timeout, check=check)


def capture(args: list[str]) -> dict:
    try:
        result = run(args, check=False)
        return {
            "exit_code": result.returncode,
            "stdout": result.stdout[-32000:],
            "stderr": result.stderr[-4000:],
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"exit_code": None, "error": str(exc)}


def roots() -> dict[str, dict[str, str]]:
    home = Path.home()
    user = run(["id", "-un"]).stdout.strip()

    def personal_directories(pattern: str) -> dict[str, str]:
        result = {}
        for entry in home.glob(pattern):
            personal = entry.resolve() / user
            if personal.is_dir() and os.access(personal, os.W_OK | os.X_OK):
                result[entry.name] = str(personal)
        return result

    return {
        "project": personal_directories("project_*"),
        "scratch": personal_directories("scratch_*"),
    }


def inspect(payload: dict) -> dict:
    user = run(["id", "-un"]).stdout.strip()
    course = user.split("_", 1)[0]
    commands = {
        "associations": [
            "sacctmgr",
            "-n",
            "-P",
            "show",
            "associations",
            f"user={user}",
            "format=Account,Partition,QOS,DefaultQOS",
        ],
        "partitions": ["sinfo", "-h", "-o", "%P|%G|%l|%D|%a|%C"],
        "jobs": ["squeue", "-h", "-u", user, "-o", "%i|%j|%T|%P|%b|%R"],
        "recent_jobs": [
            "sacct",
            "-n",
            "-P",
            "-u",
            user,
            "--starttime",
            "now-7days",
            "--format",
            "JobID,JobName,State,ExitCode,Partition,AllocTRES",
        ],
        "group_quota": ["getquota", "-g", course],
        "group_jobs": ["squeue", "-h", "-A", course, "-o", "%i|%u|%j|%T|%P|%b|%R"],
        "gpu_nodes": [
            "sinfo",
            "-N",
            "-h",
            "-p",
            "gpu_h100,gpu_h200,gpu_rtx6000",
            "-o",
            "%P|%N|%t|%G|%C",
        ],
        "filesystem": ["df", "-h", str(Path.home())],
    }
    slurm = {name: capture(args) for name, args in commands.items()}
    quota = slurm["group_quota"]
    if quota["exit_code"] == 0:
        lines = quota["stdout"].splitlines()
        own = [line for line in lines if user in line]
        summary = quota["stdout"].split("Quota Summary for", 1)
        quota["stdout"] = "\n".join(own) + (
            "\nQuota Summary for" + summary[1] if len(summary) == 2 else ""
        )
    project = roots()["project"].get(f"project_{course}")
    base = Path(project) / f"selfplay-opsd-{user}" if project else None
    workspace_state: dict = {"exists": bool(base and any((base / "state").glob("sync-*.json")))}
    if (
        base
        and SHA.fullmatch(payload.get("commit") or "")
        and DIGEST.fullmatch(payload.get("manifest_sha256") or "")
    ):
        marker = base / "state" / f"sync-{payload['commit']}.json"
        workspace_state["current_snapshot"] = marker.is_file() and json.loads(
            marker.read_text()
        ) == {
            "commit": payload["commit"],
            "manifest_sha256": payload["manifest_sha256"],
        }
    if base and base.is_dir():
        workspace_state["checkouts"] = sorted(
            p.name for p in (base / "checkouts").glob("*") if p.is_dir()
        )[:20]
        workspace_state["datasets"] = sorted(
            p.name for p in (base / "datasets").glob("*") if p.is_dir()
        )[:20]
        workspace_state["backend_proofs"] = sorted(
            p.name for p in (base / "runs/compatibility").glob("*.json")
        )[:20]
    return {
        "user": user,
        "host": run(["hostname"]).stdout.strip(),
        "home": str(Path.home()),
        "groups": capture(["id", "-Gn"]),
        "storage": roots(),
        "workspace": workspace_state,
        "software": {
            name: shutil.which(name)
            for name in ("git", "rsync", "apptainer", "sbatch", "sacct", "squeue", "sinfo", "flock")
        },
        "slurm": slurm,
    }


def context(_: dict) -> dict:
    user = run(["id", "-un"]).stdout.strip()
    return {
        "user": user,
        "storage": roots(),
        "slurm": {
            "associations": capture(
                [
                    "sacctmgr",
                    "-n",
                    "-P",
                    "show",
                    "associations",
                    f"user={user}",
                    "format=Account,Partition,QOS,DefaultQOS",
                ]
            )
        },
    }


def workspace(payload: dict) -> Path:
    storage = Path(payload["storage_root"]).resolve()
    if str(storage) not in roots()["project"].values():
        raise ValueError("Storage root is not a discovered project directory")
    user = run(["id", "-un"]).stdout.strip()
    if not SAFE_NAME.fullmatch(user):
        raise ValueError("Unsafe Yale username")
    return storage / f"selfplay-opsd-{user}"


def paths(payload: dict) -> dict[str, Path]:
    base = workspace(payload)
    commit = payload["commit"]
    manifest = payload["manifest_sha256"]
    if not SHA.fullmatch(commit) or not DIGEST.fullmatch(manifest):
        raise ValueError("Invalid immutable identity")
    return {
        "base": base,
        "bundle": base / "incoming" / f"{commit}.bundle",
        "checkout": base / "checkouts" / commit,
        "dataset": base / "datasets" / manifest / "prepared",
        "runs": base / "runs",
        "state": base / "state",
        "cache": cache_root(payload)
        / f"selfplay-opsd-{run(['id', '-un']).stdout.strip()}"
        / "huggingface",
    }


def cache_root(payload: dict) -> Path:
    scratch = Path(payload["scratch_root"]).resolve()
    if str(scratch) not in roots()["scratch"].values():
        raise ValueError("Scratch root is not a discovered scratch directory")
    return scratch


def prepare(payload: dict) -> dict:
    p = paths(payload)
    for key in ("base", "bundle", "checkout", "dataset", "runs", "state", "cache"):
        directory = p[key].parent if key in {"bundle", "checkout"} else p[key]
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    return {key: str(value) for key, value in p.items()}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("w") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finalize(payload: dict) -> dict:
    p = paths(payload)
    if (
        not p["bundle"].is_file()
        or sha256(p["dataset"] / "manifest.json") != payload["manifest_sha256"]
    ):
        raise ValueError("Bundle or matching dataset manifest missing")
    if not (p["checkout"] / ".git").is_dir():
        staged = p["checkout"].with_name(f".{payload['commit']}.{uuid.uuid4().hex}.pending")
        try:
            run(
                ["git", "clone", "--quiet", "--no-checkout", str(p["bundle"]), str(staged)],
                timeout=120,
            )
            run(["git", "-C", str(staged), "checkout", "--quiet", "--detach", payload["commit"]])
            os.replace(staged, p["checkout"])
        finally:
            if staged.exists():
                shutil.rmtree(staged)
    head = run(["git", "-C", str(p["checkout"]), "rev-parse", "HEAD"]).stdout.strip()
    if head != payload["commit"]:
        raise ValueError("Remote checkout commit mismatch")
    data_link = p["checkout"] / "data" / "prepared"
    data_link.parent.mkdir(exist_ok=True)
    if not data_link.exists() and not data_link.is_symlink():
        data_link.symlink_to(p["dataset"], target_is_directory=True)
    if data_link.resolve() != p["dataset"].resolve():
        raise ValueError("Remote data symlink points elsewhere")
    runs_link = p["checkout"] / "runs"
    if not runs_link.exists() and not runs_link.is_symlink():
        runs_link.symlink_to(p["runs"], target_is_directory=True)
    if runs_link.resolve() != p["runs"].resolve():
        raise ValueError("Remote runs symlink points elsewhere")
    # Git's directory-only `runs/` ignore does not match a managed symlink.
    # Keep operational links in local Git metadata, preserving a clean checkout.
    exclude = p["checkout"] / ".git/info/exclude"
    existing = exclude.read_text() if exclude.exists() else ""
    additions = [name for name in ("/runs", "/data/prepared") if name not in existing.splitlines()]
    if additions:
        exclude.parent.mkdir(parents=True, exist_ok=True)
        exclude.write_text(existing.rstrip() + "\n" + "\n".join(additions) + "\n")
    marker = {"commit": head, "manifest_sha256": payload["manifest_sha256"]}
    write_json(p["state"] / f"sync-{payload['commit']}.json", marker)
    return {
        "ready": True,
        "checkout": str(p["checkout"]),
        "runs": str(p["runs"]),
        "dataset": str(p["dataset"]),
    }


@contextmanager
def locked(path: Path):
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        yield


def ledger_path(payload: dict) -> Path:
    run_id = payload["run_id"]
    if not SAFE_NAME.fullmatch(run_id):
        raise ValueError("Invalid run ID")
    return workspace(payload) / "state" / f"{run_id}.json"


def read_ledger(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {"attempts": [], "retries": 0}


def job_state(job_id: str) -> dict:
    if not re.fullmatch(r"\d+", job_id):
        raise ValueError("Invalid Slurm job ID")
    queued = capture(["squeue", "-h", "-j", job_id, "-o", "%T"])
    if queued["exit_code"] == 0 and queued["stdout"].strip():
        return {"state": queued["stdout"].strip().splitlines()[0], "source": "squeue"}
    accounting = capture(["sacct", "-n", "-P", "-j", job_id, "--format", "JobIDRaw,State,ExitCode"])
    if accounting["exit_code"] != 0:
        return {
            "state": "UNKNOWN",
            "source": "sacct",
            "error": accounting.get("stderr") or accounting.get("error"),
        }
    for line in accounting["stdout"].splitlines():
        fields = line.split("|")
        if len(fields) >= 3 and fields[0] == job_id:
            return {
                "state": fields[1].split()[0].rstrip("+"),
                "exit_code": fields[2],
                "source": "sacct",
            }
    return {"state": "UNKNOWN", "source": "sacct"}


def status(payload: dict) -> dict:
    path = ledger_path(payload)
    if not path.is_file():
        return {"run_id": payload["run_id"], "ledger": read_ledger(path), "attempts": []}
    with locked(path.with_suffix(".lock")):
        ledger = reconcile_intent(read_ledger(path), path)
    attempts = [
        {**attempt, "scheduler": job_state(attempt["job_id"])} for attempt in ledger["attempts"]
    ]
    return {"run_id": payload["run_id"], "ledger": ledger, "attempts": attempts}


def reconcile_intent(ledger: dict, path: Path) -> dict:
    intent = ledger.get("pending_intent")
    if not intent:
        return ledger
    user = run(["id", "-un"]).stdout.strip()
    queued = capture(["squeue", "-h", "-u", user, "-o", "%i|%k"])
    history = capture(
        [
            "sacct",
            "-n",
            "-P",
            "-u",
            user,
            "--starttime",
            "now-7days",
            "--format",
            "JobIDRaw,Comment",
        ]
    )
    if queued["exit_code"] != 0 or history["exit_code"] != 0:
        return ledger
    found: set[str] = set()
    for report in (queued, history):
        for line in report["stdout"].splitlines():
            fields = line.split("|", 1)
            if len(fields) == 2 and fields[1] == intent and re.fullmatch(r"\d+", fields[0]):
                found.add(fields[0])
    if len(found) == 1:
        job_id = found.pop()
        ledger["attempts"].append(
            {
                "job_id": job_id,
                "intent": intent,
                "resume": ledger.get("pending_resume"),
                "operation": ledger["operation"],
            }
        )
        bind_budget_job(ledger.get("specification", {}), intent, job_id)
        ledger.pop("pending_intent")
        ledger.pop("pending_resume", None)
        write_json(path, ledger)
    return ledger


def logs(payload: dict) -> dict:
    state = status(payload)
    if not state["attempts"]:
        raise ValueError("No job attempts for this run")
    count = int(payload.get("lines", 100))
    if count < 1 or count > 1000:
        raise ValueError("Log line count must be 1 through 1000")
    job_id = state["attempts"][-1]["job_id"]
    path = workspace(payload) / "state" / f"{payload['run_id']}-{job_id}.out"
    if not path.is_file():
        return {"run_id": payload["run_id"], "job_id": job_id, "available": False}
    with path.open(errors="replace") as stream:
        tail = "".join(deque(stream, maxlen=count))[-64000:]
    return {"run_id": payload["run_id"], "job_id": job_id, "available": True, "tail": tail}


def verified_checkpoint(payload: dict) -> str:
    p = paths(payload)
    study = payload["study_id"] + (
        payload.get("pilot_suffix", "-pilot") if payload["operation"] == "pilot" else ""
    )
    run_dir = p["runs"] / study / f"{payload['method']}-seed{payload['seed']}"
    latest = json.loads((run_dir / "checkpoints/latest.json").read_text())
    name = latest["path"]
    if not re.fullmatch(r"\d{6}", name):
        raise ValueError("Invalid latest checkpoint pointer")
    checkpoint = run_dir / "checkpoints" / name
    metadata = check_checkpoint(checkpoint, payload)
    if metadata.get("iteration") != latest["iteration"]:
        raise ValueError("Latest checkpoint iteration mismatch")
    return str(checkpoint)


def check_checkpoint(checkpoint: Path, payload: dict) -> dict:
    marker = json.loads((checkpoint / "complete.json").read_text())
    for filename, expected in marker["files"].items():
        if Path(filename).name != filename or sha256(checkpoint / filename) != expected:
            raise ValueError("Latest checkpoint verification failed")
    metadata = json.loads((checkpoint / "metadata.json").read_text())
    if (
        payload.get("runtime_identity")
        and metadata.get("runtime_identity") != payload["runtime_identity"]
    ):
        raise ValueError("Checkpoint runtime identity mismatch")
    if metadata.get("scientific_hash") != payload.get("scientific_hash") or metadata.get(
        "manifests_hash"
    ) != payload.get("manifests_hash"):
        raise ValueError("Checkpoint scientific or dataset identity mismatch")
    return metadata


def recover(payload: dict) -> dict:
    request = payload
    path = ledger_path(payload)
    ledger = read_ledger(path)
    if ledger.get("specification"):
        payload = ledger["specification"] | {
            "explicit_continue": bool(request.get("explicit_continue"))
        }
    if ledger.get("identity") != payload["identity"] or not ledger["attempts"]:
        raise ValueError("Run identity or initial submission missing")
    if ledger.get("pending_intent"):
        return {"submitted": False, "state": "UNKNOWN", "reason": "unreconciled_submission"}
    state = job_state(ledger["attempts"][-1]["job_id"])["state"]
    if state in ACTIVE or state == "UNKNOWN":
        return {"submitted": False, "state": state}
    planned = bool(payload.get("explicit_continue")) and state in {"COMPLETED", "TIMEOUT"}
    if state == "COMPLETED" and not planned:
        return {"submitted": False, "state": state, "reason": "inspect_scientific_completion"}
    if not planned and state not in {"PREEMPTED", "NODE_FAIL", "BOOT_FAIL"}:
        return {"submitted": False, "state": state, "reason": "not_transient"}
    if not planned and ledger.get("retries", 0) >= 2:
        return {"submitted": False, "state": state, "reason": "retry_limit"}
    health_path = Path(ledger.get("run_dir", "")) / "state/health.json"
    if health_path.is_file() and json.loads(health_path.read_text()).get("phase") == "failed":
        return {"submitted": False, "state": state, "reason": "training_reported_failure"}
    try:
        checkpoint = verified_checkpoint(payload)
    except (OSError, KeyError, ValueError) as exc:
        return {"submitted": False, "state": state, "reason": f"checkpoint_unavailable: {exc}"}
    metadata = check_checkpoint(Path(checkpoint), payload)
    if metadata.get("iteration", 0) >= payload.get(
        "target_iterations", payload.get("iterations", 0)
    ):
        return {"submitted": False, "state": state, "reason": "scientifically_complete"}
    if planned and (
        not health_path.is_file() or json.loads(health_path.read_text()).get("phase") != "stopped"
    ):
        return {"submitted": False, "state": state, "reason": "not_a_clean_stop"}
    if planned and ledger.get("continuations", 0) >= payload.get("max_continuations", 0):
        return {"submitted": False, "state": state, "reason": "continuation_limit"}
    # Reserve a retry under the same lock as submission; a failed reservation is
    # left auditable and cannot cause an unbounded scheduler loop.
    with locked(path.with_suffix(".lock")):
        current = read_ledger(path)
        previous_job = ledger["attempts"][-1]["job_id"]
        if (
            (not planned and current.get("retries", 0) >= 2)
            or current["attempts"][-1]["job_id"] != previous_job
            or current.get("recovery_from_job") == previous_job
        ):
            return {"submitted": False, "state": state, "reason": "concurrent_recovery"}
        counter = "continuations" if planned else "retries"
        if planned and current.get(counter, 0) >= payload.get("max_continuations", 0):
            return {"submitted": False, "state": state, "reason": "continuation_limit"}
        current[counter] = current.get(counter, 0) + 1
        current["recovery_from_job"] = previous_job
        write_json(path, current)
    result = submit(
        {
            **payload,
            "resume": checkpoint,
            "stop_after_iteration": 0
            if metadata["iteration"] >= payload.get("stop_after_iteration", 0)
            else payload.get("stop_after_iteration", 0),
            "time": payload.get("continuation_time", payload["time"]),
        }
    )
    return {**result, "state": state, "resume": checkpoint}


def allowed(payload: dict) -> dict:
    account = payload["account"]
    partition = payload["partition"]
    gpu_type = payload["gpu_type"]
    gpu_count = payload["gpu_count"]
    if not all(SAFE_NAME.fullmatch(x) for x in (account, partition, gpu_type)):
        raise ValueError("Unsafe Slurm resource name")
    if gpu_count not in (0, 1, 2, 4, 8) or (gpu_count and gpu_type not in GPU_VRAM_GB):
        raise ValueError("GPU type or count is unsupported")
    user = run(["id", "-un"]).stdout.strip()
    assoc = run(
        [
            "sacctmgr",
            "-n",
            "-P",
            "show",
            "associations",
            f"user={user}",
            "format=Account,Partition,QOS,DefaultQOS",
        ]
    ).stdout
    rows = [line.split("|") for line in assoc.splitlines() if line.strip()]
    matches = [
        row for row in rows if len(row) >= 2 and row[0] == account and row[1] in ("", partition)
    ]
    if not matches:
        raise ValueError("Requested account/partition is not in the user's Slurm associations")
    qos = payload.get("qos")
    if qos and (
        not SAFE_NAME.fullmatch(qos)
        or not any(qos in (row[2] + "," + row[3]).split(",") for row in matches if len(row) >= 4)
    ):
        raise ValueError("Requested QOS is not in the user's Slurm associations")
    info = run(["scontrol", "show", "partition", partition, "-o"]).stdout
    if f"PartitionName={partition}" not in info:
        raise ValueError("Requested partition is unavailable")
    gres = run(["sinfo", "-h", "-p", partition, "-o", "%G"]).stdout
    available_counts = [
        int(m.group(1)) for m in re.finditer(rf"(?:^|[,\n])gpu:{re.escape(gpu_type)}:(\d+)", gres)
    ]
    if gpu_count and (not available_counts or max(available_counts) < gpu_count):
        raise ValueError("Requested GPU type/count is not shown on a single partition node")
    if not (1 <= int(payload["cpus"]) <= 128 and 1 <= int(payload["memory_gib"]) <= 512):
        raise ValueError("CPU or memory request exceeds controller bounds")
    time_match = re.fullmatch(r"(\d{2}):(\d{2}):(\d{2})", payload["time"])
    if (
        not time_match
        or int(time_match[1]) > 23
        or int(time_match[2]) > 59
        or int(time_match[3]) > 59
    ):
        raise ValueError("Wall time exceeds the controller's 23-hour bound")
    # Aggregated CPU partitions use values such as "48+|376832+". Per-node
    # records avoid interpreting heterogeneous minima as exact node shapes.
    nodes = run(["sinfo", "-N", "-h", "-p", partition, "-o", "%c|%m|%G"]).stdout
    fits_node = False
    for row in nodes.splitlines():
        fields = row.split("|", 2)
        if len(fields) == 3 and fields[0].isdigit() and fields[1].isdigit():
            match = re.search(rf"(?:^|,)gpu:{re.escape(gpu_type)}:(\d+)", fields[2])
            if (
                (not gpu_count or (match and int(match[1]) >= gpu_count))
                and int(fields[0]) >= payload["cpus"]
                and int(fields[1]) >= payload["memory_gib"] * 1024
            ):
                fits_node = True
    if not fits_node:
        raise ValueError("No advertised node fits requested GPUs, CPUs and memory")
    return {
        "association": matches,
        "partition": partition,
        "gpu_type": gpu_type,
        "max_gpus_per_node": max(available_counts, default=0),
    }


def submit(payload: dict) -> dict:
    p = paths(payload)
    path = ledger_path(payload)
    sync = json.loads((p["state"] / f"sync-{payload['commit']}.json").read_text())
    if sync != {"commit": payload["commit"], "manifest_sha256": payload["manifest_sha256"]}:
        raise ValueError("Remote snapshot identity mismatch")
    if payload["operation"] not in {
        "image-convert",
        "image-build",
        "worker-check",
        "worker-diagnose",
    }:
        if not Path(payload["image"]).is_file() or not DIGEST.fullmatch(payload["image_sha256"]):
            raise ValueError("Pinned trainer SIF missing")
    precheck(payload)
    if payload.get("secrets_file"):
        secret_path = Path(payload["secrets_file"])
        if not secret_path.is_file() or secret_path.stat().st_mode & 0o077:
            raise ValueError("Secrets file must exist and be private to this user")
    # Hash the large image on the allocated compute node, before GPU work.
    if not (p["checkout"] / payload["settings"]).is_file():
        raise ValueError("Settings file missing from immutable checkout")
    if payload["operation"] in {"evaluate", "diagnose"} and not payload.get("base"):
        checkpoint = payload.get("checkpoint", "")
        expected = (
            f"/runs/{payload['study_id']}/{payload['method']}-seed{payload['seed']}/checkpoints/"
        )
        if not checkpoint.startswith(expected) or not re.fullmatch(
            r"\d{6}", checkpoint.removeprefix(expected)
        ):
            raise ValueError("Checkpoint must be a numbered state in this run")
        host_checkpoint = p["runs"] / checkpoint.removeprefix("/runs/")
        check_checkpoint(host_checkpoint, payload)
    if payload["gpu_count"] and (
        payload["gpu_count"] != payload["profile_gpu_count"]
        or GPU_VRAM_GB[payload["gpu_type"]] < payload["min_vram_gib"]
    ):
        raise ValueError("GPU request does not satisfy the selected hardware profile")
    resources = allowed(payload)
    with locked(p["state"] / ".submit.lock"):
        ledger = read_ledger(path)
        ledger = reconcile_intent(ledger, path)
        if ledger.get("identity") and ledger["identity"] != payload["identity"]:
            raise ValueError("Run ID already belongs to a different immutable identity")
        if ledger["attempts"]:
            latest = ledger["attempts"][-1]
            state = job_state(latest["job_id"])
            if state["state"] == "UNKNOWN" or state["state"] in {
                "PENDING",
                "CONFIGURING",
                "RUNNING",
                "COMPLETING",
                "SUSPENDED",
                "REQUEUED",
            }:
                return {
                    "run_id": payload["run_id"],
                    "submitted": False,
                    "reason": "existing_or_uncertain_job",
                    "job_id": latest["job_id"],
                    "scheduler": state,
                }
            if state["state"] == "COMPLETED" and payload["operation"] not in {"pilot", "train"}:
                return {
                    "submitted": False,
                    "reason": "already_completed",
                    "job_id": latest["job_id"],
                }
            setup_retry = (
                payload.get("retry_setup")
                and (
                    payload["gpu_count"] == 0
                    or payload["operation"] in {"allocation-check", "backend-check"}
                )
                and state["state"] in {"PREEMPTED", "NODE_FAIL", "BOOT_FAIL", "TIMEOUT"}
                and (payload["gpu_count"] == 0 or state["state"] != "TIMEOUT")
                and ledger.get("setup_retries", 0) < 2
            )
            if not payload.get("resume") and not setup_retry:
                raise ValueError("Existing run requires explicit checkpoint resume")
            if setup_retry:
                ledger["setup_retries"] = ledger.get("setup_retries", 0) + 1
            if state["state"] not in {
                "PREEMPTED",
                "NODE_FAIL",
                "BOOT_FAIL",
                "TIMEOUT",
                "FAILED",
                "CANCELLED",
                "COMPLETED",
            }:
                raise ValueError("Previous job state does not permit resume")
        elif payload.get("resume"):
            raise ValueError("Cannot resume a run with no previous job")
        # If a prior sbatch response was lost, the pending intent remains. Reconcile
        # its unique Slurm comment before any further attempt.
        intent = ledger.get("pending_intent")
        if intent:
            return {
                "run_id": payload["run_id"],
                "submitted": False,
                "reason": "unreconciled_submission",
                "intent": intent,
            }
        token = "selfplay-" + uuid.uuid4().hex[:20]
        ledger["identity"] = payload["identity"]
        ledger["operation"] = payload["operation"]
        artifact = artifact_dir(payload)
        slot = (
            p["state"]
            / "slots"
            / (
                hashlib.sha256(
                    (
                        "image-" + payload["image_metadata"]["kind"]
                        if payload["operation"] in {"image-convert", "image-build"}
                        else str(artifact) + ":" + payload["operation"]
                    ).encode()
                ).hexdigest()
                + ".json"
            )
        )
        legacy_slot = (
            p["state"] / "slots" / (hashlib.sha256(str(artifact).encode()).hexdigest() + ".json")
        )
        parent_slot = slot
        checked_slots = [slot, legacy_slot]
        if payload["operation"] == "method-smoke":
            # A/B/C write separate reports and checkpoints. The D launcher runs
            # the entire suite, so it must conflict with every selected method.
            def method_slot(method):
                key = f"{artifact}:method-smoke:{method}"
                return p["state"] / "slots" / (hashlib.sha256(key.encode()).hexdigest() + ".json")

            method = payload["method"]
            if method not in {"A", "B", "C", "D"}:
                raise ValueError("Unknown method acceptance selection")
            slot = method_slot(method)
            peers = "ABCD" if method == "D" else method + "D"
            checked_slots = [slot, parent_slot, legacy_slot, *[method_slot(m) for m in peers]]
        for candidate_slot in dict.fromkeys(checked_slots):
            if not candidate_slot.exists():
                continue
            owner = json.loads(candidate_slot.read_text())
            if owner["run_id"] != payload["run_id"]:
                prior = status({**payload, "run_id": owner["run_id"]})
                prior_operation = prior["ledger"].get("operation") or prior["ledger"].get(
                    "specification", {}
                ).get("operation")
                if (
                    candidate_slot == legacy_slot
                    and prior_operation is not None
                    and prior_operation != payload["operation"]
                ):
                    # An old shared directory slot cannot serialize unrelated
                    # work. Equivalent old/uncertain submissions still block.
                    continue
                prior_method = prior["ledger"].get("specification", {}).get("method")
                if (
                    payload["operation"] == "method-smoke"
                    and prior_operation == "method-smoke"
                    and payload["method"] in "ABC"
                    and prior_method in {"A", "B", "C"}
                    and payload["method"] != prior_method
                ):
                    continue
                states = [a["scheduler"]["state"] for a in prior["attempts"]]
                if (
                    prior["ledger"].get("last_rejection")
                    and not prior["ledger"].get("pending_intent")
                    and not prior["attempts"]
                ):
                    # Older controllers retained slots after a definitive
                    # sbatch rejection. No accepted/uncertain job owns these.
                    continue
                if (
                    prior["ledger"].get("pending_intent")
                    or not states
                    or states[-1] in ACTIVE | {"UNKNOWN"}
                ):
                    return {
                        "submitted": False,
                        "reason": "slot_busy",
                        "state": states[-1] if states else "UNKNOWN",
                        "run_id": owner["run_id"],
                    }
                if payload["operation"] in {"train", "pilot"}:
                    raise ValueError(
                        "Existing scientific run belongs to a different immutable identity"
                    )
        reserve_budget(payload, token)
        write_json(slot, {"run_id": payload["run_id"], "identity": payload["identity"]})
        ledger.setdefault(
            "specification",
            {k: v for k, v in payload.items() if k not in {"resume", "explicit_continue"}},
        )
        ledger["run_dir"] = str(artifact)
        ledger["pending_intent"] = token
        ledger["pending_resume"] = payload.get("resume")
        write_json(path, ledger)
        exports = {
            "SELFPLAY_PROJECT": str(p["checkout"]),
            "SELFPLAY_WORKER_SIF": payload.get("worker_image", ""),
            "SELFPLAY_WORKER_SHA256": payload.get("worker_sha256", ""),
            "SELFPLAY_WORKER_SETUP_ID": payload.get("worker_setup_id", ""),
            "SELFPLAY_WORKER_CONFIG": base64.b64encode(
                json.dumps(payload.get("worker_config", {})).encode()
            ).decode(),
            "SELFPLAY_STOP_AFTER_ITERATION": str(payload.get("stop_after_iteration", 0)),
            "SELFPLAY_ARCHIVE": payload.get("archive", ""),
            "SELFPLAY_ARCHIVE_SHA256": payload.get("archive_sha256", ""),
            "SELFPLAY_IMAGE_TARGET": payload.get("image_target", ""),
            "SELFPLAY_IMAGE_MANIFEST": payload.get("image_manifest", ""),
            "SELFPLAY_IMAGE_METADATA": base64.b64encode(
                json.dumps(payload.get("image_metadata", {})).encode()
            ).decode(),
            "SELFPLAY_BUILD_KEY": payload.get("build_key", ""),
            "SELFPLAY_BUILD_DEFINITION": payload.get("definition_path", ""),
            "SELFPLAY_TRAINER_SIF": payload["image"],
            "SELFPLAY_TRAINER_SHA256": payload["image_sha256"],
            "SELFPLAY_DATA": str(p["dataset"]),
            "SELFPLAY_RUNS": str(p["runs"]),
            "SELFPLAY_CACHE": str(p["cache"]),
            "SELFPLAY_OPERATION": payload["operation"],
            "SELFPLAY_METHOD": payload["method"],
            "SELFPLAY_HARDWARE": payload["profile"],
            "SELFPLAY_GPU_COUNT": str(payload["gpu_count"]),
            "SELFPLAY_ALLOCATION_CPUS": str(payload["cpus"]),
            "SELFPLAY_ALLOCATION_MEMORY_MIB": str(payload["memory_gib"] * 1024),
            "SELFPLAY_SETTINGS": payload["settings"],
            "SELFPLAY_MAIN_STUDY": payload["study_id"],
            "SELFPLAY_PILOT_SUFFIX": payload.get("pilot_suffix", "-pilot"),
            "SELFPLAY_GPU_TYPE": payload["gpu_type"],
            "SELFPLAY_STUDY": payload["study_id"]
            + (
                payload.get("pilot_suffix", "-pilot")
                if payload["operation"] == "pilot"
                else "-method-smoke"
                if payload["operation"] == "method-smoke"
                else ""
            ),
            "SELFPLAY_SEED": str(payload["seed"]),
            "SELFPLAY_RUN_ID": payload["run_id"],
            "SELFPLAY_RESUME": (payload.get("resume") or "").replace(str(p["runs"]), "/runs", 1),
            "SELFPLAY_OUTPUT": payload.get("output") or "",
            "SELFPLAY_REFERENCE": payload.get("reference") or "",
            "SELFPLAY_ITERATIONS": str(payload.get("iterations") or 10),
            "SELFPLAY_BASE": "1" if payload.get("base") else "0",
            "SELFPLAY_SPLIT": payload.get("split") or "dev_monitor",
            "SELFPLAY_ENDPOINT": payload.get("endpoint") or "primary",
            "SELFPLAY_CHECKPOINT": payload.get("checkpoint") or "",
            "SELFPLAY_SECRETS_FILE": payload.get("secrets_file") or "",
        }
        for value in exports.values():
            if "," in value or "\n" in value:
                raise ValueError("Slurm export value contains a delimiter")
        script = p["checkout"] / "scripts/yale.slurm"
        args = [
            "sbatch",
            "--parsable",
            f"--account={payload['account']}",
            f"--partition={payload['partition']}",
            f"--cpus-per-task={payload['cpus']}",
            f"--mem={payload['memory_gib']}G",
            f"--time={payload['time']}",
            f"--comment={token}",
            f"--signal=B:USR1@{payload.get('signal_seconds', 300)}",
            f"--output={p['state']}/{payload['run_id']}-%j.out",
        ]
        if payload["gpu_count"]:
            args.append(f"--gpus={payload['gpu_type']}:{payload['gpu_count']}")
        if payload["operation"] in {"allocation-check", "backend-check"}:
            args.append("--no-requeue")
        if payload.get("exclude_nodes"):
            args.append("--exclude=" + ",".join(payload["exclude_nodes"]))
        if payload.get("qos"):
            args.append(f"--qos={payload['qos']}")
        args += ["--export=" + ",".join(f"{k}={v}" for k, v in exports.items()), str(script)]
        result = run(args, check=False)
        if result.returncode:
            ledger.pop("pending_intent")
            ledger.pop("pending_resume", None)
            release_budget(payload, token)
            ledger["last_rejection"] = result.stderr[-2000:]
            write_json(path, ledger)
            slot.unlink(missing_ok=True)
            return {
                "run_id": payload["run_id"],
                "submitted": False,
                "reason": "scheduler_rejected",
                "stderr": result.stderr[-2000:],
            }
        if not re.fullmatch(r"\d+(?:;[^\s]+)?", result.stdout.strip()):
            return {
                "run_id": payload["run_id"],
                "submitted": False,
                "reason": "unreconciled_submission",
                "intent": token,
                "stderr": result.stderr[-2000:],
            }
        job_id = result.stdout.strip().split(";", 1)[0]
        bind_budget_job(payload, token, job_id)
        ledger.pop("pending_intent", None)
        ledger.pop("pending_resume", None)
        ledger["attempts"].append(
            {
                "job_id": job_id,
                "intent": token,
                "resume": payload.get("resume"),
                "operation": payload["operation"],
            }
        )
        write_json(path, ledger)
        return {
            "run_id": payload["run_id"],
            "submitted": True,
            "job_id": job_id,
            "resources": resources,
            "ledger": str(path),
        }


def artifact_dir(payload: dict) -> Path:
    runs = paths(payload)["runs"]
    study = payload["study_id"]
    if not SAFE_NAME.fullmatch(study):
        raise ValueError("Unsafe study ID")
    operation = payload["operation"]
    if operation == "protocol-check":
        return runs / (study + "-protocol-probe") / f"B-seed{payload['seed']}"
    if operation == "protocol-revision":
        return runs / "data-revisions" / study
    if operation == "image-convert":
        return (
            workspace(payload)
            / "images"
            / payload["image_metadata"]["kind"]
            / payload["archive_sha256"]
        )
    if operation == "image-build":
        return (
            workspace(payload) / "images" / payload["image_metadata"]["kind"] / payload["build_key"]
        )
    if operation == "method-smoke":
        return runs / "readiness" / study / "method-smokes"
    if operation in {"rollout-parity", "numerical-audit"}:
        return runs / "readiness" / study / operation / payload["run_id"]
    if operation == "wandb-verify":
        return runs / "readiness" / study / operation / payload["run_id"]
    if operation in {
        "worker-check",
        "runtime-check",
        "readiness-check",
        "allocation-check",
        "worker-diagnose",
    }:
        return runs / "readiness" / study
    if operation == "backend-check":
        if payload.get("backend_identity", {}).get("schema_version") == 2:
            return runs / "compatibility/backend-v2" / payload["profile"]
        return runs / "compatibility" / payload["profile"]
    if operation == "freeze-study":
        return runs / study
    if operation == "evaluate" and payload.get("base"):
        return runs / study / "base"
    if operation == "diagnose":
        return runs / study / "diagnostics" / Path(payload["checkpoint"]).name
    return (
        runs
        / (study + (payload.get("pilot_suffix", "-pilot") if operation == "pilot" else ""))
        / f"{payload['method']}-seed{payload['seed']}"
    )


def budget_path(payload: dict) -> Path:
    if not SAFE_NAME.fullmatch(payload["study_id"]):
        raise ValueError("Unsafe study ID")
    # Readiness authorization spans protocol/study revisions. Never let a new
    # scientific identity create a fresh allowance.
    return workspace(payload) / "state" / "budget-readiness-cpsc4880.json"


def refreshed_budget(payload: dict) -> dict:
    path = budget_path(payload)
    budget = json.loads(path.read_text()) if path.exists() else {"limit": 24.0, "reservations": {}}
    authorization = "cpsc4880-readiness-24h-v1"
    if budget.get("authorization_id", authorization) != authorization:
        raise ValueError("Readiness authorization identity mismatch")
    budget["authorization_id"] = authorization
    # Import every legacy study, not just the caller's study. Keep source hashes
    # so a legacy writer cannot silently fork the allowance after migration.
    imported = budget.setdefault("legacy_sources", {})
    for legacy in sorted(path.parent.glob("budget-*.json")):
        if legacy == path:
            continue
        checksum = sha256(legacy)
        if legacy.name in imported:
            if imported[legacy.name] != checksum:
                raise ValueError(
                    "Legacy budget changed after migration; reconcile before allocation"
                )
            continue
        previous = json.loads(legacy.read_text())
        if previous.get("limit") != budget["limit"]:
            raise ValueError("Conflicting historical readiness limits")
        for token, record in previous["reservations"].items():
            existing = budget["reservations"].get(token)
            if existing is not None and existing != record:
                raise ValueError("Conflicting historical readiness reservation")
            # A job can occur in multiple copied ledgers. Only exact duplicates
            # have sufficient evidence to be charged once.
            same_job = [
                item
                for item in budget["reservations"].values()
                if record.get("job_id") and item.get("job_id") == record["job_id"]
            ]
            if same_job and any(item != record for item in same_job):
                raise ValueError("Conflicting historical job accounting")
            if not same_job:
                budget["reservations"][token] = record
        imported[legacy.name] = checksum
    for item in budget["reservations"].values():
        if not item.get("job_id") or item.get("charged") is not None:
            continue
        result = capture(
            [
                "sacct",
                "-n",
                "-P",
                "-j",
                item["job_id"],
                "--format",
                "JobIDRaw,State,ElapsedRaw,AllocTRES,Start",
            ]
        )
        if result["exit_code"] != 0:
            continue
        for row in result["stdout"].splitlines():
            parts = row.split("|")
            if (
                len(parts) < 4
                or parts[0] != item["job_id"]
                or parts[1].split()[0].rstrip("+") in ACTIVE
            ):
                continue
            if not parts[2].isdigit():
                continue
            gpu = re.search(r"(?:^|,)gres/gpu=(\d+)(?:,|$)", parts[3])
            # Keep the reservation unless the actual allocation can be verified.
            if gpu:
                item["charged"] = int(parts[2]) * int(gpu[1]) / 3600
            elif (
                len(parts) >= 5
                and parts[2] == "0"
                and not parts[3].strip()
                and parts[4] in {"None", "Unknown", "N/A"}
                and parts[1].split()[0] in {"CANCELLED", "FAILED"}
            ):
                # A confirmed never-started job consumed no allocated GPU time.
                # Missing/delayed accounting without this proof stays reserved.
                item["charged"] = 0.0
    budget["committed_gpu_hours"] = sum(
        i["reserved"] if i.get("charged") is None else i["charged"]
        for i in budget["reservations"].values()
    )
    return budget


def reserve_budget(payload: dict, token: str):
    with locked(workspace(payload) / "state" / ".budget.lock"):
        if not payload.get("readiness_budget") or not payload["gpu_count"]:
            return
        limit = float(payload["readiness_budget"])
        if not 0 < limit <= 24:
            raise ValueError("Readiness budget must be positive and at most 24 GPU-hours")
        budget = refreshed_budget(payload)
        if budget["reservations"] and budget["limit"] != limit:
            raise ValueError("Cannot change the recorded readiness budget")
        hours, minutes, seconds = map(int, payload["time"].split(":"))
        reserved = payload["gpu_count"] * (hours + minutes / 60 + seconds / 3600)
        if budget["committed_gpu_hours"] + reserved > limit + 1e-9:
            raise ValueError("Readiness GPU-hour budget exhausted; no job submitted")
        if payload["operation"] == "pilot":
            spent = sum(
                i["reserved"] if i.get("charged") is None else i["charged"]
                for i in budget["reservations"].values()
                if i.get("operation") == "pilot"
            )
            if spent + reserved > 12 + 1e-9:
                raise ValueError("Pilot allocation budget exhausted")
        budget["limit"] = limit
        if token in budget["reservations"]:
            raise ValueError("Readiness reservation token already exists")
        budget["reservations"][token] = {
            "reserved": reserved,
            "charged": None,
            "run_id": payload["run_id"],
            "operation": payload["operation"],
        }
        write_json(budget_path(payload), budget)


def bind_budget_job(payload: dict, token: str, job_id: str):
    if not payload.get("study_id"):
        return
    with locked(workspace(payload) / "state" / ".budget.lock"):
        path = budget_path(payload) if payload.get("study_id") else None
        if not path or not path.exists():
            return
        budget = json.loads(path.read_text())
        if token in budget["reservations"]:
            budget["reservations"][token]["job_id"] = job_id
            write_json(path, budget)


def release_budget(payload: dict, token: str):
    if not payload.get("study_id"):
        return
    with locked(workspace(payload) / "state" / ".budget.lock"):
        path = budget_path(payload) if payload.get("study_id") else None
        if path and path.exists():
            budget = json.loads(path.read_text())
            budget["reservations"].pop(token, None)
            write_json(path, budget)


def stage_status(base: Path, study: str, gates: dict, reports: dict) -> dict:
    """Summarize durable attempts without treating a scheduler success as evidence."""
    operations = {
        "worker": "worker-check",
        "runtime": "runtime-check",
        "single": "backend-check",
        "target": "backend-check",
        "base": "evaluate",
        "pilot": "pilot",
        "methods": "method-smoke",
        "freeze": "freeze-study",
    }
    ledgers = []
    for path in (base / "state").glob("*.json"):
        item = json.loads(path.read_text())
        if item.get("specification", {}).get("study_id") == study:
            ledgers.append((path.stat().st_mtime_ns, item))
    states = {}
    for gate, passed in gates.items():
        present = (
            any(reports.get(gate, {}).values()) if gate == "methods" else bool(reports.get(gate))
        )
        state = {
            "state": "passed" if passed else "stale" if present else "pending",
            "next_action": None if passed else "Run hpc prepare-study --watch",
        }
        candidates = []
        for modified, ledger in ledgers:
            spec = ledger["specification"]
            if spec.get("operation") != operations[gate]:
                continue
            if gate in {"single", "target"} and (spec.get("profile") == "single_gpu_80gb") != (
                gate == "single"
            ):
                continue
            if gate == "base" and not spec.get("base"):
                continue
            candidates.append((modified, ledger))
        if candidates and not passed:
            ledger = max(candidates, key=lambda item: item[0])[1]
            run_id = ledger["specification"]["run_id"]
            state["run_id"] = run_id
            if ledger.get("pending_intent"):
                state.update(
                    state="pending",
                    failure_category="uncertain_submission",
                    next_action=f"Reconcile hpc status {run_id}; do not replace this intent",
                )
            elif ledger.get("attempts"):
                job = ledger["attempts"][-1]["job_id"]
                scheduler = job_state(job)["state"]
                state.update(job_id=job, scheduler_state=scheduler)
                if scheduler in ACTIVE:
                    state["state"] = "pending" if scheduler == "PENDING" else "running"
                elif scheduler == "UNKNOWN":
                    state.update(state="pending", failure_category="uncertain_submission")
                elif scheduler != "COMPLETED":
                    attempt = base / "runs/readiness" / study / f"attempt-{job}.json"
                    details = json.loads(attempt.read_text()) if attempt.is_file() else {}
                    log = base / "state" / f"{run_id}-{job}.out"
                    message = json.dumps(reports.get(gate, {})).lower()
                    if log.is_file():
                        with log.open("rb") as stream:
                            stream.seek(max(0, log.stat().st_size - 64000))
                            message += stream.read().decode(errors="replace").lower()
                    allocation = base / "runs/readiness" / study / f"gpu-allocation-{job}.json"
                    mapped = allocation.is_file() and json.loads(allocation.read_text()).get(
                        "ready"
                    )
                    inferred = (
                        "site_setup"
                        if "dbus_session_bus_address" in message or "delegation" in message
                        else "allocation"
                        if "gpu mapping" in message
                        or "slurm/nvidia" in message
                        or "unexpected gpu count" in message
                        or (allocation.is_file() and not mapped)
                        else "cuda_oom"
                        if "cuda" in message and "out of memory" in message and mapped
                        else "application"
                    )
                    category = details.get("failure_category") or (
                        "scheduler_interrupt"
                        if scheduler in {"PREEMPTED", "NODE_FAIL", "BOOT_FAIL"}
                        else inferred
                    )
                    state.update(
                        state="failed",
                        failure_category=category,
                        evidence=str(attempt if attempt.is_file() else log),
                        next_action=f"Inspect hpc logs {run_id} and hpc status {run_id}; repair before retry",
                    )
        states[gate] = state
    return states


def allocation_compatible(proof: dict, expected: str | None, base: Path) -> bool:
    if not expected:
        return False
    if proof.get("allocation_code_hash") == expected:
        return True
    # Legacy proofs hashed the whole job launcher. Reuse only after checking its
    # immutable source checksum and the exact verifier function ASTs. No report
    # is rewritten and unrelated launcher changes do not invalidate device proof.
    commit = proof.get("execution_commit", "")
    if not SHA.fullmatch(commit):
        return False
    source = base / "checkouts" / commit / "scripts/hpc_job.py"
    if not source.is_file() or sha256(source) != proof.get("allocation_code_hash"):
        return False
    names = {
        "sha256",
        "write",
        "diagnostic_command",
        "device_access",
        "allocated_gpus",
        "resolve_gpus",
        "verify_container_gpus",
    }
    tree = ast.parse(source.read_text())
    for node in ast.walk(tree):
        if hasattr(node, "type_params") and not node.type_params:
            node.__dict__["_fields"] = tuple(name for name in node._fields if name != "type_params")
    definitions = {
        n.name: ast.dump(n, include_attributes=False)
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in names
    }
    actual = hashlib.sha256(
        json.dumps(definitions, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return set(definitions) == names and actual == expected


def recover_allocation(payload: dict) -> dict:
    """Publish completed raw ownership proof; never rerun a GPU probe on login."""
    base = workspace(payload)
    run_id = payload["run_id"]
    if not SAFE_NAME.fullmatch(run_id):
        raise ValueError("Unsafe run ID")
    ledger = read_ledger(base / "state" / f"{run_id}.json")
    spec = ledger.get("specification", {})
    if (
        spec.get("operation") != "allocation-check"
        or spec.get("account") != "cpsc4880"
        or ledger.get("pending_intent")
        or not ledger.get("attempts")
    ):
        raise ValueError("Recovery requires a reconciled cpsc4880 allocation diagnostic")
    job = ledger["attempts"][-1]["job_id"]
    if job_state(job)["state"] not in {"COMPLETED", "FAILED"}:
        raise ValueError("Allocation diagnostic is active or uncertain")
    study, profile, commit = spec["study_id"], spec["profile"], spec["commit"]
    if (
        not SAFE_NAME.fullmatch(study)
        or not SAFE_NAME.fullmatch(profile)
        or not SHA.fullmatch(commit)
    ):
        raise ValueError("Invalid diagnostic identity")
    raw_path = base / "runs/readiness" / study / f"gpu-allocation-{job}.json"
    raw = json.loads(raw_path.read_text())
    source = base / "checkouts" / commit / "scripts/hpc_job.py"
    proof = raw | {
        "execution_commit": commit,
        "allocation_code_hash": sha256(source),
        "gpu_type": spec["gpu_type"],
        "gpu_count": spec["gpu_count"],
    }
    visible = raw.get("visible", [])
    if not (
        raw.get("ready")
        and raw.get("ownership_verified")
        and raw.get("job_id") == job
        and raw.get("image_sha256") == spec["image_sha256"]
        and len(raw.get("allocated", [])) == spec["gpu_count"]
        and len(visible) == spec["gpu_count"]
        and [d["uuid"].removeprefix("GPU-").lower() for d in raw["allocated"]]
        == [d["uuid"].removeprefix("GPU-").lower() for d in visible]
        and all(
            0 < d["free_bytes"] <= d["total_bytes"]
            and d["total_bytes"] - d["free_bytes"] <= 2 * 1024**3
            for d in visible
        )
        and allocation_compatible(proof, payload["allocation_code_hash"], base)
    ):
        raise ValueError("Raw allocation proof is incomplete, incompatible or occupied")
    proof["publication_recovery"] = {
        "run_id": run_id,
        "source_report_sha256": sha256(raw_path),
        "publisher_commit": payload["commit"],
    }
    target = raw_path.parent / f"allocation-{profile}.json"
    with locked(base / "state/.allocation-publication.lock"):
        write_json(target, proof)
    return {"recovered": True, "job_id": job, "evidence": str(target), "recovery_gpu_hours": 0}


def study_state(payload: dict) -> dict:
    base = workspace(payload)
    study = payload["study_id"]
    if not SAFE_NAME.fullmatch(study):
        raise ValueError("Unsafe study ID")
    runs = base / "runs"

    def read(path):
        return json.loads(path.read_text()) if path.is_file() else {}

    images = {kind: read(base / "images" / f"{kind}.json") for kind in ("worker", "trainer")}
    evidence = runs / "readiness" / study
    identity = payload.get("runtime_identity")
    reports = {
        "worker": read(evidence / "worker.json"),
        "runtime": read(evidence / "runtime.json"),
        "single": read(runs / "compatibility/backend-v2/single_gpu_80gb.json"),
        "target": read(
            runs / "compatibility/backend-v2" / (payload.get("profile", "dual_gpu_80gb") + ".json")
        ),
        "base": read(runs / study / "base/evaluation_complete.json"),
        "pilot": read(
            runs
            / f"{study}{payload.get('pilot_suffix', '-pilot')}/D-seed{payload.get('seed', 2026)}/pilot_report.json"
        ),
        "freeze": read(runs / study / "study_frozen.json"),
        "methods": {m: read(evidence / "method-smokes" / f"{m}.json") for m in "ABC"},
        "telemetry": read(evidence / "wandb-verification.json"),
    }
    allocation_checks = {}
    for key, profile, count in (
        ("single", "single_gpu_80gb", 1),
        ("target", payload.get("profile", "dual_gpu_80gb"), payload.get("profile_gpu_count", 2)),
    ):
        proof = read(evidence / f"allocation-{profile}.json")
        allocation_checks[key] = bool(
            proof.get("ready")
            and proof.get("ownership_verified")
            and proof.get("image_sha256") == images["trainer"].get("sha256")
            and allocation_compatible(
                proof, payload.get("backend_identity", {}).get("allocation_code_hash"), base
            )
            and proof.get("gpu_count") == count
            and proof.get("gpu_type") == payload.get("gpu_type", "h100")
        )
    worker = reports["worker"]
    gates = {
        "worker": bool(
            worker.get("ready")
            and images["worker"].get("sha256")
            and worker.get("image_digest") == "sha256:" + images["worker"]["sha256"]
            and worker.get("supervisor_hash")
            == payload.get("supervisor_hash", worker.get("supervisor_hash"))
        )
    }
    if worker.get("execution_contract", {}).get("process") == "single_process":
        gates["worker"] = bool(
            gates["worker"]
            and worker.get("checks", {}).get("allocation_termination_cleanup")
            and worker.get("checks", {}).get("oom_candidate_cleanup")
            and worker.get("termination_proof", {}).get("ready")
        )
    for name in ("runtime", "single", "target", "base", "pilot", "freeze"):
        report = reports[name]
        backend = name in {"single", "target"}
        expected = payload.get("backend_identity") if backend else identity
        field = "backend_identity" if backend else "runtime_identity"
        gates[name] = bool(report and expected and report.get(field) == expected)
        if name in {"runtime", "single", "target", "base"}:
            gates[name] = gates[name] and bool(report.get("ready"))
        if name == "target":
            gates[name] = gates[name] and report.get("gpu_type") == payload.get("gpu_type", "h100")
        if backend:
            count = 1 if name == "single" else payload.get("profile_gpu_count", 2)
            profile = (
                "single_gpu_80gb" if name == "single" else payload.get("profile", "dual_gpu_80gb")
            )
            gates[name] = bool(
                gates[name]
                and report.get("gpu_count") == count
                and report.get("hardware_profile") == profile
                and report.get("objective_probes") == list("ABCD")
                and (count == 1 or report.get("single_process_reference_compared"))
            )
    readiness = read(runs / study / "readiness.json")
    gates["methods"] = all(
        report.get("ready")
        and identity
        and report.get("runtime_identity") == identity
        and report.get("method") == method
        and report.get("checks")
        and all(report["checks"].values())
        and report.get("gpu_type") == payload.get("gpu_type", "h100")
        for method, report in reports["methods"].items()
    )
    eligible = bool(
        readiness.get("eligible_to_submit_main")
        and all(gates.values())
        and identity
        and readiness.get("runtime_identity") == identity
        and readiness.get("evidence_files")
    )
    if eligible:
        for relative, expected in readiness["evidence_files"].items():
            path = runs / relative
            if (
                not path.resolve().is_relative_to(runs.resolve())
                or not path.is_file()
                or sha256(path) != expected
            ):
                eligible = False
                break
    memory_fit_failure = False
    for ledger_file in (base / "state").glob("*.json"):
        ledger = read(ledger_file)
        spec = ledger.get("specification", {})
        if (
            spec.get("study_id") != study
            or spec.get("gpu_type") != "h100"
            or not ledger.get("attempts")
        ):
            continue
        health = read(Path(ledger.get("run_dir", "")) / "state/health.json")
        error = str(health.get("error", "")).lower()
        if spec.get("operation") == "backend-check":
            log = base / "state" / f"{spec['run_id']}-{ledger['attempts'][-1]['job_id']}.out"
            if log.is_file():
                with log.open("rb") as stream:
                    stream.seek(max(0, log.stat().st_size - 64000))
                    error = stream.read().decode(errors="replace").lower()
        if (
            "out of memory" in error
            and "cuda" in error
            and job_state(ledger["attempts"][-1]["job_id"])["state"] == "FAILED"
        ):
            allocation = read(evidence / f"gpu-allocation-{ledger['attempts'][-1]['job_id']}.json")
            # An occupied or incorrectly mapped GPU is not model memory-fit
            # evidence. Older attempts without allocation proof are ineligible.
            memory_fit_failure = bool(
                memory_fit_failure
                or (
                    allocation.get("ready")
                    and allocation.get("image_sha256") == spec.get("image_sha256")
                    and len(allocation.get("allocated", [])) == spec.get("gpu_count")
                )
            )
    return {
        "study_id": study,
        "h100_memory_fit_failure": memory_fit_failure,
        "images": images,
        "gates": gates,
        "allocation_checks": allocation_checks,
        "stage_status": stage_status(base, study, gates, reports),
        "reports": reports,
        "eligible_to_submit_main": eligible,
        "readiness": readiness,
        "budget": refreshed_budget(payload),
        "base": str(base),
        "artifact_directories": {
            name: str(path)
            for name, path in {
                study: runs / study,
                study + "-pilot": runs / (study + "-pilot"),
                study + "-pilot-h200": runs / (study + "-pilot-h200"),
                "compatibility": runs / "compatibility",
                "readiness": evidence,
                study + "-method-smoke": runs / (study + "-method-smoke"),
            }.items()
            if path.is_dir()
        },
    }


def precheck(payload: dict):
    operation = payload["operation"]
    excluded = payload.get("exclude_nodes", [])
    if excluded and (
        operation not in {"allocation-check", "backend-check"}
        or not isinstance(excluded, list)
        or len(excluded) > 16
        or any(
            not isinstance(node, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9-]{0,63}", node)
            for node in excluded
        )
    ):
        raise ValueError("Invalid diagnostic node exclusions")
    if payload.get("partition") == "scavenge_gpu" and operation not in {
        "allocation-check",
        "backend-check",
    }:
        raise ValueError("scavenge_gpu is enabled only for trusted stateless diagnostics")
    if operation not in {
        "image-convert",
        "image-build",
        "method-smoke",
        "protocol-check",
        "protocol-revision",
        "backend-check",
        "rollout-parity",
        "numerical-audit",
        "wandb-verify",
        "allocation-check",
        "worker-diagnose",
        "worker-check",
        "runtime-check",
        "train",
        "pilot",
        "evaluate",
        "diagnose",
        "freeze-study",
        "readiness-check",
    }:
        raise ValueError("Unsupported submission operation")
    if operation == "image-build":
        validate_source_input(payload)
        if payload["image_metadata"]["kind"] == "trainer":
            proof = workspace(payload) / "images/build-check.json"
            report = json.loads(proof.read_text()) if proof.is_file() else {}
            if not report.get("ready") or report.get("source_commit") != payload["commit"]:
                raise ValueError("Run the Yale image build --check-only first for this commit")
        return
    if operation == "image-convert":
        if not Path(payload["archive"]).is_file() or not DIGEST.fullmatch(
            payload["archive_sha256"]
        ):
            raise ValueError("Image archive missing")
        return
    if operation == "wandb-verify":
        if (
            payload.get("gpu_count") != 0
            or payload.get("time") != "00:15:00"
            or payload.get("account") != "cpsc4880"
        ):
            raise ValueError("W&B verification requires bounded CPU-only cpsc4880 allocation")
        return
    if operation == "worker-diagnose":
        if payload.get("gpu_count") != 0:
            raise ValueError("Worker diagnostics must be CPU-only")
        return
    if operation in {"backend-check", "allocation-check"}:
        # This entry point only scores trusted synthetic fixtures. Worker state,
        # including stale optional worker paths, must not couple this diagnostic.
        if not Path(payload.get("image", "")).is_file() or not DIGEST.fullmatch(
            payload.get("image_sha256", "")
        ):
            raise ValueError("Pinned trainer SIF missing")
        if operation == "allocation-check" and (
            payload.get("gpu_count") not in {1, 2}
            or payload.get("gpu_type") not in {"h100", "h200"}
            or payload.get("time") != "00:05:00"
            or payload.get("account") != "cpsc4880"
            or payload.get("readiness_budget") != 24
        ):
            raise ValueError("Allocation diagnostics require bounded cpsc4880 H100/H200 readiness")
        return
    if operation in {"rollout-parity", "numerical-audit"}:
        if (
            payload.get("account") != "cpsc4880"
            or payload.get("gpu_count") != 1
            or payload.get("gpu_type") != "h200"
            or payload.get("time") != "01:00:00"
            or payload.get("readiness_budget") != 24
            or not Path(payload.get("image", "")).is_file()
            or not DIGEST.fullmatch(payload.get("image_sha256", ""))
            or not study_state(payload)["gates"]["target"]
        ):
            raise ValueError("Rollout parity requires accepted one-H200 backend and bounded budget")
        return
    uses_worker = operation in {
        "worker-check",
        "runtime-check",
        "train",
        "pilot",
        "method-smoke",
        "protocol-check",
        "protocol-revision",
        "evaluate",
        "diagnose",
    }
    if uses_worker and not payload.get("worker_image"):
        raise ValueError("Yale jobs require the verified allocation worker")
    if uses_worker:
        if not Path(payload["worker_image"]).is_file() or not DIGEST.fullmatch(
            payload.get("worker_sha256", "")
        ):
            raise ValueError("Pinned worker SIF missing")
    if operation == "worker-check":
        prior_path = paths(payload)["runs"] / "readiness" / payload["study_id"] / "worker.json"
        if prior_path.is_file():
            prior = json.loads(prior_path.read_text())
            detail = json.dumps(prior.get("details", {})).lower()
            if (
                not prior.get("ready")
                and ("dbus_session_bus_address" in detail or "delegat" in detail)
                and payload.get("worker_setup_id", "") == prior.get("worker_setup_id", "")
                and payload.get("supervisor_hash") == prior.get("supervisor_hash")
            ):
                raise ValueError(
                    "Worker site setup is unchanged; obtain YCRC-supported setup and supply a new worker setup ID"
                )
    if operation in {"protocol-check", "protocol-revision"}:
        if payload.get("account") != "cpsc4880" or not study_state(payload)["gates"]["worker"]:
            raise ValueError("Protocol work requires accepted cpsc4880 worker evidence")
        if operation == "protocol-check" and (
            payload.get("gpu_count") != 1
            or payload.get("time") != "02:00:00"
            or payload.get("readiness_budget") != 24
        ):
            raise ValueError("Protocol probe requires its fixed two-GPU-hour ceiling")
    if operation == "train" and payload.get("logging_mode") == "online":
        telemetry = study_state(payload)["reports"]["telemetry"]
        if (
            not telemetry.get("ready")
            or not telemetry.get("cloud_verified")
            or telemetry.get("sequence") != 2
            or telemetry.get("telemetry_identity") != payload.get("telemetry_identity")
            or telemetry.get("telemetry_identity")
            != study_state(payload)["reports"]["freeze"].get("telemetry_identity")
        ):
            raise ValueError(
                "Online main launch requires allocated-node cloud metric and resume verification"
            )
        if not payload.get("secrets_file"):
            raise ValueError("Online main launch requires a persistent private secrets file")
    if operation in {"train", "pilot", "method-smoke", "evaluate", "diagnose", "freeze-study"}:
        state = study_state(payload)
        if not state["gates"]["worker"] or not state["gates"]["runtime"]:
            raise ValueError("Runtime isolation and target data validation have not passed")
        if operation == "freeze-study":
            plan_path = workspace(payload) / "runs" / payload["study_id"] / "resource_plan.json"
            if not plan_path.is_file():
                raise ValueError("A measured resource plan is required before Yale study freeze")
        if operation == "train" or (
            operation == "evaluate" and payload.get("split") in {"final_id", "external"}
        ):
            if operation == "train" and not state["reports"]["freeze"].get("resource_plan"):
                raise ValueError("Frozen study lacks a measured resource plan for main training")
            if (
                not state["eligible_to_submit_main"]
                or state["readiness"].get("common_config") != payload.get("common_config")
                or state["reports"]["freeze"].get("execution_commit") != payload.get("commit")
                or state["readiness"].get("gpu_type") != payload.get("gpu_type")
            ):
                raise ValueError("Study is not eligible for main submission")


def preview(payload: dict) -> dict:
    precheck(payload)
    return {
        "dry_run": True,
        "submitted": False,
        "resources": allowed(payload),
        "specification": payload,
        "artifact_directory": str(artifact_dir(payload)),
    }


def prepare_image(payload: dict) -> dict:
    base = workspace(payload)
    kind, archive_hash = payload["kind"], payload["archive_sha256"]
    if kind not in {"worker", "trainer"} or not DIGEST.fullmatch(archive_hash):
        raise ValueError("Invalid image identity")
    target = base / "images" / kind / archive_hash
    incoming = base / "incoming" / f"{archive_hash}.tar"
    for directory in (target, incoming.parent, base / "state"):
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    size = int(payload["archive_bytes"])
    if size <= 0 or shutil.disk_usage(base).free < size * 2:
        raise ValueError("Insufficient project capacity for archive and SIF")
    return {
        "archive": str(incoming),
        "image_target": str(target / f"{kind}.sif"),
        "image_manifest": str(base / "images" / f"{kind}.json"),
    }


def validate_source_input(payload: dict):
    metadata = payload["image_metadata"]
    key = hashlib.sha256(
        json.dumps(
            metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()
    if (
        metadata.get("kind") not in {"trainer", "build-check"}
        or metadata.get("build_mode") != "yale"
        or metadata.get("source_commit") != payload["commit"]
        or metadata.get("platform") != "linux/amd64"
        or key != payload["build_key"]
        or hashlib.sha256(payload["definition"].encode()).hexdigest()
        != metadata["definition_sha256"]
    ):
        raise ValueError("Invalid source build identity")


def prepare_source_image(payload: dict) -> dict:
    validate_source_input(payload)
    base = workspace(payload)
    kind, key = payload["image_metadata"]["kind"], payload["build_key"]
    target = base / "images" / kind / key
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    definition = target / "image.def"
    with locked(base / "state/.source-images.lock"):
        if definition.exists() and definition.read_text() != payload["definition"]:
            raise ValueError("Conflicting immutable image definition")
        if not definition.exists():
            temporary = target / (uuid.uuid4().hex + ".pending.def")
            temporary.write_text(payload["definition"])
            os.replace(temporary, definition)
    return {
        "definition_path": str(definition),
        "image_target": str(target / f"{kind}.sif"),
        "image_manifest": str(base / "images" / f"{kind}.json"),
    }


def reuse_worker(payload: dict) -> dict:
    """Publish dependency-checked containment evidence with original provenance."""
    source_study = payload["source_study"]
    if not SAFE_NAME.fullmatch(source_study):
        raise ValueError("Unsafe source study")
    source = workspace(payload) / "runs/readiness" / source_study / "worker.json"
    report = json.loads(source.read_text())
    state = study_state(payload | {"study_id": source_study})
    if not state["gates"]["worker"] or not all(report.get("checks", {}).values()):
        raise ValueError("Worker evidence dependencies differ or acceptance is incomplete")
    if report.get("worker_setup_id", "") != payload.get("worker_setup_id", ""):
        raise ValueError("Worker setup changed")
    limits = {
        key: value
        for key, value in report.get("limits", {}).items()
        if key not in {"backend", "image", "runtime", "remote_url", "remote_token_env", "pids"}
    }
    if limits != payload.get("execution_limits"):
        raise ValueError("Worker execution limits changed")
    target = workspace(payload) / "runs/readiness" / payload["study_id"] / "worker.json"
    if not SAFE_NAME.fullmatch(payload["study_id"]):
        raise ValueError("Unsafe destination study")
    reused = report | {"reuse_provenance": {"source": str(source), "sha256": sha256(source)}}
    with locked(workspace(payload) / "state/.worker-evidence.lock"):
        if target.exists() and json.loads(target.read_text()) != reused:
            raise ValueError("Destination already has different worker evidence")
        write_json(target, reused)
    return {"ready": True, "evidence": str(target), "provenance": reused["reuse_provenance"]}


def publish_resource_plan(payload: dict) -> dict:
    """Publish a measured plan once, before the immutable study freeze."""
    study = payload["study_id"]
    if not SAFE_NAME.fullmatch(study):
        raise ValueError("Unsafe study name")
    if payload.get("pilot_suffix") not in {"-pilot", "-pilot-h200"}:
        raise ValueError("Invalid pilot suffix")
    if type(payload.get("seed")) is not int or payload["seed"] < 0:
        raise ValueError("Invalid seed")

    def read(path):
        return json.loads(path.read_text()) if path.is_file() else {}

    base = workspace(payload)
    root = base / "runs" / study
    evidence = base / "runs/readiness" / study / "method-smokes"
    plan = payload["plan"]
    pilot = read(
        base
        / "runs"
        / (study + payload["pilot_suffix"])
        / f"D-seed{payload['seed']}/pilot_report.json"
    )
    baseline = read(root / "base/evaluation_complete.json")
    methods = {m: read(evidence / f"{m}.json") for m in "ABC"}

    def digest_json(value):
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    if (
        not pilot
        or not baseline.get("ready")
        or not all(r.get("ready") for r in methods.values())
        or plan.get("runtime_identity") != pilot.get("runtime_identity")
        or baseline.get("runtime_identity") != plan.get("runtime_identity")
        or any(r.get("runtime_identity") != plan.get("runtime_identity") for r in methods.values())
        or plan.get("common_config") != payload["common_config"]
        or plan.get("hardware") != payload["hardware"]
        or plan.get("checkpoint_every") != payload["checkpoint_every"]
        or plan.get("gpu_type") != payload["gpu_type"]
        or plan.get("pilot_hash") != digest_json(pilot)
        or plan.get("baseline_hash") != digest_json(baseline)
        or plan.get("method_hashes") != {m: digest_json(r) for m, r in methods.items()}
    ):
        raise ValueError("Resource plan does not match measured study evidence")
    for method in "ABCD":
        entry = plan["per_method"][method]
        if not (
            0 < entry["planning_projection_200_hours"] < float("inf")
            and type(entry["signal_seconds"]) is int
            and 300 <= entry["signal_seconds"] <= 65535
            and type(entry["max_continuations"]) is int
            and 1 <= entry["max_continuations"] <= 32
        ):
            raise ValueError("Invalid resource plan continuation envelope")
    path = root / "resource_plan.json"
    with locked(base / "state/.resource-plan.lock"):
        if (root / "study_frozen.json").exists():
            raise ValueError("Study is frozen; use a new study ID")
        if path.exists() and read(path) != plan:
            raise ValueError("A different resource plan already exists")
        write_json(path, plan)
    return {"ready": True, "path": str(path), "sha256": sha256(path)}


def reuse_study(payload: dict) -> dict:
    """Reuse a frozen study only when all scientific/runtime dependencies match.

    Historical checkpoints are verified in place. No hash inside an original
    report or checkpoint is rewritten to pretend it belongs to a new study.
    """
    source_study, study = payload["source_study"], payload["study_id"]
    if (
        not SAFE_NAME.fullmatch(source_study)
        or not SAFE_NAME.fullmatch(study)
        or source_study == study
    ):
        raise ValueError("Distinct safe study names required")
    state = study_state(payload | {"study_id": source_study})
    frozen = state["reports"]["freeze"]
    plan = frozen.get("resource_plan", {})

    def common(value):
        return {k: v for k, v in value.items() if k != "study_id"}

    if (
        not state["eligible_to_submit_main"]
        or not plan
        or frozen.get("runtime_identity") != payload["runtime_identity"]
        or common(frozen["common_config"]) != common(payload["common_config"])
        or plan.get("hardware") != payload["hardware"]
        or plan.get("checkpoint_every") != payload["checkpoint_every"]
        or frozen.get("gpu_type") != payload["gpu_type"]
    ):
        raise ValueError("Source study is not frozen, ready or dependency-equivalent")
    base = workspace(payload)
    runs = base / "runs"
    source = runs / source_study
    target = runs / study
    evidence = runs / "readiness" / study
    original_evidence = runs / "readiness" / source_study
    if (target / "study_frozen.json").exists():
        raise ValueError("Target study already frozen")
    files = {}
    for name in (
        "worker.json",
        "runtime.json",
        "validation.json",
        "allocation-single_gpu_80gb.json",
    ):
        path = original_evidence / name
        if path.is_file():
            files[evidence / name] = json.loads(path.read_text())
    for name in ("evaluation_complete.json", "runtime_identity.json"):
        files[target / "base" / name] = json.loads((source / "base" / name).read_text())
    for path in (source / "base/events").glob("*.jsonl"):
        # Copy exact event bytes separately below; only trusted baseline records.
        files[target / "base/events" / path.name] = path
    pilot_relative = f"{payload['pilot_suffix']}/D-seed{payload['seed']}/pilot_report.json"
    pilot_source = runs / (source_study + pilot_relative)
    files[runs / (study + pilot_relative)] = json.loads(pilot_source.read_text())
    for method in "ABC":
        path = original_evidence / "method-smokes" / f"{method}.json"
        original = json.loads(path.read_text())
        if original.get("reuse_provenance"):
            raise ValueError("Reuse must point directly to original method evidence")
        files[evidence / "method-smokes" / f"{method}.json"] = original | {
            "reuse_provenance": {"source_study": source_study, "sha256": sha256(path)}
        }

    def digest_json(value):
        return hashlib.sha256(
            json.dumps(
                value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
            ).encode()
        ).hexdigest()

    new_plan = dict(plan)
    new_plan["common_config"] = payload["common_config"]
    new_plan["method_hashes"] = {
        m: digest_json(files[evidence / "method-smokes" / f"{m}.json"]) for m in "ABC"
    }
    new_plan["reuse_provenance"] = {
        "source_study": source_study,
        "freeze_sha256": sha256(source / "study_frozen.json"),
        "resource_plan_sha256": sha256(source / "resource_plan.json"),
    }
    files[target / "resource_plan.json"] = new_plan
    manifest = {
        "source_study": source_study,
        "target_study": study,
        "runtime_identity": payload["runtime_identity"],
        "source_freeze_sha256": sha256(source / "study_frozen.json"),
        "note": "Scientific dependencies identical; historical checkpoints retained in place.",
    }
    files[evidence / "reuse.json"] = manifest
    with locked(base / "state/.study-reuse.lock"):
        # Validate all existing destinations before writing any of them.
        for destination, value in files.items():
            if destination.exists():
                matches = (
                    (destination.read_bytes() == value.read_bytes())
                    if isinstance(value, Path)
                    else json.loads(destination.read_text()) == value
                )
                if not matches:
                    raise ValueError("Conflicting destination evidence")
        for destination, value in files.items():
            if isinstance(value, Path):
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(value.read_bytes())
            else:
                write_json(destination, value)
    return {
        "ready": True,
        "reuse": manifest,
        "next_action": "CPU freeze-study, then four main previews",
    }


COMMANDS = {
    "inspect": inspect,
    "context": context,
    "study": study_state,
    "preview": preview,
    "prepare_image": prepare_image,
    "prepare_source_image": prepare_source_image,
    "resources": allowed,
    "prepare": prepare,
    "finalize": finalize,
    "status": status,
    "logs": logs,
    "submit": submit,
    "recover": recover,
    "recover_allocation": recover_allocation,
    "reuse_worker": reuse_worker,
    "reuse_study": reuse_study,
    "publish_resource_plan": publish_resource_plan,
}


def main() -> None:
    command = sys.argv[1]
    payload = json.loads(base64.b64decode(sys.argv[2])) if len(sys.argv) > 2 else {}
    try:
        print(json.dumps(COMMANDS[command](payload), sort_keys=True))
    except (
        KeyError,
        ValueError,
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as exc:
        print(json.dumps({"error": str(exc), "type": type(exc).__name__}, sort_keys=True))
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
