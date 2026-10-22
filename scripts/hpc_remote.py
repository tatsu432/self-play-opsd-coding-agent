#!/usr/bin/env python3
"""Standard-library Yale control endpoint, sent over SSH on stdin by hpc.py.

The login node only performs metadata queries, small file transfers and Slurm
submission. All model and sandbox work belongs to allocated compute nodes.
"""

from __future__ import annotations

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
    tmp.write_text(json.dumps(value, sort_keys=True) + "\n")
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
    if payload["operation"] not in {"image-convert", "worker-check"}:
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
                and payload["gpu_count"] == 0
                and state["state"] in {"PREEMPTED", "NODE_FAIL", "BOOT_FAIL", "TIMEOUT"}
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
                        if payload["operation"] == "image-convert"
                        else str(artifact)
                    ).encode()
                ).hexdigest()
                + ".json"
            )
        )
        if slot.exists():
            owner = json.loads(slot.read_text())
            if owner["run_id"] != payload["run_id"]:
                prior = status({**payload, "run_id": owner["run_id"]})
                states = [a["scheduler"]["state"] for a in prior["attempts"]]
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
            "SELFPLAY_TRAINER_SIF": payload["image"],
            "SELFPLAY_TRAINER_SHA256": payload["image_sha256"],
            "SELFPLAY_DATA": str(p["dataset"]),
            "SELFPLAY_RUNS": str(p["runs"]),
            "SELFPLAY_CACHE": str(p["cache"]),
            "SELFPLAY_OPERATION": payload["operation"],
            "SELFPLAY_METHOD": payload["method"],
            "SELFPLAY_HARDWARE": payload["profile"],
            "SELFPLAY_GPU_COUNT": str(payload["gpu_count"]),
            "SELFPLAY_SETTINGS": payload["settings"],
            "SELFPLAY_MAIN_STUDY": payload["study_id"],
            "SELFPLAY_PILOT_SUFFIX": payload.get("pilot_suffix", "-pilot"),
            "SELFPLAY_GPU_TYPE": payload["gpu_type"],
            "SELFPLAY_STUDY": payload["study_id"]
            + (payload.get("pilot_suffix", "-pilot") if payload["operation"] == "pilot" else ""),
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
    if operation == "image-convert":
        return (
            workspace(payload)
            / "images"
            / payload["image_metadata"]["kind"]
            / payload["archive_sha256"]
        )
    if operation in {"worker-check", "runtime-check", "readiness-check"}:
        return runs / "readiness" / study
    if operation == "backend-check":
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
    return workspace(payload) / "state" / f"budget-{payload['study_id']}.json"


def refreshed_budget(payload: dict) -> dict:
    path = budget_path(payload)
    budget = json.loads(path.read_text()) if path.exists() else {"limit": 24.0, "reservations": {}}
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
                "JobIDRaw,State,ElapsedRaw,AllocTRES",
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
        "single": read(runs / "compatibility/single_gpu_80gb.json"),
        "target": read(
            runs / "compatibility" / (payload.get("profile", "dual_gpu_80gb") + ".json")
        ),
        "base": read(runs / study / "base/evaluation_complete.json"),
        "pilot": read(
            runs
            / f"{study}{payload.get('pilot_suffix', '-pilot')}/D-seed{payload.get('seed', 2026)}/pilot_report.json"
        ),
        "freeze": read(runs / study / "study_frozen.json"),
    }
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
    for name in ("runtime", "single", "target", "base", "pilot", "freeze"):
        report = reports[name]
        gates[name] = bool(report and identity and report.get("runtime_identity") == identity)
        if name in {"runtime", "single", "target", "base"}:
            gates[name] = gates[name] and bool(report.get("ready"))
        if name == "target":
            gates[name] = gates[name] and report.get("gpu_type") == payload.get("gpu_type", "h100")
    readiness = read(runs / study / "readiness.json")
    eligible = bool(
        readiness.get("eligible_to_submit_main")
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
            memory_fit_failure = True
    return {
        "study_id": study,
        "h100_memory_fit_failure": memory_fit_failure,
        "images": images,
        "gates": gates,
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
            }.items()
            if path.is_dir()
        },
    }


def precheck(payload: dict):
    operation = payload["operation"]
    if operation == "image-convert":
        if not Path(payload["archive"]).is_file() or not DIGEST.fullmatch(
            payload["archive_sha256"]
        ):
            raise ValueError("Image archive missing")
        return
    if payload.get("worker_image"):
        if not Path(payload["worker_image"]).is_file() or not DIGEST.fullmatch(
            payload.get("worker_sha256", "")
        ):
            raise ValueError("Pinned worker SIF missing")
    if payload["gpu_count"] or operation == "freeze-study":
        if not payload.get("worker_image"):
            raise ValueError("Yale jobs require the verified allocation worker")
        state = study_state(payload)
        if not state["gates"]["worker"] or not state["gates"]["runtime"]:
            raise ValueError("Runtime isolation and target data validation have not passed")
        if operation == "train" or (
            operation == "evaluate" and payload.get("split") in {"final_id", "external"}
        ):
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


COMMANDS = {
    "inspect": inspect,
    "context": context,
    "study": study_state,
    "preview": preview,
    "prepare_image": prepare_image,
    "resources": allowed,
    "prepare": prepare,
    "finalize": finalize,
    "status": status,
    "logs": logs,
    "submit": submit,
    "recover": recover,
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
