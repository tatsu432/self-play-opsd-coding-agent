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
    return {
        "project": {p.name: str(p.resolve()) for p in home.glob("project_*") if p.is_dir()},
        "scratch": {p.name: str(p.resolve()) for p in home.glob("scratch*") if p.is_dir()},
    }


def inspect(_: dict) -> dict:
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
    study = payload["study_id"] + ("-pilot" if payload["operation"] == "pilot" else "")
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
    if metadata.get("scientific_hash") != payload.get("scientific_hash") or metadata.get(
        "manifests_hash"
    ) != payload.get("manifests_hash"):
        raise ValueError("Checkpoint scientific or dataset identity mismatch")
    return metadata


def recover(payload: dict) -> dict:
    path = ledger_path(payload)
    ledger = read_ledger(path)
    if ledger.get("identity") != payload["identity"] or not ledger["attempts"]:
        raise ValueError("Run identity or initial submission missing")
    if ledger.get("pending_intent"):
        return {"submitted": False, "state": "UNKNOWN", "reason": "unreconciled_submission"}
    state = job_state(ledger["attempts"][-1]["job_id"])["state"]
    if state in ACTIVE or state == "UNKNOWN":
        return {"submitted": False, "state": state}
    if state == "COMPLETED":
        return {"submitted": False, "state": state}
    if state not in {"PREEMPTED", "NODE_FAIL", "BOOT_FAIL"}:
        return {"submitted": False, "state": state, "reason": "not_transient"}
    if ledger.get("retries", 0) >= 2:
        return {"submitted": False, "state": state, "reason": "retry_limit"}
    health_path = Path(ledger.get("run_dir", "")) / "state/health.json"
    if health_path.is_file() and json.loads(health_path.read_text()).get("phase") == "failed":
        return {"submitted": False, "state": state, "reason": "training_reported_failure"}
    try:
        checkpoint = verified_checkpoint(payload)
    except (OSError, KeyError, ValueError) as exc:
        return {"submitted": False, "state": state, "reason": f"checkpoint_unavailable: {exc}"}
    # Reserve a retry under the same lock as submission; a failed reservation is
    # left auditable and cannot cause an unbounded scheduler loop.
    with locked(path.with_suffix(".lock")):
        current = read_ledger(path)
        previous_job = ledger["attempts"][-1]["job_id"]
        if (
            current.get("retries", 0) >= 2
            or current["attempts"][-1]["job_id"] != previous_job
            or current.get("recovery_from_job") == previous_job
        ):
            return {"submitted": False, "state": state, "reason": "concurrent_recovery"}
        current["retries"] = current.get("retries", 0) + 1
        current["recovery_from_job"] = previous_job
        write_json(path, current)
    result = submit({**payload, "resume": checkpoint})
    return {**result, "state": state, "resume": checkpoint}


def allowed(payload: dict) -> dict:
    account = payload["account"]
    partition = payload["partition"]
    gpu_type = payload["gpu_type"]
    gpu_count = payload["gpu_count"]
    if not all(SAFE_NAME.fullmatch(x) for x in (account, partition, gpu_type)):
        raise ValueError("Unsafe Slurm resource name")
    if gpu_type not in GPU_VRAM_GB or gpu_count not in (1, 2, 4, 8):
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
    if not available_counts or max(available_counts) < gpu_count:
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
    nodes = run(["sinfo", "-h", "-p", partition, "-o", "%c|%m|%G"]).stdout
    fits_node = False
    for row in nodes.splitlines():
        fields = row.split("|", 2)
        if len(fields) == 3 and fields[0].isdigit() and fields[1].isdigit():
            match = re.search(rf"(?:^|,)gpu:{re.escape(gpu_type)}:(\d+)", fields[2])
            if (
                match
                and int(match[1]) >= gpu_count
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
        "max_gpus_per_node": max(available_counts),
    }


def submit(payload: dict) -> dict:
    p = paths(payload)
    path = ledger_path(payload)
    sync = json.loads((p["state"] / f"sync-{payload['commit']}.json").read_text())
    if sync != {"commit": payload["commit"], "manifest_sha256": payload["manifest_sha256"]}:
        raise ValueError("Remote snapshot identity mismatch")
    if not Path(payload["image"]).is_file() or not DIGEST.fullmatch(payload["image_sha256"]):
        raise ValueError("Pinned trainer SIF missing")
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
    if (
        payload["gpu_count"] != payload["profile_gpu_count"]
        or GPU_VRAM_GB[payload["gpu_type"]] < payload["min_vram_gib"]
    ):
        raise ValueError("GPU request does not satisfy the selected hardware profile")
    resources = allowed(payload)
    with locked(path.with_suffix(".lock")):
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
            if not payload.get("resume"):
                raise ValueError("Existing run requires explicit checkpoint resume")
            if state["state"] not in {
                "PREEMPTED",
                "NODE_FAIL",
                "BOOT_FAIL",
                "TIMEOUT",
                "FAILED",
                "CANCELLED",
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
        ledger["run_dir"] = str(
            p["runs"]
            / (payload["study_id"] + ("-pilot" if payload["operation"] == "pilot" else ""))
            / f"{payload['method']}-seed{payload['seed']}"
        )
        ledger["pending_intent"] = token
        ledger["pending_resume"] = payload.get("resume")
        write_json(path, ledger)
        exports = {
            "SELFPLAY_PROJECT": str(p["checkout"]),
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
            "SELFPLAY_STUDY": payload["study_id"]
            + ("-pilot" if payload["operation"] == "pilot" else ""),
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
            f"--gpus={payload['gpu_type']}:{payload['gpu_count']}",
            f"--cpus-per-task={payload['cpus']}",
            f"--mem={payload['memory_gib']}G",
            f"--time={payload['time']}",
            f"--comment={token}",
            f"--output={p['state']}/{payload['run_id']}-%j.out",
        ]
        if payload.get("qos"):
            args.append(f"--qos={payload['qos']}")
        args += ["--export=NONE," + ",".join(f"{k}={v}" for k, v in exports.items()), str(script)]
        result = run(args, check=False)
        if result.returncode:
            ledger.pop("pending_intent")
            ledger.pop("pending_resume", None)
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


COMMANDS = {
    "inspect": inspect,
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
