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
import math
import os
import re
import shutil
import subprocess
import sys
import uuid
import zlib
from collections import deque
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

GPU_VRAM_GB = {"h100": 80, "h200": 141, "rtx_pro_6000_blackwell": 96}
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
SHA = re.compile(r"^[0-9a-f]{40}$")
DIGEST = re.compile(r"^[0-9a-f]{64}$")
TASK19_OPERATIONS = {
    "task19-qualify",
    "task19-qualify-uncached",
    "task19-base",
    "task19-pilot",
    "task19-train",
}
TASK20_OPERATIONS = {
    "task20-screen",
    "task20-solver",
    "task20-calibrate",
    "task20-shared-pilot",
    "task20-main",
    "task20-evaluate",
}

TASK21_OPERATIONS = {
    "task21-feasibility",
    *(f"task21-{stage.removeprefix('task20-')}" for stage in TASK20_OPERATIONS),
}
STAGED_OPERATIONS = TASK20_OPERATIONS | TASK21_OPERATIONS
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
            run(
                ["git", "-C", str(staged), "checkout", "--quiet", "--detach", payload["commit"]],
                timeout=600,
            )
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
    run_dir = (
        artifact_dir(payload)
        if payload["operation"]
        in TASK19_OPERATIONS | STAGED_OPERATIONS | {"task17-student", "task18-student"}
        else p["runs"] / study / f"{payload['method']}-seed{payload['seed']}"
    )
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
    if payload.get("operation") in TASK19_OPERATIONS | STAGED_OPERATIONS:
        identity = metadata.get("identity", {})
        if (
            identity.get("study") != payload["study_id"]
            or identity.get("arm") != payload["method"]
            or metadata.get("manifests_hash") != payload["manifests_hash"]
            or identity.get("execution", {}).get("image") != payload["image_sha256"]
        ):
            raise ValueError("Shared-policy checkpoint study, dataset or image differs")
        if (
            payload["operation"] in STAGED_OPERATIONS
            and metadata.get("scientific_hash") != payload["scientific_hash"]
        ):
            raise ValueError("Staged checkpoint scientific control differs")
        return metadata
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


def timeout_disconnect_evidence(job_id: str, health: dict) -> dict | None:
    """Recognize only an RPC disconnect contemporaneous with a Slurm timeout."""
    message = health.get("error", "")
    if not re.fullmatch(
        r"vLLM RPC closed before a response; see /runs/[A-Za-z0-9_./-]+/state/vllm\.log \(child exit status None\)",
        message,
    ) or not isinstance(health.get("timestamp"), (int, float)):
        return None
    result = capture(["sacct", "-n", "-P", "-j", job_id, "--format", "JobIDRaw,State,End"])
    if result["exit_code"] != 0:
        return None
    for line in result["stdout"].splitlines():
        fields = line.split("|")
        if (
            len(fields) >= 3
            and fields[0] == job_id
            and fields[1].split()[0].rstrip("+") == "TIMEOUT"
        ):
            try:
                end = datetime.fromisoformat(fields[2]).timestamp()
            except ValueError:
                return None
            if abs(health["timestamp"] - end) <= 2:
                return {"job_id": job_id, "scheduler_end": fields[2], "health": health}
    return None


def reviewed_task21_timeout_disconnect(payload: dict, job_id: str, health: dict) -> dict | None:
    """Recognize the reviewed worker shutdown in the 2026 Control allocation."""
    reviewed = {
        "34298915": "237fb774dc01bcce763e228c",
        "34300641": "09e87387be4df8df15ec5f17",
    }
    if (
        job_id not in reviewed
        or payload.get("operation") != "task21-main"
        or payload.get("run_id") != reviewed[job_id]
        or health.get("allocation") != job_id
        or health.get("phase") != "failed"
        or health.get("error_type") != "ExperimentError"
    ):
        return None
    log = ledger_path(payload).parent / f"{payload['run_id']}-{job_id}.out"
    try:
        text = log.read_text()
    except OSError:
        return None
    if "selfplay_opsd.core.ExperimentError: [Errno 111] Connection refused" not in text:
        return None
    result = capture(["sacct", "-n", "-P", "-j", job_id, "--format", "JobIDRaw,State,End"])
    if result["exit_code"] != 0:
        return None
    rows = {
        fields[0]: fields
        for line in result["stdout"].splitlines()
        if len(fields := line.split("|")) >= 3
    }
    job, worker = rows.get(job_id), rows.get(f"{job_id}.0")
    if not job or not worker or job[1].split()[0].rstrip("+") != "TIMEOUT" or job[2] != worker[2]:
        return None
    return {
        "job_id": job_id,
        "scheduler_end": job[2],
        "worker_end": worker[2],
        "log_sha256": sha256(log),
        "health": health,
        "reason": "reviewed verifier disconnect when scheduler ended worker",
    }


def reconcile_rejected_specification(ledger: dict, specification: dict) -> None:
    if ledger.get("last_rejection") and not ledger.get("attempts") and ledger.get("specification"):
        previous = ledger["specification"]
        changed = {
            key
            for key in previous.keys() | specification.keys()
            if previous.get(key) != specification.get(key)
        }
        if changed == {"cpus"} and 1 <= specification["cpus"] < previous["cpus"]:
            ledger.setdefault("rejected_resource_revisions", []).append(
                {
                    "field": "cpus",
                    "before": previous["cpus"],
                    "after": specification["cpus"],
                    "rejection": ledger["last_rejection"],
                }
            )
            ledger["specification"] = specification
        elif changed:
            raise ValueError("Rejected submission may only reduce CPUs before retry")
    ledger.setdefault("specification", specification)


def recover(payload: dict) -> dict:
    request = payload
    path = ledger_path(payload)
    ledger = read_ledger(path)
    if ledger.get("specification"):
        payload = ledger["specification"] | {
            "explicit_continue": bool(request.get("explicit_continue")),
            "stop_after_iteration": ledger.get(
                "active_stop_after_iteration",
                ledger["specification"].get("stop_after_iteration", 0),
            ),
        }
    requested_stop = request.get("continuation_stop")
    if requested_stop is not None and not request.get("explicit_continue"):
        raise ValueError("A continuation stop requires explicit clean continuation")
    if ledger.get("identity") != payload["identity"] or not ledger["attempts"]:
        raise ValueError("Run identity or initial submission missing")
    if ledger.get("pending_intent"):
        return {"submitted": False, "state": "UNKNOWN", "reason": "unreconciled_submission"}
    state = job_state(ledger["attempts"][-1]["job_id"])["state"]
    if state in ACTIVE or state == "UNKNOWN":
        return {"submitted": False, "state": state}
    reviewed_screen_failure = (
        bool(payload.get("explicit_continue"))
        and state == "FAILED"
        and payload.get("operation") == "task21-screen"
        and ledger["attempts"][-1]["job_id"] == "34123330"
    )
    planned = bool(payload.get("explicit_continue")) and (
        state in {"COMPLETED", "TIMEOUT"} or reviewed_screen_failure
    )
    if payload.get("operation") in STAGED_OPERATIONS and completed_job_is_final(payload):
        return {"submitted": False, "state": state, "reason": "scientifically_complete"}
    if payload.get("operation") in TASK19_OPERATIONS:
        if completed_job_is_final(payload):
            return {"submitted": False, "state": state, "reason": "scientifically_complete"}
        if payload["operation"] in {"task19-qualify", "task19-qualify-uncached"}:
            if not planned:
                return {
                    "submitted": False,
                    "state": state,
                    "reason": "qualification_requires_review",
                }
            if (
                not ledger.get("run_dir")
                or Path(ledger["run_dir"]).resolve() != artifact_dir(payload).resolve()
            ):
                return {
                    "submitted": False,
                    "state": state,
                    "reason": "qualification_directory_mismatch",
                }
            progress = Path(ledger["run_dir"]) / "qualification-progress.json"
            health_path = progress.parent / "state/health.json"
            if not progress.is_file() or not health_path.is_file():
                return {
                    "submitted": False,
                    "state": state,
                    "reason": "qualification_progress_missing",
                }
            health = json.loads(health_path.read_text())
            if health.get("allocation") != ledger["attempts"][-1]["job_id"] or health.get(
                "phase"
            ) not in ({"stopped"} if state == "COMPLETED" else {"stopped", "running"}):
                return {
                    "submitted": False,
                    "state": state,
                    "reason": "qualification_requires_review",
                }
            saved = json.loads(progress.read_text())
            if not saved.get("identity") or not saved.get("snapshot"):
                return {
                    "submitted": False,
                    "state": state,
                    "reason": "qualification_progress_invalid",
                }
            with locked(path.with_suffix(".lock")):
                current = read_ledger(path)
                previous_job = ledger["attempts"][-1]["job_id"]
                if (
                    current["attempts"][-1]["job_id"] != previous_job
                    or current.get("recovery_from_job") == previous_job
                    or current.get("continuations", 0) >= payload.get("max_continuations", 0)
                ):
                    return {
                        "submitted": False,
                        "state": state,
                        "reason": "concurrent_or_limited_recovery",
                    }
                current["continuations"] = current.get("continuations", 0) + 1
                current["recovery_from_job"] = previous_job
                current.setdefault("qualification_recoveries", []).append(
                    {
                        "job_id": previous_job,
                        "progress_sha256": sha256(progress),
                    }
                )
                write_json(path, current)
            result = submit(
                {
                    **payload,
                    "resume": str(progress),
                    "time": payload.get("continuation_time", payload["time"]),
                }
            )
            return {**result, "state": state, "resume": str(progress)}
    # These exact evaluation allocations failed together while the shared
    # filesystem was full. The evaluator checks each saved task's frozen
    # identity before reuse; never generalize this review to other failures.
    reviewed_evaluations = {
        "8f744246aa6dd66285b1f9b1": "34518622",
        "d5f96e070b2811ba5c5fe7b0": "34519629",
        "67d7e6a9b020e135da98c893": "34520327",
        "4605dfc4b20207b32332c25a": "34520385",
        "df336c7530d75c96f183ca7c": "34520388",
        "ee1f5b80df8c80b1cc6bc77d": "34520679",
    }
    previous_job = ledger["attempts"][-1]["job_id"]
    if (
        payload.get("explicit_continue")
        and payload.get("operation") == "task21-evaluate"
        and state == "FAILED"
        and reviewed_evaluations.get(payload["run_id"]) == previous_job
    ):
        if Path(ledger.get("run_dir", "")).resolve() != artifact_dir(payload).resolve():
            return {"submitted": False, "state": state, "reason": "evaluation_directory_mismatch"}
        if shutil.disk_usage(workspace(payload)).free < 20 * 1024**3:
            return {"submitted": False, "state": state, "reason": "shared_storage_below_20_gib"}
        try:
            checkpoint = verified_checkpoint(payload)
            metadata = check_checkpoint(Path(checkpoint), payload)
            if metadata["iteration"] != payload["task20_checkpoint_iteration"]:
                raise ValueError("Evaluation checkpoint iteration differs")
        except (OSError, KeyError, ValueError) as exc:
            return {"submitted": False, "state": state, "reason": f"checkpoint_unavailable: {exc}"}
        with locked(path.with_suffix(".lock")):
            current = read_ledger(path)
            if (
                current["attempts"][-1]["job_id"] != previous_job
                or current.get("recovery_from_job") == previous_job
                or current.get("continuations", 0) >= payload.get("max_continuations", 0)
            ):
                return {
                    "submitted": False,
                    "state": state,
                    "reason": "concurrent_or_limited_recovery",
                }
            current["continuations"] = current.get("continuations", 0) + 1
            current["recovery_from_job"] = previous_job
            current.setdefault("reviewed_evaluation_recoveries", []).append(
                {
                    "job_id": previous_job,
                    "checkpoint": checkpoint,
                    "reason": "storage_exhaustion_incident",
                }
            )
            write_json(path, current)
        result = submit({**payload, "resume": checkpoint})
        return {**result, "state": state, "resume": checkpoint}
    if requested_stop is not None and not planned:
        return {"submitted": False, "state": state, "reason": "not_a_clean_continuation"}
    if state == "COMPLETED" and not planned:
        return {"submitted": False, "state": state, "reason": "inspect_scientific_completion"}
    if not planned and state not in {"PREEMPTED", "NODE_FAIL", "BOOT_FAIL"}:
        return {"submitted": False, "state": state, "reason": "not_transient"}
    if not planned and ledger.get("retries", 0) >= 2:
        return {"submitted": False, "state": state, "reason": "retry_limit"}
    health_path = Path(ledger.get("run_dir", "")) / "state/health.json"
    if reviewed_screen_failure:
        health = json.loads(health_path.read_text())
        if health.get("allocation") != "34123330" or health.get("phase") != "running":
            return {"submitted": False, "state": state, "reason": "screen_failure_requires_review"}
    timeout_evidence = None
    if health_path.is_file():
        health = json.loads(health_path.read_text())
        if health.get("phase") == "failed":
            if planned and state == "TIMEOUT" and payload["operation"] == "task15-online":
                timeout_evidence = timeout_disconnect_evidence(
                    ledger["attempts"][-1]["job_id"], health
                )
            if planned and state == "TIMEOUT" and payload["operation"] == "task21-main":
                timeout_evidence = reviewed_task21_timeout_disconnect(
                    payload, ledger["attempts"][-1]["job_id"], health
                )
            if timeout_evidence is None:
                return {"submitted": False, "state": state, "reason": "training_reported_failure"}
    try:
        checkpoint = verified_checkpoint(payload)
    except (OSError, KeyError, ValueError) as exc:
        return {"submitted": False, "state": state, "reason": f"checkpoint_unavailable: {exc}"}
    metadata = check_checkpoint(Path(checkpoint), payload)
    if (
        payload.get("target_episodes")
        and metadata.get("total_episodes", 0) >= payload["target_episodes"]
    ):
        return {"submitted": False, "state": state, "reason": "scientifically_complete"}
    if metadata.get("iteration", 0) >= payload.get(
        "target_iterations", payload.get("iterations", 0)
    ):
        return {"submitted": False, "state": state, "reason": "scientifically_complete"}
    checkpointed_timeout = state == "TIMEOUT" and payload[
        "operation"
    ] in TASK19_OPERATIONS | STAGED_OPERATIONS | {
        "task15-online",
        "task15-stabilize",
    }
    if (
        planned
        and not checkpointed_timeout
        and not reviewed_screen_failure
        and (
            not health_path.is_file()
            or json.loads(health_path.read_text()).get("phase") != "stopped"
        )
    ):
        return {"submitted": False, "state": state, "reason": "not_a_clean_stop"}
    if planned and ledger.get("continuations", 0) >= payload.get("max_continuations", 0):
        return {"submitted": False, "state": state, "reason": "continuation_limit"}
    if requested_stop is not None and (
        not isinstance(requested_stop, int)
        or isinstance(requested_stop, bool)
        or not metadata["iteration"] < requested_stop < payload.get("target_iterations", 0)
    ):
        raise ValueError("Continuation stop must be after the checkpoint and before the target")
    next_stop = (
        requested_stop
        if requested_stop is not None
        else 0
        if planned and metadata["iteration"] >= payload.get("stop_after_iteration", 0)
        else payload.get("stop_after_iteration", 0)
    )
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
        if timeout_evidence is not None:
            current.setdefault("timeout_disconnect_recoveries", []).append(
                {**timeout_evidence, "checkpoint": checkpoint}
            )
        if reviewed_screen_failure:
            current.setdefault("reviewed_screen_recoveries", []).append(
                {"job_id": previous_job, "checkpoint": checkpoint, "reason": "storage_quota_full"}
            )
        current["active_stop_after_iteration"] = next_stop
        write_json(path, current)
    result = submit(
        {
            **payload,
            "resume": checkpoint,
            "stop_after_iteration": next_stop,
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


def can_supersede_empty_pilot(payload: dict, prior: dict, owner: dict, artifact: Path) -> bool:
    """Permit an explicit replacement only for an observed preflight-only failure."""
    ledger = prior["ledger"]
    attempts = prior["attempts"]
    old = ledger.get("specification", {})
    if (
        payload.get("supersede_empty_pilot") != owner.get("run_id")
        or payload.get("operation") != "pilot"
        or old.get("operation") != "pilot"
        or ledger.get("pending_intent")
        or len(attempts) != 1
        or attempts[-1]["scheduler"]["state"] != "FAILED"
        or artifact.exists()
        or not old.get("commit")
        or old.get("commit") == payload.get("commit")
    ):
        return False
    keys = (
        "study_id",
        "method",
        "seed",
        "scientific_hash",
        "manifests_hash",
        "runtime_identity",
        "image_sha256",
        "worker_sha256",
        "gpu_type",
        "profile",
    )
    if any(key not in old or key not in payload or old[key] != payload[key] for key in keys):
        return False
    try:
        tail = logs(
            {"storage_root": payload["storage_root"], "run_id": owner["run_id"], "lines": 1000}
        )
    except (OSError, ValueError, KeyError):
        return False
    return (
        "Pilot preflight blocked: Backend acceptance proof stale or missing single-process gradient comparison"
        in tail.get("tail", "")
    )


def can_supersede_empty_task19_base(
    payload: dict, prior: dict, owner: dict, artifact: Path
) -> bool:
    """Replace an explicitly identified terminal base that saved no scientific state."""
    ledger, attempts = prior["ledger"], prior["attempts"]
    old = ledger.get("specification", {})
    if (
        payload.get("supersede_empty_base") != owner.get("run_id")
        or payload.get("operation") != "task19-base"
        or old.get("operation") != "task19-base"
        or ledger.get("pending_intent")
        or len(attempts) != 1
        or attempts[-1]["scheduler"]["state"] not in {"CANCELLED", "FAILED", "TIMEOUT"}
        or any(path.is_file() or path.is_symlink() for path in artifact.rglob("*"))
        or not SHA.fullmatch(old.get("commit", ""))
        or not SHA.fullmatch(payload.get("commit", ""))
        or old["commit"] == payload["commit"]
    ):
        return False
    keys = (
        "study_id",
        "method",
        "seed",
        "scientific_hash",
        "manifests_hash",
        "runtime_identity",
        "image_sha256",
        "worker_sha256",
        "gpu_type",
        "profile",
        "task19_control_sha256",
        "common_config",
    )
    if any(key not in old or key not in payload or old[key] != payload[key] for key in keys):
        return False
    new_checkout = paths(payload)["checkout"]
    old_checkout = new_checkout.parent / old["commit"]
    names = (
        "src/selfplay_opsd/task19.py",
        "src/selfplay_opsd/task19_policy.py",
        "src/selfplay_opsd/task19_protocol.py",
        "src/selfplay_opsd/task19_qualification.py",
        "src/selfplay_opsd/task19_rollout.py",
        "src/selfplay_opsd/task19_train.py",
        "scripts/task19_experiment.py",
        "docker/task19/uv.lock",
        "docker/task19.Dockerfile",
    )
    try:
        return all(sha256(old_checkout / name) == sha256(new_checkout / name) for name in names)
    except OSError:
        return False


def completed_job_is_final(payload: dict) -> bool:
    """Task 15 training may continue only with an explicitly verified checkpoint."""
    operation = payload["operation"]
    if operation in STAGED_OPERATIONS:
        return (artifact_dir(payload) / stage_marker(payload)).is_file()
    if operation in TASK19_OPERATIONS:
        directory = artifact_dir(payload)
        if operation.startswith("task19-qualify"):
            complete = directory / "calibrated-control.json"
        elif operation == "task19-base":
            complete = directory / "complete-base.json"
        else:
            endpoint = 32
            if operation == "task19-train":
                endpoint = json.loads((directory.parent / "main-control.json").read_text())[
                    "endpoint"
                ]
            complete = directory / f"complete-{endpoint}.json"
        return complete.is_file()
    return operation not in {"pilot", "train"} and not (
        operation in {"task15-online", "task15-stabilize", "task17-student", "task18-student"}
        and payload.get("resume")
    )


def sequential_dependency(payload: dict) -> list[str]:
    dependency = payload.get("afterok")
    if dependency is None:
        return []
    if (
        payload["operation"]
        not in {"task17-interface", "task17-normalized", "task18-normalized", "task18-native"}
        or not re.fullmatch(r"[1-9][0-9]{0,11}", str(dependency.get("job_id", "")))
        or not re.fullmatch(r"[a-f0-9]{24}", str(dependency.get("run_id", "")))
    ):
        raise ValueError("Invalid sequential interface dependency")
    return [f"--dependency=afterok:{dependency['job_id']}"]


def task17_walltime_allowed(payload: dict) -> bool:
    operation = payload["operation"]
    if operation in {"task17-interface", "task17-normalized"}:
        # Keep previously committed two-hour submissions valid while supporting
        # measured shorter allocations; both retain the same global ceiling.
        return payload.get("time") in {"01:00:00", "02:00:00"}
    expected = (
        "01:00:00"
        if operation in {"task17-prepare", "task17-replay"}
        else "06:00:00"
        if operation == "task17-student"
        else "02:00:00"
    )
    return payload.get("time") == expected


def submit(payload: dict) -> dict:
    dependency_args = sequential_dependency(payload)
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
    if payload["operation"] in {"task13-probe", "task13-counterfactual"} and payload.get(
        "checkpoint"
    ):
        checkpoint = payload["checkpoint"]
        expected_method = {"A": "A", "B": "B", "C": "C", "D": "B"}[payload["method"]]
        prefix = f"/runs/{payload['study_id']}/{expected_method}-seed{payload['seed']}/checkpoints/"
        if not checkpoint.startswith(prefix) or not re.fullmatch(
            r"\d{6}", checkpoint.removeprefix(prefix)
        ):
            raise ValueError("Probe checkpoint must be a numbered corrected-study checkpoint")
        if not (p["runs"] / checkpoint.removeprefix("/runs/") / "complete.json").is_file():
            raise ValueError("Probe checkpoint is absent or incomplete")
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
            if state["state"] == "COMPLETED" and completed_job_is_final(payload):
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
                        else str(artifact)
                        + ":"
                        + (
                            stage_marker(payload)
                            if payload["operation"] in STAGED_OPERATIONS
                            else payload["operation"]
                        )
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
        if payload["operation"] in {"task19-pilot", "task19-train"}:
            other = "task19-train" if payload["operation"] == "task19-pilot" else "task19-pilot"
            other_slot = (
                p["state"]
                / "slots"
                / (hashlib.sha256(f"{artifact}:{other}".encode()).hexdigest() + ".json")
            )
            checked_slots.append(other_slot)
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
                if payload["operation"] in TASK19_OPERATIONS and candidate_slot == slot:
                    if can_supersede_empty_task19_base(payload, prior, owner, artifact):
                        ledger["supersedes_empty_base_run"] = owner["run_id"]
                        continue
                    if (
                        payload["operation"] == "task19-qualify"
                        and states[-1] == "FAILED"
                        and not artifact.exists()
                        and "ValueError: Unsupported allocation operation"
                        in logs(
                            {
                                "storage_root": payload["storage_root"],
                                "run_id": owner["run_id"],
                                "lines": 100,
                            }
                        ).get("tail", "")
                    ):
                        ledger["supersedes_preflight_only_run"] = owner["run_id"]
                        continue
                    raise ValueError("Task 19 stage already has an immutable controller run")
                if payload["operation"] in STAGED_OPERATIONS and candidate_slot == slot:
                    raise ValueError("Task 20 stage already has an immutable controller run")
                if payload["operation"] in {"train", "pilot", "task15-online", "task15-stabilize"}:
                    if can_supersede_empty_pilot(payload, prior, owner, artifact):
                        ledger["supersedes_preflight_only_run"] = owner["run_id"]
                        continue
                    raise ValueError(
                        "Existing scientific run belongs to a different immutable identity"
                    )
        specification = {
            k: v for k, v in payload.items() if k not in {"resume", "explicit_continue"}
        }
        reconcile_rejected_specification(ledger, specification)
        reserve_budget(payload, token)
        write_json(slot, {"run_id": payload["run_id"], "identity": payload["identity"]})
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
            "SELFPLAY_TASK19_RUN_ID": payload["run_id"]
            if payload["operation"] in TASK19_OPERATIONS
            else "",
            "SELFPLAY_TASK19_COMMIT": payload["commit"]
            if payload["operation"] in TASK19_OPERATIONS
            else "",
            "SELFPLAY_TASK20_CHECKPOINT_ITERATION": (
                str(payload["task20_checkpoint_iteration"])
                if payload.get("task20_checkpoint_iteration") is not None
                else ""
            ),
            "SELFPLAY_TASK15_FREEZE": payload.get("task15_freeze") or "",
            "SELFPLAY_AUDIT_SOURCE": payload.get("audit_source") or "",
            "SELFPLAY_DECISION": payload.get("decision") or "",
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
        args += dependency_args
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
    if operation in {"rollout-parity", "numerical-audit", "proposer-audit"}:
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
            if (
                payload.get("output")
                == f"/runs/compatibility/backend-v2/studies/{study}/{payload['profile']}"
            ):
                return runs / "compatibility/backend-v2/studies" / study / payload["profile"]
            return runs / "compatibility/backend-v2" / payload["profile"]
        return runs / "compatibility" / payload["profile"]
    if operation in {"freeze-study", "task15-freeze"}:
        return runs / study
    if operation == "evaluate" and payload.get("base"):
        return runs / study / "base"
    if operation == "diagnose":
        return runs / study / "diagnostics" / Path(payload["checkpoint"]).name
    if operation == "task13-probe":
        return runs / "repair-diagnosis-2026-probe-v1" / payload["run_id"]
    if operation == "task14-probe":
        return runs / "repair-diagnosis-2026-task14-probe-v1" / payload["run_id"]
    if operation in {"task17-student", "task18-student"}:
        return runs / payload["study_id"] / f"{payload['effective_method']}-seed{payload['seed']}"
    if operation in TASK19_OPERATIONS:
        suffix = {
            "task19-qualify": "qualification",
            "task19-qualify-uncached": "qualification-uncached",
            "task19-base": "base",
            "task19-pilot": f"{payload['method']}-seed2026",
            "task19-train": f"{payload['method']}-seed2026",
        }[operation]
        return runs / study / suffix
    if operation in STAGED_OPERATIONS:
        stage = operation.split("-", 1)[1]
        name = payload["method"] if stage in {"main", "evaluate"} else stage
        return runs / study / f"{name}-seed{payload['seed']}"
    if operation.startswith("task18-"):
        return runs / study / payload["run_id"]
    if operation.startswith("task17-"):
        suffix = (
            "prepared"
            if operation == "task17-prepare"
            else "calibration"
            if operation == "task17-calibrate"
            else payload["run_id"]
        )
        return runs / "repair-selfplay-2026-task17-interface-v1" / suffix
    if operation == "task14-student":
        return runs / "repair-diagnosis-2026-task14-student-v1" / payload["run_id"]
    if operation == "task14-counterfactual":
        return runs / "repair-diagnosis-2026-task14-counterfactual-v1" / payload["run_id"]
    if operation == "task14-counterfactual-replay":
        return runs / "repair-diagnosis-2026-task14-counterfactual-replay-v1" / payload["run_id"]
    if operation == "task14-calibrate":
        return runs / "repair-diagnosis-2026-task14-calibrate-v1" / payload["run_id"]
    if operation == "task14-numerical":
        return runs / "repair-diagnosis-2026-task14-numerical-v1" / payload["run_id"]
    if operation == "task14-numerical-components":
        return runs / "repair-diagnosis-2026-task14-numerical-components-v1" / payload["run_id"]
    if operation == "task14-proposal-student":
        return runs / "repair-diagnosis-2026-task14-proposal-student-v1" / payload["run_id"]
    if operation == "task14-static":
        return runs / "repair-diagnosis-2026-task14-static-v1" / payload["run_id"]
    if operation == "task15-numerical":
        return runs / "repair-diagnosis-2026-task15-numerical-v1" / payload["run_id"]
    if operation == "task14-online":
        return runs / "repair-selfplay-2026-task14-online-v1" / payload["run_id"]
    if operation == "task13-model-probe":
        return runs / "repair-diagnosis-2026-model-probe-v1" / payload["run_id"]
    if operation == "task13-header-replay":
        return runs / "repair-diagnosis-2026-header-replay-v1" / payload["run_id"]
    if operation == "task13-static":
        return runs / "repair-diagnosis-2026-static-v1" / payload["run_id"]
    if operation == "task13-static-continuation":
        return runs / "repair-diagnosis-2026-static-continuation-v1" / payload["run_id"]
    if operation == "task13-counterfactual":
        return runs / "repair-diagnosis-2026-counterfactual-v1" / payload["run_id"]
    return (
        runs
        / (study + (payload.get("pilot_suffix", "-pilot") if operation == "pilot" else ""))
        / f"{payload['method']}-seed{payload['seed']}"
    )


def task17_budget(payload: dict) -> bool:
    return payload.get("operation", "").startswith("task17-") or payload.get(
        "study_id", ""
    ).startswith("repair-selfplay-2026-task17-")


def task18_budget(payload: dict) -> bool:
    return payload.get("operation", "").startswith("task18-") or payload.get(
        "study_id", ""
    ).startswith("repair-selfplay-2026-task18-")


def task19_budget(payload: dict) -> bool:
    return payload.get("operation") in TASK19_OPERATIONS or payload.get("study_id", "").startswith(
        "repair-selfplay-2026-task19-"
    )


def task20_budget(payload: dict) -> bool:
    return payload.get("study_id", "").startswith("repair-selfplay-2026-task20-")


def task21_budget(payload: dict) -> bool:
    return payload.get("study_id", "").startswith("repair-selfplay-2026-task21-")


def staged_budget(payload: dict) -> bool:
    return task20_budget(payload) or task21_budget(payload)


def readiness_limit(payload: dict) -> int:
    if task19_budget(payload):
        return 80
    if task18_budget(payload):
        return 12
    if not task17_budget(payload):
        return 24
    return (
        24
        if payload.get("operation") in {"task17-calibrate", "task17-student"}
        or "-objectives-" in payload.get("study_id", "")
        else 10
    )


def budget_path(payload: dict) -> Path:
    if not SAFE_NAME.fullmatch(payload["study_id"]):
        raise ValueError("Unsafe study ID")
    # Readiness authorization spans protocol/study revisions. Never let a new
    # scientific identity create a fresh allowance.
    if task19_budget(payload):
        return workspace(payload) / "state" / "task19-budget.json"
    if task21_budget(payload):
        return workspace(payload) / "state" / "task21-budget.json"
    if task20_budget(payload):
        return workspace(payload) / "state" / "task20-budget.json"
    if task18_budget(payload):
        return workspace(payload) / "state" / "task18-budget.json"
    if task17_budget(payload):
        phase = "phase2" if readiness_limit(payload) == 24 else "phase1"
        return workspace(payload) / "state" / f"task17-{phase}-budget.json"
    return workspace(payload) / "state" / "budget-readiness-cpsc4880.json"


def refreshed_budget(payload: dict) -> dict:
    path = budget_path(payload)
    task17 = task17_budget(payload)
    limit = payload.get("readiness_budget") if staged_budget(payload) else readiness_limit(payload)
    budget = json.loads(path.read_text()) if path.exists() else {"limit": limit, "reservations": {}}
    authorization = (
        "cpsc4880-task21-v1"
        if task21_budget(payload)
        else "cpsc4880-task20-v1"
        if task20_budget(payload)
        else "cpsc4880-task19-80h-v1"
        if task19_budget(payload)
        else "cpsc4880-task18-12h-v1"
        if task18_budget(payload)
        else f"cpsc4880-task17-{limit}h-v1"
        if task17
        else "cpsc4880-readiness-24h-v1"
    )
    if budget.get("authorization_id", authorization) != authorization:
        raise ValueError("Readiness authorization identity mismatch")
    budget["authorization_id"] = authorization
    # Import every legacy study, not just the caller's study. Keep source hashes
    # so a legacy writer cannot silently fork the allowance after migration.
    imported = budget.setdefault("legacy_sources", {})
    for legacy in (
        []
        if task17 or task18_budget(payload) or task19_budget(payload) or staged_budget(payload)
        else sorted(path.parent.glob("budget-*.json"))
    ):
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
                if (task19_budget(payload) or staged_budget(payload)) and len(parts) >= 5:
                    item["allocation_start"] = parts[4]
                    item["allocated_gpu_seconds"] = int(parts[2]) * int(gpu[1])
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
        if not payload["gpu_count"] or (
            not payload.get("readiness_budget") and not staged_budget(payload)
        ):
            return
        raw_limit = payload.get("readiness_budget")
        limit = float(raw_limit) if raw_limit is not None else None
        if task18_budget(payload) and limit != 12:
            raise ValueError("Task 18 has a hard 12 GPU-hour ceiling")
        if task19_budget(payload):
            if limit != 80:
                raise ValueError("Task 19 uses its shared 80 H200-hour ceiling")
        elif staged_budget(payload):
            if limit is not None and (not math.isfinite(limit) or limit <= 0):
                raise ValueError("Staged budget must be positive when supplied")
        elif limit is None or not 0 < limit <= 24:
            raise ValueError("Readiness budget must be positive and at most 24 GPU-hours")
        budget = refreshed_budget(payload)
        if budget["reservations"] and budget["limit"] != limit:
            raise ValueError("Cannot change the recorded readiness budget")
        hours, minutes, seconds = map(int, payload["time"].split(":"))
        reserved = payload["gpu_count"] * (hours + minutes / 60 + seconds / 3600)
        spend_cap = limit
        if task19_budget(payload):
            control = workspace(payload) / "runs" / payload["study_id"] / "main-plan.json"
            if control.is_file():
                spend_cap = min(limit, json.loads(control.read_text())["cap"])
        if spend_cap is not None and budget["committed_gpu_hours"] + reserved > spend_cap + 1e-9:
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


TASK18_VERIFIER_FILES = (
    "uv.lock",
    "src/selfplay_opsd/core.py",
    "src/selfplay_opsd/codec.py",
    "src/selfplay_opsd/config.py",
    "src/selfplay_opsd/data.py",
    "src/selfplay_opsd/evaluation.py",
    "src/selfplay_opsd/mutations.py",
    "src/selfplay_opsd/protocol.py",
    "src/selfplay_opsd/environment.py",
    "src/selfplay_opsd/policy.py",
    "src/selfplay_opsd/readiness.py",
    "src/selfplay_opsd/task18_protocol.py",
    "src/selfplay_opsd/task17_recovery.py",
    "src/selfplay_opsd/sandbox.py",
    "src/selfplay_opsd/worker.py",
    "src/selfplay_opsd/remote_worker.py",
    "src/selfplay_opsd/worker_runtime.py",
    "src/selfplay_opsd/worker_policy.py",
    "scripts/hpc_job.py",
)


def task18_verifier_compatible(report: dict, expected: dict, payload: dict, base: Path) -> bool:
    if not task18_budget(payload) or not report.get("ready"):
        return False
    actual = report.get("runtime_identity", {})
    ignored = {"code_hash", "scientific_code_hash"}
    if not expected or {k: v for k, v in actual.items() if k not in ignored} != {
        k: v for k, v in expected.items() if k not in ignored
    }:
        return False
    previous = report.get("provenance", {}).get("git_commit", "")
    current = payload.get("commit", "")
    if not SHA.fullmatch(previous) or not SHA.fullmatch(current):
        return False
    old, new = base / "checkouts" / previous, base / "checkouts" / current
    for relative in TASK18_VERIFIER_FILES:
        a, b = old / relative, new / relative
        if not a.is_file() or not b.is_file() or sha256(a) != sha256(b):
            return False
    validation = report.get("validation", {})
    return bool(
        validation.get("ready")
        and validation.get("executed")
        and report.get("isolation", {}).get("ready")
    )


def study_state(payload: dict) -> dict:
    base = workspace(payload)
    study = payload["study_id"]
    if not SAFE_NAME.fullmatch(study):
        raise ValueError("Unsafe study ID")
    runs = base / "runs"

    def read(path):
        return json.loads(path.read_text()) if path.is_file() else {}

    images = {
        kind: read(base / "images" / f"{kind}.json")
        for kind in ("worker", "trainer", "task19-trainer")
    }
    evidence = runs / "readiness" / study
    identity = payload.get("runtime_identity")

    def backend_report(profile):
        scoped = runs / "compatibility/backend-v2/studies" / study / f"{profile}.json"
        legacy = runs / "compatibility/backend-v2" / f"{profile}.json"
        if not scoped.is_file() and study == "repair-selfplay-2026-task15-stabilization-v1":
            source = (
                runs
                / "compatibility/backend-v2/studies/repair-selfplay-2026-task15-ab-v1"
                / f"{profile}.json"
            )
            if source.is_file():
                # The normal identity, hardware and ready gates below still apply.
                # Return the immutable source proof without changing its contents.
                return read(source) | {"reused_from": str(source), "reused_sha256": sha256(source)}
        return read(scoped if scoped.is_file() else legacy)

    reports = {
        "worker": read(evidence / "worker.json"),
        "runtime": read(evidence / "runtime.json"),
        "single": backend_report("single_gpu_80gb"),
        "target": backend_report(payload.get("profile", "dual_gpu_80gb")),
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
        if name == "runtime" and not gates[name]:
            gates[name] = task18_verifier_compatible(report, expected or {}, payload, base)
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


def stage_marker(payload: dict) -> str:
    if payload["operation"] in {"task20-evaluate", "task21-evaluate"}:
        return f"complete-evaluate-{payload['task20_checkpoint_iteration']:06d}.json"
    return f"complete-{payload['operation'].split('-', 1)[1]}.json"


def task20_solver_evidence(solver: dict, directory: Path) -> bool:
    """Check learning evidence while retaining the original recorded gate."""
    if solver["result"]["gate"]["learnable"]:
        return True
    results = solver["result"]["solver_train"]
    diagnostic = json.loads((directory / "one-step.json").read_text())
    step = diagnostic.get("step", {})
    rows = [
        json.loads(path.read_text())["metrics"]
        for path in sorted((directory / "iterations").glob("*.json"))
        if path.stem != "000000"
    ]
    return (
        solver.get("effective_rl_updates", 0) >= 2
        and len(results) == 2
        and len(rows) == solver["effective_rl_updates"]
        and step.get("optimizer_step") is True
        and step.get("update_norm", 0) > 0
        and step.get("weighted_logprob_change", 0) > 0
        and all(row["shared"]["optimizer_step"] and row["rl"]["norm"] > 0 for row in rows)
        and all(row["rl"]["support_exits"] == 0 for row in rows)
        and all(result["after"] > result["before"] for result in results.values())
    )


def check_task21(payload: dict) -> None:
    operation = payload["operation"]
    stage = operation.removeprefix("task21-")
    path = artifact_dir(payload)
    control = (paths(payload)["checkout"] / payload["settings"]).with_suffix(".json")
    wall = re.fullmatch(r"([0-9]{2}):([0-5][0-9]):00", payload.get("time", ""))
    minutes = int(wall[1]) * 60 + int(wall[2]) if wall else 0
    if (
        not task21_budget(payload)
        or payload.get("account") != "cpsc4880"
        or payload.get("gpu_count") != 1
        or payload.get("gpu_type") != "h200"
        or payload.get("profile") != "single_gpu_80gb"
        or not 30 <= minutes <= 720
        or payload.get("output") != f"/runs/{payload['study_id']}/{path.name}"
        or sha256(control) != payload.get("task20_control_sha256")
    ):
        raise ValueError("Task 21 requires a bounded H200 allocation and frozen control")
    config = json.loads(control.read_text())
    if config.get("study") != payload["study_id"] or config.get("seed") != payload["seed"]:
        raise ValueError("Task 21 control identity is inconsistent")
    if not study_state(payload)["gates"]["worker"]:
        raise ValueError("Task 21 requires the accepted isolated worker")
    if stage not in {"main", "evaluate"} and payload["method"] != "control":
        raise ValueError("Task 21 diagnostic stages require the control arm")
    manifest = json.loads((paths(payload)["dataset"] / "manifest.json").read_text())
    if (
        manifest.get("benchmark") != "pdb-native-v1"
        or manifest.get("bundle_sha256") != sha256(paths(payload)["dataset"] / "pdb.json")
        or not manifest.get("counts", {}).get("feasibility")
    ):
        raise ValueError("Task 21 requires its verified PDB task bundle")
    needed = {"screen": "screen", "solver": "solver_train", "main": "dev_monitor"}.get(stage)
    if needed and not manifest["counts"].get(needed):
        raise ValueError(f"Task 21 stage requires the {needed} partition")
    if stage == "screen":
        root = path.parent / f"feasibility-seed{payload['seed']}"
        marker = root / "complete-feasibility.json"
        if not marker.is_file():
            raise ValueError("Task 21 screen requires a completed model feasibility pilot")
        feasibility = json.loads(marker.read_text())
        prior_data = json.loads((paths(payload)["dataset"] / "pdb.json").read_text())["provenance"]
        accepted = {payload["manifests_hash"], prior_data.get("feasibility_manifest")}
        if (
            feasibility["identity"]["data"] not in accepted
            or feasibility["identity"]["execution"]["image"] != payload["image_sha256"]
            or not feasibility["result"]["count"]
            or feasibility["checkpoint_sha256"]
            != sha256(root / "checkpoints" / feasibility["checkpoint"] / "complete.json")
        ):
            raise ValueError("Task 21 feasibility marker or checkpoint differs")
    if payload["method"] in {"C", "D"} and not config.get("alignment_gate_hash"):
        raise ValueError("Task 21 C/D requires separately verified alignment evidence")
    if stage in {"main", "evaluate"}:
        if not isinstance(config.get("endpoint"), int) or not config.get("shared_gate_hash"):
            raise ValueError("Task 21 main comparison requires frozen scientific evidence")
        if stage == "evaluate":
            checkpoint = payload.get("task20_checkpoint_iteration")
            if not isinstance(checkpoint, int) or checkpoint < 0:
                raise ValueError("Task 21 evaluation requires a verified checkpoint")
            check_checkpoint(path / "checkpoints" / f"{checkpoint:06d}", payload)


def check_task20(payload: dict) -> None:
    operation = payload["operation"]
    path = artifact_dir(payload)
    control = (paths(payload)["checkout"] / payload["settings"]).with_suffix(".json")
    wall = re.fullmatch(r"([0-9]{2}):([0-5][0-9]):00", payload.get("time", ""))
    minutes = int(wall[1]) * 60 + int(wall[2]) if wall else 0
    limit = payload.get("readiness_budget")
    if (
        not task20_budget(payload)
        or payload.get("account") != "cpsc4880"
        or payload.get("gpu_count") != 1
        or payload.get("gpu_type") != "h200"
        or payload.get("profile") != "single_gpu_80gb"
        or payload.get("method") not in {"control", "A", "B", "C", "D"}
        or (
            limit is not None
            and (
                not isinstance(limit, (int, float))
                or not math.isfinite(limit)
                or minutes / 60 > limit
            )
        )
        or not 30 <= minutes <= 720
        or payload.get("output") != f"/runs/{payload['study_id']}/{path.name}"
        or sha256(control) != payload.get("task20_control_sha256")
    ):
        raise ValueError("Task 20 requires an authorized H200 reservation and frozen control")
    config = json.loads(control.read_text())
    if config.get("study") != payload["study_id"] or config.get("seed") != payload["seed"]:
        raise ValueError("Task 20 control identity is inconsistent")
    if not study_state(payload)["gates"]["worker"]:
        raise ValueError("Task 20 requires the accepted isolated worker")
    if operation not in {"task20-main", "task20-evaluate"} and payload["method"] != "control":
        raise ValueError("Task 20 diagnostic stages require the control arm")
    manifest = json.loads((paths(payload)["dataset"] / "manifest.json").read_text())
    if manifest.get("benchmark") != "task20-evalplus-harder-v1":
        raise ValueError("Expected the versioned Task 20 benchmark")
    if operation == "task20-solver" and not DIGEST.fullmatch(
        manifest.get("difficulty_screen_sha256") or ""
    ):
        raise ValueError("Task 20 solver requires a frozen screened curriculum")
    if operation in {"task20-calibrate", "task20-shared-pilot"}:
        solver_dir = path.parent / f"solver-seed{payload['seed']}"
        solver = json.loads((solver_dir / "complete-solver.json").read_text())
        if (
            not task20_solver_evidence(solver, solver_dir)
            or solver["identity"]["data"] != payload["manifests_hash"]
        ):
            raise ValueError("Task 20 shared work requires the matching solver learnability gate")
    if operation == "task20-shared-pilot" and config.get("shared_pilot_mode", "shared") == "shared":
        calibration = json.loads(
            (path.parent / f"calibrate-seed{payload['seed']}/complete-calibrate.json").read_text()
        )
        if (
            not calibration["result"]["ready"]
            or calibration["identity"]["data"] != payload["manifests_hash"]
            or calibration["result"]["proposer_scale"] != config.get("proposer_scale")
        ):
            raise ValueError("Shared pilot requires the matching measured proposer scale")
    if operation in {"task20-main", "task20-evaluate"}:
        for key in (
            "qualification_hash",
            "calibration_hash",
            "solver_gate_hash",
            "shared_gate_hash",
        ):
            if not DIGEST.fullmatch(config.get(key) or ""):
                raise ValueError("Task 20 main and evaluation require recorded gate evidence")
        if not isinstance(config.get("endpoint"), int):
            raise ValueError("Task 20 main and evaluation require a frozen endpoint")
        if payload["method"] in {"C", "D"} and not DIGEST.fullmatch(
            config.get("alignment_gate_hash") or ""
        ):
            raise ValueError("Task 20 C/D require separate alignment evidence")
        for stage, key in (
            ("solver", "solver_gate_hash"),
            ("calibrate", "calibration_hash"),
            ("shared-pilot", "shared_gate_hash"),
        ):
            evidence = path.parent / f"{stage}-seed{payload['seed']}" / f"complete-{stage}.json"
            if not evidence.is_file() or sha256(evidence) != config[key]:
                raise ValueError(f"Task 20 frozen {stage} evidence is missing or differs")
    if operation == "task20-evaluate":
        iteration = payload.get("task20_checkpoint_iteration")
        if not isinstance(iteration, int) or iteration < 0:
            raise ValueError("Task 20 evaluation requires a numbered checkpoint")
        marker = json.loads((path / "complete-main.json").read_text())
        metadata = check_checkpoint(path / "checkpoints" / f"{iteration:06d}", payload)
        final = marker["episodes"]
        milestone = metadata["episodes"]
        if milestone not in {0, final}:
            summaries = sorted((path / "evaluation").glob("*/dev_monitor/summary.json"))
            ranked = []
            for item in summaries:
                row = json.loads(item.read_text())
                if row["episodes"] in {*config["dev_milestones"], final}:
                    ranked.append((row["verified"] / row["count"], -row["episodes"]))
            if not ranked or milestone != -max(ranked)[1]:
                raise ValueError(
                    "Task 20 secondary final evaluation must use earliest best development"
                )


def check_task19(payload: dict) -> None:
    operation = payload["operation"]
    path = artifact_dir(payload)
    control = paths(payload)["checkout"] / Path(payload["settings"]).parent / "task19-control.json"
    wall = re.fullmatch(r"([0-9]{2}):([0-5][0-9]):00", payload.get("time", ""))
    minutes = int(wall[1]) * 60 + int(wall[2]) if wall else 0
    maximum = (
        720
        if operation in {"task19-pilot", "task19-train"}
        else 360
        if operation == "task19-base"
        else 240
    )
    if (
        payload.get("account") != "cpsc4880"
        or not payload.get("study_id", "").startswith("repair-selfplay-2026-task19-")
        or payload.get("gpu_count") != 1
        or payload.get("gpu_type") != "h200"
        or payload.get("readiness_budget") != 80
        or payload.get("seed") != 2026
        or payload.get("method") not in {"control", "A", "B", "C", "D"}
        or not 30 <= minutes <= maximum
        or payload.get("output") != f"/runs/{payload['study_id']}/{path.name}"
        or sha256(control) != payload.get("task19_control_sha256")
    ):
        raise ValueError("Task 19 requires the bounded H200 allocation and pinned initial control")
    if not study_state(payload)["gates"]["worker"]:
        raise ValueError("Task 19 requires accepted worker evidence")
    if (
        operation in {"task19-qualify", "task19-qualify-uncached", "task19-base"}
        and payload["method"] != "control"
    ):
        raise ValueError("Task 19 qualification and base require the shared control initialization")
    if operation == "task19-qualify-uncached":
        cached = path.parent / "qualification"
        if not (cached / "uncached-control.json").is_file() or json.loads(
            (cached / "numerical.json").read_text()
        ).get("passed"):
            raise ValueError("Uncached fallback requires a recorded cached qualification failure")
    if operation in {"task19-base", "task19-pilot", "task19-train"}:
        choices = [
            path.parent / folder
            for folder in ("qualification", "qualification-uncached")
            if (path.parent / folder / "calibrated-control.json").is_file()
        ]
        if len(choices) != 1:
            raise ValueError("Task 19 requires one executed calibration and qualification")
        evidence = choices[0]
        cfg = json.loads((evidence / "calibrated-control.json").read_text())
        if (
            sha256(evidence / "calibration.json") != cfg.get("calibration_hash")
            or sha256(evidence / "numerical.json") != cfg.get("qualification_hash")
            or not json.loads((evidence / "numerical.json").read_text()).get("passed")
        ):
            raise ValueError("Task 19 qualification evidence differs")
        if operation != "task19-base" and not (path.parent / "base/complete-base.json").is_file():
            raise ValueError("Task 19 requires completed matched base evaluations before training")
        if operation == "task19-train":
            main = json.loads((path.parent / "main-control.json").read_text())
            endpoint = main.pop("endpoint", None)
            cfg.pop("endpoint", None)
            if (
                main != cfg
                or type(endpoint) is not int
                or not 64 <= endpoint <= 1024
                or endpoint % 32
            ):
                raise ValueError("Task 19 common endpoint or scientific settings differ")
            pilot = json.loads((path / "complete-32.json").read_text())
            info = path / "checkpoints" / pilot["checkpoint"]
            if (
                not re.fullmatch(r"\d{6}", pilot["checkpoint"])
                or sha256(info / "complete.json") != pilot["checkpoint_sha256"]
            ):
                raise ValueError("Task 19 main requires a verified pilot checkpoint")
            check_checkpoint(info, payload)


def precheck(payload: dict):
    operation = payload["operation"]
    stop = payload.get("stop_after_iteration", 0)
    if (
        type(stop) is not int
        or stop < 0
        or (
            stop
            and (
                operation
                not in {
                    "train",
                    "pilot",
                    "task15-online",
                    "task15-stabilize",
                    "task17-student",
                    "task18-student",
                }
                or stop >= payload.get("target_iterations", 0)
            )
        )
    ):
        raise ValueError("Invalid bounded iteration stop")
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
    if operation not in TASK19_OPERATIONS | STAGED_OPERATIONS and operation not in {
        "image-convert",
        "image-build",
        "method-smoke",
        "protocol-check",
        "protocol-revision",
        "backend-check",
        "rollout-parity",
        "numerical-audit",
        "proposer-audit",
        "wandb-verify",
        "allocation-check",
        "worker-diagnose",
        "worker-check",
        "runtime-check",
        "train",
        "pilot",
        "evaluate",
        "diagnose",
        "task13-probe",
        "task14-probe",
        "task14-student",
        "task17-prepare",
        "task17-replay",
        "task17-interface",
        "task17-normalized",
        "task17-calibrate",
        "task17-student",
        "task18-numerical",
        "task18-normalized",
        "task18-native",
        "task18-student",
        "task14-counterfactual",
        "task14-counterfactual-replay",
        "task14-calibrate",
        "task14-numerical",
        "task14-numerical-components",
        "task14-proposal-student",
        "task14-static",
        "task14-online",
        "task15-online",
        "task15-stabilize",
        "task15-evaluate",
        "task15-numerical",
        "task15-freeze",
        "task13-model-probe",
        "task13-header-replay",
        "task13-static",
        "task13-static-continuation",
        "task13-counterfactual",
        "freeze-study",
        "readiness-check",
    }:
        raise ValueError("Unsupported submission operation")
    if operation == "image-build":
        validate_source_input(payload)
        if payload["image_metadata"]["kind"] in {"trainer", "task19-trainer"}:
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
            or payload.get("readiness_budget") != readiness_limit(payload)
        ):
            raise ValueError("Allocation diagnostics require bounded cpsc4880 H100/H200 readiness")
        return
    if operation in {"rollout-parity", "numerical-audit", "proposer-audit"}:
        if (
            payload.get("account") != "cpsc4880"
            or payload.get("gpu_count") != 1
            or payload.get("gpu_type") != "h200"
            or payload.get("time")
            != (
                "00:30:00"
                if operation == "proposer-audit"
                else "00:50:00"
                if operation == "numerical-audit"
                else "01:00:00"
            )
            or payload.get("readiness_budget") != readiness_limit(payload)
            or not Path(payload.get("image", "")).is_file()
            or not DIGEST.fullmatch(payload.get("image_sha256", ""))
            or not study_state(payload)["gates"]["target"]
        ):
            raise ValueError("Rollout parity requires accepted one-H200 backend and bounded budget")
        return
    uses_worker = operation in TASK19_OPERATIONS | STAGED_OPERATIONS or operation in {
        "worker-check",
        "runtime-check",
        "train",
        "pilot",
        "method-smoke",
        "protocol-check",
        "protocol-revision",
        "evaluate",
        "diagnose",
        "task13-probe",
        "task14-probe",
        "task14-student",
        "task17-prepare",
        "task17-replay",
        "task17-interface",
        "task17-normalized",
        "task17-calibrate",
        "task17-student",
        "task18-numerical",
        "task18-normalized",
        "task18-native",
        "task18-student",
        "task14-counterfactual",
        "task14-counterfactual-replay",
        "task14-calibrate",
        "task14-numerical",
        "task14-numerical-components",
        "task14-proposal-student",
        "task14-static",
        "task14-online",
        "task15-online",
        "task15-stabilize",
        "task15-evaluate",
        "task15-numerical",
        "task13-model-probe",
        "task13-header-replay",
        "task13-static",
        "task13-static-continuation",
        "task13-counterfactual",
    }
    if uses_worker and not payload.get("worker_image"):
        raise ValueError("Yale jobs require the verified allocation worker")
    if uses_worker:
        if not Path(payload["worker_image"]).is_file() or not DIGEST.fullmatch(
            payload.get("worker_sha256", "")
        ):
            raise ValueError("Pinned worker SIF missing")
    if operation in TASK19_OPERATIONS:
        check_task19(payload)
        return
    if operation in TASK21_OPERATIONS:
        check_task21(payload)
        return
    if operation in TASK20_OPERATIONS:
        check_task20(payload)
        return
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
            or payload.get("readiness_budget") != readiness_limit(payload)
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
    if operation in {
        "train",
        "pilot",
        "method-smoke",
        "evaluate",
        "diagnose",
        "freeze-study",
        "task13-probe",
        "task14-probe",
        "task14-student",
        "task17-prepare",
        "task17-replay",
        "task17-interface",
        "task17-normalized",
        "task17-calibrate",
        "task17-student",
        "task18-numerical",
        "task18-normalized",
        "task18-native",
        "task18-student",
        "task14-counterfactual",
        "task14-counterfactual-replay",
        "task14-calibrate",
        "task14-numerical",
        "task14-numerical-components",
        "task14-proposal-student",
        "task14-static",
        "task14-online",
        "task15-online",
        "task15-stabilize",
        "task15-evaluate",
        "task15-numerical",
        "task13-model-probe",
        "task13-header-replay",
        "task13-static",
        "task13-static-continuation",
        "task13-counterfactual",
    }:
        state = study_state(payload)
        if not state["gates"]["worker"] or (
            operation != "task17-prepare" and not state["gates"]["runtime"]
        ):
            raise ValueError("Runtime isolation and target data validation have not passed")
        if operation in {
            "task13-probe",
            "task14-probe",
            "task14-student",
            "task14-counterfactual",
            "task14-counterfactual-replay",
            "task14-calibrate",
            "task14-numerical",
            "task14-numerical-components",
            "task14-proposal-student",
        } and (
            payload.get("account") != "cpsc4880"
            or payload.get("gpu_count") != 1
            or payload.get("gpu_type") != "h200"
            or payload.get("time") != "02:00:00"
            or payload.get("method") not in {"A", "B", "C", "D"}
            or payload.get("iterations") not in {8, 10}
        ):
            raise ValueError("Task 13 probe requires bounded one-H200 cpsc4880 allocation")
        if operation in {
            "task14-probe",
            "task14-student",
            "task14-counterfactual",
            "task14-counterfactual-replay",
            "task14-calibrate",
            "task14-numerical",
            "task14-numerical-components",
            "task14-proposal-student",
        } and (
            payload.get("method")
            not in (
                {"A", "B", "C", "D"}
                if operation
                in {
                    "task14-probe",
                    "task14-student",
                    "task14-proposal-student",
                    "task14-numerical",
                    "task14-numerical-components",
                }
                else {"A", "B"}
            )
            or payload.get("iterations") != 10
            or payload.get("checkpoint")
        ):
            raise ValueError("Task 14 probe requires a fixed base-model arm")
        if operation.startswith("task18-"):
            expected_time = (
                "02:00:00" if operation in {"task18-numerical", "task18-student"} else "00:30:00"
            )
            control_path = paths(payload)["checkout"] / "configs/task18-control.json"
            if (
                payload.get("account") != "cpsc4880"
                or not payload.get("study_id", "").startswith("repair-selfplay-2026-task18-")
                or payload.get("gpu_count") != 1
                or payload.get("gpu_type") != "h200"
                or payload.get("readiness_budget") != 12
                or (
                    payload.get("time") not in {"00:30:00", "01:00:00", "02:00:00"}
                    if operation == "task18-numerical"
                    else payload.get("time") != expected_time
                )
                or payload.get("seed") not in {2026, 2027}
                or not control_path.is_file()
                or sha256(control_path) != payload.get("task18_control_sha256")
            ):
                raise ValueError(
                    "Task 18 requires its bounded one-H200 allocation and frozen control"
                )
        if operation.startswith("task17-"):
            cpu = operation in {"task17-prepare", "task17-replay"}
            phase2 = operation in {"task17-calibrate", "task17-student"}
            if (
                payload.get("account") != "cpsc4880"
                or not payload.get("study_id", "").startswith("repair-selfplay-2026-task17-")
                or payload.get("gpu_count") != (0 if cpu else 1)
                or (not cpu and payload.get("gpu_type") != "h200")
                or payload.get("readiness_budget") != (24 if phase2 else 10)
                or not task17_walltime_allowed(payload)
            ):
                raise ValueError("Task 17 requires its bounded isolated cpsc4880 allocation")
        if operation == "task14-static" and (
            payload.get("account") != "cpsc4880"
            or payload.get("gpu_count") != 1
            or payload.get("gpu_type") != "h200"
            or payload.get("time") != "06:00:00"
            or payload.get("method") not in {"A", "B", "C"}
            or payload.get("iterations") not in {26, 27, 28}
            or not (paths(payload)["checkout"] / "configs/task14-static-control.json").is_file()
        ):
            raise ValueError(
                "Task 14 static requires a bounded H200 arm and frozen calibration control"
            )
        if operation == "task15-numerical":
            run, job = (
                ("cd5dae1d517b8b78d2535071", "31618995")
                if payload["method"] == "A"
                else ("0078cd3bbbb89eb3c5282ddd", "31619326")
            )
            expected = f"/runs/repair-diagnosis-2026-task14-student-v1/{run}"
            source = paths(payload)["runs"] / expected.removeprefix("/runs/")
            prior = job_state(job)
            if (
                payload.get("audit_source") != expected
                or prior.get("state") != "COMPLETED"
                or prior.get("exit_code") != "0:0"
                or not (source / "complete.json").is_file()
            ):
                raise ValueError(
                    "Task 15 numerical source is not the completed training-only probe"
                )
        if operation == "task15-evaluate":
            frozen = json.loads(
                (paths(payload)["runs"] / payload["study_id"] / "freeze.json").read_text()
            )
            if (
                frozen.get("status") != "frozen"
                or frozen.get("execution_commit") != payload["commit"]
                or frozen.get("common_configs", {}).get(str(payload["seed"]))
                != payload["common_config"]
            ):
                raise ValueError("Task 15 final evaluation differs from the frozen study")
            if not payload.get("base"):
                checkpoint = paths(payload)["runs"] / payload["checkpoint"].removeprefix("/runs/")
                metadata = check_checkpoint(checkpoint, payload)
                target = payload["common_config"]["online_supply"]["target_episodes"]
                if not target <= metadata.get("total_episodes", 0) <= target + 12:
                    raise ValueError(
                        "Task 15 final evaluation requires the completed exposure endpoint"
                    )
        if operation in {"task15-online", "task15-stabilize"}:
            if payload.get("logging_mode") == "online" and not payload.get("secrets_file"):
                raise ValueError(
                    "Online Task 15 logging requires the existing private secrets-file path"
                )
            if (
                payload.get("account") != "cpsc4880"
                or payload.get("gpu_count") != 1
                or payload.get("gpu_type") != "h200"
                or payload.get("time") != "06:00:00"
                or payload.get("method") not in {"A", "B"}
                or not state["gates"]["target"]
                or not payload.get("target_episodes")
                or payload.get("exposure_policy", {}).get("target_episodes")
                != payload["target_episodes"]
            ):
                raise ValueError("Task 15 requires accepted bounded H200 and an exposure policy")
            if operation == "task15-online":
                freeze_path = paths(payload)["runs"] / payload["study_id"] / "freeze.json"
                frozen = json.loads(freeze_path.read_text())
                if (
                    frozen.get("status") != "frozen"
                    or frozen.get("common_configs", {}).get(str(payload["seed"]))
                    != payload["common_config"]
                    or frozen.get("execution_commit") != payload["commit"]
                ):
                    raise ValueError("Task 15 main launch does not match its immutable freeze")
        if operation == "task14-online" and (
            payload.get("account") != "cpsc4880"
            or payload.get("gpu_count") != 1
            or payload.get("gpu_type") != "h200"
            or payload.get("time") != "06:00:00"
            or payload.get("method") not in {"A", "B", "C", "D"}
            or payload.get("iterations") != 20
            or not payload.get("secrets_file")
            or not state["gates"]["target"]
        ):
            raise ValueError(
                "Task 14 online arm requires bounded H200, verified backend and online logging"
            )
        if operation in {"task14-numerical", "task14-numerical-components"}:
            if operation == "task14-numerical-components" and payload["method"] not in {"A", "C"}:
                raise ValueError("Numerical decomposition requires a diff student arm")
            source_run, source_job = {
                "A": ("cd5dae1d517b8b78d2535071", "31618995"),
                "B": ("6efc9d94e4f5e2e7b0b0fd04", "31618998"),
                "C": ("0078cd3bbbb89eb3c5282ddd", "31619326"),
                "D": ("bce7b4ed210efe8445488fbd", "31619331"),
            }[payload["method"]]
            source = paths(payload)["runs"] / "repair-diagnosis-2026-task14-student-v1" / source_run
            state = job_state(source_job)
            if (
                state.get("state") != "COMPLETED"
                or state.get("exit_code") != "0:0"
                or not (source / "complete.json").is_file()
            ):
                raise ValueError("Numerical probe requires a completed student source arm")
        if operation == "task14-proposal-student":
            source_run, source_job = {
                "A": ("276931a7445252e963f37b1b", "31618956"),
                "B": ("36db89f113049ef37c32b79d", "31618960"),
                "C": ("d113371007fb12d958f92ccf", "31619234"),
                "D": ("5c34adbea7983c01775ef5a8", "31619324"),
            }[payload["method"]]
            source = paths(payload)["runs"] / "repair-diagnosis-2026-task14-probe-v1" / source_run
            status = job_state(source_job)
            if (
                status.get("state") != "COMPLETED"
                or status.get("exit_code") != "0:0"
                or not (source / "complete.json").is_file()
            ):
                raise ValueError("Student rollout probe requires a completed proposer source")
        if operation == "task14-counterfactual-replay":
            source = (
                paths(payload)["runs"]
                / "repair-diagnosis-2026-task14-counterfactual-v1/e74ecdee52efe104590c60b3"
            )
            status = job_state("31619094")
            if (
                payload.get("method") != "B"
                or status.get("state") != "COMPLETED"
                or status.get("exit_code") != "0:0"
                or not (source / "complete.json").is_file()
            ):
                raise ValueError("Counterfactual replay requires the completed structured source")
        if operation == "task13-header-replay" and (
            payload.get("account") != "cpsc4880"
            or payload.get("gpu_count") != 0
            or payload.get("time") != "01:00:00"
            or payload.get("method") != "C"
            or payload.get("checkpoint")
        ):
            raise ValueError("Header replay requires the bounded CPU-only worker allocation")
        if operation == "task13-static-continuation":
            source = (
                paths(payload)["runs"] / "repair-diagnosis-2026-static-v1/9a44124b61147aee02c685d6"
            )
            original = job_state("31535783")
            if (
                payload.get("account") != "cpsc4880"
                or payload.get("gpu_count") != 1
                or payload.get("gpu_type") != "h200"
                or payload.get("time") != "06:00:00"
                or payload.get("method") != "B"
                or not (
                    (
                        original.get("state") == "COMPLETED"
                        and original.get("exit_code") == "0:0"
                        and (source / "stopped.json").is_file()
                    )
                    or (
                        original.get("state") == "TIMEOUT"
                        and any(
                            p.is_file()
                            for p in source.glob(
                                "repair-diagnosis-2026-static-*/B-seed2026/checkpoints/[0-9][0-9][0-9][0-9][0-9][0-9]/complete.json"
                            )
                        )
                    )
                )
                or (source / "complete.json").exists()
            ):
                raise ValueError(
                    "Static continuation requires the original clean stopped allocation or timeout with a joint checkpoint"
                )
        if operation == "task13-model-probe" and (
            payload.get("account") != "cpsc4880"
            or payload.get("gpu_count") != 1
            or payload.get("gpu_type") != "h200"
            or payload.get("time") != "02:00:00"
            or payload.get("method") not in {"A", "B", "C", "D"}
            or payload.get("checkpoint")
        ):
            raise ValueError("Task 13 model probe requires bounded base-model inference")
        if operation == "task13-model-probe":
            models = {
                "A": (
                    "Qwen/Qwen2.5-Coder-7B-Instruct",
                    "c03e6d358207e414f1eca0bb1891e29f1db0e242",
                    "diff",
                ),
                "B": (
                    "Qwen/Qwen2.5-Coder-7B-Instruct",
                    "c03e6d358207e414f1eca0bb1891e29f1db0e242",
                    "structured",
                ),
                "C": ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218", "diff"),
                "D": ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218", "structured"),
            }
            name, revision, representation = models[payload["method"]]
            expected = {
                "name": name,
                "revision": revision,
                "representation": representation,
                "non_thinking": payload["method"] in {"C", "D"},
                "backend": "vllm-inference-only",
            }
            if payload.get("probe_model") != expected:
                raise ValueError("Task 13 model identity does not match the pinned manifest")
        if operation == "task13-static" and (
            payload.get("account") != "cpsc4880"
            or payload.get("gpu_count") != 1
            or payload.get("gpu_type") != "h200"
            or payload.get("time") != "06:00:00"
            or payload.get("method") != "B"
        ):
            raise ValueError(
                "Task 13 static comparison requires bounded one-H200 cpsc4880 allocation"
            )
        if operation == "task13-counterfactual" and (
            payload.get("account") != "cpsc4880"
            or payload.get("gpu_count") != 1
            or payload.get("gpu_type") != "h200"
            or payload.get("time") != "02:00:00"
            or payload.get("method") != "C"
            or not payload.get("checkpoint", "").endswith("/000099")
        ):
            raise ValueError("Task 13 counterfactual requires one H200 and C checkpoint 99")
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
        metadata.get("kind") not in {"trainer", "task19-trainer", "build-check"}
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


def reuse_runtime(payload: dict) -> dict:
    """Reuse executed verifier acceptance across Task 17 loss/interface settings."""
    source_study, study = payload["source_study"], payload["study_id"]
    if source_study == study or not all(
        SAFE_NAME.fullmatch(s) and s.startswith("repair-selfplay-2026-task17-")
        for s in (source_study, study)
    ):
        raise ValueError("Distinct Task 17 studies required for runtime reuse")
    base = workspace(payload)
    source = base / "runs/readiness" / source_study / "runtime.json"
    raw = source.read_bytes()
    report = json.loads(raw)
    expected, actual = payload["runtime_identity"], report.get("runtime_identity", {})
    if (
        not actual.get("common_hash")
        or not expected.get("common_hash")
        or {k: v for k, v in actual.items() if k != "common_hash"}
        != {k: v for k, v in expected.items() if k != "common_hash"}
    ):
        raise ValueError("Runtime code/model/image/worker/data dependencies changed")
    validation = report.get("validation", {})
    if not (
        report.get("ready")
        and report.get("isolation", {}).get("ready")
        and validation.get("ready")
        and validation.get("executed")
        and validation.get("checked_tasks")
        and all(int(n) > 0 for n in validation["checked_tasks"].values())
        and validation.get("manifest_hash") == expected["manifest_hash"]
        and validation.get("execution_limits") == payload["execution_limits"]
        and validation.get("sandbox_image_digest") == "sha256:" + expected["worker_sha256"]
        and validation.get("family_audit") == validation.get("split_audit") == "pass"
    ):
        raise ValueError("Executed verifier/isolation acceptance is missing or limits changed")
    content_hash = hashlib.sha256(raw).hexdigest()
    archive = base / "runs/compatibility/task17-runtime" / (content_hash + ".json")
    target = base / "runs/readiness" / study / "runtime.json"
    reused = report | {
        "runtime_identity": expected,
        "original_runtime_identity": actual,
        "reuse_provenance": {"source": str(archive), "sha256": content_hash},
        "reuse_scope": "Identical executed data/verifier dependencies; no new loss, model-generation or learning acceptance",
    }
    with locked(base / "state/.task17-runtime-evidence.lock"):
        archive.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if archive.exists() and archive.read_bytes() != raw:
            raise ValueError("Immutable runtime proof archive differs")
        if not archive.exists():
            archive.write_bytes(raw)
        if target.exists() and json.loads(target.read_text()) != reused:
            previous = target.read_bytes()
            old = archive.with_name(hashlib.sha256(previous).hexdigest() + ".json")
            if old.exists() and old.read_bytes() != previous:
                raise ValueError("Previous runtime proof archive differs")
            if not old.exists():
                old.write_bytes(previous)
        write_json(target.with_name("validation.json"), validation)
        write_json(target, reused)
    return {
        "ready": True,
        "evidence": str(target),
        "provenance": reused["reuse_provenance"],
        "scope": reused["reuse_scope"],
    }


def reuse_telemetry(payload: dict) -> dict:
    """Reuse the actual allocated-node cloud proof when its dependencies match."""
    source_study, study = payload["source_study"], payload["study_id"]
    if (
        not SAFE_NAME.fullmatch(source_study)
        or not SAFE_NAME.fullmatch(study)
        or source_study == study
    ):
        raise ValueError("Distinct safe study names required")
    base = workspace(payload)
    if (base / "runs" / study / "study_frozen.json").exists():
        raise ValueError("Target study is frozen; evidence cannot be replaced")
    source = base / "runs/readiness" / source_study / "wandb-verification.json"
    report = json.loads(source.read_text())
    frozen = json.loads((base / "runs" / source_study / "study_frozen.json").read_text())
    if (
        not report.get("ready")
        or not report.get("cloud_verified")
        or report.get("sequence") != 2
        or report.get("entity") != payload.get("entity")
        or report.get("telemetry_identity") != payload.get("telemetry_identity")
        or frozen.get("telemetry_identity") != payload.get("telemetry_identity")
        or frozen.get("runtime_identity", {}).get("trainer_sha256") != payload.get("trainer_sha256")
    ):
        raise ValueError("Telemetry proof dependencies differ or cloud verification is incomplete")
    reused = report | {"reuse_provenance": {"source": str(source), "sha256": sha256(source)}}
    target = base / "runs/readiness" / study / "wandb-verification.json"
    with locked(base / "state/.telemetry-evidence.lock"):
        if target.exists() and json.loads(target.read_text()) != reused:
            raise ValueError("Destination already has different telemetry evidence")
        write_json(target, reused)
    return {"ready": True, "evidence": str(target), "provenance": reused["reuse_provenance"]}


def reuse_allocation(payload: dict) -> dict:
    """Reuse allocation verification only when the image and verifier match."""
    source_study, study, profile = payload["source_study"], payload["study_id"], payload["profile"]
    if (
        not all(SAFE_NAME.fullmatch(v) for v in (source_study, study, profile))
        or source_study == study
    ):
        raise ValueError("Distinct safe study and profile names required")
    base = workspace(payload)
    if (base / "runs" / study / "study_frozen.json").exists():
        raise ValueError("Target study is frozen; evidence cannot be replaced")
    source = base / "runs/readiness" / source_study / f"allocation-{profile}.json"
    proof = json.loads(source.read_text())
    if (
        not proof.get("ready")
        or not proof.get("ownership_verified")
        or proof.get("image_sha256") != payload.get("trainer_sha256")
        or proof.get("gpu_type") != payload.get("gpu_type")
        or proof.get("gpu_count") != payload.get("gpu_count")
        or not allocation_compatible(proof, payload.get("allocation_code_hash"), base)
    ):
        raise ValueError("Allocation proof dependencies differ or ownership is unverified")
    reused = proof | {"reuse_provenance": {"source": str(source), "sha256": sha256(source)}}
    target = base / "runs/readiness" / study / f"allocation-{profile}.json"
    with locked(base / "state/.allocation-evidence.lock"):
        if target.exists() and json.loads(target.read_text()) != reused:
            raise ValueError("Destination already has different allocation evidence")
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


def task19_budget_status(payload: dict) -> dict:
    if not re.fullmatch(r"repair-selfplay-2026-task19-[A-Za-z0-9_.-]+", payload["study_id"]):
        raise ValueError("Invalid Task 19 study")
    with locked(workspace(payload) / "state/.budget.lock"):
        return refreshed_budget(payload)


def task20_budget_status(payload: dict) -> dict:
    if not task20_budget(payload):
        raise ValueError("Task 20 budget status requires a Task 20 study")
    with locked(workspace(payload) / "state/.budget.lock"):
        return refreshed_budget(payload)


def task21_budget_status(payload: dict) -> dict:
    if not task21_budget(payload):
        raise ValueError("Task 21 budget status requires a Task 21 study")
    with locked(workspace(payload) / "state/.budget.lock"):
        return refreshed_budget(payload)


def task21_verify_endpoint(payload: dict) -> dict:
    report = status(payload)
    spec = report["ledger"]["specification"]
    if spec.get("operation") not in TASK21_OPERATIONS or not report["attempts"]:
        raise ValueError("Endpoint verification requires a Task 21 allocation")
    attempt = report["attempts"][-1]
    if attempt["scheduler"]["state"] != "COMPLETED":
        raise ValueError("Task 21 allocation is not terminal and successful")
    root = artifact_dir(spec)
    marker = json.loads((root / stage_marker(spec)).read_text())
    if not re.fullmatch(r"\d{6}", marker["checkpoint"]):
        raise ValueError("Invalid committed checkpoint ID")
    checkpoint = root / "checkpoints" / marker["checkpoint"]
    record = check_checkpoint(checkpoint, spec)
    if (
        marker["checkpoint_sha256"] != sha256(checkpoint / "complete.json")
        or marker["identity"] != record["identity"]
        or marker["episodes"] != record["episodes"]
        or marker["snapshot"] != record["snapshot"]
    ):
        raise ValueError("Task 21 committed marker and checkpoint disagree")
    return {
        **marker,
        "run_id": report["run_id"],
        "job_id": attempt["job_id"],
        "scheduler": attempt["scheduler"],
    }


def stage_task19_baseline(payload: dict) -> dict:
    """Reuse the verified initial monitoring tasks for the five identical initial policies."""
    base = workspace(payload)
    study = payload["study_id"]
    if not re.fullmatch(r"repair-selfplay-2026-task19-[A-Za-z0-9_-]+", study):
        raise ValueError("Invalid Task 19 study")
    root = base / "runs" / study
    source = root / "base"
    with locked(base / "state/.submit.lock"):
        slot = hashlib.sha256(f"{source}:task19-base".encode()).hexdigest()
        owner = json.loads((base / "state/slots" / f"{slot}.json").read_text())["run_id"]
        state = status(payload | {"run_id": owner})
        if (
            state["ledger"].get("pending_intent")
            or not state["attempts"]
            or state["attempts"][-1]["scheduler"]["state"] != "COMPLETED"
        ):
            raise ValueError("Initial monitoring reuse requires a completed baseline")
        marker = json.loads((source / "complete-base.json").read_text())
        summary_path = source / "evaluation/0000/dev_monitor/summary.json"
        summary = json.loads(summary_path.read_text())
        choices = [
            root / name
            for name in ("qualification", "qualification-uncached")
            if (root / name / "calibrated-control.json").is_file()
        ]
        if len(choices) != 1:
            raise ValueError("Initial monitoring requires one qualified control")
        qualified = choices[0]
        qualification = json.loads((qualified / "calibration.json").read_text())
        calibrated = json.loads((qualified / "calibrated-control.json").read_text())
        numerical = json.loads((qualified / "numerical.json").read_text())
        identity = marker["identity"]
        if (
            any(
                identity.get(key) != qualification["identity"].get(key)
                for key in ("study", "arm", "data", "execution", "initial", "worker")
            )
            or calibrated.get("calibration_hash") != sha256(qualified / "calibration.json")
            or calibrated.get("qualification_hash") != sha256(qualified / "numerical.json")
            or numerical.get("passed") is not True
            or identity.get("study") != study
            or identity.get("arm") != "control"
            or marker["snapshot"] != identity.get("initial")
            or summary.get("identity") != identity
            or summary.get("snapshot") != marker["snapshot"]
            or summary.get("episodes") != 0
            or summary.get("split") != "dev_monitor"
        ):
            raise ValueError("Initial monitoring identity differs from qualification")
        task_ids = [row["task_id"] for row in summary["tasks"]]
        summary_rows = {row["task_id"]: row for row in summary["tasks"]}
        if (
            not task_ids
            or len(task_ids) != summary["count"]
            or len(set(task_ids)) != len(task_ids)
            or any(not DIGEST.fullmatch(task) for task in task_ids)
        ):
            raise ValueError("Initial monitoring task coverage differs")
        source_tasks = summary_path.parent / "tasks"
        records = {}
        for task in task_ids:
            path = source_tasks / f"{task}.json"
            if path.is_symlink() or not path.is_file():
                raise ValueError("Initial monitoring task record missing")
            row = json.loads(path.read_text())
            row_identity = row["identity"]
            summary_row = summary_rows[task]
            if (
                row_identity.get("task") != task
                or row_identity.get("data") != identity["data"]
                or row_identity.get("science") != identity["science"]
                or row_identity.get("snapshot") != marker["snapshot"]
                or row.get("repair", {}).get("parent_id") != summary_row.get("parent_id")
                or row.get("metrics")
                != {k: v for k, v in summary_row.items() if k not in {"task_id", "parent_id"}}
            ):
                raise ValueError("Initial monitoring task identity differs")
            records[task] = sha256(path)
        if {p.stem for p in source_tasks.glob("*.json")} != set(task_ids):
            raise ValueError("Initial monitoring task files do not match summary")
        arms = ("control", "A", "B", "C", "D")
        for arm in arms:
            target = root / f"{arm}-seed2026"
            for operation in ("task19-pilot", "task19-train"):
                key = hashlib.sha256(f"{target}:{operation}".encode()).hexdigest()
                if (base / "state/slots" / f"{key}.json").exists():
                    raise ValueError("Cannot stage baseline after an arm has been submitted")
        for arm in arms:
            target = root / f"{arm}-seed2026/evaluation/0000/dev_monitor/tasks"
            target.mkdir(parents=True, exist_ok=True, mode=0o700)
            for task, expected in records.items():
                destination = target / f"{task}.json"
                if destination.exists():
                    if destination.is_symlink() or sha256(destination) != expected:
                        raise ValueError("Existing staged task differs from baseline")
                    continue
                temp = target / f".{task}.{uuid.uuid4().hex}.tmp"
                try:
                    with (source_tasks / f"{task}.json").open("rb") as src, temp.open("xb") as dst:
                        os.chmod(temp, 0o600)
                        shutil.copyfileobj(src, dst)
                        dst.flush()
                        os.fsync(dst.fileno())
                    if sha256(temp) != expected:
                        raise ValueError("Staged task checksum differs")
                    os.replace(temp, destination)
                finally:
                    temp.unlink(missing_ok=True)
            write_json(
                target.parent / "reused-base.json",
                {
                    "base_run": owner,
                    "base_marker_sha256": sha256(source / "complete-base.json"),
                    "summary_sha256": sha256(summary_path),
                    "calibration_sha256": calibrated["calibration_hash"],
                    "numerical_sha256": calibrated["qualification_hash"],
                    "snapshot": marker["snapshot"],
                    "task_sha256": records,
                },
            )
        return {"base_run": owner, "tasks": len(records), "arms": list(arms)}


def publish_task19_endpoint(payload: dict) -> dict:
    base = workspace(payload)
    study = payload["study_id"]
    cap = payload["cap"]
    if type(cap) not in {int, float} or not 0 < cap <= 80:
        raise ValueError("Use the verified total Yale allowance, up to 80 H200-hours")
    root = base / "runs" / study
    with locked(base / "state/.submit.lock"):
        budget = task19_budget_status(payload)
        if (root / "main-control.json").is_file():
            plan = json.loads((root / "main-plan.json").read_text())
            if plan["cap"] != cap:
                raise ValueError("Task 19 already has a frozen common endpoint")
            return plan | {"control": json.loads((root / "main-control.json").read_text())}
        qualified = [
            path
            for path in (root / "qualification", root / "qualification-uncached")
            if (path / "calibrated-control.json").is_file()
        ]
        if len(qualified) != 1:
            raise ValueError("Expected one successful Task 19 qualification")
        control = json.loads((qualified[0] / "calibrated-control.json").read_text())

        def stage(operation: str, directory: Path, marker: str) -> tuple[dict, float, str]:
            slot = hashlib.sha256(f"{directory}:{operation}".encode()).hexdigest()
            run_id = json.loads((base / "state/slots" / f"{slot}.json").read_text())["run_id"]
            state = status(payload | {"run_id": run_id})
            if (
                state["ledger"].get("pending_intent")
                or not state["attempts"]
                or state["attempts"][-1]["scheduler"]["state"] != "COMPLETED"
            ):
                raise ValueError(
                    "Pilot/base allocation lacks reconciled successful Slurm completion"
                )
            costs = [
                item["charged"]
                for item in budget["reservations"].values()
                if item["run_id"] == run_id
            ]
            if not costs or any(cost is None for cost in costs):
                raise ValueError("Endpoint freeze requires actual allocated GPU time")
            proof = json.loads((directory / marker).read_text())
            return proof, sum(costs), run_id

        basemarker, base_hours, base_run = stage("task19-base", root / "base", "complete-base.json")
        pilots = {}
        initial = basemarker["identity"]["initial"]
        for arm in ("control", "A", "B", "C", "D"):
            directory = root / f"{arm}-seed2026"
            pilot, hours, run_id = stage("task19-pilot", directory, "complete-32.json")
            name = pilot["checkpoint"]
            if not re.fullmatch(r"\d{6}", name) or pilot["episodes"] != 32 or hours <= 0:
                raise ValueError("Pilot exposure, checkpoint name or actual allocation differs")
            cp = directory / "checkpoints" / name
            if sha256(cp / "complete.json") != pilot["checkpoint_sha256"]:
                raise ValueError("Pilot checkpoint checksum differs")
            spec = read_ledger(base / "state" / f"{run_id}.json")["specification"]
            meta = check_checkpoint(cp, spec)
            identity = meta["identity"]
            if (
                identity != pilot["identity"]
                or identity["initial"] != initial
                or any(
                    identity[key] != basemarker["identity"][key]
                    for key in (
                        "study",
                        "science",
                        "data",
                        "worker",
                        "execution",
                    )
                )
            ):
                raise ValueError("Pilot/base initialization or common identity differs")
            pilots[arm] = {
                "controller_run": run_id,
                "gpu_hours": hours,
                "checkpoint": name,
                "checkpoint_sha256": pilot["checkpoint_sha256"],
            }
        student_rate = max(pilots[arm]["gpu_hours"] for arm in ("control", "A", "B")) / 32
        alignment_rate = max(pilots[arm]["gpu_hours"] for arm in ("C", "D")) / 32
        fixed_hours = 5 * base_hours
        remaining = (cap - budget["committed_gpu_hours"]) / 1.2 - fixed_hours
        block_hours = 32 * (3 * student_rate + 2 * alignment_rate)
        endpoint = min(1024, 32 + 32 * int(max(0.0, remaining) / block_hours))
        plan = {
            "study": study,
            "cap": cap,
            "spent_gpu_hours": budget["committed_gpu_hours"],
            "base": {"controller_run": base_run, "gpu_hours": base_hours},
            "pilots": pilots,
            "endpoint": endpoint,
            "ready": endpoint > 32,
            "planned_evaluation_hours": fixed_hours,
            "student_hours_per_1000_episodes": student_rate * 1000,
            "alignment_hours_per_1000_episodes": alignment_rate * 1000,
            "contingency": 0.2,
        }
        if endpoint > 32:
            main_control = control | {"endpoint": endpoint}
            write_json(root / "main-plan.json", plan)
            write_json(root / "main-control.json", main_control)
            return plan | {"control": main_control}
        return plan


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
    "reuse_runtime": reuse_runtime,
    "reuse_telemetry": reuse_telemetry,
    "reuse_allocation": reuse_allocation,
    "reuse_study": reuse_study,
    "publish_resource_plan": publish_resource_plan,
    "task19_budget_status": task19_budget_status,
    "task20_budget_status": task20_budget_status,
    "task21_budget_status": task21_budget_status,
    "task21_verify_endpoint": task21_verify_endpoint,
    "publish_task19_endpoint": publish_task19_endpoint,
    "stage_task19_baseline": stage_task19_baseline,
}


def main() -> None:
    command = sys.argv[1]
    if len(sys.argv) > 2:
        argument = sys.argv[2]
        raw = base64.b64decode(argument[2:] if argument.startswith("z:") else argument)
        payload = json.loads(zlib.decompress(raw) if argument.startswith("z:") else raw)
    else:
        payload = {}
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
