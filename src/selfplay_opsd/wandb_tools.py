"""Explicit cloud verification and aggregate-only recovery; never loads a model."""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Literal

from selfplay_opsd.config import RunConfig
from selfplay_opsd.storage import CheckpointManager, active_events, atomic_json, read_json
from selfplay_opsd.telemetry import (
    METHOD_NAMES,
    canonical_payload,
    canonical_snapshot,
    update_canonical_summary,
)


def sdk_settings():
    import wandb

    return wandb.Settings(
        init_timeout=15, disable_code=True, disable_git=True, console="off", x_disable_stats=True
    )


def verify(
    output: Path,
    *,
    entity: str | None = None,
    run_id: str | None = None,
    sequence: int = 1,
    timeout: float = 45,
) -> dict:
    """Invoke twice with the returned ID to validate cross-process resume.

    A successful run requires a cloud-side history value, never merely init().
    Even a missing-key report includes compute-node HTTPS reachability.
    """
    import httpx
    import wandb

    output.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "ready": False,
        "cloud_verified": False,
        "sequence": sequence,
        "allocated_gpu_count": 0,
        "hostname": __import__("socket").gethostname(),
    }
    base_url = os.getenv("WANDB_BASE_URL", "https://api.wandb.ai").rstrip("/")
    if base_url != "https://api.wandb.ai":
        raise ValueError("Verification only supports the approved public W&B endpoint")
    try:
        response = httpx.get(base_url + "/", timeout=10)
        report["https_reachable"] = True
        report["https_status"] = response.status_code
    except Exception as exc:
        report.update(https_reachable=False, network_error_type=type(exc).__name__)
    if not os.getenv("WANDB_API_KEY"):
        report["required_action"] = (
            "Run scripts/setup_wandb.py in a terminal; pass --secrets-file on Yale"
        )
        atomic_json(output / "wandb-verification.json", report)
        return report
    run_id = run_id or ("telemetry-check-" + uuid.uuid4().hex[:16])
    run = None
    try:
        api = wandb.Api(timeout=15)
        # The authenticated API determines the entity; no guessed namespace.
        viewer = api.viewer
        selected = entity or os.getenv("WANDB_ENTITY") or api.default_entity
        report["authenticated_username"] = viewer.username
        if not selected:
            raise ValueError("Specify an authorized W&B entity")
        run = wandb.init(
            project="self-play-opsd-coding-agent",
            entity=selected,
            id=run_id,
            resume="must" if sequence > 1 else "never",
            mode="online",
            group="telemetry-verification",
            name=run_id,
            dir=str(output),
            settings=sdk_settings(),
            config={"synthetic_telemetry_check": True},
        )
        url, path = run.url, "/".join((run.entity, run.project, run.id))
        run.log({"verification/sequence": sequence, "verification/timestamp": time.time()})
        run.finish()
        run = None
        deadline = time.monotonic() + timeout
        sequences: set[int] = set()
        while True:
            api.flush()
            remote = api.run(path)
            sequences = {
                int(row["verification/sequence"])
                for row in remote.scan_history(keys=["verification/sequence"])
            }
            if sequence in sequences and (sequence == 1 or sequence - 1 in sequences):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Cloud history did not contain expected verification metrics")
            time.sleep(2)
        report.update(
            ready=True,
            cloud_verified=True,
            entity=selected,
            run_id=run_id,
            url=url,
            project_url=f"https://wandb.ai/{selected}/self-play-opsd-coding-agent",
            cloud_sequences=sorted(sequences),
        )
    except Exception as exc:
        report["error_type"] = type(exc).__name__
    finally:
        if run is not None:
            try:
                run.finish()
            except Exception:
                pass
    atomic_json(output / "wandb-verification.json", report)
    return report


def reconcile(
    root: Path, *, entity: str | None = None, mode: Literal["online", "offline"] = "online"
) -> dict:
    """One writer per logical run. Never run alongside an active trainer sink."""
    import fcntl

    import wandb

    from selfplay_opsd.wandb_tools import sdk_settings

    events = active_events(root)
    resolved = next(e["payload"] for e in events if e["kind"] == "run_config")
    config = RunConfig.model_validate(
        {k: v for k, v in resolved.items() if k in RunConfig.model_fields}
    )
    logical = next(e["logical_run_id"] for e in events if "logical_run_id" in e)
    pointer = read_json(root / "checkpoints/latest.json")
    CheckpointManager.verify(root / "checkpoints" / pointer["path"])
    with (root / "state/telemetry-writer.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        snapshot = canonical_snapshot(root, config)
        run = wandb.init(
            project=config.logging.project,
            entity=entity or config.logging.entity,
            id=logical if mode == "online" else uuid.uuid4().hex,
            resume="allow" if mode == "online" else None,
            mode=mode,
            group=config.study_id,
            name=METHOD_NAMES.get(config.method, config.method),
            dir=str(root),
            settings=sdk_settings(),
        )
        try:
            payload = canonical_payload(wandb, snapshot)
            run.log(payload)
            update_canonical_summary(run, payload)
            url = run.url
        finally:
            run.finish()
        report = {
            "sdk_delivered_iteration": snapshot["summary"]["run/checkpoint_iteration"],
            "canonical_revision": snapshot["revision"],
            "url": url,
            "mode": mode,
            "cloud_verified_iteration": None,
        }
        if mode == "online":
            api = wandb.Api(timeout=15)
            remote = api.run("/".join((run.entity, run.project, run.id)))
            if remote.summary.get("run/canonical_revision") != snapshot["revision"]:
                raise RuntimeError(
                    "Cloud canonical revision not yet verified; retry reconciliation"
                )
            report["cloud_verified_iteration"] = report["sdk_delivered_iteration"]
        atomic_json(root / "state/telemetry-reconciliation.json", report)
        return report
