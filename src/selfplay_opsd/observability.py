"""Bounded best-effort W&B delivery; scientific records always stay local."""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import torch

from selfplay_opsd.config import RunConfig
from selfplay_opsd.storage import EventStore


def distribution(values: list[float]) -> dict:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "p05": None,
            "p50": None,
            "p90": None,
            "p95": None,
        }
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise FloatingPointError("Nonfinite metric distribution")
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "p05": float(np.quantile(array, 0.05)),
        "p50": float(np.quantile(array, 0.5)),
        "p90": float(np.quantile(array, 0.9)),
        "p95": float(np.quantile(array, 0.95)),
    }


def flatten(value: dict, prefix: str = "") -> dict:
    output = {}
    for key, child in value.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(child, dict):
            output.update(flatten(child, name))
        elif isinstance(child, (float, int, str, bool)) or child is None:
            output[name] = child
    return output


class WandbSink:
    """Network work stays on one bounded daemon; local events are the spool.

    Notifications coalesce rather than queueing copies of trajectories/metrics.
    Checkpoint callbacks are the publication boundary, independent of optional
    artifact uploads. No model/RNG/optimizer code runs in the delivery thread.
    """

    def __init__(self, config: RunConfig, store: EventStore):
        self.config, self.store = config, store
        self.errors: queue.Queue = queue.Queue(maxsize=32)
        self.thread: threading.Thread | None = None
        self.wake = threading.Event()
        self.stopping = threading.Event()
        self.dropped = 0
        self.last_revision = None
        self.status: dict = {
            "sdk_delivered_iteration": -1,
            "cloud_verified_iteration": None,
            "delivery_errors": 0,
            "local_data_preserved": True,
        }
        if config.logging.mode != "disabled":
            self.thread = threading.Thread(target=self._worker, name="wandb-delivery", daemon=True)
            self.thread.start()

    def _warning(self, exc):
        # Exception text/URLs may contain credentials or model context.
        self.status["delivery_errors"] += 1
        try:
            self.errors.put_nowait(type(exc).__name__)
        except queue.Full:
            self.dropped += 1

    def _write_status(self):
        from selfplay_opsd.storage import atomic_json

        atomic_json(self.store.root / "state/telemetry.json", self.status)

    def _worker(self):
        import fcntl

        try:
            with (self.store.root / "state/telemetry-writer.lock").open("a+") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._deliver()
        except Exception as exc:
            self._warning(exc)

    def _deliver(self):
        from selfplay_opsd.telemetry import (
            METHOD_NAMES,
            canonical_payload,
            canonical_snapshot,
            update_canonical_summary,
        )

        backoff = 5
        while True:
            run = None
            try:
                import wandb

                run = wandb.init(
                    project=self.config.logging.project,
                    entity=self.config.logging.entity,
                    group=self.config.study_id,
                    name=METHOD_NAMES.get(self.config.method, self.config.method),
                    id=self.store.identity["logical_run_id"]
                    if self.config.logging.mode == "online"
                    else self.store.segment,
                    resume="allow" if self.config.logging.mode == "online" else None,
                    mode=self.config.logging.mode,
                    dir=str(self.store.root),
                    config={
                        "study_id": self.config.study_id,
                        "method": self.config.method,
                        "seed": self.config.seed,
                        "iterations": self.config.iterations,
                        "scientific_hash": self.config.scientific_hash,
                        "model_revision": self.config.model.revision,
                        "history_semantics": "append_only_audit; canonical tables are authoritative",
                    },
                    tags=[self.config.method, self.config.hardware.profile],
                    settings=wandb.Settings(
                        init_timeout=15,
                        disable_code=True,
                        disable_git=True,
                        console="off",
                        x_disable_stats=True,
                    ),
                )
                run.define_metric("train/outer_iteration")
                run.define_metric("compute/allocated_gpu_hours")
                for group in (
                    "student/*",
                    "proposer/*",
                    "distill/*",
                    "alignment/*",
                    "perf/*",
                    "system/*",
                    "telemetry/*",
                    "eval_by_iteration/*",
                ):
                    run.define_metric(group, step_metric="train/outer_iteration")
                run.define_metric("eval_by_gpu_hours/*", step_metric="compute/allocated_gpu_hours")
                self.status["url"] = run.url
                # Rebuild canonical tables after every init, including uncertain
                # delivery. The append-only scalar audit may repeat; tables do not.
                self.last_revision = None
                refresh = True
                snapshot = None
                while True:
                    self.wake.clear()
                    if refresh or snapshot is None:
                        snapshot = canonical_snapshot(self.store.root, self.config)
                    if snapshot["revision"] != self.last_revision:
                        payload = canonical_payload(wandb, snapshot)
                        payload.update(
                            {
                                "run/canonical_revision": snapshot["revision"],
                                "run/logical_run_id": self.store.identity["logical_run_id"],
                                "run/current_segment_id": self.store.segment,
                                "telemetry/sdk_enqueue_timestamp": time.time(),
                                "telemetry/delivery_errors": self.status["delivery_errors"],
                            }
                        )
                        run.log(payload)
                        # Summary tables give a replaceable authoritative view.
                        update_canonical_summary(run, payload)
                        self.last_revision = snapshot["revision"]
                        self.status.update(
                            sdk_delivered_iteration=snapshot["summary"]["run/checkpoint_iteration"],
                            sdk_enqueue_timestamp=time.time(),
                            canonical_revision=snapshot["revision"],
                        )
                    self._write_status()
                    backoff = 5
                    if self.stopping.is_set():
                        break
                    refresh = self.wake.wait(30)
                break
            except Exception as exc:
                self._warning(exc)
                try:
                    self._write_status()
                except OSError:
                    pass
            finally:
                if run is not None:
                    try:
                        run.finish()
                    except Exception as exc:
                        self._warning(exc)
            if self.stopping.wait(backoff):
                break
            backoff = min(60, backoff * 2)

    def _drain_warnings(self):
        while not self.errors.empty():
            self.store.emit(
                "logging_warning",
                {
                    "error_type": self.errors.get_nowait(),
                    "local_data_preserved": True,
                    "sdk_delivered_iteration": self.status["sdk_delivered_iteration"],
                    "cloud_verified_iteration": self.status["cloud_verified_iteration"],
                    "latest_local_iteration": self.store.iteration,
                    "recovery": "selfplay wandb-reconcile --run-dir RUN_DIR",
                },
                committed=True,
            )

    def log(self, metrics: dict):
        # The caller's metrics may precede a durable checkpoint. Only reconstruct
        # checkpoint-backed aggregate events; never send this dictionary directly.
        self.wake.set()
        self._drain_warnings()

    def finish(self):
        self.stopping.set()
        self.wake.set()
        if self.thread is not None:
            self.thread.join(timeout=15)
        self._drain_warnings()

    def checkpoint(self, path: Path, iteration: int):
        self.wake.set()

    def table(self, name: str, columns: list[str], rows: list[list]):
        # Raw examples remain local even if logging.upload_checkpoints is true.
        self.store.emit(
            "sample_table",
            {"name": name, "columns": columns, "rows": rows[: self.config.logging.table_rows]},
            committed=True,
        )


def system_metrics(output_root: Path) -> dict:
    process = psutil.Process()
    disk = psutil.disk_usage(output_root)
    children_rss = 0
    child_pids = []
    for child in process.children(recursive=True):
        try:
            children_rss += child.memory_info().rss
            child_pids.append(child.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    result = {
        "host_rss_bytes": process.memory_info().rss,
        "descendant_rss_bytes": children_rss,
        "descendant_pids": child_pids,
        "host_available_bytes": psutil.virtual_memory().available,
        "disk_free_bytes": disk.free,
        "cpu_percent": process.cpu_percent(),
        "timestamp": time.time(),
    }
    if torch.cuda.is_available():
        result["gpu"] = {
            str(i): {
                "allocated_bytes": torch.cuda.memory_allocated(i),
                "reserved_bytes": torch.cuda.memory_reserved(i),
                "peak_bytes": torch.cuda.max_memory_allocated(i),
                "device_used_bytes_at_sample": torch.cuda.mem_get_info(i)[1]
                - torch.cuda.mem_get_info(i)[0],
            }
            for i in range(torch.cuda.device_count())
        }
    return result


class HealthMonitor:
    def __init__(self):
        self.history: list[dict] = []

    def inspect(self, metrics: dict) -> list[str]:
        self.history.append(metrics)
        self.history = self.history[-10:]
        messages = []
        if len(self.history) >= 5 and all(
            m["proposer"]["valid_fraction"] < 0.25 for m in self.history[-5:]
        ):
            messages.append("valid_fraction_below_0.25_for_5_iterations")
        if len(self.history) == 10 and all(
            m["student"]["mixed_group_fraction"] == 0 for m in self.history
        ):
            messages.append("no_mixed_groups_for_10_iterations")
        for key, threshold in (("gap_clip_fraction", 0.05), ("gate_saturation_fraction", 0.95)):
            value = metrics["distill"].get(key)
            if key == "gate_saturation_fraction" and not metrics["distill"].get(
                "gating_enabled", True
            ):
                continue
            if value is not None and value > threshold:
                messages.append(key)
        mismatch = metrics["student"].get("pre_update_logprob_abs_max")
        if mismatch is not None and mismatch > 0.05:
            messages.append("rollout_scoring_mismatch")
        if metrics["alignment"].get("reference_norm") == 0:
            messages.append("zero_reference_gradient")
        if len(self.history) >= 5 and all(
            (m["alignment"].get("near_zero_fraction") or 0) > 0.8 for m in self.history[-5:]
        ):
            messages.append("alignment_near_zero_fraction_above_0.8_for_5_iterations")
        if (
            len(self.history) >= 5
            and sum(m["student"].get("invalid_action_fraction", 0) for m in self.history[-5:]) / 5
            > 0.2
        ):
            messages.append("invalid_action_fraction_above_0.2_over_5_iterations")
        return messages
