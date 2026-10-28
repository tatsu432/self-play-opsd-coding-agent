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
    def __init__(self, config: RunConfig, store: EventStore):
        self.config, self.store = config, store
        self.queue: queue.Queue = queue.Queue(maxsize=128)
        self.errors: queue.Queue = queue.Queue(maxsize=32)
        self.thread: threading.Thread | None = None
        self.dropped = 0
        if config.logging.mode != "disabled":
            self.thread = threading.Thread(target=self._worker, name="wandb-delivery", daemon=True)
            self.thread.start()

    def _worker(self):
        run = None
        try:
            import wandb

            run = wandb.init(
                project=self.config.logging.project,
                entity=self.config.logging.entity,
                group=self.config.study_id,
                name=f"{self.config.study_id}-{self.config.method}-seed{self.config.seed}-{self.store.segment[:8]}",
                id=self.store.identity["logical_run_id"]
                if self.config.logging.mode == "online"
                else self.store.segment,
                resume="allow" if self.config.logging.mode == "online" else None,
                mode=self.config.logging.mode,
                dir=str(self.config.run_dir),
                config=self.config.resolved(),
                tags=[
                    self.config.method,
                    self.config.hardware.profile,
                    self.config.model.revision,
                    f"logical:{self.store.identity['logical_run_id']}",
                ],
                settings=wandb.Settings(init_timeout=15),
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
            ):
                run.define_metric(group, step_metric="train/outer_iteration")
            run.define_metric("eval_by_iteration/*", step_metric="train/outer_iteration")
            run.define_metric("eval_by_gpu_hours/*", step_metric="compute/allocated_gpu_hours")
            run.log(
                {
                    "run/segment_id": self.store.segment,
                    "run/rollback_to": self.store.rollback_to,
                    "run/logical_run_id": self.store.identity["logical_run_id"],
                }
            )
            while True:
                item = self.queue.get()
                try:
                    if item is None:
                        break
                    if "__artifact__" in item:
                        entry = item["__artifact__"]
                        artifact = wandb.Artifact(
                            entry["name"], type="adapter-checkpoint", metadata=entry["metadata"]
                        )
                        artifact.add_file(entry["path"])
                        run.log_artifact(artifact)
                        continue
                    if "__table__" in item:
                        table = item.pop("__table__")
                        item[table["name"]] = wandb.Table(
                            columns=table["columns"], data=table["rows"]
                        )
                    run.log(item)
                finally:
                    self.queue.task_done()
        except Exception as exc:
            try:
                self.errors.put_nowait(f"{type(exc).__name__}: {exc}")
            except queue.Full:
                self.dropped += 1
        finally:
            if run is not None:
                try:
                    run.finish()
                except Exception as exc:
                    self.dropped += 1
                    try:
                        self.errors.put_nowait(str(exc))
                    except queue.Full:
                        self.dropped += 1

    def log(self, metrics: dict):
        if self.thread is not None and self.thread.is_alive():
            try:
                self.queue.put_nowait(
                    flatten(metrics)
                    | {
                        "run/segment_id": self.store.segment,
                        "run/local_event_index": self.store.index,
                        "run/committed": True,
                    }
                )
            except queue.Full:
                self.dropped += 1
        while not self.errors.empty():
            self.store.emit(
                "logging_warning",
                {"error": self.errors.get_nowait(), "local_data_preserved": True},
                committed=True,
            )

    def finish(self):
        if self.thread is not None and self.thread.is_alive():
            try:
                self.queue.put(None, timeout=1)
            except queue.Full:
                self.dropped += 1
            self.thread.join(timeout=15)
        while not self.errors.empty():
            self.store.emit(
                "logging_warning",
                {"error": self.errors.get_nowait(), "local_data_preserved": True},
                committed=True,
            )
        if self.dropped:
            self.store.emit(
                "logging_warning",
                {"dropped_remote_events": self.dropped, "local_data_preserved": True},
                committed=True,
            )

    def checkpoint(self, path: Path, iteration: int):
        if not self.config.logging.upload_checkpoints or self.thread is None:
            return
        try:
            self.queue.put_nowait(
                {
                    "__artifact__": {
                        "name": f"adapters-{self.store.identity['logical_run_id']}",
                        "path": str(path / "inference_adapters.pt"),
                        "metadata": {
                            "iteration": iteration,
                            "model_revision": self.config.model.revision,
                        },
                    }
                }
            )
        except queue.Full:
            self.dropped += 1

    def table(self, name: str, columns: list[str], rows: list[list]):
        bounded = rows[: self.config.logging.table_rows]
        self.store.emit(
            "sample_table", {"name": name, "columns": columns, "rows": bounded}, committed=True
        )
        if self.thread is not None and self.thread.is_alive():
            try:
                self.queue.put_nowait(
                    {
                        "__table__": {"name": name, "columns": columns, "rows": bounded},
                        "train/outer_iteration": self.store.iteration,
                    }
                )
            except queue.Full:
                self.dropped += 1


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
