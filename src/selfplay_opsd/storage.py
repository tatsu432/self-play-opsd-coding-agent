"""Local-first scientific events, bounded shards and atomic verified checkpoints."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import random
import shutil
import subprocess
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from selfplay_opsd.core import ExperimentError, canonical, digest, read_json


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temp.open("x") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def code_hash() -> str:
    """Include untracked implementation files, unlike git diff alone."""
    package = Path(__file__).resolve().parent
    content = {f"selfplay_opsd/{p.name}": file_hash(p) for p in sorted(package.glob("*.py"))}
    project = package.parents[1]
    content.update(
        {
            p.name: file_hash(p)
            for p in (project / "pyproject.toml", project / "uv.lock")
            if p.exists()
        }
    )
    return digest(content)


def scientific_code_hash() -> str:
    package = Path(__file__).resolve().parent
    names = (
        "core.py",
        "config.py",
        "codec.py",
        "objectives.py",
        "protocol.py",
        "environment.py",
        "policy.py",
        "verl_backend.py",
        "data.py",
        "evaluation.py",
        "mutations.py",
        "trainer.py",
        "rollout.py",
        "sandbox.py",
        "worker.py",
        "remote_worker.py",
        "worker_runtime.py",
        "worker_policy.py",
        "vllm_worker.py",
        "pilot.py",
    )
    return digest({name: file_hash(package / name) for name in names})


def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ExperimentError(
                "CUDA RNG world size changed; resume needs explicit resource migration"
            )
        torch.cuda.set_rng_state_all(state["cuda"])


def provenance() -> dict:
    import importlib.metadata
    import platform

    def git(*args):
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=False
        ).stdout.strip()

    versions: dict[str, str | None] = {}
    for package in ("torch", "transformers", "peft", "verl", "vllm", "evalplus", "wandb", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "code_hash": code_hash(),
        "git_commit": git("rev-parse", "HEAD"),
        "dirty_diff_hash": digest(git("diff", "--", ".", ":(exclude)docs")),
        "untracked": git("ls-files", "--others", "--exclude-standard").splitlines(),
        "lock_hash": file_hash(Path("uv.lock")) if Path("uv.lock").exists() else None,
        "versions": versions,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda": torch.version.cuda,
        "image_digest": os.getenv("SELFPLAY_IMAGE_DIGEST"),
        "allocation": {
            k: os.environ[k]
            for k in (
                "CUDA_VISIBLE_DEVICES",
                "SLURM_JOB_ID",
                "SLURM_CPUS_PER_TASK",
                "SLURM_JOB_GPUS",
            )
            if k in os.environ
        },
    }


class EventStore:
    def __init__(self, root: Path, identity: dict, rollback_to: int | None = None):
        self.root, self.identity = root, identity
        self.segment = uuid.uuid4().hex
        self.rollback_to = rollback_to
        self.index = 0
        self.iteration = 0
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.chmod(0o700)
        for directory in (
            "events",
            "state",
            "trajectories",
            "tables",
            "summaries",
            "plots",
            "manifests",
        ):
            (root / directory).mkdir(exist_ok=True)
        self.event_path = root / "events" / f"{self.segment}.jsonl"
        self.emit("segment_start", {"rollback_to": rollback_to}, committed=True)

    def emit(
        self, kind: str, payload: Any, *, committed: bool = False, phase: str = "train"
    ) -> dict:
        self.index += 1
        record = {
            "schema_version": 1,
            **self.identity,
            "segment_id": self.segment,
            "event_index": self.index,
            "completed_iteration": self.iteration,
            "wall_timestamp": time.time(),
            "phase": phase,
            "kind": kind,
            "committed": committed,
            "payload": payload,
        }
        # One bounded JSONL event; large trajectories go in separate shards.
        line = canonical(record)
        if len(line.encode()) > 16 * 1024 * 1024:
            raise ValueError("Event exceeds 16 MiB; write a trajectory shard")
        with self.event_path.open("a") as stream:
            stream.write(line + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return record

    def shard(self, iteration: int, name: str, records: list[dict]) -> str:
        path = self.root / "trajectories" / self.segment / f"{iteration:06d}-{name}.json"
        atomic_json(path, records)
        return str(path.relative_to(self.root))

    def health(self, phase: str, **details):
        atomic_json(
            self.root / "state/health.json",
            {
                "phase": phase,
                "iteration": self.iteration,
                "segment_id": self.segment,
                "timestamp": time.time(),
                **details,
            },
        )

    @contextlib.contextmanager
    def phase(self, name: str) -> Iterator[dict]:
        self.health(name)
        start = time.monotonic()
        metrics: dict = {}
        try:
            yield metrics
        finally:
            self.emit(
                "phase_time",
                {"seconds": time.monotonic() - start, **metrics},
                committed=True,
                phase=name,
            )

    def stage_path(self, iteration: int, snapshot: str) -> Path:
        return self.root / "state/pending" / f"{iteration:06d}-{snapshot}.json"

    def materialize(self) -> None:
        records = active_events(self.root)
        kinds = {r["kind"] for r in records}
        for kind in kinds:
            rows = [r for r in records if r["kind"] == kind]
            # Payload JSON retains sparse/nested schemas without Arrow inferring
            # a conflicting struct schema from different status branches.
            flattened = [
                {
                    **{k: v for k, v in r.items() if k != "payload"},
                    "payload_json": canonical(r["payload"]),
                }
                for r in rows
            ]
            pq.write_table(
                pa.Table.from_pylist(flattened), self.root / "tables" / f"{kind}.parquet"
            )


def all_events(root: Path) -> list[dict]:
    records = []
    for path in (root / "events").glob("*.jsonl"):
        with path.open() as stream:
            for line in stream:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    # Only a torn final append may be ignored; do not hide corruption.
                    if stream.read():
                        raise ExperimentError(f"Corrupt nonterminal event in {path}") from None
    return sorted(records, key=lambda r: (r["wall_timestamp"], r["segment_id"], r["event_index"]))


def active_events(root: Path) -> list[dict]:
    records = all_events(root)
    active: list[dict] = []
    for r in records:
        if r["kind"] == "segment_start" and r["payload"].get("rollback_to") is not None:
            watermark = r["payload"]["rollback_to"]
            active = [
                old
                for old in active
                if old["completed_iteration"] <= watermark
                or old["kind"] in {"phase_time", "resource", "segment_start", "error"}
            ]
        active.append(r)
    commits = {
        (r["segment_id"], r["completed_iteration"])
        for r in active
        if r["kind"] == "iteration_commit"
    }
    return [
        r
        for r in active
        if r["committed"] or (r["segment_id"], r["completed_iteration"]) in commits
    ]


class CheckpointManager:
    def __init__(self, root: Path):
        self.root = root / "checkpoints"
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, iteration: int, state: dict, metadata: dict) -> Path:
        target = self.root / f"{iteration:06d}"
        temp = self.root / f".pending-{uuid.uuid4().hex}"
        temp.mkdir()
        try:
            torch.save(state, temp / "state.pt")
            if "policy" in state and "adapters" in state["policy"]:
                torch.save(state["policy"]["adapters"], temp / "inference_adapters.pt")
            with (temp / "state.pt").open("rb") as stream:
                os.fsync(stream.fileno())
            atomic_json(temp / "metadata.json", metadata)
            hashes = {p.name: file_hash(p) for p in temp.iterdir() if p.is_file()}
            atomic_json(temp / "complete.json", {"files": hashes, "iteration": iteration})
            self.verify(temp)
            if target.exists():
                # Preserve previous same-iteration lineage for audit.
                os.replace(target, self.root / f".superseded-{iteration:06d}-{uuid.uuid4().hex}")
            os.replace(temp, target)
            atomic_json(self.root / "latest.json", {"path": target.name, "iteration": iteration})
            return target
        except BaseException:
            shutil.rmtree(temp, ignore_errors=True)
            raise

    @staticmethod
    def verify(path: Path) -> dict:
        marker = read_json(path / "complete.json")
        for name, expected in marker["files"].items():
            if Path(name).name != name or file_hash(path / name) != expected:
                raise ExperimentError(f"Checkpoint checksum mismatch: {path}/{name}")
        return read_json(path / "metadata.json")

    @classmethod
    def load(cls, path: Path) -> tuple[dict, dict]:
        metadata = cls.verify(path)
        # Only trusted, checksummed project checkpoints. Never load model-produced
        # files with torch.load or accept checkpoint locations from an agent.
        return torch.load(path / "state.pt", map_location="cpu", weights_only=False), metadata

    def prune(self, protected: set[int]) -> None:
        checkpoints = sorted(p for p in self.root.iterdir() if p.name.isdigit())
        keep = {p.name for p in checkpoints[-2:]} | {f"{i:06d}" for i in protected}
        for checkpoint in checkpoints:
            if checkpoint.name not in keep:
                shutil.rmtree(checkpoint)
        for old in self.root.glob(".superseded-*"):
            # A verified replacement exists before superseded state is retired.
            with (self.root / "retired.jsonl").open("a") as stream:
                stream.write(
                    canonical(
                        {
                            "path": old.name,
                            "metadata": read_json(old / "metadata.json"),
                            "checksums": read_json(old / "complete.json"),
                        }
                    )
                    + "\n"
                )
            shutil.rmtree(old)
