"""Task 18 identity and bounded experiment configuration; historical defaults stay intact."""

from __future__ import annotations

import json
from pathlib import Path

from selfplay_opsd.config import RunConfig
from selfplay_opsd.storage import file_hash
from selfplay_opsd.task18_protocol import Task18EpisodeConfig

STUDY = "repair-selfplay-2026-task18-v1"
OPERATIONS = ("numerical", "normalized", "native", "student")


class Task18Config(RunConfig):
    task18_control_sha256: str

    def scientific_settings(self, **kwargs):
        values = super().scientific_settings(**kwargs)
        values["episode"] = self.episode.model_dump(mode="json")
        return values

    def resolved(self):
        return super().resolved() | {"episode": self.episode.model_dump(mode="json")}


def configure_operation(cfg, operation: str, arm: str, control_path: Path):
    if operation not in OPERATIONS or arm not in {"A", "B", "D"}:
        raise ValueError(
            "Task 18 supports A=nominal ungated, B=gated, D=GRPO; interface A=base/B=trained"
        )
    control = json.loads(control_path.read_text())
    if control["study_id"] != STUDY or control["max_gpu_hours"] != 12:
        raise ValueError("Task 18 requires its separate 12-hour authorization")
    if control["pool_sha256"] != "602eaf506f8ac2cf17924596dae7c0ff3891ef5034a546d4d3751b9803412c83":
        raise ValueError("Task 18 must reuse the immutable verified Task 17 pool")
    cfg = Task18Config.model_validate(
        cfg.model_dump() | {"task18_control_sha256": file_hash(control_path)}
    )
    if cfg.seed not in {2026, 2027}:
        raise ValueError("Task 18 is bounded to two paired seeds")
    cfg.study_id = STUDY
    cfg.method = "B"
    cfg.max_gpu_hours = 12
    cfg.sd_gap_definition = "same_scorer_v2"
    cfg.episode = Task18EpisodeConfig(**cfg.episode.model_dump())
    cfg.episode.edit_protocol = "replace_text"
    if operation in {"normalized", "native"} and arm == "D":
        raise ValueError("Frozen interface snapshots are A=base and B=trained B2026")
    if operation == "normalized":
        cfg.episode.edit_protocol = "normalized_diff"
    if operation == "native":
        cfg.episode.tool_serialization = "qwen_native"
    if operation == "student":
        if not control.get("training_ready"):
            raise ValueError("Freeze a numerical/interface decision before new training")
        for item in control.get("evidence", []):
            if file_hash(Path(item["path"])) != item["sha256"]:
                raise ValueError("Training decision evidence checksum differs")
        if not control.get("evidence") or not control.get("substantive_change"):
            raise ValueError("Reuse Task 17 unless new evidence justifies a changed study")
        endpoint = control["student_episodes"]
        if type(endpoint) is not int or endpoint != 80:
            raise ValueError(
                "Freeze an 80-episode target; use the supported iteration-4 stop for pilots"
            )
        cfg.episode.edit_protocol = control["selected_interface"]
        cfg.episode.tool_serialization = control["selected_serialization"]
        if (
            cfg.episode.tool_serialization == "qwen_native"
            and cfg.episode.edit_protocol != "replace_text"
        ):
            raise ValueError("Native serialization currently supports replace_text only")
        cfg.method = {"A": "A", "B": "B", "D": "GRPO"}[arm]
        cfg.study_id = f"{STUDY}-objectives-seed{cfg.seed}"
        cfg.iterations = 10
        cfg.parents_per_iteration = cfg.proposals_per_parent = 2
        cfg.evaluate_every = 11
        cfg.gradient_diagnostics = True
        cfg.objective.sd_lambda = 0.1
    return cfg
