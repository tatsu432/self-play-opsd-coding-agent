"""Explicit tiny/fixture diagnostics, never a substitute for study results."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Literal

from selfplay_opsd.codec import encode
from selfplay_opsd.core import Case, ParentProgram
from selfplay_opsd.data import TaskRepository, build_task
from selfplay_opsd.environment import canonical_trajectory, run_episode, score_teacher
from selfplay_opsd.mutations import make_patch, validate_mutation
from selfplay_opsd.objectives import alignment
from selfplay_opsd.policy import HFPolicy
from selfplay_opsd.sandbox import SandboxBackend
from selfplay_opsd.storage import atomic_json, file_hash
from selfplay_opsd.trainer import SelfPlayTrainer, reference_examples, student_examples


def fixture_repository(root: Path, sandbox: SandboxBackend, policy: HFPolicy) -> TaskRepository:
    """Artificial programs labelled fixture_only. No benchmark metrics are inferred."""
    (root / "private").mkdir(parents=True, exist_ok=True, mode=0o700)
    parents, task_hashes, anchor_hashes = [], {}, {}
    for index, split in enumerate(
        ["train"] * 4 + ["reference", "dev_monitor", "dev_diagnostic", "final_id", "external"]
    ):
        source = f"def add_{index}(a, b):\n    return a + b + {index}\n"
        parent = ParentProgram(
            parent_id=f"fixture/{index}",
            source=source,
            specification=f"Implement add_{index}(a, b), returning a + b + {index} for integers.",
            entry_point=f"add_{index}",
            dataset="fixture",
            family_id=f"fixture-family-{index}",
            provenance={"fixture_only": True, "license": "project MIT"},
        )
        public = [Case(args=encode([1, 2]), expected=encode(3 + index))]
        hidden = [
            Case(args=encode([-1, 4]), expected=encode(3 + index)),
            Case(args=encode([5, -2]), expected=encode(3 + index)),
        ]
        parents.append(
            {
                "parent": parent.model_dump(mode="json"),
                "split": split,
                "public": [c.model_dump(mode="json") for c in public],
                "hidden": [c.model_dump(mode="json") for c in hidden],
            }
        )
        if split == "train":
            continue
        mutation = validate_mutation(source, make_patch(source, source.replace("a + b", "a - b")))
        task, oracle = build_task(parent, mutation, public, hidden, split, 2026 + index, sandbox)
        path = root / "private" / f"tasks-{split}.json"
        atomic_json(
            path, [{"task": task.model_dump(mode="json"), "oracle": oracle.model_dump(mode="json")}]
        )
        task_hashes[path.name] = file_hash(path)
        if split in {"reference", "dev_diagnostic"}:
            anchor = canonical_trajectory(policy, sandbox, task, oracle, policy.config.episode)
            path = root / "private" / f"anchors-{split}.json"
            atomic_json(path, [anchor])
            anchor_hashes[split] = file_hash(path)
    atomic_json(root / "private/parents.json", parents)
    atomic_json(
        root / "manifest.json",
        {
            "schema_version": 1,
            "fixture_only": True,
            "ready": True,
            "parents_hash": file_hash(root / "private/parents.json"),
            "task_hashes": task_hashes,
            "anchor_hashes": anchor_hashes,
            "image_digest": sandbox.identity(),
        },
    )
    return TaskRepository(root)


def run_smoke(
    output: Path,
    sandbox: SandboxBackend,
    *,
    wandb_mode: Literal["offline", "online", "disabled"] = "disabled",
) -> dict:
    import torch

    from selfplay_opsd.config import load_config

    torch.set_num_threads(1)
    config = load_config(
        study_id="smoke",
        hardware_profile="cpu_debug",
        output_root=output,
        data_root=output / "fixture_data",
        iterations=1,
        proposer_max_tokens=16,
    )
    config.logging.mode = wandb_mode
    policy = HFPolicy(config)
    sandbox.preflight()
    repository = fixture_repository(config.data_root, sandbox, policy)
    report: dict[str, Any] = {
        "fixture_only": True,
        "real_tiny_generation": True,
        "actual_sandbox": sandbox.attestation,
        "iterations": {},
        "alignment_witness": None,
    }
    for method in ("A", "B", "C", "D"):
        cfg = config.model_copy(deep=True)
        cfg.method = method
        model = HFPolicy(cfg)
        trainer = SelfPlayTrainer(cfg, model, repository, sandbox)
        trainer.checkpoint()
        record = trainer.run_iteration(1)
        trainer.checkpoint()
        assert trainer.store is not None and trainer.sink is not None
        trainer.store.materialize()
        trainer.sink.finish()
        report["iterations"][method] = {
            "attempted": record.attempted,
            "valid": record.valid,
            "student_steps": record.student_steps,
            "proposer_steps": record.proposer_steps,
            "note": "Random tiny-model proposals may all be invalid; this is an explicit no-op, not learning evidence",
        }
    config.episode.max_turns = 1
    config.episode.max_turn_generated_tokens = 16
    task, oracle = repository.tasks("dev_diagnostic")[0]
    episodes = [
        score_teacher(
            policy,
            run_episode(policy, sandbox, task, oracle, config.episode, 2026 + i),
            oracle,
            config.episode,
        )[0]
        for i in range(4)
    ]
    config.method = "B"
    c_examples, _ = student_examples(episodes, config, kind="rl")
    d_examples, _ = student_examples(episodes, config)
    reference, _ = policy.gradients(reference_examples(repository.anchors()))
    c, mc = policy.gradients(c_examples)
    d, md = policy.gradients(d_examples)
    report["alignment_witness"] = {
        "c": dataclasses.asdict(alignment(c, reference)),
        "d": dataclasses.asdict(alignment(d, reference)),
        "rl_norm": mc["grad_norm"],
        "combined_norm": md["grad_norm"],
        "episodes": len(episodes),
        "tokens": sum(len(t.target_ids) for e in episodes for t in e.turns),
    }
    atomic_json(output / "smoke_report.json", report)
    policy.close()
    return report
