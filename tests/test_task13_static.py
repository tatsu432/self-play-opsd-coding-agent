"""Fixed-task batches must not produce a proposer policy update."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from selfplay_opsd.config import load_config
from selfplay_opsd.trainer import proposer_examples

spec = importlib.util.spec_from_file_location("task13_static", Path("scripts/task13_static.py"))
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_fixed_batches_cycle_and_have_no_proposer_tokens():
    trainer = object.__new__(module.StaticTrainer)
    trainer.policy = SimpleNamespace(version=lambda role: "initial")
    tasks = [SimpleNamespace(task_id=f"task{i}", parent_id=f"parent{i}") for i in range(4)]
    trainer.pool = [(task, object(), f"key{i}") for i, task in enumerate(tasks)]
    first, first_tasks = trainer.proposals(1)
    second, second_tasks = trainer.proposals(2)
    assert list(first_tasks) == ["task0", "task1"]
    assert list(second_tasks) == ["task2", "task3"]
    assert [p.valid for p in first] == [True, False, True, False]
    cfg = load_config(Path("configs/task12-corrected.yaml"), method="GRPO")
    cfg.parents_per_iteration = 2
    cfg.proposals_per_parent = 2
    examples = proposer_examples(first, [0, -1, 1, -1], cfg)
    assert len(examples) == 4
    assert not any(example.targets for example in examples)
