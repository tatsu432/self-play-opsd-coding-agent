import importlib.util
from pathlib import Path


def test_static_evaluation_follows_episode_count_not_optimizer_steps(monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    spec = importlib.util.spec_from_file_location("task14_static", Path("scripts/task14_static.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    trainer = object.__new__(module.ExposureTrainer)
    trainer.total_episodes = 72
    trainer.student_steps = 0
    trainer.stop_requested = False
    evaluations = []
    trainer.evaluate_development = lambda: evaluations.append(trainer.total_episodes)

    def iteration(self, _):
        self.total_episodes += 8
        return "record"

    monkeypatch.setattr(module.StaticTrainer.__bases__[0], "run_iteration", iteration)
    for step in range(1, 12):
        trainer.run_iteration(step)
    assert evaluations == [80, 160]
    assert trainer.student_steps == 0
    assert trainer.stop_requested
