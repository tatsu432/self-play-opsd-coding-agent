"""Matched-exposure fixed-task trainer for Task 14 experiments."""

from __future__ import annotations

from task13_static import StaticTrainer


class ExposureTrainer(StaticTrainer):
    """Evaluate at fixed episode counts, regardless of skipped optimizer steps."""

    def run_iteration(self, k):
        # Avoid Task 13's step-count stop and evaluation rules.
        record = super(StaticTrainer, self).run_iteration(k)
        if self.total_episodes in {80, 160}:
            self.evaluate_development()
        if self.total_episodes >= 160:
            self.stop_requested = True
        return record
