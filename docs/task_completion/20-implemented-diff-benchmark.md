# Task 20: learning-signal rescue and shared-policy gate

Completed the Task 20 investigation through its prespecified shared-policy
no-go decision on November 20, 2026. The matched Control/A/B comparison and
C/D alignment experiments were not run: the shared pilot failed the viability
gate. This is a research limitation, not evidence that OPSD or shared
self-play cannot work.

The work includes an audit of Task 19 reward signals, an isolated EvalPlus
fallback with verified tasks and separated partitions, a solver-only Dr. GRPO
positive control, gradient calibration, and a shared-policy pilot. The solver
control made eight nonzero updates and improved both small training checks;
the separate training probe was flat and the diagnostic has a documented
support-exit limitation. The shared pilot made one effective solver update in
64 episodes; none of five proposer groups ranked valid candidates by reward.
Its development score changed 25/41 to 26/41 (two gains, one loss). The
[decision](../investigations/task20-final-decision.md),
[solver evidence](../investigations/task20-positive-control.md), and
[shared diagnosis](../investigations/task20-shared-diagnosis.md) describe
methods, figures and limitations.

Actual allocation was 63,218 H200-seconds (17.560556 hours). All four Task 20
jobs completed and must not be duplicated. The endpoint checkpoint, iteration
batches, and the shared run's two W&B aggregate rows were verified. Local
validation passed 483 tests, with 38 skipped, and Ruff and mypy passed. The
[status record](../investigations/task20-status.md) identifies the jobs and
accounting. Private datasets, trajectories and checkpoints remain ignored.
