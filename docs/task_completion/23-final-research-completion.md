# Task 23 · Final research study completed

The audit reproduced all 12 original checkpoint cells and 60 fixed
development measurements. It reconstructed 206 committed training iterations
with comparable student and proposer histories, paired task-family inference,
and a private checkpoint and Slurm registry. The historical Task 21/22 ledger
reconciled 64 allocations and 322.530 H200-hours.

The full matched inference-budget study evaluated 768 checkpoint-task
outcomes. Raising the per-turn cap left aggregate verified repairs at 300,
increased truncation failures from 404 to 414, and narrowed B's original
advantage over Control from +10.16 to +0.78 percentage points. The original
budget remains the primary comparison.

The independent transfer study evaluated the same 17 verified BigCodeBench
source-shift tasks and 32 verified LiveCodeBench two-bug tasks in each of
12 cells. The two shifts gave different observed rankings. The joint
task-family intervals remain conditional on the two checkpoint seeds.
C and D paired candidate gradients changed from 264/512 near zero to 0/512.
This establishes a signal mechanism in the archived diagnostics and leaves
its effect on downstream repair accuracy unresolved.

Task 23 added 50.580 H200-hours across 27 terminal allocations including both
reused timing jobs and the zero-GPU verifier qualification. No account jobs
remained active. A′, further training continuations, and additional seeds were
deferred. The retained private evidence and cohort freeze remain ignored.

- [Results, discussion, and reproduction](../report/task23-results-and-discussion.md)
- [Five-minute presentation outline](../report/task23-presentation.md)
- [Execution and accounting](../investigations/task23-status.md)
- [Full aggregate statistics](../figures/task23/statistics.json)
- [Consolidated W&B run](https://wandb.ai/tatsu432s/self-play-opsd-coding-agent/runs/task23-final-aggregates)
