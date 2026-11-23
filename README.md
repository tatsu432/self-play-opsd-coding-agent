# Self-play OPSD coding-agent experiment

We study whether coding agents improve at repairing Python programs through
self-play, and whether improvement transfers to other problem families. A
proposer introduces verified bugs and a student repairs them using tools and
tests. We compare verifier-driven reinforcement learning with ungated or gated
on-policy self-distillation, and difficulty-based with gradient-alignment
proposer rewards. See [methods](docs/methods.md).

## Progress and results

Task 21 adds an isolated adapter for the Precise Debugging Benchmark (PDB)
while retaining Qwen-native exact-text repair. The first CPU pilot verified
11 tasks from BigCodeBench and LiveCodeBench, with 22/22 original and
canonical verdicts matching the pinned scorers. Qwen3.5-4B repaired 5/11 in
the training-only feasibility pilot. The completed difficulty screen found
mixed rewards on 12/24 four-rollout groups and 17/24 eight-rollout groups.
Four rollouts gave more mixed groups per H200-hour. The solver-only pilot
completed eight effective RL updates; paired training samples improved from
24/48 to 30/48 and development from 7/24 to 8/24. These small, single-run
results do not establish generalization.
In the completed shared pilot, seven of 16 solver groups were mixed, but valid
proposals had no within-parent reward differences; development changed from
7/24 to 6/24. A revised follow-up produced four solver updates and valid-only
reward differences on five parents; development changed from 7/24 to 8/24.
Matched 512-episode Control, A and B runs are underway for seed 2026; the
seed 2027 control is running and A/B are queued. Objective results are not
yet available.
[Task 21 status and reproduction](docs/investigations/task21-status.md)

Task 20 audited the Task 19 training batches and found just two to four
iterations with nonzero solver RL gradients per arm. It prepared 205
oracle-verified harder EvalPlus tasks and added a solver-only Dr. GRPO
control. Its base screen found mixed rewards in 5/32 tasks. The solver
positive control completed eight updates and improved its small training set;
its separate probe was flat. A 64-episode shared pilot produced one effective
solver update and no within-group reward diversity among valid proposals, so
the matched method comparison is gated pending a viable intervention. Its
development score changed from 25/41 to 26/41. See the
[positive-control results](docs/investigations/task20-positive-control.md),
[shared-pilot diagnosis](docs/investigations/task20-shared-diagnosis.md),
[research decision](docs/investigations/task20-final-decision.md),
[Task 20 status](docs/investigations/task20-status.md),
[root-cause audit](docs/investigations/task20-root-cause.md), and
[benchmark decision](docs/investigations/task20-benchmark-decision.md). Its
[completion record](docs/task_completion/20-implemented-diff-benchmark.md)
describes the deferred comparisons and next steps.

The completed Task 19 study uses Qwen3.5-4B, a shared proposer/student LoRA,
Dr. GRPO, and native thinking/tool calls. All five seed-2026 arms reached 96
attempted training episodes. The base solved 88/115 ID and 93/116 external
repairs; control/A/B/C/D solved 87/88/90/88/85 ID and 92/92/92/88/92 external.
The previously inspected external families are exploratory. Family-clustered
intervals conditional on this single run do not establish a gating or proposer
reward winner. Total allocation was **64.879 H200-hours**, including failed
and resumed jobs. See [results, figures and decision](docs/investigations/task19-results.md),
[execution and reproduction](docs/investigations/task19-status.md), and the
[completion record](docs/task_completion/19-implemented-qwen35-shared-selfplay.md).

Earlier Qwen3 studies used a different model and interface. Task 15's three-seed
A/B runs improved ID repair but left external results near zero; Task 17's
matched student-objective study did not establish a robust gating advantage.
Task 18 selected native exact-text calls on inspected tasks and identified
optimizer sensitivity that motivated Task 19's separate FP32 numerical
qualification. See [historical A/B](docs/investigations/task15-final-results.md),
[objective](docs/investigations/task17-objective-results.md), and
[interface](docs/investigations/task18-results.md) evidence.

## Getting started

```bash
uv sync --frozen
uv run selfplay doctor --settings configs/local.yaml --isolation
uv run selfplay smoke --output runs/local-smoke
```

See the [operations guide](docs/operations.md) for setup, Yale execution,
checkpoints and evaluation. Private datasets, trajectories and checkpoints
remain under ignored local or Yale paths.
