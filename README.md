# Self-play OPSD coding-agent experiment

We investigate whether a coding agent can learn to repair Python programs
through self-play, and how gated on-policy self-distillation (OPSD) interacts
with a proposer that creates repair tasks. A shared-parameter proposer and
student compare verifier-driven Dr. GRPO, ungated or gated OPSD, and
proposer rewards based on task difficulty or student-gradient alignment.
[Methods](docs/methods.md)

## Current results

The matched seed-2027 C/D replication is in progress; see
[execution status](docs/investigations/task22-status.md).

On the Precise Debugging Benchmark (PDB), Qwen3.5-4B completed matched
512-episode self-play runs. On 64 held-out single-bug tasks, the initial model,
Control, ungated A and gated B verified 21/22/26/28 repairs in seed 2026;
the gradient-alignment arms C and D verified 33/25. Control/A/B also have a
second training seed (21/21/28); C/D are single-seed exploratory results.
D minus C was −12.5 percentage points, with a task-family bootstrap interval
of [−25, 0] conditional on those runs. Paired gradient diagnostics show that
D supplies a non-negligible candidate gradient more often than C, but they do
not establish an accuracy benefit. The limited seeds, truncation, small transfer
sets and 19–21 optimizer steps prevent broad generalization or convergence
claims. Further matched seeds and longer training are needed.

Task 22 used 54.552 incremental H200-hours, including failed allocations,
continuations and endpoint evaluations. The earlier Task 21 study used
217.155 H200-hours. See [Task 22 results](docs/investigations/task22-results.md),
[mechanistic analysis](docs/investigations/task22-mechanistic-analysis.md),
[execution and reproduction](docs/investigations/task22-status.md), and
[Task 21 results](docs/investigations/task21-results.md).
Earlier [Task 20](docs/investigations/task20-final-decision.md),
[Task 19](docs/investigations/task19-results.md), and
[interface](docs/investigations/task18-results.md) investigations document
benchmark and numerical choices.

## Getting started

```bash
uv sync --frozen
uv run selfplay doctor --settings configs/local.yaml --isolation
uv run selfplay smoke --output runs/local-smoke
```

See the [operations guide](docs/operations.md) for setup, Yale execution,
checkpoints and evaluation. Private datasets, trajectories and checkpoints
remain in ignored local or Yale paths.
