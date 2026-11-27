# Self-play OPSD coding-agent experiment

We investigate whether a coding agent can learn to repair Python programs
through self-play, and how gated on-policy self-distillation (OPSD) interacts
with a proposer that creates repair tasks. A shared-parameter proposer and
student compare verifier-driven Dr. GRPO, ungated or gated OPSD, and
proposer rewards based on task difficulty or student-gradient alignment.
[Methods](docs/methods.md)

## Current results

On the adapted Precise Debugging Benchmark, Qwen3.5-4B completed five
512-episode self-play configurations in each of two seeds. Gated OPSD (B)
verified 28/64 held-out repairs in both seeds, ahead of ungated OPSD
(26/64 and 21/64) and verifier-only Control (22/64 and 21/64).
The C/D gradient-alignment ordering reversed between seeds. Adding gated
OPSD to the candidate gradient reduced paired near-zero counts from 264/512
to 0/512. Raising the per-turn generation cap in a matched evaluation left
aggregate repairs flat and narrowed B’s mean advantage over Control from
10.16 to 0.78 percentage points.

The consolidated study records 322.530 historical H200-hours, including
pilots, failures, resumes, training and evaluation. Two training seeds, a
short optimization horizon, high truncation, and small original transfer
cohorts limit generalization and convergence claims.
[Results and discussion](docs/report/task23-results-and-discussion.md) ·
[Five-minute presentation](docs/report/task23-presentation.md) ·
[Evaluation status](docs/investigations/task23-status.md)

## Getting started

```bash
uv sync --frozen
uv run selfplay doctor --settings configs/local.yaml --isolation
uv run selfplay smoke --output runs/local-smoke
```

See the [operations guide](docs/operations.md) for setup, Yale execution,
checkpoints and evaluation. Private datasets, trajectories and checkpoints
remain in ignored local or Yale paths.
