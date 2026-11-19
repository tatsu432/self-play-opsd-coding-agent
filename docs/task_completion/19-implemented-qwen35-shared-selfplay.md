# Task 19 completion: Qwen3.5 shared self-play

The bounded one-seed five-arm study completed the common 96-episode endpoint,
endpoint monitoring, 115-task final ID and 116-task previously inspected
external evaluations. Full outcomes, clustered intervals, the research
decision, failure breakdown and figures are in [Task 19 results](../investigations/task19-results.md).
The reproducible execution history, numerical qualifications, private paths and
failed attempts are in [Task 19 status](../investigations/task19-status.md).

The implementation pins Qwen3.5-4B and its tokenizer revision, uses a shared
FP32 LoRA and optimizer for proposer and student, Dr. GRPO, native tool calling
and thinking, strict isolated verification, ungated/gated OPSD, and RL-only or
RL+OPSD proposer gradient alignment. It preserves rollout behavior denominators,
transactional checkpoints, data identities, and a common frozen experiment
control. H200 cached FP32 passed the bounded numerical replay and calibration;
its measured replay, gradient and update differences are recorded in the results.

| Arm | Pilot controller | Main controller | Endpoint checkpoint |
|---|---|---|---|
| Control | `826f6ae808dc186b477bc945` | `562da119c0ee91f60ff3c932` | `000007` |
| A | `2ae1a78b54cd011fc21623ef` | `ad1f59aa48704da5e8887dcd` | `000007` |
| B | `9016ab34ca892ee6f6ae633d` | `51f3b27715d01a0b843ec36f` | `000007` |
| C | `f5c5a6c6f900e5c11b9a9f34` | `ae25de421c2c804155ac8bec` | `000007` |
| D | `fd4b69da407f990319580e8d` | `4ce3511806b176d79c16a3a3` | `000007` |

All five endpoint checkpoint files passed local checksum verification and
matched their completion markers and 96-episode exposure. A's final continuation
`33902234` completed exit 0 in 1,117 H200-seconds; do not duplicate. All
allocations, including unsuccessful qualification and baseline attempts, cost
233,565 H200-seconds (64.879167 hours). The shared cap was 80 hours and no
account jobs were active at the final check. Fourteen aggregate W&B history
rows per arm were independently verified after guarded replay from offline
files, with original timing and segment steps retained.

The analysis handles the pilot-skipped 24-episode nominal milestone explicitly:
the observed curves use 0/32/48/72/96 and still require all reachable summaries.
The analysis manifest, final ledger, artifacts and reproducible generated report
remain under ignored `runs/task19/`; the figures and aggregate conclusions are
published in the repository. Parent-family intervals are conditional on one
training run; the previously inspected external families are exploratory. The
observed differences do not establish an objective or proposer-reward winner.

Final local checks passed: 476 pytest tests, with 38 skipped; Ruff lint and
format; and mypy for 61 source files. The completed five-arm analysis ran
against the checksum-verified endpoint checkpoints and reconciled ledger.
