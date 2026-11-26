# Task 22 PDB gradient-alignment results

Results are exploratory: C and D each have one training seed (2026), while the
previously completed Control, A and B each have two. Every arm below uses the
same frozen 64-task primary held-out set; the 2027 C/D replication is in
progress and its endpoint evaluations are not yet available. See the
[execution status](task22-status.md) for current progress.

| Arm | Seed 2026 verified / 64 | Seed 2027 verified / 64 |
| --- | ---: | ---: |
| Initial | 21 | 20 |
| Control | 22 | 21 |
| A: ungated OPSD | 26 | 21 |
| B: gated OPSD | 28 | 28 |
| C: RL gradient alignment | 33 | — |
| D: RL + gated OPSD alignment | 25 | — |

The paired D-minus-C difference is −8/64 (−12.5 percentage points), with
four tasks gained and 12 lost relative to C. A 95% paired task-family bootstrap
interval is [−25, 0] percentage points. C-minus-B is +5/64 (+7.8125 points),
interval [−4.6875, 21.875], and D-minus-B is −3/64 (−4.6875 points), interval
[−17.1875, 7.8125]. These intervals describe task variation conditional on the
observed checkpoints; they do not quantify variation across training seeds or
prove that either proposer objective is superior in general.

C's matched transfer evaluation verified 1/8 harder PDB tasks, 1/1 cross-source
task and 5/9 separate mutants; D verified 1/8, 1/1 and 6/9, respectively.
These small sets do not support a broad transfer claim.

The complete Task 22 study used 196,387 H200-seconds (54.5519 H200-hours),
including qualification, timeouts, continuations and all endpoint evaluations,
within the 95-hour incremental cap. Its optional matched Control/A/B extension
was deferred: extrapolating their observed 512-episode training costs requires
approximately 37 hours for another 256 episodes across all three arms, before
endpoint evaluation or contingency, exceeding the roughly 40.4 hours remaining
in the budget. Checkpoints and original records remain available for a later
authorized extension.

The final private analysis is `runs/task22/final-analysis/summary.json`;
the [aggregate W&B run](https://wandb.ai/tatsu432s/self-play-opsd-coding-agent/runs/task22-8b6b398847d1)
was independently checked for its history and three aggregate tables. The local
verified receipts and reconciled ledger remain authoritative. Reproduce the
analysis from the private manifest and receipts:

```bash
PYTHONPATH=src .venv/bin/python scripts/analyze_task22.py \
  --qualification runs/task22/qualification-verified.json \
  --manifest runs/task22/analysis-manifest.json \
  --budget runs/task22/budget-final.json --output runs/task22/final-analysis
```

See the [primary comparison](../figures/task22/baseline/02-primary.pdf),
[learning and efficiency curves](../figures/task22/baseline/03-learning-efficiency.pdf),
and [failure categories](../figures/task22/baseline/06-failure-modes.pdf).

## Mechanism and failure modes

The matched gradient qualification compared candidate rewards on the same
snapshot and tasks. C had five near-zero gradients among 14 candidates, while
D had zero. This supports the proposed signal mechanism but is not an accuracy
result. The full C training reported 97 near-zero alignment events, and D
reported none; their trajectories differ, so that comparison is not a
controlled same-task intervention. Within the paired diagnostic records from
the C trajectory, C's candidate gradient was near zero on 63/128 candidates,
versus 0/128 for D's candidate gradient on those same candidates. On the D
trajectory the respective counts were 70/128 and 0/128. These are diagnostics
on trajectory-selected tasks rather than independent performance tests. See
[mechanistic analysis](task22-mechanistic-analysis.md)
and [qualification](task22-alignment-qualification.md).

Primary failures are substantial: 28/64 C episodes and 33/64 D episodes were
classified as truncated, using the analysis' mutually exclusive categories.
Those counts do not establish that truncation caused the difference. The
shared student/proposer adapter, limited optimization steps and single-seed
alignment arms constrain the conclusions; learning curves are not proof of
convergence.
