# Task 22 seed-2027 replication

The matched C/D replication completed 512 attempted episodes per arm and the
same 64-task primary benchmark as the earlier study. Initial adapters, data,
model and evaluation protocol were matched to the corresponding seed-2027
Control/A/B runs. The namespace extension changed execution code identity but
not the objective or frozen scientific settings. C and D use different sampled
training trajectories.

| Arm | Seed 2026 | Seed 2027 |
| --- | ---: | ---: |
| Initial | 21/64 | 20/64 |
| Control | 22/64 | 21/64 |
| A: ungated OPSD | 26/64 | 21/64 |
| B: gated OPSD | 28/64 | 28/64 |
| C: RL gradient alignment | 33/64 | 24/64 |
| D: RL + gated OPSD alignment | 25/64 | 31/64 |

The ordering of C and D reverses across the two runs. In 2026, D minus C was
−8/64 (−12.5 percentage points), with a paired task-family bootstrap 95%
interval of [−25, 0]. In 2027 it was +7/64 (+10.94 points), with 15 tasks
gained and eight lost, interval [−3.13, 26.56]. The 2027 C-minus-B and
D-minus-B differences were −6.25 points [−18.75, 4.69] and +4.69 points
[−7.81, 18.75]. These intervals describe variation across benchmark families
conditional on the checkpoints; they do not measure uncertainty across training
seeds. Two seeds and repeatedly inspected tasks do not establish a winner.
See the [seed-2027 primary figure](../figures/task22/seed2027/02-primary.pdf)
and the [seed-2026 analysis](task22-results.md).

For 2027, C/D verified 1/1 and 1/1 cross-source tasks, 4/9 and 7/9 mutants,
and 2/8 and 2/8 harder tasks. These small sets support no broad transfer claim.
On the same candidate tasks from the C trajectory, the C/D candidate gradients
were near zero on 68/128 and 0/128; on the D trajectory they were 63/128 and
0/128. This is a paired mechanism diagnostic on trajectory-selected candidates,
not evidence that the gradient change caused the accuracy difference.

The replication used 182,963 H200-seconds (50.823056 hours), including
qualification, both timeouts, continuations and all evaluations. Training alone
used 25.176667 hours for C and 20.419167 for D. The original seed-2026 study
used 196,387 seconds; together Task 22 used 379,350 seconds (105.375 hours).
All replication reservations are reconciled, endpoint checkpoints and
evaluation identities are verified, and no new outcome was counted twice.
Worker W&B credentials were unavailable during training. Verified aggregate
results were published to the [replication W&B run](https://wandb.ai/tatsu432s/self-play-opsd-coding-agent/runs/task22-seed2027-4b1ee552aa6d);
its history and all three aggregate tables were independently verified through
the W&B API. The private local receipts and ledger are authoritative.

Reproduce the analysis with the private fetched artifacts:

```bash
PYTHONPATH=src .venv/bin/python scripts/analyze_task22_replication.py \
  --manifest runs/task22/seed2027-analysis-manifest.json \
  --budget runs/task22/seed2027-budget-final.json \
  --output runs/task22/seed2027-final-analysis
```

The December 2 five-minute presentation is a soft milestone for findings so
far. The final report is due December 12, 2026. Further seeds, stronger
independent evaluation and investigation of truncation remain useful future
work; the observed reversal argues against selecting C or D as a general winner.
