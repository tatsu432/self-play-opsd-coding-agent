# Task 20 root-cause audit

The six- or seven-step Task 19 comparison showed little effective solver RL
exposure on the specialized repair benchmark. The archive establishes the
mechanism for zero-centered advantages on unanimous rewards. It leaves open
whether a longer run on this distribution, a harder curriculum, or a different
shared gradient scale would have improved verified accuracy.

| Arm | Groups at 0/1/2/3/4 successes | Mixed groups | Tokens in mixed groups | Nonzero solver RL iterations | Joint optimizer steps | Steps with zero solver RL |
|---|---|---:|---:|---:|---:|---:|
| Control | 4/2/1/3/14 | 6/24 | 56,017/146,840 | 3 | 6 | 3 |
| A | 6/2/1/2/13 | 5/24 | 55,126/153,764 | 2 | 7 | 5 |
| B | 3/3/2/2/14 | 7/24 | 84,418/158,598 | 4 | 7 | 3 |
| C | 4/3/0/3/14 | 6/24 | 51,886/144,730 | 3 | 7 | 4 |
| D | 4/3/3/1/13 | 7/24 | 72,636/147,774 | 4 | 7 | 3 |

Mixed-group tokens are the sampled tokens eligible for centered RL terms.
The archive records actual nonzero gradients per batch. It does not identify
per-token gradient support or cancellation. The earlier 1/29 mixed-group count
came from a different qualification run.

1. **Low effective exposure and an easy benchmark.** All arms had at least
   13/24 all-success groups. The base solved 88/115 primary ID tasks and 93/116
   previously inspected external tasks. The attempted-episode endpoint does
   not establish a learning plateau or convergence.
2. **Limited proposer discrimination.** Valid-only rewards differed in just
   1/18, 1/18, 2/18, 3/18, and 8/18 groups across Control/A/B/C/D. C recorded
   29 near-zero alignment candidates. In Control the proposer contributed
   the only current gradient on three updates. This makes the shared control
   unsuitable as a solver-only positive control.
3. **Scaling is plausible and unproven.** Mean response lengths ranged from
   1,508 to 1,652 tokens, 12.3–13.4% of the fixed 12,288-token denominator.
   That establishes fixed-denominator attenuation relative to per-token
   normalization. AdamW, clipping, and the relative scales of auxiliary
   terms determine the actual step. The recorded mean RL norms were
   0.00545–0.01043 including zeros. Mean weighted proposer norms were
   0.00249–0.00279. A's mean weighted OPSD norm was 0.00558 versus 0.00134
   for B. These measurements do not isolate normalization as a cause.
4. **Shared gradient interference is unresolved.** Recorded cosines between
   the combined student gradient and proposer ranged from −0.0234 to +0.0247
   across arms. The archive does not give every unweighted component cosine.
   The causal test is a restored-state student-only versus joint step on an
   independent training probe.
5. **A large scoring mismatch has weaker support.** Task 19's cached FP32
   qualification passed all 64 recorded checks. The main batches logged zero
   clipping and zero support exits. Batch means cannot exclude token-level
   mismatches or evaluation and interface limitations.

Reproduce the aggregate and per-iteration audit from the private archive.
The [training-signal figure](../figures/task20/task19-training-signal.png)
summarizes both the reward histograms and the source of optimizer steps.

```bash
uv run --frozen python scripts/audit_task20.py \
  --manifest runs/task19/analysis-manifest.json \
  --control runs/task19/main-control.json \
  --output runs/task20/task19-audit.json \
  --figure docs/figures/task20/task19-training-signal.png
```

Each batch is checked against its archived SHA-256. The output includes
reward histograms, token counts, proposer families and rewards, failure
categories, stored component norms, clipping, and parameter displacement.
For qualification and evaluation limits see [Task 19 results](task19-results.md).
