# Task 19: Qwen3.5 shared-adapter self-play results

All five arms reached the preselected common 96-attempted-episode endpoint on
seed 2026, and completed the same endpoint evaluations. The common model is
`Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, with
one FP32 shared LoRA and native thinking/tool calls. The completed training and
evaluation jobs used scientific source commit
`e2fbe083a3c0aff7a20c8d8a7a2d6c5547910190` for pilots and
`7150a8b31cc94c876998c6f5362186596c59af29` for main segments; their scientific
source files and execution identities match. The last A continuation used the
same persisted immutable main specification. See [status and reproduction](task19-status.md)
for qualification, data, controls, artifact paths and failed allocations.

## Main results

Counts are independently verified repair outcomes. Monitoring is development,
not a final test. The external families were previously inspected in historical
work and are exploratory, not untouched confirmation.

| Arm | Student / proposer | Monitoring /41 | Final ID /115 | External /116 | ID generated tokens | External generated tokens | Arm H200-hours |
|---|---|---:|---:|---:|---:|---:|---:|
| Base | Initial policy | 31 | 88 | 93 | 235,661 | 265,086 | 5.0564 |
| Control | Dr. GRPO / difficulty | 28 | 87 | 92 | 199,200 | 238,105 | 7.5769 |
| A | Ungated OPSD / difficulty | 30 | 88 | 92 | 252,489 | 289,750 | 9.4969 |
| B | Gated OPSD / difficulty | 29 | 90 | 92 | 226,697 | 258,717 | 8.6847 |
| C | Gated OPSD / RL alignment | 31 | 88 | 88 | 227,644 | 258,362 | 8.7469 |
| D | Gated OPSD / RL+OPSD alignment | 28 | 85 | 92 | 225,055 | 273,580 | 9.0147 |

Arm hours include its pilot, all main segments, and final evaluations; base
includes its cancelled empty attempt and successful replacement. The shared
qualification and failure costs are not allocated to arms. All 27 reservations,
including failures and continuations, were reconciled to 233,565 H200-seconds
(**64.879167 H200-hours**) against the shared 80-hour cap. No account jobs were
active at final reconciliation. The retained private budget and analysis are
`runs/task19/budget-final.json` and `runs/task19/analysis/`.

The 95% intervals below bootstrap parent families, with 52 ID and 113 external
families. They quantify task/family variation conditional on one observed
training run, not variation across seeds. Values are percentage points, first
method minus second:

| Comparison | Final ID difference [95% interval] | External difference [95% interval] |
|---|---:|---:|
| A − control | +0.87 [−5.65, +7.77] | 0.00 [−7.83, +7.89] |
| B − control | +2.61 [−2.44, +7.69] | 0.00 [−6.09, +6.14] |
| B − A | +1.74 [−4.59, +8.41] | 0.00 [−6.84, +6.84] |
| C − B | −1.74 [−8.73, +5.69] | −3.45 [−10.75, +4.35] |
| D − C | −2.61 [−9.17, +3.31] | +3.45 [−4.24, +10.35] |

These results do not establish that OPSD improves over Dr. GRPO, that gating
beats ungated OPSD, or that either alignment reward improves over difficulty
or the other alignment reward. The point estimates and broad intervals support
uncertainty rather than equivalence. On this model/interface the base is already
strong, and the 96-episode endpoint shows no demonstrated improvement over it.
Task 15–18 used a different model or interface and are not a controlled model
comparison. No generalization gain over the new base is established.

## Signals and failure modes

The seven logged training updates attempted 49/49/49/49/50 proposer candidates
for control/A/B/C/D, of which 37/36/37/37/37 were valid. Valid reward diversity
was limited: distinct-valid-reward groups were 1/18, 1/18, 2/18, 3/18 and 8/18.
C recorded 29 near-zero alignments and D zero. Thus RL-only alignment often
lacked a usable signal, while adding OPSD produced more differentiable proposer
rewards; this did not establish an endpoint benefit or causal mechanism.
Difficulty arms' valid-versus-invalid separation can still produce updates even
when valid rewards do not differ. Verified training rollout repairs were
69/62/69/68/64 out of 96; these are training diagnostics, not independent
accuracy. Control made six optimizer steps and each other arm seven.

At the endpoint, observed non-success categories were truncation and
public-pass/hidden-fail: ID truncations were 12/15/18/16/17/16 and public-pass/
hidden-fail 15/13/9/9/10/14 for base/control/A/B/C/D. External truncations were
13/10/17/11/21/13 and public-pass/hidden-fail 10/14/7/13/7/11. These labels
are observed outcomes, not proof of causes. Generated tokens and wall time also
vary by arm, so the table reports efficiency alongside accuracy.

The cached FP32 qualification on H200 passed 64 numerical comparisons: maximum
replay difference and clipping changes were zero, relative gradient difference
was 1.548e-5, and relative optimizer displacement difference was 8.292e-6.
This qualifies the measured execution path; it does not prove all numerical
causes are absent. Actual rollout behavior probabilities remain the PPO
denominator. Earlier BF16/uncached failed or stopped attempts and their costs
are preserved in the [status record](task19-status.md).
Across the seven recorded updates per arm, joint-gradient and update norms were
finite, reported clipping fractions and support exits were zero, and the largest
absolute batch-mean behavior drift was 8.88e-8. This is a batch mean, not a
maximum token discrepancy; some batches had no reported drift.

## Figures and telemetry

1. [Learning and efficiency curves](../figures/task19/01-learning-curves.pdf)
   show 0, 32, 48, 72 and 96 attempted episodes, sampled tokens and allocation.
   The nominal 24-episode milestone was skipped by the completed 32-episode
   pilot; the analysis deliberately does not fabricate an observation.
2. [Final generalization and family intervals](../figures/task19/02-generalization.pdf).
3. [OPSD diagnostics](../figures/task19/03-opsd.pdf).
4. [Proposer reward and alignment diagnostics](../figures/task19/04-proposer.pdf).
5. [Observed failure taxonomy](../figures/task19/05-failure-taxonomy.pdf).

[Control](https://wandb.ai/tatsu432s/self-play-opsd-coding-agent/runs/073799c84d9418b708518264),
[A](https://wandb.ai/tatsu432s/self-play-opsd-coding-agent/runs/250fce494fbb23cf6fe05214),
[B](https://wandb.ai/tatsu432s/self-play-opsd-coding-agent/runs/bbad5488a9736820fcc2b972),
[C](https://wandb.ai/tatsu432s/self-play-opsd-coding-agent/runs/6e8841e8445be7647ea1aa23),
and [D](https://wandb.ai/tatsu432s/self-play-opsd-coding-agent/runs/f0d2b590a9260ba557f5a009)
were recorded offline on Yale and replayed as aggregate scalar histories from
the authenticated local workspace. The API confirmed 14 rows per arm; original
timestamps and runtime are explicit metrics, and continuation records also
retain their original segment steps. A's repeated final ID publication on resume was omitted once; its
external metric was appended. Original offline records and replay logs remain
private under `runs/task19/`.

## Decision

Keep this as a completed bounded one-seed study, not evidence of a winner.
For a future follow-up, improve proposer reward diversity and investigate the
large number of near-zero RL-only alignment candidates before allocating more
training; pre-register any additional seeds and budget separately. Do not
reclassify the already inspected external benchmark as unseen confirmation.
No further GPU run is implied by this decision.
