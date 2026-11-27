# Demystifying gated on-policy self-distillation in coding-agent self-play

This study separates two decisions. The first is how the student uses privileged
repair guidance on its own trajectories. The second is how the proposer scores
the tasks that the student attempts. Across the observed Qwen3.5-4B checkpoints,
gated distillation improved held-out repairs relative to ungated distillation
and verifier-only RL in both seeds. Incorporating the gated distillation
gradient into the proposer score removed near-zero gradients in paired
diagnostics. The C/D accuracy ranking reversed across seeds.

## Experimental setup

| Method | Proposer reward on scored tasks | Student objective |
| --- | --- | --- |
| Initial | — | Initial model |
| Control | Difficulty, `4p(1-p)` | Dr. GRPO |
| A | Difficulty | Dr. GRPO + ungated OPSD |
| B | Difficulty | Dr. GRPO + gated OPSD |
| C | Reference CE cosine with RL gradient | Dr. GRPO + gated OPSD |
| D | Reference CE cosine with RL and gated OPSD gradient | Dr. GRPO + gated OPSD |

The two seeds share a frozen adapted PDB dataset and use the matching
initialization within each seed. Each trained arm attempts 512 student
episodes, four student rollouts per task, with native exact-text editing and
fresh executable verification. The Qwen3.5-4B proposer and student share a
rank-32, alpha-64 FP32 LoRA adapter. The clipped AdamW update uses learning
rate 1e−5, zero weight decay, ratio clip 0.2 and combined norm clip 1.0.
Nominal distillation and proposer scales are 0.1717172 and 0.3035281.
Evaluation uses the recorded cached student sampler and the original limits
of eight turns, 4,096 generated tokens per turn, 12,288 generated tokens per
episode, and 20,480 student-context tokens. The teacher's context cap during
training is 24,576. [Executed objective definitions](../methods.md#matched-pdb-study-tasks-2123)

Fixed training-only probes and reference-gradient tasks are separate from
development, primary and transfer evaluation. The historical solver-only
probe was a pilot under a different curriculum. Shared-run capability curves
use the 24 development tasks, repeatedly monitored at 0, 64, 128, 256, 384 and
512 attempted episodes. The primary set contains 64 held-out single-bug tasks.
We report the adapted PDB and BigCodeBench-derived repair tasks under their
own names. [Framework](../figures/task23/01-framework.pdf)

## Main results

| Method | 2026 / 64 | 2027 / 64 | Mean | Seed SD (pp) | Δ Initial (pp) | Δ Control (pp) | Training H200 h, 2026 / 2027 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Initial | 21 | 20 | 32.03% | 1.10 | 0.00 | −1.56 | 0 / 0 |
| Control | 22 | 21 | 33.59% | 1.10 | +1.56 | 0.00 | 25.667 / 24.235 |
| A | 26 | 21 | 36.72% | 5.52 | +4.69 | +3.13 | 20.984 / 22.439 |
| B | 28 | 28 | 43.75% | 0.00 | +11.72 | +10.16 | 27.515 / 24.701 |
| C | 33 | 24 | 44.53% | 9.94 | +12.50 | +10.94 | 24.221 / 25.177 |
| D | 25 | 31 | 43.75% | 6.63 | +11.72 | +10.16 | 24.375 / 20.419 |

| Contrast | 2026 (pp) | 2027 (pp) | Mean (pp) | Conditional 95% task-family interval (pp) |
| --- | ---: | ---: | ---: | --- |
| A − Control | +6.25 | 0.00 | +3.13 | [−9.38, +15.63] |
| B − A | +3.13 | +10.94 | +7.03 | [−3.13, +16.41] |
| B − Control | +9.38 | +10.94 | +10.16 | [+0.78, +20.31] |
| C − B | +7.81 | −6.25 | +0.78 | [−8.59, +10.16] |
| D − C | −12.50 | +10.94 | −0.78 | [−7.81, +6.25] |
| D − B | −4.69 | +4.69 | 0.00 | [−9.38, +10.16] |

Bootstrap intervals come from 2,000 draws of task families with replacement.
Each pooled draw uses the same sampled families across both seeds, keeps
within-family tasks together and weights by the number of resampled tasks.
Initial outcomes are counted once per seed even where earlier evaluations ran
separately. These intervals hold the observed checkpoints fixed. Seed SD
describes the two observed scores. The conditional B–Control interval favors
B for these checkpoints under this task resampling procedure. Training-seed
uncertainty and multiple comparisons still limit the strength of any broader
claim. [Primary results](../figures/task23/02-primary.pdf) ·
[Detailed results](../figures/task23/table-2-main.csv) ·
[Paired contrasts](../figures/task23/table-3-contrasts.csv)

| Method | PDB-Multi / 8 | PDB cross-source / 1 | BigCodeBench mutants / 9 |
| --- | ---: | ---: | ---: |
| Initial | 1 / 0 | 1 / 1 | 4 / 2 |
| Control | 0 / 0 | 1 / 1 | 3 / 4 |
| A | 3 / 1 | 1 / 1 | 6 / 7 |
| B | 0 / 3 | 1 / 1 | 5 / 5 |
| C | 1 / 2 | 1 / 1 | 5 / 4 |
| D | 1 / 2 | 1 / 1 | 6 / 7 |

Transfer entries show verified repairs for seeds 2026 / 2027. The single
cross-source task has no variation in correctness. The other original transfer
sets are exploratory. [Transfer figure](../figures/task23/03-transfer.pdf)

## Learning dynamics and mechanism

D finishes at 14/24 and 15/24 on development, the highest endpoint in each
seed. Its primary scores, 25/64 and 31/64, illustrate the limits of that
monitor. The two-seed mean development curves align by exposure. Token,
optimizer-step and compute plots use each run's actual coordinates. H200-hour
coordinates include the relevant training-job allocation up to the monitor,
with startup, inline evaluation and interruption overhead.
[Development curves](../figures/task23/04-development.pdf) ·
[Seed panels](../figures/task23/04-development-by-seed.pdf)

Online binary reward uses the evolving task distribution. For example,
seed-2026 A verifies 351/512 online attempts and B verifies 328/512, whereas
B repairs more held-out tasks. The rolling curve averages the latest 64
committed attempt slots and retains raw iteration points. Proposer rewards
appear separately for difficulty and cosine. Novelty counts use distinct
scored task IDs. Duplicate flags can overlap proposal validity. Verifier
statuses distinguish model failures from infrastructure failure or retries.
[Online dynamics](../figures/task23/05-online-dynamics.pdf)

| Paired diagnostic group | Candidates | C near-zero | D near-zero | D positive cosine |
| --- | ---: | ---: | ---: | ---: |
| All failed | 79 | 79 | 0 | 79 |
| Mixed | 248 | 0 | 0 | 169 |
| All successful | 185 | 185 | 0 | 184 |
| Total | 512 | 264 | 0 | 432 |

The paired diagnostics score the same candidate episodes and reference
direction at the same snapshot. D yields distinct rewards in 169 of 259
recorded proposer groups, versus 120 for C. Most recovered zero-gradient cases
are already-solved groups, although every all-failed case in the paired
sample also acquires a positive reference cosine. A reference cosine measures
direction relative to training-reference CE in LoRA space. It does not
include the joint update, AdamW preconditioning or clipping, and it cannot
establish held-out repair benefit. [Gradient sparsity and alignment](../figures/task23/06-alignment.pdf)

A/B median weighted OPSD-to-RL gradient ratios are 0.674/0.177 and
0.758/0.214. Corresponding median RL–OPSD cosines are +0.058/−0.020 and
+0.034/−0.008. These compare different on-policy trajectories. The sigmoid
weights both positive and negative teacher gaps and changes auxiliary
magnitude and direction. A strength-matched ungated comparison remains the
direct test of whether selective weighting contributes beyond attenuation.
[Distillation diagnostics](../figures/task23/07-distillation.pdf) ·
[Mechanistic and efficiency statistics](../figures/task23/table-4-mechanisms.csv)

## Failure modes and interpretation

The original evaluations include 404 truncation failures across 768
checkpoint-task outcomes. The per-turn token cap accounts for 243, the turn cap
for 160, and the episode cap for one. In 64 cases the model hit a limit and
still produced a verified repair. Among truncation failures, 14 ended with
an accepted edit and passing public tests. Hidden-test verification governs
every success count. These observations motivate the budget sensitivity
comparison. [Failure categories and budget](../figures/task23/08-failure-and-budget.pdf)

The study uses two seeds, short optimization horizons, small evaluation
partitions, and shared proposer/student parameters. Generated task
distributions and compute differ among equal-exposure arms. B's advantage
mixes gating and effective auxiliary strength. C/D alignment uses a first-order
proxy. The development set was repeatedly inspected. Control has a self-play
proposer, so the primary comparisons cannot estimate an advantage over a
fixed curriculum. The available results support conclusions about objective
design in this setting.

The student objective builds on Dr. GRPO's group-centered reward formulation
and OPSD's use of privileged context on a student's own sampled trajectory.
SIPO also combines verifier rewards with self-distillation through a
contrastive teacher. Our comparisons vary a sigmoid token gate and whether the
proposers' reference-gradient scoring includes OPSD. PDB supplies the basis
for the adapted repair setting. [Dr. GRPO](https://arxiv.org/abs/2503.20783) ·
[OPSD](https://arxiv.org/abs/2601.18734) ·
[SIPO](https://arxiv.org/abs/2609.36742) ·
[PDB](https://arxiv.org/abs/2604.17338)

## Reproduction and compute

The reconciled Tasks 21 and 22 ledgers contain 64 distinct allocations
totaling 1,161,107 H200-seconds (322.530 hours). This includes qualification,
pilot, failed, continued, training and evaluation allocations. Per-cell
end-to-end totals count the allocations associated with the corresponding
training and evaluation owners. The three repeated Initial evaluations are
included in physical cost.

From the repository root with the archived Task 21/22 receipts, frozen PDB
bundle and fetched HPC histories in their two original directory roots or
read-only links, run:

```bash
uv sync --frozen --group build
uv run --frozen python scripts/analyze_task23.py --output docs/figures/task23
```

The command writes a private logical-run registry and produces aggregate
[figure data](../figures/task23/statistics.json). The live execution and
follow-up accounting are in the [status record](../investigations/task23-status.md).
