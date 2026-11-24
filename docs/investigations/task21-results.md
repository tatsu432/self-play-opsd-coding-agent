# Task 21 final held-out results

The frozen 512-episode study completed Control (Dr. GRPO), A (ungated
OPSD), and B (gated OPSD), with two matched training seeds. Every endpoint
checkpoint and all 12 baseline/endpoint evaluations were verified. The same
held-out tasks are reused for all arms; the seeds are separate training runs,
not separate test sets. The eight PDB-Multi tasks and the BigCodeBench cells
are small, so their results are exploratory.

| Seed | Checkpoint | PDB single / 64 | PDB multi / 8 | PDB BigCodeBench / 1 | BigCodeBench mutants / 9 |
|---|---|---:|---:|---:|---:|
| 2026 | Initial | 21 | 1 | 1 | 4 |
| 2026 | Control | 22 | 0 | 1 | 3 |
| 2026 | A | 26 | 3 | 1 | 6 |
| 2026 | B | 28 | 0 | 1 | 5 |
| 2027 | Initial | 20 | 0 | 1 | 2 |
| 2027 | Control | 21 | 0 | 1 | 4 |
| 2027 | A | 21 | 1 | 1 | 7 |
| 2027 | B | 28 | 3 | 1 | 5 |

The initial scores in each seed match task by task across the three arms. On
PDB single, B minus Control was +6/64 (+9.4 percentage points) for 2026 and
+7/64 (+10.9 points) for 2027. Family bootstrap 95% intervals conditional on
each training run were [−3.1, +21.9] and [−1.6, +23.4] points. A minus Control
was +4/64 and 0/64, with intervals [−7.8, +20.3] and [−15.6, +15.6] points.
These task-based intervals do not quantify training-seed uncertainty. Two
seeds and the limited training duration are insufficient to establish a robust
objective winner; a zero-width empirical interval on a tiny cell does not
establish equivalence. The 1- and 9-example BigCodeBench cells are not a broad
external benchmark. The PDB-Multi results vary substantially between seeds.

All arms attempted 512 episodes, but generated task distributions differed by
method, and training required only 19–21 optimizer steps. The study does not
establish convergence or that this was enough optimization. Each initial
checkpoint was evaluated separately per arm, which consumed extra compute;
its repeated initial outcomes should not be counted as independent examples.
Endpoint training allocations were 25.667/20.984/27.515 H200-hours for
Control/A/B in 2026 and 24.235/22.439/24.701 in 2027. These include job
continuations and are not equal-compute endpoints. The full study allocated
**217.155 H200-hours**, including pilots, evaluations, failed attempts, and
resumptions; all reservations were reconciled. Six first endpoint evaluation
attempts failed while the shared project filesystem was full; the precise
failure cause was not established. Existing partial records were checked and
resumed, and the failed allocations remain included in accounting.

The 64-task primary endpoint had 29–38 truncations per trained arm. Limits on
turns, context, or tokens are therefore a major observable failure mode.
Training rewards use evolving proposer-generated tasks and are not a fixed-set
learning curve. The 24-task development set was repeatedly inspected; it is a
monitor, not fresh confirmation. Two seeds offer a limited replication check,
but a deeper single-seed study could better address the unresolved optimization
depth question. The checkpoints save the adapter, Adam optimizer and random state, and the
training loop restores them; the endpoint is excluded from the scientific
hash. This makes a longer continuation technically plausible. A new endpoint
still needs a checked controller/ledger workflow and a small resume validation,
since extending a completed run was not tested. No extra training is implied
by this result.

Figures: [difficulty screen](../figures/task21/difficulty.pdf),
[2026 learning/efficiency](../figures/task21/learning-and-efficiency-2026.pdf),
[2027 learning/efficiency](../figures/task21/learning-and-efficiency-2027.pdf),
[2026 paired intervals](../figures/task21/paired-intervals-2026.pdf), and
[2027 paired intervals](../figures/task21/paired-intervals-2027.pdf).
Aggregate [W&B final results](https://wandb.ai/tatsu432s/self-play-opsd-coding-agent/runs/task21-final-aggregates),
[learning curves](https://wandb.ai/tatsu432s/self-play-opsd-coding-agent/runs/e3684c8e0122b912f97c238a),
and [held-out progress](https://wandb.ai/tatsu432s/self-play-opsd-coding-agent/runs/4e507c652d2e258a12289799)
are available; the latter records counts, not endpoint scores. Raw trajectories,
private datasets, checkpoints and detailed analysis remain in ignored storage.
See [status and reproducibility](task21-status.md).

## Earlier pilot and monitoring evidence

# Task 21 research results — preliminary

The matched Control/A/B training and independent final evaluations are still
running. This document records the verified benchmark and pilot evidence; it
does not yet answer which student objective performs best. Run identities,
failure and recovery records, private artifact locations, and reproduction
instructions are in the [study status](task21-status.md).

## Benchmark and scope

We adapted Precise Debugging Benchmark (PDB) tasks to the existing isolated
Qwen-native editing and verification workflow. The initial feasibility pilot
contained 11 parent-disjoint single-bug tasks: eight LiveCodeBench-derived and
three BigCodeBench-derived. The pinned scorer and isolated adapter agreed on
22/22 original and canonical-source verdicts on that pilot. The base model
repaired 5/11 tasks in one attempt each. This is a small feasibility sample,
not an estimate of benchmark-wide performance or model generalization.

The main study uses a small adapted, parent-partitioned dataset, with separate
development, held-out single-bug, PDB-Multi, adapted BigCodeBench-derived,
and independently generated BigCodeBench-mutant groups. These are distinct
evaluation categories and should not be combined into an official PDB score.
Private fixtures and trajectories remain outside the public repository.
The [status document](task21-status.md#reproduction) records pinned upstream
revisions, verifier scope, and reproduction commands.

## Learning signal before the main study

The completed 24-parent screen found mixed rewards in 12/24 four-rollout
groups and 17/24 eight-rollout groups. Including both screen allocations,
four and eight rollouts produced 3.70 and 2.62 mixed groups per H200-hour,
respectively. Four rollouts were selected for the solver-only curriculum.

The verified 128-episode solver-only pilot made eight nonzero RL updates.
On 12 training tasks sampled four times each, verified repairs rose from
24/48 to 30/48; on eight separate probe tasks, they rose from 8/32 to 10/32.
The task-cluster bootstrap intervals for the changes were 0 to +27.1 and
−9.4 to +28.1 percentage points, respectively. They describe task variation
conditional on one trained model and do not establish generalization or
training-seed uncertainty.

The first 64-episode shared-policy pilot produced seven mixed solver groups
out of 16, but no parent with distinct rewards among valid proposals. A
revised prompt and larger candidate cap were tested together; its verified
112-episode pilot produced 17 mixed groups out of 28 and five distinct parents
with different valid-proposal rewards. The two changes are bundled, so this
improvement cannot be attributed to either individually. These results show
that the revised setup can produce both types of signal, not that self-play
improves independent repair performance.

## Matched comparison and interpretation

Two matched seeds each compare verifier-driven Control, ungated OPSD A, and
gated OPSD B at 512 attempted episodes. The within-seed arms share their
initialization, data, and frozen study settings; independent endpoint scoring
and resource accounting remain pending. Development-monitor results reuse
24 tasks and are not held-out final results.

Reconciled matched-training allocations, including timed-out and resumed
attempts, total approximately 74.167 H200-hours for seed 2026 and 71.375
for seed 2027. These are training allocations for the three arms per seed;
pilot, screening and held-out evaluation allocations are additional.

The A and B arms use the same nominal distillation coefficient, calibrated
to the gated gradient. At the seeded initial calibration, the ungated and
gated gradient norms differed substantially. Thus an A/B difference would
not isolate gating from effective objective strength. Two training seeds and
the finite test sets also limit the precision of any comparison. The final
report will add verified endpoint scores, paired uncertainty, training-signal
and efficiency curves, actual allocations, and the supported research
decision after the evaluations finish.

All six runs have completed and verified 512 training attempts. The following
aggregate diagnostics are from their committed iterations. Each
mixed group contains four attempts with both successes and failures.

| Seed | Method | Verified repairs / 512 | First / last 128 | Mixed groups / 128 | Nonzero RL updates |
|---|---|---:|---:|---:|---:|
| 2026 | Control | 300 | 73 / 70 | 58 | 19 |
| 2026 | A | 351 | 85 / 90 | 60 | 19 |
| 2026 | B | 328 | 81 / 87 | 74 | 19 |
| 2027 | Control | 276 | 70 / 73 | 61 | 20 |
| 2027 | A | 313 | 89 / 71 | 58 | 19 |
| 2027 | B | 313 | 79 / 71 | 61 | 20 |

The Control/A/B runs performed 19/19/20 optimizer steps for seed 2026 and
21/21/21 for seed 2027. Different valid-proposal counts produced different
batch counts despite identical attempted-episode exposure.

These training rewards are on evolving proposer-generated tasks, so early and
late values are not a controlled learning comparison. Recorded joint gradient
norms were finite and reported support exits and worker retries were zero.
However, 137–212 attempts per run failed after reaching a turn, context, or
token limit, and scheduler timeouts required recorded continuations. Those
limitations and the held-out scores must be considered in the final decision.
The [W&B learning-curves run](https://wandb.ai/tatsu432s/self-play-opsd-coding-agent/runs/e3684c8e0122b912f97c238a)
contains an overlaid development-monitor chart and a per-arm training-health
table. The separate [evaluation-progress run](https://wandb.ai/tatsu432s/self-play-opsd-coding-agent/runs/4e507c652d2e258a12289799)
tracks task-record counts without exposing partial scores.

The 384-episode development-monitor scores below reuse the same 24 tasks at
each milestone. They are shown to document the observed trajectory, not as
independent endpoint estimates or evidence that a method wins.

| Seed | Baseline | Control | A | B |
|---|---:|---:|---:|---:|
| 2026 | 7/24 | 7/24 | 8/24 | 11/24 |
| 2027 | 8/24 | 11/24 | 7/24 | 14/24 |

Both A arms subsequently completed 512 episodes and scored 12/24 and 10/24
for seeds 2026 and 2027, respectively, on this same monitor. Their held-out
evaluations remain pending. The seed-2027 Control endpoint also completed,
scoring 8/24 versus 8/24 at baseline on the reused monitor; seed-2027 B
scored 11/24 versus the same 8/24 baseline. Seed-2026 Control completed
with 7/24 versus 7/24 at baseline, and seed-2026 B scored 13/24 versus
7/24. All held-out evaluations remain pending.
