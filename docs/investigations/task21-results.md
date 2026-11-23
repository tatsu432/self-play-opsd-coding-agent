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

The A and B arms use the same nominal distillation coefficient, calibrated
to the gated gradient. At the seeded initial calibration, the ungated and
gated gradient norms differed substantially. Thus an A/B difference would
not isolate gating from effective objective strength. Two training seeds and
the finite test sets also limit the precision of any comparison. The final
report will add verified endpoint scores, paired uncertainty, training-signal
and efficiency curves, actual allocations, and the supported research
decision after the evaluations finish.
