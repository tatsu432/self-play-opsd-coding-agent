# Task 20 research decision

The evidence does not support launching the matched shared Control/A/B
comparison or C/D alignment arms on the current candidate distribution. The
shared pilot reached 64 episodes with one effective solver update and no
within-group reward variation among valid proposals. The small sample does not
prove that shared self-play or OPSD is ineffective; it fails the stated
viability gate for a defensible method comparison.

## What was investigated

The [Task 19 audit](task20-root-cause.md) reconstructed sparse solver signals
from the original runs: only two to four iterations per arm had a nonzero
solver RL gradient. This is evidence of limited informative updates, not proof
of convergence or of the cause of the final scores. The
[benchmark decision](task20-benchmark-decision.md) records why the PDB source
constraints were not admitted by the existing sandbox and how 205 verified
EvalPlus fallback tasks were prepared with separate development and final
partitions. The screen found five mixed four-rollout groups among 32 tasks;
on the paired eight-task subset, eight rollouts found three mixed groups versus
one with four rollouts. The five mixed tasks were fixed as a training-only
curriculum.

The [solver-only control](task20-positive-control.md) made eight nonzero RL
updates in 256 episodes. Two paired training checks improved 9/20 to 11/20 and
8/20 to 13/20, while separate training probes were 27/32 to 27/32 and 27/32 to
26/32. Development changed 25/41 to 28/41. These observations support
learnability on the small screened training set, not generalization. The
original one-step diagnostic excluded nonfinite changes associated with 29
combined support exits and cannot separate pre- from post-update exits; all
eight same-policy training rescores had zero support exits. The post-hoc
reviewed gate and its difference from the original saved marker are disclosed
in the linked report.

The [shared-pilot diagnosis](task20-shared-diagnosis.md) compares role gradients
and examines the actual proposal groups. It found two mixed solver groups in
its first batch and none in its second. All valid proposals within each group
received the same reward. The nonzero proposer gradients therefore distinguished
invalid from valid proposals, rather than ranking valid tasks. The only
measurable solver–proposer cosine was near zero (0.0026); one measurement does
not establish the presence or absence of harmful interference. Initial
41-task development outcomes matched the solver-only baseline task by task;
the endpoint was 26/41, with two gains and one loss. These exploratory results
do not establish a generalization improvement.

## Decision and limits

Keep the qualified shared FP32 path and authentic behavior-policy denominators.
Do not interpret these single-seed pilots or task-level intervals as uncertainty
across training seeds. Do not choose a checkpoint or tune a future intervention
on the untouched final partitions. A next study would need a prespecified
training-only intervention that increases candidate difficulty and creates
within-parent reward diversity among valid unique proposals, then repeat the
viability check under a separate scientific identity. This recommendation is a
hypothesis, not evidence that such an intervention succeeds. After viability is
shown, a matched Control/A/B comparison can test the two OPSD objectives; C/D
requires its additional alignment evidence.

Completion status, allocation, telemetry and reproducibility details are in the
[status record](task20-status.md). Private datasets, trajectories and
checkpoints remain in ignored storage. No final ID or transfer evaluation is
claimed for the incomplete matched comparison. Actual Task 20 allocation was
63,218 H200-seconds (17.560556 hours). Both shared-pilot aggregate rows were
verified in [W&B](https://wandb.ai/tatsu432s/self-play-opsd-coding-agent/runs/64112872d3b8d79c595ae0c2).
