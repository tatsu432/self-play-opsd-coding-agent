# Task 17 controlled student-objective comparison

The [generalization specification](17-generalization-spec.md) governs this study.
Its question is whether gated on-policy self-distillation improves Python repair
relative to GRPO alone or ungated distillation when student exposure and task
supply are held fixed. The completed Task 15 A/B runs do not isolate this effect
because their proposer and student coevolved separately.

The Phase 1 training-only paired comparison selected exact text for a fresh base
student. Base strict/text verified repairs were 5/40 and 19/40; the trained B2026
snapshot instead scored 27/40 and 21/40. The single trained-checkpoint result and
its uncertainty remain important limitations. See the
[interface investigation](../investigations/task17-repair-interface-results.md).

Use the pinned 4B model, original behavior-policy probabilities, `same_scorer_v2`,
unchanged isolation, one verified fixed curriculum, four rollouts per task and
80 attempted episodes per arm. Two curriculum tasks are consumed per iteration;
all arms use the same order and keyed seeds. The common control and evidence hashes
are in `configs/task17-objective-control.json`. Student labels are:

| Label | Student objective | Nominal SD coefficient |
|---|---|---:|
| D | GRPO | zero effective SD |
| B | GRPO + gated OPSD | 0.1 |
| A | GRPO + ungated OPSD | 0.1 |
| C | GRPO + calibrated ungated OPSD | 0.02958647361953192 |

Calibration used eight other training-only families and four rollouts each. The
coefficient matches the initial RMS auxiliary gradient norm to gated OPSD; it does
not match later gradient directions, optimizer state or update magnitudes. Its
private artifact SHA-256 is
`704e1fad2b53daa360dce14a53ba359689bdc50a21bb5870f8c8230f5d0280fc`.

Evaluate the existing 41-task monitoring set and a separate 16-family development
set at baseline, 40 episodes and 80 episodes. Neither Task 15 final set is
untouched confirmation. Begin with seed-2026 GRPO/gated pilots stopped after 32
episodes; review checkpoints, mixed groups, gradient finiteness, denominator
warnings, exposure and cost before extending. Prioritize matching GRPO and gated
across seeds 2026–2028. Add ungated controls as the authorized 24 H200-hour
Phase 2 ceiling permits, including failed allocations. Use family-aware task
uncertainty separately from variation across training seeds. Do not launch full
C/D proposer studies. See the [objective results](../investigations/task17-objective-results.md)
for actual execution and the [execution plan](../plan/17-repair-interface-plan.md).
