# Task 17 controlled student objectives — in progress

The [frozen study specification](../task/17-objective-comparison-spec.md) uses
exact text, a common fixed curriculum and an 80-attempted-episode endpoint.
This document records actual results; objective superiority is not yet measured.

## Calibration

The initial calibration allocation failed after four saved rollouts because the
backend did not supply an `update_norm` metric. Its run `11610fb16bd9af1f197221b1`
(job `32872812`) is archived, including the original ledger; it consumed 265
H200-seconds. The corrected runner measures actual adapter displacement and
restores adapter/optimizer state after each diagnostic update. Focused tests cover
normal and exceptional restoration. A new matching CPU runtime check passed.

Corrected calibration run `06982710d021357786a605aa` (job `32877712`) completed
32 rollouts over eight distinct reserved training families. It used 556
H200-seconds. The gated squared gradient-norm sum was 0.1054657523 and the
ungated-unit squared sum 120.4827979370, yielding matched initial coefficient
**0.02958647361953192**. Its artifact SHA-256 is
`704e1fad2b53daa360dce14a53ba359689bdc50a21bb5870f8c8230f5d0280fc`.
Twenty-five of 32 calibration rollouts passed; calibration outcomes are not
independent evaluation results. The final control is immutable for continuation
and has SHA-256 `1969665967250fa8e7d8698c9b7f8de4042bec2a97a1c28fae5b5efc38a2e091`.

## Seed-2026 bounded pilots

| Objective | Baseline monitoring | Baseline independent development | Rollout successes | Mixed groups | Optimizer updates | GPU-seconds |
|---|---:|---:|---:|---:|---:|---:|
| GRPO | 21/41 | 7/16 | 26/32 | 5/8 | 3 | 1,851 |
| Gated OPSD | 21/41 | 7/16 | 27/32 | 3/8 | 4 | 1,904 |

Both baseline outcome sets match task-by-task. The objective run manifests also
match in initial adapter, model, pool, tasks, seed, interface, budgets, prompt,
control and scientific code. Rollout outcomes are on-policy training data, not
an independent objective comparison. Both pilots stopped cleanly at iteration 4;
the complete joint checkpoints' metadata and file checksums were verified.
All recorded gradients were finite. One uniform-reward GRPO iteration produced
no update; gated OPSD produced an auxiliary-driven update on the corresponding
uniform-reward first iteration. This is a mechanistic observation, not evidence
of superior generalization.

The maximum rollout/rescoring log-probability differences were 1.1153 (GRPO)
and 0.8137 (gated); each iteration emitted a mismatch warning. The original
behavior-policy denominators remain intact, and the causal impact is unknown.
Matched continuation to 40 and 80 episodes is warranted by nonzero learning
signals, but any conclusion must account for this numerical limitation.

GRPO continuation `32898057` completed exit 0 at 80 episodes, consuming 3,268
additional H200-seconds. Its monitoring curve was 21/41 → 25/41 → 25/41; its
independent-development curve was 7/16 → 9/16 → 10/16. At 80, three original
failures became successes and none of the original successes regressed. It made
five optimizer steps over 80 episodes; 54/80 training rollouts succeeded, with
seven mixed groups out of 20. The maximum scoring difference reached 2.9407;
0.59% of sampled tokens in that iteration differed by more than 0.2. The final
checkpoint files and metadata were checksum-verified. These are single-seed
development observations, not an objective comparison.

Gated continuation `32908569` completed exit 0 in 3,168 H200-seconds; its
iteration-10 checkpoint files and metadata were verified. At 0/40/80 episodes,
monitoring was 21/27/28 out of 41 and independent development was 7/9/10 out of
16. GRPO was 21/25/25 and 7/9/10, respectively. At the endpoint each method
solved one independent task the other missed. The gated-minus-GRPO paired
difference was 0/16 with a family-bootstrap 95% interval of −18.75 to +18.75
percentage points, conditional on seed 2026. This small development set and
one training seed cannot establish equivalence or robust superiority.

GRPO/gated used 1.422/1.409 H200-hours and generated 24,990/22,481 training
tokens respectively. On-policy training success was 54/80 versus 58/80;
these are not independent evaluations. GRPO made five updates with seven mixed
groups out of 20; gated made ten updates with six mixed groups. Maximum
behavior/rescoring differences were 2.9407 and 1.8313, respectively. Gated's
auxiliary updates on uniform groups establish a difference in learning signal,
not a causal generalization advantage. The unchanged behavior denominators and
unverified impact of scoring differences remain limitations.

![Single-seed development curves and GPU allocation](../figures/task17/objective-learning-efficiency-seed2026.png)

Both pilots and the full seed-2026 pair support proceeding with seeds 2027 and
2028 before interpreting training-seed variation. GRPO seed-2027 job `32917167`
(run `95735e5ad673444230f30717`) was submitted once to 80 episodes; its pending
wall was reduced to three hours, with its original reservation retained. Inspect
its ledger and never duplicate it. Total Phase 2 allocated time
through these endpoints is **11,012 H200-seconds (3.05889 GPU-hours)**, including
the failed calibration. The objective analysis is at
`runs/task17/objective-analysis-seed2026-v3/`; its family-bootstrap interval
reflects task sampling conditional on one training seed, not seed variation.

## Reproduction after endpoints

Fetch each existing run with `uv run selfplay hpc fetch RUN_ID`, verify its
checkpoint and scheduler/ledger identities, and create a private manifest with
`runs` entries `{arm, seed, run_dir, allocated_gpu_hours}`. Hours must sum actual
Slurm allocations for that logical run, including failed/interrupted attempts;
retain the accounting evidence. Do not include a run before its endpoint is
complete. Then use a fresh output directory:

```bash
uv run python scripts/analyze_task17_objectives.py --manifest runs/task17/objective-manifest.json --pool runs/hpc/yale-hpc-cpsc4880/86e2df49bb04a9321f7d755d/pool.json --control configs/task17-objective-control.json --output runs/task17/objective-analysis-v1
```

The script checks the final control, initial adapter pairing within seed, shared
scientific configuration, full exposure and development milestones. Its family
bootstrap is conditional on each training seed; training-seed variation is
reported separately. Private source and per-task records remain under `runs/`.
