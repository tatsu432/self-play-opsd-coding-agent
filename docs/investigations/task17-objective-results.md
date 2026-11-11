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

GRPO continuation `32898057` was submitted once from verified checkpoint 000004
for the 80-episode endpoint. Inspect its live ledger; never duplicate it. The
pending six-hour wall was reduced to three hours while retaining the original
budget reservation. Continue the gated arm sequentially after GRPO review. Total
Phase 2 allocated time through the pilots is **4,576 H200-seconds (1.27111
GPU-hours)**, including the failed calibration; the new continuation is pending.

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
