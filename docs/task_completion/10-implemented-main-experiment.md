# Task 10 — main launch engineering

Status: implementation and live validation in progress. **Main A–D training has
not been submitted and remains unauthorized.** Do not interpret an offline SDK
test or the prior Task 09 scientific readiness as verified online monitoring.

## Implemented behavior

The online successor is `repair-selfplay-2026-main-online-v1` with
`configs/task10-online.yaml`. Scientific settings excluding study ID, model,
data, prompts, images, hardware and conservative throughput settings match
Task 09 v2 exactly. Scientific code identity remains
`e3260b72cdfe3fa41a2f9ef509343f3c3143bf624f0d8faa047871de6b118e8f`.
Task 09 freeze 31197007 and execution `93a8533beeae26e54bb1510caa12991fb2471337`
remain intact as the separately frozen baseline.

Implemented aggregate-only, checkpoint-backed W&B projection, stable logical
run IDs and names, explicit iteration/physical-GPU-hour axes, bounded background
notifications/reconnect waits, private raw examples, canonical rollback tables,
clear stale summaries, and checkpoint-verified reconciliation. The pinned SDK's
custom charts are logged through history rather than assigned directly to
summary. Main previews require allocated-node cloud/readback/resume proof bound
to the frozen telemetry implementation and a persistent secrets-file path.

Added standard `wandb login` credential provisioning, `wandb-verify`,
`wandb-reconcile`, CPU-only controller verification, dependency-checked
`hpc reuse-study`, saved-token `numerical-audit`, and health cutoffs at 5/10/20.
Slurm accounting sampling counts all attempts once without double-counting
GPU TRES. No worker, objective, prompt, sampler or optimizer implementation was
changed. There is no new automatic scientific-underperformance stop.

## Validation and operations

Local CPU suite passed 239 tests with 33 environment-dependent skips and three
deselections before the latest provenance regression; lint and mypy passed.
Focused tests cover private upload boundaries, checkpoints, rollback, stable IDs,
SDK failure isolation, pinned SDK table/chart serialization, scientific RNG,
all A–D numerical counterfactuals without steps, accounting, and exact reuse.
Hosted Linux/Docker CI is still being monitored. CPU tests alone do not prove
online W&B, H200 numerical results or worker readiness.

Dependency-checked evidence reuse from Task 09 succeeded. CPU freeze attempt
31205612 failed before publishing a freeze because the allocation's
`SELFPLAY_MAIN_STUDY` redirected the historical smoke-report source lookup.
The lookup is now explicit and has a regression test under the real environment
variable. No historical report/checkpoint was changed to make reuse pass.

Standard W&B login was opened in a local Terminal. Local and remote credential
inspection initially found no W&B credentials. User supplied username `tatsu432`
and candidate entities `tatsu432`/`tatsu432s`; the authenticated API must establish
the actual entity. Cloud URLs will only be reported after cloud verification.

Live Slurm inspection found the cpsc4880 association with interactive/normal
QOS. `gpu_h200` has 72 GPUs across nine nodes; partition QOS `part_gpu_h200`
reports a 48-hour wall limit, 16 GPUs/user and six jobs/user. These reported
limits permit A–D independent allocations in principle, subject to inherited
limits, current scheduling, fairshare and explicit authorization. User fairshare
was 0.810131 at inspection; this is not a queue-time guarantee.

## Numerical assessment

Static inspection retains exact sampled action IDs; causal target logits use
positions immediately preceding each target, left padding has explicit masks
and positions, teacher replay keeps exact targets/context suffixes, and policy
versions bind adapters. vLLM samples at temperature 1, top-p 1, top-k disabled,
min-p 0 and repetition penalty 1 and extracts the sampled token's log probability.
No correctness defect warranting denominator replacement has been identified.
BF16/kernel differences are a plausible explanation, not an established cause.

Existing H200 evidence has initial/later mean absolute discrepancy
0.00369/0.00330, p99 0.1004/0.0996, max 1.138/0.728 and clip-interval crossings
18/3621 and 21/4054 tokens. One mixed group showed 9.9% relative RL-gradient L2
change with cosine 0.9985 under a trainer-rescored counterfactual. These counts
alone do not measure OPSD or C/D reward effects. The new bounded replay covers
those consequences using complete saved groups, both roles and multiple
snapshots without generation, candidate execution or optimizer steps.

Replacing old probabilities changes both the GRPO importance estimator and the
teacher-minus-old OPSD coefficient/gate. The original behavior denominator is
retained. No scientific correction has been adopted.

## Runtime scenarios and staged authorization

| Method | Pilot-yield GPU-hours | Planning GPU-hours | First main measured rate | Projected finish |
|---|---:|---:|---|---|
| A | 17.69 | 82.06 | Not run | Not established |
| B | 17.33 | 79.94 | Not run | Not established |
| C | 18.00 | 84.82 | Not run | Not established |
| D | 15.80 | 97.95 | Not run | Not established |

Total planning scenario: 344.78 GPU-hours including monitoring, excluding final
evaluation. Estimates are conditional, not upper bounds. Queue delay is
separate. Preserve 23-hour segments, per-iteration joint checkpoints, stop leads
4616/4540/4717/5211 seconds, clean continuations 5/5/5/6 and two bounded transient
retries. Use manual verified continuation after SSH loss. Main A first requires
explicit GPU-hour authorization; B/C/D require review and separate authorization.

The [launch/recovery workflow](../task10-launch.md) contains copyable commands,
the metric mapping, privacy boundaries, interpretation and missing measurements.
Executed versus prepared commands and final live evidence will be recorded here
before declaring Task 10 complete.
