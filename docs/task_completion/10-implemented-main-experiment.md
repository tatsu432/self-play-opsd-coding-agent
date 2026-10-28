# Task 10 — main launch engineering

Status: online monitoring, freeze and four actual main previews are validated;
final numerical assessment is running. **Main A–D training has not been submitted
and remains unauthorized.**

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

Local CPU suite passed 240 tests with 33 environment-dependent skips and three
deselections; lint and mypy passed. Subsequent controller/rollback checks passed
39 focused tests.
Focused tests cover private upload boundaries, checkpoints, rollback, stable IDs,
SDK failure isolation, pinned SDK table/chart serialization, scientific RNG,
all A–D numerical counterfactuals without steps, accounting, and exact reuse.
Frozen-source hosted Linux/Docker CI **40784025848** passed **272 tests**,
one skipped and three deselected. Controller CI **40784625055** also passed.
The opt-in cloud sink test passed all four methods with real cloud readback.
A subsequent injected delivery-interruption test verified reconnect/replay. CPU tests alone do not prove
online W&B, H200 numerical results or worker readiness.

Dependency-checked evidence reuse from Task 09 succeeded. CPU freeze attempt
31205612 failed before publishing a freeze because the allocation's
`SELFPLAY_MAIN_STUDY` redirected the historical smoke-report source lookup.
The lookup is now explicit and has a regression test under the real environment
variable. No historical report/checkpoint was changed to make reuse pass.

CPU freeze **31206522**, controller run `7e9a7c9ab6f73107387bdd04`, completed
successfully with execution **`54f087fdc5cf5170800d825bf1ee92b89a43c8ed`**.
All scientific readiness gates pass with unchanged dependencies. Resource-plan
hash is `47ed0332e6f873022a5aa743b63c5aeb10d53c33ba9981f56dc0d49d3f0544cb`;
frozen telemetry identity is
`dfdb26ba5320da92155a5ccf7e754725ad201211724b1953ffae1c83ccbcd9ef`.
The controller separately blocks online launch when cloud/resume proof is absent.

W&B authentication is verified as username **`tatsu432`**, default entity
**`tatsu432s`**. The 0.22.2 interactive login rejected the new 86-character key;
login succeeded with the isolated **0.22.3** tool. The frozen 0.22.2 trainer SDK
then demonstrably authenticated and delivered metrics with the key supplied
through its supported environment path. No dependency lock or SIF was changed.
Credentials are in user-owned mode-600 files outside Git, including the persistent
Yale secrets path; no secret values were printed, uploaded or passed as arguments.

CPU-only compute job **31208617**, controller run
`5c7e1e247ccfdfe81889b225`, passed on **c1104u05n01**. HTTPS and actual cloud
history were verified; a second independent process resumed the same run and
read back both sequence values `[1, 2]`. This proof matches the frozen telemetry
identity. Actual URLs:

- [Project](https://wandb.ai/tatsu432s/self-play-opsd-coding-agent)
- [Yale compute-node verification](https://wandb.ai/tatsu432s/self-play-opsd-coding-agent/runs/telemetry-check-ffbb587a663649ed)
- [Local transport verification](https://wandb.ai/tatsu432s/self-play-opsd-coding-agent/runs/telemetry-check-0594601936814586)

Four actual cloud sink fixtures verified display names, common grouping, a
checkpoint restart, a previously uploaded iteration rollback, corrected canonical
row counts, and replay to iteration two. Their group is explicitly
`task10-synthetic-telemetry-verification`; these are **synthetic validation data,
not main training outcomes**:

- [A validation](https://wandb.ai/tatsu432s/self-play-opsd-coding-agent/runs/task10check-5a4b972d36cbA)
- [B validation](https://wandb.ai/tatsu432s/self-play-opsd-coding-agent/runs/task10check-af0d1b01e731B)
- [C validation](https://wandb.ai/tatsu432s/self-play-opsd-coding-agent/runs/task10check-1f32a8443a96C)
- [D validation](https://wandb.ai/tatsu432s/self-play-opsd-coding-agent/runs/task10check-1db95b379bb0D)

All four actual online main previews passed with `submitted=false`, frozen
execution `54f087f`, 200 iterations and the measured resource envelope:

| Method | Controller preview ID | Stop lead (s) | Clean continuations |
|---|---|---:|---:|
| A | `a87f3772b284d35a375cf69e` | 4616 | 5 |
| B | `0b05980057d59ae164ab70bd` | 4540 | 5 |
| C | `b3375c49bcf2afb16abd7860` | 4717 | 5 |
| D | `ca8427236f53c75161749aab` | 5211 | 6 |

Actual main run URLs do not exist yet; they are emitted after the authorized
main starts. No synthetic fixture was inserted into a main namespace. Main
A/B/C/D artifact directories and submission ledgers remain absent.

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

Replay attempt 31207286 was cancelled while pending (zero allocated seconds)
to correct saved-proposal parsing. Attempt 31207388 failed before model load
because the new CLI hardware argument was positional while the controller used
`--hardware`; it charged 92 seconds (0.025555556 GPU-hours). This avoidable
launcher defect was fixed and has a direct CLI regression. The replacement
**31208618**, run `ba904652d21c9dfa5205bbda`, has a **50-minute cap**, keeping
all actual Task 10 diagnostic allocations together below one GPU-hour. It is a
manual corrected launch after confirmed failure, not an automatic retry or a
repeat for a different stochastic outcome. The replacement performs no
sampling, candidate execution or optimizer steps.

Initial and later saved-token replay statistics reproduce Task 09 exactly.
New proposer measurements are mean absolute discrepancy 0.002691/0.002187,
p99 0.07377/0.04848, maximum 0.4873/0.3087, with 9/1690 and 5/1228 tokens
outside the ratio clipping interval. Complete RL/OPSD/C/D consequences are
being measured by the same bounded allocation.

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
Executed: synchronization, evidence reuse, two CPU freeze attempts, local and
allocated-node cloud verification, the opt-in four-method cloud sink test,
four main previews, fetch/status/log inspection, and bounded numerical diagnostics.
Prepared only: actual main A/B/C/D submissions, their continuation/watch commands,
and final evaluations. Main spending and final-evaluation spending require
separate authorization. Branch: `feat/10-main-online`; [draft PR #1](https://github.com/tatsu432/self-play-opsd-coding-agent/pull/1).
No persistent scheduler service, network tunnel or login-node daemon was added.
