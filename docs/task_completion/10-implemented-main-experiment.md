# Task 10 — main launch engineering

Status: **the user accepted the numerical sensitivity for a bounded initial A run; main GPU-hour authorization remains required.** Online monitoring, mechanical readiness, freeze, recovery and four
actual main previews are validated. Main A–D training remains unauthorized and
unsubmitted. The numerical investigation is complete within its bounded budget;
it found material proposer sensitivity, not a proven indexing/serialization bug.

| Decision item | Result |
|---|---|
| Numerical issue | Quantified; material proposer gradient/update sensitivity remains |
| Training/checkpoint/recovery mechanics | Passed existing real evidence and new scoped tests |
| Frozen scientific design | Preserved exactly; no denominator/objective correction adopted |
| Live W&B | Verified from allocated compute and four synthetic method fixtures |
| Compute plan | Bound to freeze; conditional scenarios remain reasonable, not upper bounds |
| Main launch | Numerical sensitivity accepted for bounded A; separate GPU-hour authorization required |

The user accepted the documented estimator sensitivity for the bounded initial A run.
The controller's eligibility flag measures its existing immutable readiness gates;
it is currently true. It does not supersede this Task 10 scientific recommendation
or provide spending authorization. No new arbitrary success-rate or automatic
scientific stopping gate was introduced.

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
`hpc reuse-study`, saved-token `numerical-audit`/`proposer-audit`, and health
cutoffs at 5/10/20.
Slurm accounting sampling counts all attempts once without double-counting
GPU TRES. No worker, objective, prompt, sampler or optimizer implementation was
changed. There is no new automatic scientific-underperformance stop.

## Validation and operations

Final local CPU suite passed **244 tests**, with 33 environment-dependent skips
and four deselections; lint, format checks and mypy passed. The separate opt-in
real cloud test passed. Scoped proposer/CLI/controller tests passed 19 tests,
and the injected reconnect/SDK test passed.
Focused tests cover private upload boundaries, checkpoints, rollback, stable IDs,
SDK failure isolation, pinned SDK table/chart serialization, scientific RNG,
all A–D numerical counterfactuals without steps, accounting, and exact reuse.
Frozen-source hosted Linux/Docker CI **40784025848** passed **272 tests**,
one skipped and three deselected. Controller CI **40784625055** also passed.
Final diagnostic-source hosted CI **40788699300** and PR CI **40788704401**
also passed for commit `3798eea`.
The opt-in cloud sink test passed all four methods with real cloud readback.
Offline aggregate archive creation followed by online reconciliation also passed
with cloud revision readback ([synthetic recovery run](https://wandb.ai/tatsu432s/self-play-opsd-coding-agent/runs/task10recovery-0ba19f4105fe)).
An injected delivery-interruption test verified reconnect/replay. CPU tests alone
do not prove online W&B, H200 numerical results or worker readiness; the actual
cloud and allocated diagnostics supply those dependency-scoped observations.

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
**31208618**, run `ba904652d21c9dfa5205bbda`, completed in **911 seconds**,
with a 50-minute cap. It is a
manual corrected launch after confirmed failure, not an automatic retry or a
repeat for a different stochastic outcome. The replacement performs no
sampling, candidate execution or optimizer steps.

Initial and later saved-token replay statistics reproduce Task 09 exactly.
New proposer measurements are mean absolute discrepancy 0.002691/0.002187,
p99 0.07377/0.04848, maximum 0.4873/0.3087, with 9/1690 and 5/1228 tokens
outside the ratio clipping interval. The same allocation completed the full student/OPSD/alignment assessment.
All comparisons below use saved D snapshots; method labels denote objective
counterfactuals at that snapshot, not four separately trained main runs.

| Student objective | RL cosine / relative L2 | OPSD cosine / relative L2 | Total cosine / relative L2 |
|---|---|---|---|
| A | 0.99852 / 10.31% | 0.99763 / 7.25% | 0.99835 / 9.63% |
| B | 0.99853 / 10.13% | 0.95193 / 30.63% | 0.99857 / 10.21% |
| C | 0.99852 / 10.16% | 0.95189 / 30.65% | 0.99857 / 10.37% |
| D | 0.99852 / 10.44% | 0.95178 / 30.68% | 0.99858 / 10.01% |

Relative L2 values use the trainer-rescored vector norm as denominator; they
are counterfactual sensitivities, not errors against a ground-truth gradient.

The mixed snapshot contains 3 tasks, 12 episodes, 93 turns and 5,397 tokens.
There were 41 tokens outside the ratio clip interval but only **3 actively
clipped student tokens**, because clipping also depends on advantage sign.
Mean/max absolute gate change was **0.000556 / 0.16314**; gap-clipping membership
changed for two tokens. C's maximum reward change was **0.0000271**; D's was
**0.0078613**. Two equal-reward C candidates remained exactly zero with neutral
reward. D's two OPSD-only candidates had relative gradient changes about 38–39%,
although alignment signs and candidate reward ordering were preserved.

Student first-token errors were tiny (max 0.00114/0.00087 at initial/later
snapshots), with outliers predominantly later. Proposer first-token errors were
larger in those snapshots (max 0.45065/0.24215), so student boundary agreement
cannot be generalized to the proposer. These saved contexts were all below
2,048 tokens; full-budget long-context numerical behavior remains unmeasured.
No unsupported long-context or bitwise-parity claim is made.

### Proposer consequence and bounded confirmation

The first replay showed a large proposer counterfactual effect even with fixed
difficulty rewards. This justified **one focused follow-up**, CPU-tested before
submission, using the same saved proposals and actual stored D rewards.
Job **31212941**, controller run `3a1da2fa9d3e4e65207a3deb`, completed in
**120 seconds** with a 30-minute cap. Each hypothetical Adam step restored both
adapters, optimizer state and RNG from the trusted iteration-one joint checkpoint;
the original proposer version was verified after restoration. No updated model
or checkpoint was promoted. Original source optimizer step counts were one.

| Fixed reward case | Behavior / rescored gradient norm | Gradient cosine | Adam delta cosine | Adam delta relative L2 |
|---|---|---:|---:|---:|
| Actual saved D rewards | 0.24680 / 0.01609 | **−0.77112** | **0.58968** | **97.38%** |
| Difficulty counterfactual at the same D checkpoint | 0.25364 / 0.00479 | 0.00819 | 0.67645 | 87.49% |

Eight proposer examples had nonzero advantages, covering 558 active tokens.
Only **one active token** was clipped with behavior denominators. The
trainer-rescored denominator gave exactly zero pre-update log-ratio discrepancy
on these tokens, ruling out a train/eval scoring mismatch in this sample.
Repeated behavior-denominator gradients were stable (cosines 0.99982/0.99986,
relative L2 1.89%/1.69%). Rescored gradients were much smaller and cancellation
sensitive: repeat cosines 0.89938/0.04002, relative L2 45.0%/138.2%.

This is **estimator sensitivity**, not a proof that the behavior-denominator
implementation is mathematically wrong. The behavior denominator represents the
actual sampler and must not be replaced merely to force ratios to one. However,
the measured effect is too large to label the proposer discrepancy harmless.
The exact BF16/attention/LoRA kernel source was not isolated. Indexing, masks,
serialization, sampled IDs and snapshot identities passed the scoped audit.
The user explicitly accepted this sensitivity for the bounded initial A run.
Any later scoring-consistency correction requires a new immutable study and
affected validation.
No additional broad readiness cycle is recommended, and no correction was
silently adopted.

### Final accounting

The existing locked controller reconciler durably updated the shared ledger:
**7.790000000 / 24 allocated GPU-hours consumed**, zero outstanding reservation,
no active user Slurm job and no watcher. Task 10 GPU use was **0.311944444 hours**
(92 + 911 + 120 allocated seconds); the cancelled pending job used zero.
Authorization remains `cpsc4880-readiness-24h-v1`. No allowance was reset.
CPU freeze and W&B checks used no GPUs.

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
four main previews, fetch/status/log inspection, and bounded numerical diagnostics including restored Adam updates.
Prepared only: actual main A/B/C/D submissions, their continuation/watch commands,
and final evaluations. Main spending and final-evaluation spending require
separate authorization. Branch: `feat/10-main-online`; [draft PR #1](https://github.com/tatsu432/self-play-opsd-coding-agent/pull/1).
No persistent scheduler service, network tunnel or login-node daemon was added.


## Code and validation references

Branch `feat/10-main-online`; draft PR #1 contains the complete diff. Key commits:
`2fd0d3b` initial telemetry/controller implementation; `54f087f` frozen operational
execution; `f868b2e` diagnostic CLI/cap correction; `6616306` four-method real cloud
integration test; `3798eea` proposer repeatability and restored-Adam diagnostic.
The original frozen source and SIFs remain available.

Changed interfaces: `wandb-verify`, `wandb-reconcile`, `hpc reuse-study`,
`numerical-audit`, `proposer-audit`, and `scientific_health --through-iteration`.
Changed subsystems: aggregate telemetry projection/sink, controller/standard-library
allocation endpoint, evidence reuse verification, CLI, scoped diagnostics,
regression/integration tests and operational documentation. `trainer.py`,
`policy.py`, `verl_backend.py`, `vllm_worker.py`, objectives, protocol, model/data,
`pyproject.toml`, `uv.lock` and container definitions were not changed.

Primary artifacts remain on Yale and were fetched under their controller run IDs.
Numerical input hashes: episode shard
`95ebcde968f4007390422dfaebbcb0ef719cdfb16343664a21c642bc204c2ae6`,
proposal shard `c5af99a252b515a0c8493b56672bcae9f77f245c75d749ce7df1c7896ba67060`,
joint checkpoint marker `a38386c566e109f19c7f557003d47c74764ed9bea3dea54d1c42cbef825197ac`.


Final report file SHA-256 values: `numerical-audit.json`
`06c5e2e0aca8bb995985cb4e74b4d154190484529de56842e387c6b3e56db458`;
`proposer-audit.json`
`a85958f5cb87d51dae9973cfa7206cfbee42f580e2d27e326d97d023e9a91e63`.
Optional batching, CUDA Graphs, prefix caching and larger scoring microbatches
remain disabled. Preserved older freezes must not be used to bypass the new
numerical review; the measured backend sensitivity is not removed by reverting
operational documentation or logging configuration.


## Bounded A launch preparation

The existing frozen trainer accepts a stop-at-iteration control without changing its
200-iteration scientific configuration. The controller exposes `--stop-after-iteration`
and checks its range; the persisted submission retains it across transient retries.
A clean stop at 20 writes a verified joint checkpoint. Only an explicit later
`resume --continue-stopped` clears that stop; no watcher or automatic continuation
is authorized for the initial allocation. Main A requires separate explicit
GPU-hour authorization before submission.

The actual controller preview for bounded A passed with `submitted=false`, run
ID `a87f3772b284d35a375cf69e`, frozen execution `54f087fdc5cf5170800d825bf1ee92b89a43c8ed`,
`target_iterations=200`, `stop_after_iteration=20`, one H200, 23-hour cap and
4,616-second stop lead. Controller commit `2df6902` passed 88 scoped tests.

## Authorized initial A allocation

The user authorized one H200 allocation with a 23 GPU-hour ceiling, stopping at
iteration 20; no automatic continuation or B/C/D is authorized. Submitted Slurm
job `31219895`, controller run `a87f3772b284d35a375cf69e`. It retains the frozen
200-iteration scientific configuration, source `54f087fdc5cf5170800d825bf1ee92b89a43c8ed`,
and explicit `stop_after_iteration=20`. Review W&B and health at 5, 10 and 20.
Inspect this existing ledger before any further scheduler action.

### Initial authorized Experiment A stage — 2026-10-28

User authorized one H200 allocation, at most 23 GPU-hours, bounded to iteration 20; no automatic continuation or B–D launch. Slurm job 31219895, run `a87f3772b284d35a375cf69e`, completed with exit 0 after 5,500 seconds (1.527778 allocated GPU-hours). The run health reports `stopped` at iteration 20, with no iteration 21; the single ledger attempt had no retries. The frozen 200-iteration scientific target remains unchanged. The latest pointer identifies joint checkpoint `000020` with a `complete.json` checksum manifest and `fully_resumable` metadata. Its file checksums and identity must be verified by the existing controller before any future resume; this closeout did not independently rehash the large checkpoint files.

The live [W&B run](https://wandb.ai/tatsu432s/self-play-opsd-coding-agent/runs/9c125e660c942541076d5d50) finished with checkpoint 20, 20 canonical training rows and zero reported delivery errors. At iteration 5, 8/80 proposals were valid and 8/32 solver episodes succeeded; both optimizers had five steps, with two mixed-outcome groups. At iteration 10, 16/160 proposals were valid and 20/64 episodes succeeded; each optimizer had nine steps, with two mixed groups. At iteration 20, 30/320 proposals were valid (9.375%) and 40/120 episodes succeeded (33.3%); 19 task groups were all-zero, nine all-one, and two mixed. Each optimizer stepped 15 times; the other five iterations had no valid proposals. Mean proposer reward across the 320 equally sized proposal batches was -0.9015625. The main invalid reasons were hunk count mismatch (78), invalid hunk line/extra file (65), duplicates (53), and patch context mismatch (51).

Student total gradient norms at update iterations were finite (maximum 1.0978); mixed outcomes were rare, and no separately sampled RL/OPSD gradient interaction is available: iteration 20 produced no valid tasks. The frozen behavior-policy probabilities and objectives were unchanged. Initial greedy development evaluation was 0/41; at iteration 20 it was 12/41, with parent-macro 0.347826 versus 0 initially. This is development evidence, not a held-out test result.

Slurm accounting includes startup and both development evaluations. Phase timing attributed approximately 2,239 seconds to the two development evaluations. A rough projection from this stage, assuming the same training workload and evaluation time, is about 12.5 GPU-hours for 200 iterations plus 11 development evaluations, before extra continuation startup and final diagnostics. Valid yield and generation length can change materially, so this is not a guarantee. W&B system event history contained no GPU utilization samples; per-iteration device memory samples are not GPU utilization and are not an allocation-wide peak. No continuation was submitted or authorized.

### Experiment A: 200-iteration completion — 2026-10-29

The user authorized continuation of A toward the frozen 200-iteration target. After checkpoint verification, continuation job `31232768` ran on one H200 and Slurm recorded `COMPLETED`, exit `0:0`, elapsed 22,229 seconds. Together with initial job `31219895` (5,500 seconds), allocated time was **27,729 seconds / 7.7025 GPU-hours**, excluding queue time. The durable trainer health reports `complete` at iteration 200, and `checkpoints/latest.json` points to `000200`. No B–D job or further continuation was submitted. The end of the Slurm log contains step-termination messages during cleanup despite the successful job exit and completed trainer state; do not infer a failed training run from those cleanup lines alone.

Fetched local scientific events contain 200 committed iterations. Of 3,200 attempted proposals, 173 were valid (5.40625%); the solver succeeded in 162 of 692 episodes (23.4104%). Across 173 outcome groups, 130 were all-failure, 38 all-success and five mixed. Each optimizer stepped 111 times. Main rejection counts were duplicate 1,103, hunk-count mismatch 730, invalid hunk line or extra file 627, and patch-context mismatch 317. Six sampled RL gradients were zero, with corresponding sampled lambda-scaled OPSD norms approximately 0.13630, 0.07061, 0.001800, 0.25197, 0.000149 and 0.0000297; these are sampled batches, not evidence that every RL gradient was zero.

Development results, task successes out of 41 at iterations 0/20/40/60/80/100/120/140/160/180/200, were **0/12/12/10/9/0/0/0/0/0/0**. The best monitoring-development parent-macro score was 0.347826 at iteration 20, with the prescribed earlier-iteration tie-break. The primary final-iteration development result was 0/41. These are development outcomes, not the locked final in-domain or external test evaluations; neither final test has been reported. A–D comparison and causal attribution remain unsupported until the remaining methods and evaluations are run under the frozen protocol.

W&B cloud delivery stopped at checkpoint 139, despite training continuing to 200. Remote `state/telemetry.json` records 86 delivery errors and `sdk_delivered_iteration: 139`; its `local_data_preserved` flag is true. The W&B URL remains https://wandb.ai/tatsu432s/self-play-opsd-coding-agent/runs/9c125e660c942541076d5d50 but its latest cloud metrics are incomplete. A trusted aggregate event fetch succeeded. The full checkpoint-inclusive recovery fetch was interrupted due to transfer size (about 7.6 GiB on Yale); no cloud reconciliation was performed. Before any W&B replay use the documented single-writer recovery procedure in `docs/task10-launch.md`, verify the checkpoint and source identity, and avoid bulk-syncing raw trajectories. The previous response claiming the run stopped at iteration 139 was based on stale cloud telemetry and is superseded by Slurm and trainer records above.
