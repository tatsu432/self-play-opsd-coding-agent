# Task 08 — Verified launch readiness

Completed on 2026-10-28 on `feat/05-yale-experiment-readiness`. **Launch readiness
is true. All ten readiness gates and four actual controller submission previews
pass. No main A–D job was submitted.** Main training still requires separate user
authorization.

## Implementation and validation

The shared format clarification resolved the empty-workload blocker while
retaining the strict parser. Historical D had 154 header errors, five JSON errors
and one hunk-count error in 160 outputs; none reached 512 tokens. Removing Git
preambles in an offline diagnostic still produced no valid mutation. Historical
student trajectories showed source reads followed by malformed edits, without
observation truncation. See the [saved-output diagnosis](../task/08-historical-protocol-diagnosis.md).

Protocol `repair-json-v2` specifies exact file headers/path, line prefixes, hunk
counts/context, final newline and JSON escaping, with one unrelated syntax-only
example shared by student and proposer. It preserves the four tools and sampled
IDs/probabilities. No semantic repair, constrained decoding, refill, model/SFT,
sampling, reward or mathematical-budget change was made.

Greedy evaluation now reaches generation on every turn in both backends.
Explicit pilot instrumentation replaces the study-name suffix check. The complete
protocol identity covers templates, output descriptions, the example and relevant
serialization, with benign comments/docstrings excluded. A stable authorization
ledger migrates all historical budgets under the existing lock, deduplicates only
proven identical jobs, and reserves unresolved accounting. A study revision
cannot reset either allowance.

A/B/C acceptance is one fixed three-iteration window per method. Nonempty real
work, finite objectives, gates/alignment, actual nonzero-gradient update paths
and joint checkpoints remain required. Separately labelled state-restored fixtures
verify optimizer mechanics when rewards legitimately give zero advantages.
Every actual A/B/C iteration here stepped both optimizers. Fixtures were not
needed as substitutes for those steps; their 140.46/153.65/150.44 seconds remain
charged to readiness and are excluded from main throughput projections.

Key commits:

- `0138dbc`, `a291098`, `e1836a0`: protocol, greedy/diagnostic fixes, shared budget,
  immutable revision, probe and bounded acceptance.
- `7866cad`, `3961fcf`, `0465b20cc8bd47fac72c99b912e67a5232050796`: corrected pilot
  counters, per-iteration checkpoints, process attribution and revised-data reuse;
  the real D pilot used `0465b20`.
- `b86fb0a`, `d5ac499`, `6af9fc31d432ce708ce25c6b5b2a0b8310af962d`, `8cce9b6`:
  conditional resource forecasting, evidence-bound freeze validation and monitoring
  stop margins. A/B/C used `6af9fc3`; CI `40636326117` passed before submission
  with identical A/B/C execution files, and exact-source CI `40636150801` also passed.
- `296ee6cc723c2806eb787412c21cc9ec89fdabd5`,
  `983a6cf57ce5caf5c9aa7ddf53f5ad7f313b4e73`: raw exposure/action auditing and
  reliable report transfers. Default fetch omits tensors and normalizes inherited
  directory modes for macOS openrsync. The frozen execution source is `983a6cf`.
- `25b07f230ef66a7136c51c6da4668ac6cecf9f4d`: main preview metadata now consistently
  says 200 iterations. The unused field previously showed the pilot CLI default
  while the training settings and target already specified 200. A regression
  checks both main/pilot budgets and frozen resource-plan propagation. Four new
  previews pass; the first results remain in `runs/task08/first-main-previews`.

Hosted [CI 40627889719](https://github.com/tatsu432/self-play-opsd-coding-agent/actions/runs/40627889719)
passed 238 tests. [CI 40636863255](https://github.com/tatsu432/self-play-opsd-coding-agent/actions/runs/40636863255)
passed 242 tests, one skipped and three deselected, including lint/format, mypy
and Linux/Docker checks. Focused final controller regressions passed 14 tests;
real nested report retrieval and strict preview consistency checks also passed.
[Frozen-source CI 40639028520](https://github.com/tatsu432/self-play-opsd-coding-agent/actions/runs/40639028520)
passed 242 tests, one skipped and three deselected.
[Final controller CI 40640124480](https://github.com/tatsu432/self-play-opsd-coding-agent/actions/runs/40640124480)
passed 244 tests, one skipped and three deselected on `25b07f2`, including the
main-budget/resource-plan regressions and the full Linux/Docker suite.

Earlier CI runs 40626463095/40626756410 exposed a sampler/subprocess mock race.
The launch-boundary mock was isolated and dedicated sampler attribution coverage
added. Historical rsync directory-mode errors did not invalidate the separately
verified checkpoint bytes; normalized receiver modes now make those transfers
complete successfully.

## Frozen scientific and data identities

Final settings: `configs/task08.yaml`. Study: `repair-selfplay-2026-protocol-v2`.
Model/tokenizer: `Qwen/Qwen3-4B-Instruct-2507`, revision
`cdbee75f17c01a7cc42f958dc650907174af0554`. Seed 2026, 200 main iterations,
4 parents × 4 attempted proposals, 4 episodes per valid task, 512 proposer tokens,
4096 student generated tokens/episode, 8 turns and all sequence/teacher budgets
remain unchanged. A–D objectives, gates, normalization, rewards and lambda 0.1
remain unchanged. Main initialization is fresh.

| Identity | Value |
|---|---|
| Protocol/prompt | `repair-json-v2` / `07968cec687ca52d8ce02c4a6d52eaca529f830855690cab1caff71c096f4322` |
| Data root | `data/protocol-v2` |
| New manifest file SHA-256 | `934412b2181ead4377d383130a19ad0d58eaab6bb9f74c5d8005416d09412fd2` |
| New canonical manifest | `d9140367e20587051b328dda20a7d93916ee5cadc7e72c2564523cf6ab6beade` |
| Original manifest file SHA-256 | `44fe6caffc0548bebb01bf39208f8b2f5de10ab89ee8487dca20dd436c0154f5` |
| Original protocol hash | `ec1f46db1df8afe308934c54222a37cb6c9d73d072995699e19ea9419ccfd9c5` |
| Scientific code | `d8526f680fbbe3aff938d8688307817bb25ff496cd3d1cc02de3daecb5a5c6e1` |
| Allocation verifier | `89730a635e803a168ac7285b1261396bda723aaaccdf664bf8a344dce86d6fc9` |
| Backend compatibility | `c819f3b09a746a9c311f8e490cd265746cb1c883ae6a8d3b5cabbcb4abdcfe42` |
| Trainer SIF | `fbe2737a6815644ecaad01df40cfa7883b05e5554d518fd0352129b1f6fd69e2` |
| Worker SIF | `76a43eef3424a6e72d516a73c967d0610e12c13288eba6cbd762d8635e860ec3` |
| Frozen record file SHA-256 | `1cac52dbce24fc4c82c69984b86b7665f1da7494ed8d1c7a0efe3ab50875f3f4` |
| Resource-plan file SHA-256 | `880bc7ee522746bc4fbcf44de49719b8ab2add1572cd92f56939645790da49f2` |
| Embedded resource-plan digest | `f6ef066f0a7f8df3ac13a45537644b364ef531fc2c62305105fee34a35e949f5` |

The targeted immutable revision preserved every parent/task/oracle/split/coverage
file byte-for-byte and regenerated prompt-dependent anchors/admission. All 306
tasks passed: 14 reference, 41 monitoring, 20 diagnostic, 115 final-ID and 116
external. No task was dropped, substituted or moved into training. Original
`data/prepared` is intact. Embedded task protocol v1 identifies creation history;
active interaction uses v2. Final/external repair outcomes were not used to tune
the clarification.

## Real acceptance and report locations

All listed new jobs completed successfully. Paths below are relative to
`/nfs/roberts/project/cpsc4880/cpsc4880_ts2236/selfplay-opsd-cpsc4880_ts2236/runs`.

| Stage | Job / controller run | Result and report |
|---|---|---|
| Worker | 31033392, reused | All 19 checks; original report SHA `b0b4781ee150260269690f074f2f509b24998c243800ce36383acf34988ebe8c`; `readiness/repair-selfplay-2026-protocol-v2/worker.json` retains provenance |
| Fixed probe | 31069728 / `7f6319f6b8b125ac11eaa7af` | 32 attempts, 5 valid tasks, 20 episodes/9,252 tokens, meaningful read/edit/test; no updates/refill; `repair-selfplay-2026-protocol-v2-protocol-probe/B-seed2026/protocol_report.json` |
| Immutable revision | 31069732 / `d4630697f6f230e3021769b7` | 306 admissions/verifications and exact original-file equality; `data-revisions/repair-selfplay-2026-protocol-v2/revision_report.json` |
| Runtime | 31071377 / `1efeedad6e8ff2df7ca56863` | Deployed 306-task/isolation proof; `readiness/repair-selfplay-2026-protocol-v2/runtime.json` |
| Backend | 31071410 / `1b79a8543eecd835654471e5` | Numerical A–D objectives, adapter synchronization, vLLM and checkpoint acceptance; `compatibility/backend-v2/single_gpu_80gb/backend_report.json` |
| Corrected greedy base | 31072777 / `7e3983549225a9d7726d8ae0` | Complete 0/41 repairs; `repair-selfplay-2026-protocol-v2/base` |
| D stop/resume | 31075857 → 31077825 / `a82cea05e53bc66d57fc1ad0` | Ten iterations, verified joint resume at two; `repair-selfplay-2026-protocol-v2-pilot-h200/D-seed2026/pilot_report.json` |
| A | 31080094 / `7c8141f2d8a76ab8486e5d4d` | Three iterations; `readiness/repair-selfplay-2026-protocol-v2/method-smokes/A.json` |
| B | 31080096 / `5bd55fdc22479238a01b97ff` | Three iterations; same report directory, `B.json` |
| C | 31080097 / `fbaa95b19742b5e25f48a522` | Three iterations including alignment; same report directory, `C.json` |
| Freeze | 31082462 / `83eb2b0a3e002a341476858a` | CPU only; all ten gates true; `repair-selfplay-2026-protocol-v2/study_frozen.json` and `readiness.json` |

A/B/C trajectories/events live in `repair-selfplay-2026-protocol-v2-method-smoke/M-seed2026`.
Local reports, raw-shard checksums, accounting, initialization proof and previews
are under `runs/task08`; general controller fetches are under
`runs/hpc/yale-hpc-cpsc4880/RUN_ID`. Detailed measurements and estimator inputs are
`runs/task08/final-analysis/workload_measurements.json` and `resource_plan.json`.
Raw Slurm accounting for all 31 GPU-history jobs and CPU freeze is retained in
`runs/task08/scheduler-accounting.txt`.
`scripts/analyze_readiness_workload.py` reproduces their construction from fetched
raw records, phase events and reconciled accounting.

The D iteration-two joint checkpoint was archived before pruning at
`repair-selfplay-2026-protocol-v2-pilot-h200/D-seed2026/acceptance/iteration-two-checkpoint`.
Its state SHA is `9840184af5963199d33e083a8c2f45f8a87955f1cdebdba72157f5871d3dcd8d`.
The real continuation loaded both adapters/optimizers, RNG, curriculum and counters.
Active commits are exactly iterations 1–10; all 72 episode IDs are unique and
exposure counters equal raw totals. This verifies clean continuation, without a
claim of bitwise distributed/vLLM reproducibility. All four initial adapter pairs
match version `1e30ec499be6c996f0d4889c6bf74490e3fadc8f9b9f850b3eebda35f4c4e55d`.

Original failed runs remain intact. The prior numerical report, 1.85-GB gradient
reference and 1.06-GB joint checkpoint are archived under
`readiness/repair-selfplay-2026/backend-v2-archive-31040984`. Reused SIF/worker
proofs are dependency-scoped; prompt-dependent runtime/backend/base/learning
proofs were actually rerun. See the [dependency matrix](../task/08-evidence-dependencies.md).

## Measured learning workload

| Method | Attempted | Valid | Episodes | Student tokens | Student / proposer steps | Alignment scores |
|---|---:|---:|---:|---:|---:|---:|
| A | 48 | 6 | 24 | 9,993 | 3 / 3 | 0 |
| B | 48 | 6 | 24 | 10,451 | 3 / 3 | 0 |
| C | 48 | 6 | 24 | 10,758 | 3 / 3 | 6 |
| D | 160 | 18 | 72 | 27,994 | 10 / 10 | 18 |

D's per-iteration validity is recorded in the pilot report. Success-group mix
(numbers are successes among four episodes) was A `0:5, 1:1`, B `0:6`,
C `0:5, 1:1`, D `0:15, 1:1, 2:1, 4:1`. These are generated training-task
outcomes, not monitoring improvement estimates. The revised greedy base is 0/41;
the previous 0/41 remains historical sampled evaluation.

D diagnostics sampled every iteration: two nonzero RL norms, eight explicit
`zero_rl_norm` cases with undefined ratio/cosine represented as null. Defined
cosines were 0.3233 and −0.6285; SD/RL norm ratios were 0.03280 and 0.03439.
Lambda-scaled SD norms ranged 0.00150–0.22567. Lambda was retained. C scored six
candidates, five near-zero; D scored 18, none near-zero. Reference norms remained
nonzero. No nonfinite failure or token-mapping error occurred. Per-iteration
teacher gaps, gate distributions, reward/advantage variation and norms remain in
the measurement JSON and raw events.

Actions remain imperfect. D recorded 183 valid reads, eight valid patches, seven
valid tests, 50 valid finishes, 240 patch tool errors and 53 invalid actions.
A recorded one valid patch/test; C one valid patch; B had valid reads/finishes and
real scored updates. No new proposer output hit 512 tokens and no recorded turn
history contained observation truncation. Dominant remaining rejection reasons
are hunk/context/structural errors and duplicates. They are retained as sampled;
no invalid slots were refilled or silently corrected.

Phase totals are seconds; allocation includes startup, shutdown, fixtures and
other uninstrumented time.

| Method | Propose/validate | Rollout + scoring | Alignment | Student + proposer updates | Checkpoints | Allocated GPU-hours |
|---|---:|---:|---:|---:|---:|---:|
| A | 213.2 | 351.2 | 0.0 | 66.2 | 22.3 | 0.240000 |
| B | 210.9 | 375.8 | 0.0 | 69.7 | 22.0 | 0.251111 |
| C | 211.7 | 378.6 | 64.4 | 66.1 | 24.0 | 0.324444 |
| D | 651.1 | 1039.8 | 191.6 | 192.6 | 63.3 | 0.769722 |

D additionally spent 190.98 s on training-parent reference preparation and 296.10 s
on pilot diagnostics; C reference preparation took 199.29 s. Checkpoint p90,
per-valid-task rates and unattributed allocation time are explicit in the plan.

## Resources, memory and conditional main cost

Selected request: **one H200**, seven trainer CPUs/128 GiB plus an independently
limited no-GRES worker CPU/1 GiB, eight CPUs/129 GiB total. Ownership, container
UUID and initial-occupancy checks passed. SIFs were reused and hashed on allocated
compute nodes. This configuration has real nonempty workload evidence; there is
no measured benefit requiring additional GPUs.

| Method | Whole-device sampled peak MiB | Actor + descendant phase RSS GiB | Device samples |
|---|---:|---:|---:|
| A | 73,527 | 27.83 | 801 |
| B | 73,527 | 27.91 | 841 |
| C | 73,529 | 27.89 | 1,097 |
| D | 73,527 | 28.45 | 2,604 |

Device capacity was 143,771 MiB (140.40 GiB); observed joint peak was about
71.81 GiB, including rollout-engine startup. Trainer GPU-process peaks were about
14.1–14.5 GiB and rollout-engine process peaks about 70.65 GiB, at different times;
individual peaks must not be added as a simultaneous peak. One 14-MiB
`nvidia_test` sample is retained as unattributed. All device usage is included.
Nominal sampling is one second plus query latency, with zero query errors.
Peaks are sampled lower bounds; phase RSS sums may double-count shared pages and
exclude the separately bounded worker. Fixture intervals are excluded from the
learning-pipeline memory summary. Full-length/future high-yield memory is not
claimed to have been measured. The old trainer-only 10.46-GiB number is not used.

The following forecasts cover **200 training iterations plus monitoring** on one
H200. They are conditional scenarios, not confidence intervals or guaranteed
future costs. All-valid scenarios retain observed response lengths/reward mix;
p90 planning rates scale variable phases to all 16 valid proposals. Main
diagnostics are charged every 20 iterations, checkpoints every iteration, and
11 full monitoring passes (including zero) each conservatively cost the entire
1,234-second baseline allocation. Startup/continuation/logger/shutdown overhead
is retained through conservative per-iteration amortization. Main A/B/C diagnostic
cost uses D's measured rate. Pilot-only reference construction and mechanics
fixtures are excluded from main forecasts but remain charged to readiness.

| Method | Observed-yield hours | All-valid median-rate hours | All-valid planning hours | Stop lead seconds | Max clean continuations |
|---|---:|---:|---:|---:|---:|
| A | 17.16 | 70.26 | 76.56 | 4503 | 5 |
| B | 17.66 | 77.10 | 79.93 | 4624 | 5 |
| C | 18.91 | 83.88 | 86.81 | 4874 | 6 |
| D | 16.51 | 73.61 | 86.14 | 4848 | 5 |
| Total | 70.24 | 304.85 | 329.45 | — | — |

Segments have a 23-hour cap. Stop lead covers twice the conditional high-load
iteration plus a full monitoring pass, checkpoint and margin. The bounded
continuation counts include additional margin. Longer contexts/responses, changed
reward mix and differences between pilot reference proxies and main reference
anchors may exceed these scenarios. This acceptance establishes operation under
the measured workload and makes those dependencies explicit. Locked final-ID,
external/secondary evaluations and separate counterfactual diagnostics are not
included in these training forecasts or executed in this task.

## Accounting, freeze and actual previews

Stable authorization: `cpsc4880-readiness-24h-v1`, persisted in
`state/budget-readiness-cpsc4880.json` under the existing budget lock. All 31
historical/new reservations reconcile to **3.234722222/24 GPU-hours**,
with **zero outstanding reservation**. Task 08 added 2.143611111 GPU-hours to the
historical 1.091111111. The shared pilot total is 1.025000/12 hours. No allowance
was reset by the new study. CPU freeze and submission previews allocated no GPU.
Final Slurm inspection found no active user job or HPC watcher.

The freeze binds science, protocol, manifest, SIFs, worker/runtime/backend,
corrected base, resumed D, A/B/C proofs, checkpoint cadence and the measured
resource-plan digest. Each successful actual controller result has `dry_run=true`
and `submitted=false`, fresh initialization and the frozen request.

| Method | Main run identity from preview | Iterations | Submitted |
|---|---|---:|---|
| A | `b88cf9bd9980d580bd806dc4` | 200 | false |
| B | `401e3d80a8c36f1cc3c99c2a` | 200 | false |
| C | `19f609f5811b66837a8fe630` | 200 | false |
| D | `dae4973178eec16beea07ab0` | 200 | false |

Exact successful commands:

```bash
uv run selfplay hpc submit --settings configs/task08.yaml --operation train --method A --profile single_gpu_80gb --gpu-type h200 --account cpsc4880 --dry-run
uv run selfplay hpc submit --settings configs/task08.yaml --operation train --method B --profile single_gpu_80gb --gpu-type h200 --account cpsc4880 --dry-run
uv run selfplay hpc submit --settings configs/task08.yaml --operation train --method C --profile single_gpu_80gb --gpu-type h200 --account cpsc4880 --dry-run
uv run selfplay hpc submit --settings configs/task08.yaml --operation train --method D --profile single_gpu_80gb --gpu-type h200 --account cpsc4880 --dry-run
```

Prepared later main commands, requiring separate user authorization:

```bash
uv run selfplay hpc submit --settings configs/task08.yaml --operation train --method A --profile single_gpu_80gb --gpu-type h200 --account cpsc4880
uv run selfplay hpc submit --settings configs/task08.yaml --operation train --method B --profile single_gpu_80gb --gpu-type h200 --account cpsc4880
uv run selfplay hpc submit --settings configs/task08.yaml --operation train --method C --profile single_gpu_80gb --gpu-type h200 --account cpsc4880
uv run selfplay hpc submit --settings configs/task08.yaml --operation train --method D --profile single_gpu_80gb --gpu-type h200 --account cpsc4880
```

These commands were not executed. Main payloads automatically pin
frozen execution commit `983a6cf`; later documentation/controller changes do not
replace the frozen trainer science. The common initialization remains fresh;
no readiness checkpoint is a main warm start.

## Monitoring and recovery

```bash
uv run selfplay hpc doctor --settings configs/task08.yaml --gpu-type h200
uv run selfplay hpc inspect
uv run selfplay hpc status RUN_ID
uv run selfplay hpc logs RUN_ID --lines 100
uv run selfplay hpc fetch RUN_ID
uv run selfplay hpc watch RUN_ID --continue-stopped
uv run selfplay hpc resume RUN_ID --continue-stopped
```

Use the last two only for an authorized run and verified clean-stop/transient
recovery under its persisted limits. Unknown scheduler/submission state must be
inspected without duplication. Scientific failure, OOM, authentication and quota
errors are not automatic retries; partial optimizer state is never resumed.
`hpc fetch RUN_ID --full` retrieves full checkpoints/raw trajectories when needed.
Ordinary Duo reauthentication uses `hpc login`; it does not require a workaround.
The [launch runbook](../yale-launch-runbook.md) is the active command reference.

Remaining limitations are scientific yield/action quality and uncertainty in
future workload/long-context cost. Additional profiling can narrow estimates;
changes to representation, model, sampling, rewards or mathematical budgets need
a further scientific decision and compatible study identity. These limitations
are distinct from the completed launch gates. Main A–D outcomes remain unmeasured.

## Implementation commit sequence

```text
0138dbccae60d7b3f2d09fe697094e91b998996e Version the common repair protocol and enforce shared readiness accounting
a291098452e995800095ea548865a5e833154ebe Bind later-turn protocol serialization and worker reuse limits
e1836a0d946892ad6d616af11d365281d3077715 Isolate telemetry tests and harden protocol revision publication
c07dd4248d8a73f4adc91e19058c8073206fde57 Record passed protocol probe and immutable data revision
db0b5954ad2ccaf5084f8bea4fa9b4fe376ec9ee Keep report fetches small and isolate selected method acceptance slots
7866cad4a99e88f6c535bd3ba25c34f58aa3c914 Report actual pilot update counts and qualify workload extrapolation
3961fcf45c5072aa4f803e982a1c9345d0748c6d Checkpoint each learning iteration and attribute sampled GPU processes
0465b20cc8bd47fac72c99b912e67a5232050796 Honor revised data roots during Yale image reuse
b86fb0a93c3f7110d234999eca630b925fca0ebb Bind conditional workload resource plans to study freezes
d5ac499033f69789b63df175514a935c846a3fbd Verify frozen resource plans reject stale measurement evidence
6af9fc31d432ce708ce25c6b5b2a0b8310af962d Summarize measured workload and avoid redundant adapter downloads
8cce9b6b3cfde901e13ab4e2a50b751e2a52bf90 Include monitoring boundaries in measured stop signal margins
3ce83b60f9a18d070384cf11d0ffc26e9c4d5629 Fetch read-only checkpoint reports without replaying directory modes
3467a5c6008a190363e4c3f1d59fb2b2a35553e9 Audit raw action validity and unique exposures in workload summaries
296ee6cc723c2806eb787412c21cc9ec89fdabd5 Cross-check continuation exposure totals against committed raw episodes
983a6cf57ce5caf5c9aa7ddf53f5ad7f313b4e73 Normalize inherited directory mode bits when fetching readiness reports
25b07f230ef66a7136c51c6da4668ac6cecf9f4d Report the frozen 200-iteration main budget in submission previews
```
