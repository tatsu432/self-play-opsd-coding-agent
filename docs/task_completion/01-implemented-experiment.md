# Experiment implementation completion

Implemented 2026-10-21 from `docs/task/01-experiment-implementation-spec.md`
and `docs/plan/01-experiment-implementation-plan.md`.

## Status

The implementation and local validation are complete. Production acceptance is
**pending**: this ARM macOS workstation has no allocated CUDA devices, Yale
allocation, AWS allocation or online W&B credentials. No main experiment, GPU
pilot, runtime projection or repair-performance result is claimed.

## Delivered

- Installable, locked uv package with strict typed configuration, CLI and CI.
  Docker worker, CPU trainer and CUDA trainer definitions; local Docker, AWS
  Docker and Yale Apptainer/Slurm launch paths for all hardware profiles.
- Pinned MBPP+/HumanEval+ ingestion, original public examples, special oracle
  comparators, cached preparation, deterministic family/split audits, finite AST
  mutations, frozen task manifests and tokenizer-specific canonical actions.
  Upstream notices and complete license texts are bundled.
- Four-action repair environment with exact retained sampled IDs across tool
  turns, privileged teacher replay of those same targets, explicit causal maps,
  fresh final verification and bounded resources. Generated code executes only in
  isolated workers; authenticated remote execution is available for Yale.
- A–D objectives, independent proposer/student adapters and optimizers, attempted
  proposal accounting, population-standardized GRPO, detached signed-gap SD and
  full-coordinate reference/candidate gradient alignment. Pure numerical tests
  verify the objectives independently of the trainer.
- Real HF/PEFT CPU debugging and production verl FSDP2/vLLM integration, bounded
  scoring microbatches, activation checkpointing, explicit adapter reloads,
  bounded overlapping tool work and canonical result ordering.
- Joint atomic checkpoints, checksums, stage caches, RNG/curriculum restoration,
  fault recovery and lineage rollback that retains consumed physical compute.
- Local-first W&B offline/online delivery, stable online logical IDs, bounded
  tables, optional inference-adapter artifacts, resource/health metrics, pilot
  reports, backend acceptance probes and frozen-study launch gates.
- Frozen greedy evaluation, primary/secondary checkpoint selection,
  counterfactual optimizer-restored diagnostics, parent-paired bootstrap,
  local tables/plots and explicit missing/incomplete results.

Operational commands are in `README.md`; equations, normalization, isolation
boundaries and production limitations are in `docs/methods.md`.

## Prepared data

Actual pinned inputs contain 378 MBPP+ and 164 HumanEval+ parents. Preparation
retained 497 eligible parents and recorded 45 exclusions. Split counts are 186
training, 32 reference, 34 monitoring-development, 12 diagnostic-development,
100 final in-distribution and 130 external parents; three additional overlapping
MBPP parents are excluded from the training pool.

| Frozen split | Actual tasks | Target tasks | Parents covered |
| --- | ---: | ---: | ---: |
| Reference | 14 | 32 | 14 |
| Monitoring development | 41 | 68 | 23 |
| Diagnostic development | 20 | 24 | 10 |
| Final in-distribution | 115 | 200 | 60 |
| External | 116 | 130 | 116 |

Total: **306 frozen tasks**. The finite mutation library, supported public-example
parser and execution/precondition exclusions reduce coverage. Held-out
reservations are preserved; no replacement silently transfers reserved parents
into training. `data/prepared/coverage.json` contains every exclusion and deficit.
These tasks are repair benchmarks derived from EvalPlus, not official generation
scores. Dataset/model downloads and preparation outputs remain in ignored `data/`.
The earlier token-serialization lineage is preserved separately at
`data/prepared-serialization-v1`; current training uses only `data/prepared`.

## Local verification evidence

Final command outputs are under `runs/verification/`; the actual model/sandbox
smoke report is `runs/final-verification/smoke_report.json`.

- `uv sync --frozen`, Ruff lint/format and mypy pass.
- Docker worker and CPU trainer images build. Current source performs real tiny
  model generation and scoring inside the CPU image.
- Actual Docker isolation passes network/credential/GPU denial, nonroot,
  read-only root and CPU/memory/PID/tmp-disk limit checks. Remote authentication
  and execution are tested locally.
- Real tiny-model A–D smoke and offline W&B finish successfully. All random tiny
  proposals are invalid, so these iterations correctly make zero updates. This
  does not demonstrate learning. Scripted-action integration tests separately
  exercise valid A–D updates using real model scoring and gradients.
- A fixture witness has zero RL gradient and nonzero gated SD gradient; C returns
  neutral alignment and D a finite nonneutral alignment score.
- GitHub Actions' filtered test command passes with forced ANSI color: **64 passed,
  3 deselected**. This reproduces the CI color environment that exposed the help
  assertion issue. Log: `runs/verification/github-workflow-final.log`.
- After expanding the scientific compatibility hash to include execution workers,
  the focused recovery/storage suite passes again: **9 passed**. Evidence:
  `runs/verification/recovery-final.log`.
- Tests cover microbatch equivalence, actual activation checkpoint execution,
  retained noncanonical token IDs, adapter independence, zero-gradient Adam
  behavior, checkpoint corruption, interrupted joint-update replay, sandbox
  boundaries, real pinned data, launch profiles and paired statistics.
- Final pytest result: **65 passed, 2 skipped**, six nonfatal upstream/dataset
  warnings, in 107.55 seconds. Skips require CUDA or explicit online W&B credentials.
  Full log: `runs/verification/pytest.log`.
- Executed validation of **all 306 current-manifest tasks** passes: restored source
  succeeds, mutants fail, and family/split/workspace audits pass. The manifest hash
  is `633a0ee3a94ff9f7efdfd48c50f1ac22ae565879f7b9739ee00b28dbca9f9658`.
  See `data/prepared/validation_report.json` and its copy under `runs/verification/`.

## Explicit choices and remaining acceptance

Public tests use a trusted function-call supervisor rather than executing an
agent-controlled pytest configuration. Correct source is AST-formatted before
mutation. Family similarity is conservative syntax-based matching, with its
semantic-duplicate limits recorded. Sampling is unconstrained JSON; generation
is serialized while independent sandbox execution overlaps.

The training-only pilot reports gradient influence but retains configured lambda;
automatic lambda rescaling is disabled. To change it, edit the common config and
repeat the pilot before freeze. Pilot alignment uses training-parent proxies,
not reserved reference/development tasks.

The production environment is pinned to torch 2.8.0, transformers 4.57.1,
PEFT 0.17.1, vLLM 0.11.0 and verl commit
`bec9ef74768dd201881cd4e54cd0385e87caae27`. Source compatibility was inspected;
GPU runtime compatibility, memory fit, adapter reload equivalence and optimizer
resharding remain unverified until `backend-check` on the target allocation.
Changing CUDA world size during recovery remains guarded rather than assumed
safe. All hardware profiles have launch/configuration coverage, not measured
GPU acceptance.

Before four main runs: establish the target worker isolation proof, validate
prepared data against that exact worker contract, run single-GPU and dual-GPU
backend checks with the shared gradient reference, evaluate the common untrained
baseline, run the measured ten-iteration training-only D pilot on Yale two 80 GB
GPUs, and freeze matching common settings. Verify online W&B there if live
tracking is required. The README contains the exact commands. Full runs remain
gated until those artifacts exist and match.

Existing `.gitignore` entries and `.DS_Store` were preserved. Plans, task
completion reports, data and run evidence are ignored by the existing repository
policy and are delivered locally; no staging, commit, deployment or main-run
launch was performed.
