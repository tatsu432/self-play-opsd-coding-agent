# Yale A–D readiness and launch

Task 08 is authoritative; the Task 07 checkpoint below is historical. Main A–D training remains unauthorized. The cumulative
readiness cap is 24 allocated GPU-hours, including historical attempts and all
unresolved reservations. Historical job states live in the Task 03–06 completion
records and the Task 07 plan; reconcile live Slurm and controller state first.

## Task 08 continuation

Task 08 authorizes the common format clarification and prompt-dependent immutable
revision. Main training remains unauthorized. Current work has no readiness claim.
Use `configs/task08-source.yaml` only for the fixed training-parent probe and data
revision, both based on unchanged original parents/tasks. After successful CI:

```bash
uv run selfplay hpc doctor --settings configs/task08-source.yaml --gpu-type h200
uv run selfplay hpc inspect
uv run selfplay hpc sync --settings configs/task08-source.yaml
uv run selfplay hpc reuse-worker repair-selfplay-2026 --settings configs/task08-source.yaml
uv run selfplay hpc submit --settings configs/task08-source.yaml --operation protocol-check --method B --profile single_gpu_80gb --gpu-type h200
uv run selfplay hpc submit --settings configs/task08-source.yaml --operation protocol-revision --method B --profile single_gpu_80gb --gpu-type h200
```

Reconcile shared accounting and inspect existing operation ledgers before any
submission. The probe is one fixed 32-proposal/two-batch window, no updates or
refills, four episodes per valid task, with a two-GPU-hour ceiling. Failed
clarified outputs stop dependent experiments; do not retry unchanged outputs.
CPU revision work may complete independently. All allocations retain accepted
containment and exact GPU ownership/initial occupancy checks.

Revision outputs live under `runs/data-revisions/STUDY`. Preserve original data,
inspect all equality/admission results, fetch the completed revision, and commit
final settings pointing at its local directory before final synchronization.
Run only invalidated runtime/backend proofs, corrected greedy monitoring base,
D's ten-iteration stop/resume pilot, three-iteration A/B/C windows, freeze, then
four actual previews. Source settings are not final main settings.

All studies share `state/budget-readiness-cpsc4880.json` under the existing lock.
Historical charges and unresolved reservations remain binding, including the
12-hour pilot sublimit. Never use a new study/operation name to bypass a limit.
GPU allocations record one-second device/process memory samples in readiness
`resource-samples-JOB.jsonl`; these are sampled lower bounds, not exact peaks.

## Historical Task 07 checkpoint — 2026-10-27

**Launch readiness is blocked by zero valid proposal yield.** The accepted Yale
execution path requires no further YCRC setup. The [Task 07 record](task_completion/07-implemented-yale-execution-blocked-readiness.md)
contains exact code, CI, identities, measurements and live evidence. Execution
code is `9980c78`; CI 40617600830 passed 220 tests.

The selected profile is one H200 in `gpu_h200`, with seven trainer CPUs/128 GiB
and an independently bounded no-GRES worker CPU/1 GiB: eight CPUs/129 GiB total.
Slurm permits at most eight CPUs per H200. The documented scheduling reason
permits H200 without manufacturing an H100 OOM. A multi-GPU configuration is not
required; adding GPUs would additionally require gradient-agreement evidence.

| Stage | Current result |
|---|---|
| Worker | All 19 checks passed, job 31033392 |
| Deployed data | All 306 frozen tasks passed, job 31045342 |
| Numerical backend | One-H200 acceptance passed, job 31040984 |
| Baseline | Completed 0/41 repairs, job 31046386 |
| D pilot | Ten iterations, verified resume at two, 0/160 valid proposals and no updates |
| A/B/C | One iteration each; all 0/16 valid proposals, no episodes or updates |
| Freeze/main previews | Blocked; all four dry runs refused eligibility |

The worker installs an irreversible AMD64 single-process/thread seccomp policy
before candidate compilation/imports. Slurm bounds its actual cpuset and memory;
Apptainer isolates filesystem/network/GPU/credentials. Private temporary storage
is 64 MiB. Real candidate OOM recovery, timeout/output/cancellation cleanup and
Slurm KILL acceptance passed. No nested user cgroup, numeric pids.max, external
worker host, subordinate UID allocator or SSH tunnel is required. Deliberate OOM
witnesses run only in CPU acceptance, not ordinary GPU preflight.

GPU ownership comes from the actual Slurm device cgroup plus exact container
CUDA UUID/occupancy verification. Pinned vLLM uses a numeric mask, proved before
model load; NVIDIA indices and Linux minors need not match. Student/proposer
master precision, independent rollout rendezvous, DTensor base verification and
method-smoke argument integration are repaired without weakening their checks.

Cumulative consumption is **1.091111111 of 24 GPU-hours**; outstanding
reservations, active jobs and watchers are zero. The D pilot's proposal-only
49.06-s median/104.53-s p90 and 10.46-GiB trainer PyTorch peak do not establish
learning throughput or memory fit. A defensible main cost estimate is unavailable.

The [shared diff-format clarification](task/07-shared-diff-format-decision.md)
is a reviewed proposal only. User authorization is pending because Task 07 freezes
prompts. No active prompt/parser, sampling, reward, model or data change has been
applied. Keep the failed results intact and do not repeat the unchanged zero-yield
pilot. A new study identity must not create a fresh readiness allowance.

## Review and authorized continuation

```bash
uv run selfplay hpc doctor --gpu-type h200
uv run selfplay hpc inspect
uv run selfplay hpc status f597e3f66092a9a9c1265ea0
uv run selfplay hpc logs f597e3f66092a9a9c1265ea0
uv run selfplay hpc fetch f597e3f66092a9a9c1265ea0
```

After an explicitly authorized scientific resolution and compatible evidence,
use `hpc sync` and `hpc prepare-study --gpu-type h200 --gpu-hours-limit 24 --watch`
for the new setup. Current passed evidence should be reused wherever relevant;
changed prompt/scientific identities require their own compatible evidence.
Never reset or fork the cumulative authorization ledger. Normal MFA, if needed,
is `hpc login` in a terminal; complete Yale's normal Duo prompt.

`doctor` reports pending/running/passed/stale/failed stages, scheduler/run identity,
evidence and next action. Reconcile an existing active or uncertain submission;
never duplicate it. Independent operation slots permit CPU validation while a
GPU diagnostic is pending. Allocation/backend acceptance executes only trusted
fixtures and does not depend on generated-code worker readiness.

Preparation validates deployed data, the selected backend, base monitoring
evaluation, a real ten-iteration D pilot stopped/resumed at iteration two, and
one full iteration each of A/B/C in fresh processes. The pilot must demonstrate
valid tasks, alignment and both updates; synthetic probes cannot establish yield,
throughput or repair performance. A degenerate pilot/smoke must be diagnosed,
without changing rewards, sampling, data boundaries or refilling proposals.
Freeze and all four successful previews are required for launch eligibility.
Preparation never submits main runs. CPU tests alone establish no target readiness.
Pilot wall-time ceilings scale with GPU count to preserve the original four
GPU-hours before the iteration-two stop plus eight GPU-hours for continuation:
one GPU receives four/eight hours; two GPUs retain two/four hours. The pilot's
12-GPU-hour and total readiness 24-GPU-hour ceilings are unchanged. The initial
real base episodes measured roughly 66 seconds each, so carrying the old
two-GPU wall limits into the one-GPU setup could truncate the required pilot.
This operational scaling preserves the same allowance. The completed pilot had
no valid tasks, so its measured empty-workload cadence is not training throughput.

All attempts share the durable 24-hour ledger. At most two confirmed transient
scheduler interruptions can restart diagnostics. Application failures, OOM,
GPU timeout, authentication and uncertain submission require inspection. A
never-started cancellation releases its reservation only after confirmed accounting;
incomplete accounting remains reserved. Do not reset the ledger.

Images are checksummed on compute nodes. Existing trainer/worker SIFs can be
reused when their pinned base, lock and installation inputs match. Original image
provenance remains intact; current immutable trainer sources and exactly three
worker entry files are mounted read-only. Controller/documentation changes do
not force image rebuilds or unrelated numerical retesting.

```bash
uv run selfplay hpc images build --kind all
```

`--force`, an explicit builder or build root requests an actual build. The normal
reuse path has been verified without allocation. If an allocation check's raw
proof passed but publication failed, `hpc recover-allocation RUN_ID` validates its
immutable verifier and raw checksum without a new GPU allocation; it preserves
original failed scheduler accounting. This does not prove backend compatibility.

## Main previews and later launch

All four commands below were tested and refused main eligibility (exit 1). They use the
selected single-GPU profile explicitly; the generic submit command defaults to a
different profile. All four must pass after freeze. Main submission additionally
requires separate user authorization.

```bash
uv run selfplay hpc submit --operation train --method A --profile single_gpu_80gb --gpu-type h200 --dry-run
uv run selfplay hpc submit --operation train --method B --profile single_gpu_80gb --gpu-type h200 --dry-run
uv run selfplay hpc submit --operation train --method C --profile single_gpu_80gb --gpu-type h200 --dry-run
uv run selfplay hpc submit --operation train --method D --profile single_gpu_80gb --gpu-type h200 --dry-run
```

After readiness and separate main authorization, remove `--dry-run` for the
corresponding method. Begin with one method at a time. Each starts fresh at seed
2026 for 200 iterations; pilot/smoke weights are never used. Meaningful learning-memory and training-time measurements remain unestablished
because the pilot had no valid tasks. No valid main projection or launch command
can be certified yet. Main GPU cost requires separate authorization.

## Monitor and recover

```bash
uv run selfplay hpc status RUN_ID
uv run selfplay hpc logs RUN_ID
uv run selfplay hpc fetch RUN_ID
uv run selfplay hpc watch RUN_ID
```

Watch resumes at most two confirmed PREEMPTED/NODE_FAIL/BOOT_FAIL attempts from
a verified joint checkpoint. `hpc resume RUN_ID --continue-stopped` permits a
recorded clean continuation. Unknown scheduler state, ambiguous submission,
scientific failure, OOM, authentication or quota problems require inspection.
Never submit a replacement while an earlier intent remains unresolved.

W&B is offline by default. Local events/checkpoints are authoritative; retain
offline segments for later `selfplay wandb-sync`. Fetch uses small reports by
default; `--full` explicitly includes checkpoints/trajectories/W&B files.

## Build without a large local archive

When image preparation reports an incompatible or missing trainer, use a clean committed checkout:

```bash
uv sync --frozen --group build
uv run selfplay hpc doctor
uv run selfplay hpc inspect
# If normal authentication has expired:
uv run selfplay hpc login
uv run selfplay hpc images build --kind trainer --build-mode yale --check-only
```

Inspect the returned run ID using `hpc status RUN_ID`, `hpc logs RUN_ID` and
`hpc fetch RUN_ID`. The feasibility allocation uses two CPUs, 4 GiB and at most
15 minutes. It exercises a small pinned image, files, RUN, ENV and entrypoint.
It does not execute generated code or prove worker isolation. After it passes:

```bash
uv run selfplay hpc images build --kind trainer --build-mode yale
```

This uses eight CPUs, 32 GiB and at most four hours. Squashfs compression is
explicitly limited to eight CPUs and 8 GiB; dependency downloads are not cached
inside the image. Personal course scratch
requires 150 GiB free; final project storage requires 30 GiB. Source, generated
definition, dependency lock, base digest, converter version and resulting SIF
are checksummed. Only the successful verified SIF is published. Large build
contexts/caches stay on Yale; the Mac receives small logs/reports. Docker Buildx
on the Mac reads only small registry configuration metadata for conversion. The previous
Docker archive path remains available with `--build-mode docker`.

If a site permission failure occurs, inspect that job and discuss it in the
existing YCRC thread. Do not repeat it without a changed supported setup.
