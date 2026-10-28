# Yale A–D readiness and launch

Task 07 is authoritative. Main A–D training remains unauthorized. The cumulative
readiness cap is 24 allocated GPU-hours, including historical attempts and all
unresolved reservations. Historical job states live in the Task 03–06 completion
records and the Task 07 plan; reconcile live Slurm and controller state first.

## Current checkpoint — 2026-10-27

The selected profile is `single_gpu_80gb`, using one entitled H200 in `gpu_h200`
for the documented scheduling reason in the Task 07 plan. H100 OOM is not required.
Seven trainer CPUs/128 GiB host RAM and a separate one-CPU/1-GiB worker step are
requested for generated-code jobs. Add GPUs only if measurements justify them;
a multi-GPU target additionally needs agreement with single-GPU gradients.
Slurm rejected the previous nine-CPU total: gpu_h200 permits at most eight CPUs
per GPU. The current total is eight, retaining independent worker enforcement.

Worker acceptance **31033392** / `639345f9a3d7a5c3953ece55` passed all 19 checks:
independent scheduler CPU/memory enforcement, single-process/thread seccomp,
filesystem/network/GPU/credential isolation, bounded storage/output/time,
candidate OOM followed by correct-program recovery, cancellation and real Slurm
step KILL with no observed processes left running. The no-GRES broker uses
`--oom-kill-step=0`; its 1-GiB kernel bound remains enforced. The deliberate OOM
witness is CPU acceptance only. Ordinary GPU preflight does not trigger an OOM.
The policy is installed before candidate compilation/imports. Private temporary
storage is 64 MiB. No nested user cgroup configuration, pids.max, subordinate UID
allocator, external service or SSH tunnel is needed. These restrictions are the
same for A–D.

CPU runtime validation **31040084** / `333e5b3f69f36d3e35221472` passed all 306
frozen tasks under the latest scientific source identity. Worker and runtime
acceptance remain compatible with the operational trainer CPU adjustment.

Allocation check **31038198** / `a38a5c80f7518cd7d26c66f4` passed actual Slurm
device ownership, numeric container mask, exact CUDA UUID and initial occupancy.
On this node NVIDIA index 0 had Linux minor 1; equality is never assumed. Pinned
vLLM 0.11.0 requires a numeric mask, so the launcher verifies its actual UUIDs
before any model load. Ambiguous, extra, wrong or occupied GPUs stop the job.

Backend **31038874** / `2eddfa162616da98b977e779` loaded the pinned model and
recognized Qwen3 in vLLM, then stalled with inherited torchrun agent-store state.
It was stopped after diagnosis; `state/task07-stop-31038874.json` records why.
Source **e0f4374** removes all TORCHELASTIC_* settings alongside parent rank and
rendezvous variables. A real CPU process-group regression reproduces the timeout
with the inherited flag and succeeds without it. Hosted CI 40607054798 passed
(212 tests). Backend **31040984** / `d3a9ac3571ccc56dc379385b`, source **6768da2**
(same scientific code), passed all A–D objective probes, adapter isolation, two
vLLM synchronization/scoring cycles and checkpoint save/load/restoration. It used
222 GPU-seconds. Base evaluation, pilot, method acceptance, freeze and main
previews are still pending. Synthetic acceptance does not establish real yield,
throughput or full-workload memory fit.
Cumulative consumption is **0.274444444 GPU-hours**, with no active GPU job at
this checkpoint. Live accounting and reservations always take precedence.

The earlier PEFT initialization mismatch is fixed: proposer adapter masters are
promoted to the student's FP32 dtype before copying, with equal initial version
hashes and independent storage. BF16 base compute, objectives, prompts, dataset,
seed 2026 and mathematical budgets are preserved.

## Continue preparation

```bash
uv run selfplay hpc doctor --gpu-type h200
uv run selfplay hpc inspect
# Only if normal SSH authentication has expired:
uv run selfplay hpc login
uv run selfplay hpc sync
uv run selfplay hpc prepare-study --stage backend --gpu-type h200 --gpu-hours-limit 24 --watch
uv run selfplay hpc prepare-study --gpu-type h200 --gpu-hours-limit 24 --watch
```

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

These are the intended commands, **not successful previews yet**. They use the
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
2026 for 200 iterations; pilot/smoke weights are never used. Measured pilot memory,
median/p90 iteration/phase times and startup costs are still pending. Its projected
200-iteration duration is an estimate for A/B/C, not measured performance. Main
GPU cost must be reviewed separately from the readiness allowance.

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
