# Yale A–D readiness and launch

Task 05 is preparing this study. **Do not launch main runs until the live doctor
and all four previews pass and main training is separately authorized.** CPU CI
does not prove GPU compatibility or generated-code isolation.

The [Task 05 checkpoint](task_completion/05-implemented-yale-experiment-readiness.md)
records the published trainer, queued reference job, exact budget and external
worker dependency. Inspect that existing job before submitting more readiness
work. If doctor reports a verified trainer already available, proceed to the
readiness sequence; rebuilding is needed only when its authoritative inputs change.

## Build without a large local archive

From the clean committed Task 05 checkout:

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

## Readiness sequence

Task 06 first establishes physical allocation ownership in short model-free
checks and collects worker setup diagnostics on a CPU node:

```bash
uv run selfplay hpc sync
uv run selfplay hpc worker-diagnose
uv run selfplay hpc allocation-check --profile single_gpu_80gb
uv run selfplay hpc allocation-check --profile dual_gpu_80gb
```

Inspect each returned run before advancing. Allocation checks each have a
five-minute ceiling and share the existing 24 GPU-hour ledger. They publish
`gpu-allocation-JOB_ID.json` even on failure, and successful profile proofs as
`allocation-PROFILE.json`. NVIDIA indices and Linux device minors may differ;
ownership comes from actual kernel device access inside the live Slurm cgroup,
followed by a container CUDA UUID/occupancy check. Extra device access fails.

Worker diagnostics execute no candidate code, establish no isolation evidence,
and publish `worker-diagnostics-JOB_ID.json`. Use these findings in YCRC ticket
79403. After a documented supported site change, identify it explicitly:

```bash
uv run selfplay hpc prepare-study --worker-setup-id TICKET-AND-SETUP-REVISION --watch
```

The setup ID records the change; actual isolation probes must still pass. An
unchanged delegation failure cannot be retried merely by restarting preparation.
The setup identity follows runtime evidence and checkpoints into main previews.
An existing approved Yale service remains a contingency pending YCRC's concrete
endpoint, authentication, cancellation and accounting contract; no service has
been provisioned or assumed available.

`doctor` includes `stage_status`: pending, running, passed, stale or failed,
with scheduler/run identity, evidence and next action. Allocation and backend
checks can retry at most two confirmed scheduler interruptions; application
failure, OOM, uncertain submission and GPU timeout require inspection.

After the trainer build and hosted CPU CI pass, backend-only GPU acceptance can
proceed independently of the worker support response:

```bash
uv run selfplay hpc sync
uv run selfplay hpc prepare-study --stage backend --gpu-hours-limit 24 --watch
```

Synchronize again after any new source commit before preparing evidence. A
successful build does not synchronize later changes automatically.

The full workflow requires a YCRC-supported worker setup and fresh successful
isolation evidence. After that setup change:

```bash
uv run selfplay hpc prepare-study --gpu-hours-limit 24 --watch
uv run selfplay hpc doctor
uv run selfplay hpc fetch-study repair-selfplay-2026
```

Preparation validates deployed data, backend reference/comparison, base
monitoring evaluation, a ten-iteration D pilot with stop/resume at 2, and
one full iteration each of A/B/C in separate fresh processes. Method smoke
states live in `repair-selfplay-2026-method-smoke`; their proofs live under
`readiness/repair-selfplay-2026/method-smokes`. All three must exercise their
expected update paths and retain intact joint checkpoints. A degenerate smoke
is a failure to establish evidence, not permission to alter generated data.

Initial reservation ceilings are reference 1, target 2, base 4, D pilot 12 and
combined A/B/C smoke 4 allocated GPU-hours. One additional GPU-hour remains;
unused reservations are released only after actual Slurm accounting. All
attempts share the cumulative 24-hour cap. H200 requires a recorded H100 CUDA OOM
on a correctly mapped, initially unoccupied allocation and fresh target proof,
within that same cap. Preparation never launches main.

GPU jobs resolve Slurm's physical allocation to UUIDs and verify the container's
actual CUDA UUIDs before loading models. Allocation proofs are retained as
`readiness/STUDY/gpu-allocation-JOB_ID.json`. Ambiguous NVIDIA/Slurm numbering,
missing masks, wrong devices or substantial initial occupancy stop the job.
Inspect this proof alongside an OOM; a device-binding failure is not evidence
that the model needs a larger GPU.

## Four main previews and submissions

The following H100 commands resolve the published SIF and frozen evidence.
Use the hardware selected by the actual readiness result; H200 fallback needs
`--gpu-type h200` and its matching evidence. Do not infer readiness from a free
GPU in a queue snapshot.

```bash
uv run selfplay hpc submit --operation train --method A --gpu-type h100 --dry-run
uv run selfplay hpc submit --operation train --method B --gpu-type h100 --dry-run
uv run selfplay hpc submit --operation train --method C --gpu-type h100 --dry-run
uv run selfplay hpc submit --operation train --method D --gpu-type h100 --dry-run
```

Only after all gates pass and main training is authorized, submit one method at
a time, initially. Remove `--dry-run` from the corresponding preview command.
Each main run starts fresh with seed 2026 and 200 iterations; pilot/smoke weights
are never used. Four concurrent runs are not needed for scientific comparability.

**Measured estimates:** pending a real D pilot. Its median/p90 phase times,
memory, startup costs and projected 200-iteration duration determine continuation
planning. The D projection is an estimate for A/B/C, not a measured guarantee.
Main GPU cost must be reviewed separately from the readiness allowance.

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
