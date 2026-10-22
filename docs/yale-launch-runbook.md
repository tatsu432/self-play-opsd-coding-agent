# Yale A–D readiness and launch

Task 05 is preparing this study. **Do not launch main runs until the live doctor
and all four previews pass and main training is separately authorized.** CPU CI
does not prove GPU compatibility or generated-code isolation.

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

This uses eight CPUs, 32 GiB and at most four hours. Personal course scratch
requires 150 GiB free; final project storage requires 30 GiB. Source, generated
definition, dependency lock, base digest, converter version and resulting SIF
are checksummed. Only the successful verified SIF is published. Large build
contexts/caches stay on Yale; the Mac receives small logs/reports. The previous
Docker archive path remains available with `--build-mode docker`.

If a site permission failure occurs, inspect that job and discuss it in the
existing YCRC thread. Do not repeat it without a changed supported setup.

## Readiness sequence

After the trainer build and hosted CPU CI pass, backend-only GPU acceptance can
proceed independently of the worker support response:

```bash
uv run selfplay hpc prepare-study --stage backend --gpu-hours-limit 24 --watch
```

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
and fresh target proof, within that same cap. Preparation never launches main.

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
