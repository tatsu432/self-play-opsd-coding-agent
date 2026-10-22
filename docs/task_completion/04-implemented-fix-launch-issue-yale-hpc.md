# 04 — Yale launch repair

Source: `docs/task/04-fix-launch-issue-yale-hpc-spec.md`; approved plan:
`docs/plan/04-fix-launch-issue-yale-hpc-plan.md`. Work is on
`fix/04-yale-launch-readiness`, based on `e70df4a`; main remains unchanged.

## Fixed issues

- Checkpoint identity uses the supplied validated TaskRepository, rejects a
  configured-root mismatch, and is cached for the trainer's lifetime. Save and
  resume retain strict manifest/scientific/prompt/runtime checks. Trainer fixtures
  explicitly use their temporary data root. Missing/corrupt data still fails.
- The original five CI failures are fixed. Commit `83d3e36` passed the exact
  correctness workflow on GitHub **before** infrastructure changes began:
  [CI checkpoint](https://github.com/tatsu432/self-play-opsd-coding-agent/actions/runs/39732823085).
- Trusted backend diagnostics no longer require worker images, endpoints,
  credentials, isolation evidence, or target execution validation. Submission
  and allocation independently enforce this distinction. Backend allocations
  do not mount the dataset or start a broker; model output is scored, never run.
- Versioned `backend_identity` binds implementation/dependencies, model/tokenizer,
  objective probe configuration and trainer image. Count/profile/device and
  single-process gradient comparisons remain checked separately. New evidence
  is in `compatibility/backend-v2`; old reports and Yale artifacts are preserved
  and cannot satisfy the new contract. Full execution evidence still binds
  dataset, prompts, worker image and scientific settings.
- `hpc prepare-study --stage backend` advances only the one-H100 reference and
  target backend check. Default `--stage all` retains complete readiness.
  Training, pilots, evaluation, generated-code diagnostics and data verification
  retain worker requirements; every generated-code allocation probes isolation.
  Evidence-only freeze/readiness commands avoid unnecessary worker startup.
- `hpc images build --build-root PATH` / `SELFPLAY_IMAGE_BUILD_ROOT` select the
  filesystem for exports, temporary archives, metadata and logs. `--builder`
  selects an existing Buildx builder. Capacity checks inspect the selected
  filesystem; archives retain checksums, provenance checks and atomic metadata.
  Interrupted exports are removed. Builder cache capacity is documented separately.
- Removed duplicate preparation payload construction, repeated trainer identity
  hashes, a redundant image-build snapshot, and the hardcoded manifest reread
  in payload construction. Scheduler/accounting reconciliation, immutable
  submissions, output locks, budget reservations and checkpoint integrity remain.

## Verification

The 52 focused identity, controller, readiness, allocation and image-build tests
pass. They cover backend execution with broken incidental worker configuration,
missing dataset and forbidden sandbox construction; generated-code gate failures;
stale/legacy evidence; cached archive corruption/provenance; interrupted export;
build-root precedence; and builder selection. The five Docker trainer transaction
and interruption tests also pass after the final source changes.

The exact CPU suite passed **117 tests**, with 3 tests excluded by the workflow's
GPU/download/online selection and no capability skips. Ruff lint/format checks,
mypy, the Docker worker build and Slurm shell syntax passed. Three focused
tests also verify the final sandbox-constructor and archive-provenance assertions.
The final branch's hosted acceptance is available in
[correctness workflow runs](https://github.com/tatsu432/self-play-opsd-coding-agent/actions/workflows/checks.yml?query=branch%3Afix%2F04-yale-launch-readiness).
CPU probes do not establish CUDA compatibility or Yale sandbox enforcement.

## Remaining blockers and next commands

All four main methods **A, B, C and D remain not ready to run**. Yale worker
isolation has not been established, the trainer SIF has not been built, and
production GPU acceptance, base evaluation, the D pilot/continuation and study
freeze remain outstanding. Backend checks can now proceed independently once
the trainer image and GPU authorization are available.

The last worker observation remains job `30218129` on
`c1104u05n03.mghpcc.ycrc.yale.edu`: cgroups v2, UID 35094, absent user D-Bus
socket, and Apptainer refusing to apply resource limits before startup. This
repair did not repeat that probe or attempt a privilege/service workaround.
The [unsent support request and proposed Yale managed service](../yale-worker-support.md)
record required security properties and the site/deployment approvals still
needed. No support message was sent and no alternative host was provisioned.

Local storage still has approximately **22 GiB** free, below the trainer export's
60 GiB threshold. No sufficiently large external build root or builder was
selected. Configurability removes the repository-filesystem requirement;
capacity must still be supplied for both the export and Docker builder storage.

The post-repair read-only `hpc doctor` confirmed a clean checkout and validated
prepared manifest `44fe6caffc0548bebb01bf39208f8b2f5de10ab89ee8487dca20dd436c0154f5`.
Remote inspection was unavailable because SSH requires keyboard-interactive
authentication. No remote readiness state was refreshed and no job was allocated.
The exact next command, in an interactive terminal, is:

```bash
uv run selfplay hpc login
```

Complete normal Yale MFA, then rerun `uv run selfplay hpc doctor`.
Before image preparation,
select sufficient export/builder storage and build the trainer with the
documented `--build-root` / `--builder` options; monitor its CPU conversion job.
The build command synchronizes the clean branch into a new immutable checkout.
Do not launch using the old main snapshot.

After image conversion and **later authorization for GPU use**, the executable
backend-only command is:

```bash
uv run selfplay hpc prepare-study --stage backend --gpu-hours-limit 24 --watch
```

It preserves the recorded readiness budget and stops after backend acceptance.
Resolve the worker architecture and regenerate full runtime/data evidence before
base evaluation, the D pilot and freeze. Main A–D launch remains a later action.

## Resource and artifact accounting

Task 04 spent **0 GPU-hours** and submitted **no Yale jobs**. No main experiments,
support contacts, site changes or remote artifact rewrites were performed.
Existing Yale evidence remains historical and auditable. Scientific configurations,
pinned dependencies and Dockerfiles are unchanged; implementation hash changes
correctly require fresh compatibility/runtime evidence before experiments.
