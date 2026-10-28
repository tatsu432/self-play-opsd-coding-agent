# Self-play OPSD coding-agent experiment

Task 09 optimization code is staged under `configs/task09.yaml` and is not
launch ready. The [Task 09 record](task_completion/09-implemented-efficiency-learnability.md)
lists CPU evidence, remaining H200 checks and the read-only health report.
Task 08's frozen launch commands remain the active accepted path.


Task 08 launch readiness is verified: all ten gates, a frozen common study and
four actual A–D submission previews pass. Follow the active [Yale launch runbook](yale-launch-runbook.md)
and [completion record](task_completion/08-implemented-task-learnability-probe.md).
Use `configs/task08.yaml` and immutable `data/protocol-v2`. Corrected greedy base
completed 0/41; the earlier 0/41 is historical sampled evidence. Shared readiness
consumption is 3.234722222/24 GPU-hours with zero outstanding reservations at
closeout. **Main A–D training remains unauthorized and unsubmitted.** Older
workflow examples below are historical where they conflict with that runbook.

Controlled Python repair self-play with gated signed-gap on-policy
self-distillation, verifiable rewards, and gradient-guided task generation.
Read [methods.md](methods.md) for the implemented equations and boundaries.

| Method | Student | Valid-task proposer reward |
|---|---|---|
| A | GRPO + ungated signed-gap SD | `4*p*(1-p)` |
| B | GRPO + gated signed-gap SD | `4*p*(1-p)` |
| C | Same as B | RL-gradient alignment with canonical edit CE |
| D | Same as B | Combined-gradient alignment with the same CE target |

Optional `GRPO` and `GRPO_KL` methods do not require a fifth main run. A–D retain
16 attempted proposals, four episodes per valid task, and one optimizer step per
mathematical batch. Scores are **MBPP+-derived repair** and
**HumanEval+-derived repair**, not official benchmark generation scores.

## Local setup and verification

Use Python 3.12 and uv 0.8.15. Run from the checkout root:

```bash
uv sync --frozen
docker build -f docker/worker.Dockerfile -t selfplay-worker:local .
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run mypy src/selfplay_opsd
uv run pytest -q
uv run selfplay doctor --settings configs/local.yaml --isolation
uv run selfplay smoke --output runs/local-smoke
uv run selfplay analyze runs/local-smoke/smoke
```

Choose a new smoke directory each time. Smoke uses a randomly initialized tiny
Qwen model, real sampling/scoring and clearly labeled artificial fixtures. Random
proposals may all be invalid, correctly causing no updates; a separate alignment
witness exercises zero-RL/nonzero-SD gradients. Tests of complete valid-proposal
A–D updates use explicitly mocked generation with real model scoring and Docker
verification. These are correctness checks, not repair-performance evidence.

Tests have separate `sandbox`, `download`, `gpu`, and `wandb_online` markers.
Missing capabilities are reported as skips/blockers. GPU tests require explicit
`SELFPLAY_RUN_GPU_TESTS=1` under an allocated `torchrun` process. CPU correctness
cannot establish FSDP/vLLM or production sandbox readiness.

## Prepare immutable data

```bash
uv run selfplay prepare-data --settings configs/data.yaml --inventory-only
uv run selfplay prepare-data --settings configs/data.yaml
```

Releases and original MBPP examples have pinned SHA-256 checks. The command
isolates correct-program execution, preserves EvalPlus comparators, records
exclusions/families, partitions parents, freezes independent mutations and builds
verified canonical reference/diagnostic actions. It loads the real tokenizer,
without allocating the 4B model. Oracle work is cached for interrupted preparation.

If requested counts cannot be met, inspect `data/prepared/coverage.json` and
explicitly acknowledge reduced coverage:

```bash
uv run selfplay prepare-data --settings configs/data.yaml --allow-reduced
uv run selfplay validate-data --settings configs/data.yaml --execute
```

No held-out parents move into training to fill deficits. Fewer valid frozen
mutants are recorded explicitly. Once finalized, select a new data root for
changes; do not rerun `--inventory-only` inside frozen data. Private oracles and
canonical answers are under `data/prepared/private`, never in student requests
or worker mounts. Public tests expose original public examples only.

## Current Yale execution contract (Task 07)

Follow the [Yale launch runbook](yale-launch-runbook.md). The initial target is one
H100, with H200 permitted for a documented practical reason within the same
cumulative 24 GPU-hour readiness cap. Main A–D remain unauthorized.

The Yale worker uses a separately limited no-GRES Slurm step, fresh Apptainer
payloads, and a trusted kernel seccomp policy prohibiting candidate-created
processes and threads. CPU/cpuset and memory limits come from Slurm; candidate
filesystem/network/GPU containment, temporary storage, output/time and cleanup
must pass real preflight. Nested user cgroups and a precise 32-process limit are
not prerequisites. Trusted allocation/backend checks do not require this worker.
Historical mechanism and H100-OOM restrictions elsewhere are superseded by Task 07.

## Containers and safe execution workers

The uv lock pins torch 2.8.0, Transformers 4.57.1, PEFT 0.17.1, verl v0.7.1 at
`bec9ef74768dd201881cd4e54cd0385e87caae27`, and vLLM 0.11.0. OpenCV is constrained
for compatible NumPy requirements. GPU packages install on Linux x86-64. Docker
base images are digest-pinned. The CPU/debug image is `docker/cpu.Dockerfile`.

On an existing GPU host:

```bash
docker build -f docker/trainer.Dockerfile -t selfplay-trainer:local .
docker image inspect selfplay-trainer:local --format '{{.Id}}'
```

Trainer containers receive no Docker socket. Run an authenticated worker service
on a dedicated execution host, reached by loopback SSH forwarding or HTTPS:

```bash
export SELFPLAY_WORKER_TOKEN="$(openssl rand -hex 32)"
uv run python -m selfplay_opsd.remote_worker --host 127.0.0.1 --port 8765 --concurrency 4
```

Supply the same token through the trainer environment. Never put it in YAML,
arguments, images or committed files. `configs/main.yaml` and `configs/yale.yaml`
use `http://127.0.0.1:8765`; non-loopback plain HTTP is rejected. On Yale, establish
the approved SSH tunnel from the **allocated compute host** to your worker host.
No machines or tunnels are created automatically.

The worker has no egress, capabilities, credentials, host/home mounts, Docker
socket or GPU. Root is read-only; CPU, RAM, PID, disk, output and wall time are
bounded. Preflight tests actual enforcement. A local Apptainer worker backend is
also implemented: configure `backend: apptainer`, `runtime: apptainer`, and its
SIF path. On Yale, use the verified Slurm-step resource backend and single-process policy;
user cgroup delegation is unnecessary. Apptainer availability alone remains
insufficient evidence. An existing approved external worker is a fallback only
when the simpler supported path proves unsuitable.

The four-tool workspace is a trusted virtual file map. Public `run_tests` invokes
the public function cases through the supervisor rather than model-controlled
pytest. Every final verification starts a fresh worker. Expected outputs remain
outside candidate execution; code can observe its currently supplied inputs.
This is OS isolation, not cryptographic test secrecy.

## Switch profiles and verify production backends

| Profile | GPUs | Initial execution policy |
|---|---:|---|
| `cpu_debug` | 0 | Tiny FP32 correctness backend |
| `single_gpu_80gb` | 1 | BF16 LoRA, streaming episodes |
| `dual_gpu_80gb` | 2 | Optional distributed target |
| `dual_gpu_141gb` | 2 | Same mathematical batches on H200-class devices |
| `multi_gpu_single_node_4` | 4 | Sharded coordinates, global reductions |
| `multi_gpu_single_node_8` | 8 | Same strategy across eight GPUs |

Method, resource, environment and logging settings are independent. GPU phases
alternate rollout/scoring/training residency with CPU offload. Every FSDP rank
evaluates the same mathematical microbatches; averaging preserves normalization
while coordinates are sharded. This conservative implementation duplicates data
compute; allocated compute still counts. Memory fit and throughput are measured,
not guaranteed. Multi-node support is not advertised.

Inside the target image/allocation, establish a one-GPU reference, then compare
each multi-GPU profile:

```bash
uv sync --frozen --group gpu
uv run torchrun --standalone --nproc-per-node=1 -m selfplay_opsd.cli backend-check --settings configs/main.yaml --hardware single_gpu_80gb --output runs/backend-single
uv run torchrun --standalone --nproc-per-node=2 -m selfplay_opsd.cli backend-check --settings configs/main.yaml --hardware dual_gpu_80gb --output runs/backend-dual --reference runs/backend-single/reference_gradients.pt
uv run selfplay doctor --settings configs/main.yaml --isolation
```

Repeat with count/profile 2/141GB, 4 or 8 as appropriate. Checks cover two
post-update weight reloads, actor/rollout target-logprob agreement, all-coordinate
gradients and checkpoint restoration. Evidence lives in `runs/compatibility`;
code/model/profile changes invalidate it. Run inside Docker with the checkout
mounted at `/workspace`, writable outputs and host networking for a loopback
worker, or inside the corresponding allocated Apptainer image.

## Base, pilot, study freeze and four main runs

Evaluate the shared untrained baseline first. Commands in this section run inside
the chosen container/allocation:

```bash
uv run torchrun --standalone --nproc-per-node=2 -m selfplay_opsd.cli evaluate --settings configs/main.yaml --base --split dev
```

The launcher runs a separate training-only D pilot, including alignment costs:

```bash
uv run python scripts/launch.py --environment aws --hardware dual_gpu_80gb --settings configs/main.yaml --method D --image selfplay-trainer:local --operation pilot --iterations 10
uv run selfplay freeze-study --settings configs/main.yaml --pilot-report runs/repair-selfplay-2026-pilot/D-seed2026/pilot_report.json
```

The pilot must exercise alignment and both updates. It reports median/p90 phase
cost, memory, gradient influence and actual exposures. Default common lambda is
.1; automatic rescaling is disabled. To change it, set the common configuration
and repeat the pilot before freezing. Main runs reset
adapters, optimizer, RNG and proposal history; A–D initialization hashes must agree.

After readiness gates pass, the exact four main launch commands are:

```bash
uv run python scripts/launch.py --environment aws --hardware dual_gpu_80gb --settings configs/main.yaml --method A --image selfplay-trainer:local
uv run python scripts/launch.py --environment aws --hardware dual_gpu_80gb --settings configs/main.yaml --method B --image selfplay-trainer:local
uv run python scripts/launch.py --environment aws --hardware dual_gpu_80gb --settings configs/main.yaml --method C --image selfplay-trainer:local
uv run python scripts/launch.py --environment aws --hardware dual_gpu_80gb --settings configs/main.yaml --method D --image selfplay-trainer:local
```

Change environment/profile flags without changing science. The implemented
`p5.4xlarge` path uses `single_gpu_80gb`; optional `p5.48xlarge` uses
`multi_gpu_single_node_8`. Supply your own region, quota, storage and current
instance price. Nothing is provisioned. `scripts/aws-cost.py --instance ...
--hours ... --price-per-hour ...` estimates instance cost, excluding storage,
network and tax. YAML `max_active_hours` / `max_gpu_hours` bound execution.

For Yale, use the controller from the local checkout. It discovers account and
partition permissions from Slurm and keeps the run ledger in project storage:

```bash
uv run selfplay hpc doctor
uv run selfplay hpc login  # complete Yale's normal MFA prompt in a terminal if needed
uv run selfplay hpc inspect
uv run selfplay hpc resources --profile dual_gpu_80gb
uv run selfplay hpc sync
uv run selfplay hpc submit --operation backend-check --method D --profile single_gpu_80gb --image /ABSOLUTE/TRAINER.sif --image-sha256 SHA256
uv run selfplay hpc status RUN_ID
uv run selfplay hpc logs RUN_ID
```

Inspection, resource checks and synchronization do not allocate GPUs. A clean committed Git checkout,
prepared validated data, authenticated SSH, discovered project/scratch storage,
and a pinned trainer SIF are required before submission. The controller transfers
a Git bundle to a commit-specific checkout under the user's writable course
project directory and stores data by manifest hash. Model caches use the user's
course scratch directory; the group fileset roots themselves are not writable.
Submit options support `backend-check`, `pilot`, `train`, `evaluate`, and
`diagnose`; run `selfplay hpc submit --help` for their arguments. The selected
profile defaults to H100 for 80 GB and H200 for 141 GB. An RTX Pro 6000
Blackwell may use an 80 GB profile after live authorization and actual backend
acceptance. No profile is assumed to fit merely because its VRAM is sufficient.
Compare measured memory, phase time and allocated GPU-hours on small verified
pilots before choosing between H100, RTX Pro 6000 Blackwell and H200. Queue
state is only a snapshot; `sinfo` does not reserve an available GPU.

The trainer SIF must be created from the pinned Docker image through a Yale
approved distribution path. `scripts/yale.slurm` hashes it on the allocated
compute node and runs the mounted immutable code with `uv`-locked dependencies
inside Apptainer. It writes results/checkpoints to durable project storage and
uses course scratch for model caches. The actual time, CPU, memory and GPU
request is checked against live Slurm associations and node advertisements.
The pre-timeout signal creates a shared stop request for checkpointing at a safe
iteration boundary. Never run model or generated-code work on a login node.

Use `selfplay hpc watch RUN_ID` for bounded
automatic recovery of a prior training submission, or `selfplay hpc resume` for
one check. At most two PREEMPTED/NODE_FAIL/BOOT_FAIL attempts may resume from a
verified joint checkpoint. A missing job ID, ambiguous `sbatch` response, OOM,
scientific error or authentication problem stops resubmission. `hpc fetch RUN_ID`
copies reports locally; `--full` also copies trajectories and checkpoints.
The default alias is `yale-hpc-cpsc4880`. Other aliases are inspection only until
their course permissions and shared limits are verified. Consult the current
[Bouchet partitions](https://docs.ycrc.yale.edu/clusters/bouchet/) before
comparing GPU options.
An optional `--secrets-file` points to a mode-600 file on the compute-visible
filesystem for a worker token or W&B key. The controller passes only its path
to Slurm and never writes the secret into the repository or scheduler arguments.

## Prepare the Yale A–D study

Task 07 continues this workflow with verified single-process workers, compatible
image/evidence reuse and real A–D acceptance checks. Follow the current
[Yale launch runbook](yale-launch-runbook.md); local large-image exports below
remain an optional compatibility path.

Task 03 authorizes Yale `cpsc4880` only and at most **24 allocated GPU-hours**
for readiness. It stops before main training. The old AWS hosts are unavailable.
Run from a clean committed checkout:

```bash
uv run selfplay hpc images build --kind worker
uv run selfplay hpc prepare-study --gpu-hours-limit 24 --watch
```

Image building explicitly targets Linux AMD64, even on an ARM64 workstation.
Docker archives are checksummed, transferred and converted to SIF in a CPU
allocation. Large trainer exports require at least 60 GiB free on the selected
artifact filesystem; worker exports require 2 GiB. Inspect the conversion job with `hpc status` and
`hpc logs`. The preparation command reports `hpc images build --kind trainer`
when the worker has passed and the trainer is needed. Resume preparation after
conversion. Cached completed archives and image manifests are reused.

The worker is a loopback authenticated service started on the allocated host.
It launches isolated Apptainer processes with the shared standard-library
supervisor, without a host training virtual environment. Its ephemeral token
is injected only into the trusted trainer. Every generated-code allocation probes
isolation; site restrictions stop those operations before model work. No Docker socket, external
worker host, privilege workaround or weaker execution contract is used.

Readiness advances through worker isolation, target execution validation,
selected one-GPU backend, distributed gradient comparison only for multi-GPU
targets, base monitoring evaluation,
a ten-iteration D pilot, and study freeze. The pilot stops at iteration 2 and
resumes its joint checkpoint to demonstrate real continuation. Main runs keep
200 iterations, seed 2026 and fresh initialization. Runtime evidence includes
code, model/tokenizer, dataset, prompts, trainer/worker images and execution
limits. Data stays read-only; target validation writes into readiness outputs.

Initial one-GPU ceilings are one hour for backend acceptance, two hours for
base evaluation, six total hours for the D stop/resume pilot, and two hours for
A/B/C acceptance. Reconcile actual consumption and all reservations against the
cumulative 24 GPU-hour cap before each request. Multi-GPU profiles multiply
allocated cost and require additional gradient-agreement evidence. Reservations count their entire requested wall time until
Slurm accounting confirms actual use; uncertain submissions/accounting retain
their reservation. A failed stage is reported with its ledger/evidence instead
of being skipped. Hardware or science changes require compatible new evidence.

`hpc resume RUN_ID --continue-stopped` explicitly permits the recorded number
of clean continuations (one for the readiness pilot). Transient retries retain
the separate limit of two. Resume and watch use the original committed
specification, even after local documentation commits. Slurm COMPLETED is not
scientific completion; the target iteration and expected reports must exist.
Each attempt uses a distinct stop marker.

Freeze executes on a CPU node against the artifacts already on Yale and is
idempotent for identical evidence. A conflicting freeze requires a new study.
Preparation then produces four main-run dry runs; it never submits them.
`hpc doctor` reports `eligible_to_submit_main`, while each actual allocation
still performs runtime preflight. To retrieve the complete report layout:

```bash
uv run selfplay hpc fetch-study repair-selfplay-2026
uv run selfplay analyze runs/hpc/yale-hpc-cpsc4880/studies/repair-selfplay-2026/repair-selfplay-2026
```

`hpc fetch RUN_ID` now follows each operation's real output directory, including
base and backend outputs. `--full` opts into checkpoint/trajectory/W&B files.
W&B remains offline; local events and checkpoints are authoritative.

### Independent backend diagnostics and build storage

Task 04 permits CPU repairs and verification only, with zero GPU-hours. The
following backend preparation command requires later GPU authorization:

```bash
uv run selfplay hpc prepare-study --stage backend --gpu-hours-limit 24 --watch
```

It runs the selected trusted backend (adding a one-GPU reference and distributed
comparison only for multiple GPUs) without a worker image,
worker broker or target execution-validation report. It checks trusted synthetic
gradient fixtures, model loading, verl/vLLM adapter synchronization and checkpoint
roundtrips; generated model output is never executed. `--stage all` remains the
default full readiness flow. Backend completion does not establish A–D readiness.
Synchronization still requires the existing prepared, validated dataset snapshot;
the backend probe itself neither opens nor mounts it.

Backend evidence now lives under `runs/compatibility/backend-v2/`. Its versioned
identity binds code/dependencies, model/tokenizer, objective settings and trainer
image. Worker/data runtime evidence remains separate and mandatory for generated
code. Old compatibility artifacts are retained but cannot satisfy the new gate.

Export storage is configurable, with precedence flag, environment, default
`runs/hpc/images`. For example, after selecting a filesystem with enough space:

```bash
uv run selfplay hpc images build --kind trainer --build-root /PATH/TO/LARGE/DISK/selfplay-images
# Equivalent default for subsequent builds:
export SELFPLAY_IMAGE_BUILD_ROOT=/PATH/TO/LARGE/DISK/selfplay-images
```

`--builder NAME` selects an existing Buildx builder. Archives, temporary exports,
metadata and logs use the selected root, which may be outside the checkout.
Cached archives are checksummed and provenance-checked before reuse; incomplete
exports are removed. BuildKit cache and Docker Desktop's disk image occupy
separate storage: changing `--build-root` does not relocate them. Select a builder
whose backing filesystem has sufficient capacity or move Docker Desktop's disk
image using its settings before building. The controller does not prune caches
or change daemon settings. See [Buildx cache storage](https://docs.docker.com/build/builders/drivers/docker-container/)
and [Docker Desktop storage settings](https://docs.docker.com/desktop/settings-and-maintenance/settings/).

The observed Yale delegation failure and support correspondence are documented
in [Yale worker support](yale-worker-support.md). Task 07 first uses Slurm steps and kernel-enforced single-process payloads.
An existing approved Yale service is a fallback only if this simpler route fails.

## Recovery and artifacts

```bash
uv run python scripts/launch.py --environment aws --hardware dual_gpu_80gb --settings configs/main.yaml --method B --image selfplay-trainer:local --resume runs/repair-selfplay-2026/B-seed2026/checkpoints/000010
```

Only load trusted project checkpoints. Checksums, model/data/prompt/code identity
and parameter order are checked. Both adapters/optimizers restore together.
Rollback can supersede several later iterations; old events remain auditable and
physical compute remains counted. Cached task order is restored from proposal
slots, preserving gradient accumulation order. Cross-count optimizer export is
implemented, but CUDA RNG/world-size migration stays guarded until explicitly
validated. Do not claim bitwise distributed reproducibility.

Runs contain resolved configuration/provenance, `state/health.json`, append-only
events, bounded trajectory shards, Parquet tables, checkpoints, plots and
summaries. `checkpoints/latest.json` points to the latest verified state. Last two,
final and best-development states are retained, plus iteration zero for probes.
`state/rollout_adapters` holds inference exports, not full resume checkpoints.
Inspect health JSON and scheduler process state together: a dead logger cannot
report its own death. Numerical/mapping/oracle errors stop the run. Model failures
are not infrastructure retries; transient infrastructure gets at most two retries.

## Frozen evaluation, diagnostics and analysis

For B's primary endpoint:

```bash
uv run torchrun --standalone --nproc-per-node=2 -m selfplay_opsd.cli evaluate --settings configs/main.yaml --checkpoint runs/repair-selfplay-2026/B-seed2026/checkpoints/000200 --split final_id --protocol frozen
uv run torchrun --standalone --nproc-per-node=2 -m selfplay_opsd.cli evaluate --settings configs/main.yaml --checkpoint runs/repair-selfplay-2026/B-seed2026/checkpoints/000200 --split external --protocol frozen
```

Use matching method settings for A/C/D. Final evaluation requires the frozen
study and common-budget endpoint. Optional `--endpoint secondary` checks the
predeclared best-development choice; test results never select checkpoints.
Evaluate the base on the same splits with `--base`.

```bash
uv run torchrun --standalone --nproc-per-node=2 -m selfplay_opsd.cli diagnose --settings configs/main.yaml --checkpoint runs/repair-selfplay-2026/B-seed2026/checkpoints/000100 --output runs/repair-selfplay-2026/diagnostics/000100
uv run selfplay analyze runs/repair-selfplay-2026
```

Repeat B diagnostics at 0/100/200, or predeclared 0/50/100 for a 100-step study.
One frozen training-only candidate pool beside these output directories is reused.
Each probe restores parameters **and optimizer**, measures separate diagnostic
development CE and leaves main checkpoints untouched. Missing probes are explicit;
constant-score correlations are undefined.

Local-only analysis generates CSV/JSON summaries, 2,000-resample paired parent
bootstrap intervals, per-task outcomes, raw learning curves, PNG/PDF figures,
diagnostic utility, metadata and deterministic success/failure cases. Missing and
incomplete runs stay explicit. Intervals do not measure training-seed variability;
A/B compare separately evolving systems.

## W&B and readiness

Default project is `self-play-opsd-coding-agent`, mode **offline**. Set
`logging.mode: online` and an existing `WANDB_API_KEY` environment variable for
live tracking. No Slurm job prompts for credentials. Offline logs are not live;
disabled mode remains functional.

```bash
uv run selfplay wandb-sync --path runs/STUDY/B-seed2026/wandb/offline-run-SEGMENT
uv run selfplay wandb-export --run-dir runs/STUDY/B-seed2026 --output runs/events-export.json
```

Online resumes reuse the logical run ID; offline segments use separate IDs. Local
lineage is authoritative if W&B contains abandoned events. Delivery is bounded
and best-effort; no full-model watch or default oracle/final-answer uploads.

Dashboard recipe: compare development success against `train/outer_iteration`
and `compute/allocated_gpu_hours`, validity/mixed groups, gaps/gates, RL/SD norms
and cosines, alignment/zero-gradient rates, phase time and VRAM. Custom axes and
bounded plain-text tables are emitted. No account dashboard is assumed created.

Main launches require finalized data, real backend acceptance, a measured target
pilot and frozen common settings. CPU tests do not establish CUDA memory fit,
online W&B, Yale isolation or measured 200-iteration runtime.
