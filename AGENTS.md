# Project and operating instructions

This repository studies controlled Python repair self-play with on-policy self
distillation (OPSD), GRPO and verifiable rewards. The authoritative scientific
specification is `docs/task/01-experiment-implementation-spec.md`; early proposal
documents are historical. Read `README.md`, `docs/methods.md` and
`docs/operations.md` before changing experiments. Related papers are in
`docs/papers`.

## Engineering invariants

- Use the pinned `uv.lock`, digest-pinned Docker images for reproducibility, and
  Apptainer for the trainer on Yale. W&B is offline by default; never put keys in
  YAML, Git, scheduler arguments or logs.
- Keep science, hardware, environment and logging settings separate. Changes to
  method, data, model, prompts or scientific code require new compatibility
  evidence and, after freeze, a new study ID. Never silently modify A–D
  comparisons or move held-out data into training.
- Checkpoints contain both adapters, optimizers, RNG and curriculum state.
  Resume only a checksummed project checkpoint with matching scientific/data/
  prompt identity. A failed partial optimizer step is never resumed directly.
- Keep `README.md` short and link detailed material from `docs/`. For each task
  spec under `docs/task/NN-...-spec.md`, create the matching plan under
  `docs/plan/NN-...-plan.md` and a completion record under
  `docs/task_completion/NN-implemented-....md` after implementation.
- Keep implementation modular and DRY. Fix code errors and run relevant tests
  before declaring readiness. CPU tests do not establish GPU backend, memory fit
  or generated-code isolation on Yale.

## Yale operations

The current task is `docs/task/08-task-learnability-probe-spec.md`; its plan is
`docs/plan/08-task-learnability-probe-plan.md`. Task 08 authorizes the common
patch-format clarification, complete protocol hashing, a new study/data revision,
and bounded acceptance windows. Task 07's worker/resource rules remain in force
where Task 08 does not supersede them. Main training remains unauthorized. The cumulative
cpsc4880 readiness allowance remains 24 allocated GPU-hours, including prior
attempts and outstanding reservations. Main A–D training and new infrastructure
remain unauthorized.

Worker safety is outcome-based: bound CPU, memory, process/thread creation,
temporary storage, output and time; isolate trainer/oracles/credentials, network
and GPUs; verify cleanup. Investigate kernel-enforced single-process execution
before subordinate UIDs or a service. Nested user cgroups, pids.max and exactly
32 processes are not requirements when equivalent protection is proven. Never
execute generated candidates before real Bouchet containment acceptance.
Trusted CPU fixtures and backend checks remain independent of worker readiness.
Reuse evidence by relevant dependencies, preserving provenance. Independent
operations must not share a submission slot merely because their report parent
is the same. Prefer H100, but entitled H200 is authorized for a documented memory,
scheduling or runtime reason without first manufacturing an H100 OOM. Preserve
GPU ownership and initial-occupancy checks. Preserve all scientific settings.

The local controller is
`uv run selfplay hpc --help`. Its standard-library remote endpoint is
`scripts/hpc_remote.py`; `scripts/yale.slurm` runs allocated jobs. The default
alias is `yale-hpc-cpsc4880`. The `cpsc4710` and `sds6190` aliases are inspection
only until each account's authorization, course policy and shared limits are
verified. All three reach Bouchet, so never assume independent GPU pools.

Task 03 authorizes CPU setup and a total of 24 allocated GPU-hours for readiness
on `cpsc4880`, starting with H100. It ends before main A–D training. The old
AWS hosts are unavailable: use Yale only and stop if its required worker
isolation cannot be enforced. Use `hpc images build --kind worker`, then
`hpc prepare-study --watch`. The preparation command identifies the next missing
image or failed evidence gate. Never treat a passed CPU test as target readiness.

1. For Task 08, start with `uv run selfplay hpc doctor --settings configs/task08.yaml --gpu-type h200` and `uv run selfplay hpc inspect`.
   The commands are read-only and do not allocate resources. Inspect the live
   account/QOS associations, partitions, GRES, queue, quotas and software.
   YCRC web documentation is guidance; Slurm is the live authority. If SSH says
   keyboard-interactive authentication is needed, run `uv run selfplay hpc login`
   in a terminal and complete the normal MFA prompt. It opens a two-hour idle
   SSH control connection for noninteractive controller calls. Never work around
   Yale authentication or guess authorization.
2. Before a run, require a clean committed Git tree and prepared, validated
   data. `uv run selfplay hpc sync` sends a Git bundle to a commit-specific
   checkout in project storage and copies the frozen dataset to a manifest-hash
   directory. Durable runs/checkpoints live in the user's personal directory
   inside course project storage; model caches live in the corresponding
   personal course scratch directory. The controller rejects a missing,
   unwritable or ambiguous storage directory.
3. A submission needs the actual account, partition, typed GPU resource,
   profile, SIF path and SHA-256. The controller checks live associations and
   node fit; `scripts/yale.slurm` hashes the SIF on the allocated compute node
   before invoking the image. Do not run model, image-hash or sandbox work on
   login nodes. Use `backend-check` before pilot, then base evaluation and a
   measured D pilot before `freeze-study` and main A–D runs.
   Use `hpc resources --profile ...` to validate a proposed request without an
   allocation. Choose GPUs using measured pilot memory, phase time and allocated
   GPU-hours, not a transient `sinfo` snapshot.
4. The current Yale worker is an authenticated loopback broker on the allocated
   compute host, inside an independently limited no-GRES Slurm step. Fresh
   Apptainer payloads install the single-process seccomp policy before candidate
   compilation/imports. No external worker host or SSH tunnel is needed. Require
   real containment, OOM recovery and scheduler-termination acceptance. Keep any
   optional persistent credentials in a user-owned mode-600 file outside Git.
5. Use `hpc status`, `hpc logs` and `hpc fetch` to inspect a submitted run.
   `hpc watch` can resume a previously authorized training run after a confirmed
   PREEMPTED, NODE_FAIL or BOOT_FAIL state, at most twice, from the latest
   verified joint checkpoint. It reconciles the durable ledger with Slurm first.
   If a submission response or scheduler state is uncertain, stop and inspect;
   never create a duplicate. Scientific failures, OOM, authentication, quota
   and unauthorized resource requests are not automatic retries.

## Task 08 execution checkpoint — 2026-10-28

Execution code `e1836a0` passed hosted CI 40627889719 (238 tests, one skipped,
three deselected). Protocol v2's common strict format clarification is applied;
model, sampling, rewards and mathematical budgets are unchanged.

Fixed probe 31069728 / `7f6319f6b8b125ac11eaa7af` passed: 32 proposals, five
valid mutations, 20 real student episodes/9,252 tokens, and a valid source-read
then patch/test action. No optimizer updates/refills occurred. Do not repeat it.
CPU revision 31069732 / `d4630697f6f230e3021769b7` passed all 306 task admissions
and target verifications, regenerated anchors, and preserved every original
parent/task/split/coverage file byte-for-byte. `data/prepared` remains unchanged.

Use `configs/task08.yaml` for the final study `repair-selfplay-2026-protocol-v2`,
with immutable data in `data/protocol-v2`. Manifest file SHA-256:
`934412b2181ead4377d383130a19ad0d58eaab6bb9f74c5d8005416d09412fd2`;
canonical manifest: `d9140367e20587051b328dda20a7d93916ee5cadc7e72c2564523cf6ab6beade`;
prompt: `07968cec687ca52d8ce02c4a6d52eaca529f830855690cab1caff71c096f4322`.
`configs/task08-source.yaml` is historical source/probe setup, not main settings.

Launch readiness remains false pending fresh runtime/backend evidence, corrected
greedy baseline, real D continuation/learning coverage, bounded A/B/C acceptance,
freeze and four previews. Worker proof and SIFs are reused with original provenance.
The probe cost 0.15 GPU-hours; reconciled cumulative consumption is 1.241111111
of 24, with no outstanding reservation at this checkpoint. Inspect live state
before new allocations. Main training remains unauthorized.

All studies share `state/budget-readiness-cpsc4880.json`, authorization
`cpsc4880-readiness-24h-v1`. Import all historical ledgers; changed legacy history
requires reconciliation. The 24-hour total and 12-hour pilot sublimit persist.
The fixed probe uses two batches, 32 proposals, four episodes per valid task,
no updates/refills, and at most two allocated GPU-hours. Do not restart for luck.
A/B/C use one three-iteration window each; clearly labelled optimizer fixtures
may establish zero-advantage mechanics but cannot replace real workload.
D retains ten iterations and the iteration-two stop/resume design.

The previous 0/41 baseline was sampled due to a lost greedy flag. It remains
historical sampled evidence and cannot satisfy corrected greedy evaluation.
The read-only queue inspection found no active user jobs; accounting must still
be reconciled before allocations. Historical consumption below is not a new grant.

## Task 07 historical checkpoint — 2026-10-27

Execution code `9980c78` passed hosted CI 40617600830 (220 tests). Real worker
acceptance 31033392 passed all 19 checks; runtime 31045342 validated all 306 tasks;
one-H200 backend 31040984 passed numerical/adapter/vLLM/checkpoint acceptance.
Base evaluation 31046386 completed 0/41 repairs. The selected request is one H200,
seven trainer CPUs/128 GiB plus one worker CPU/1 GiB (eight CPUs/129 GiB total).

D jobs 31049338/31050754 completed ten iterations with verified resume at two,
but 0/160 proposals were valid and no episodes, updates or alignment scores were
produced. Independent A/B/C jobs 31054795/31055413/31056122 each committed one
iteration and a joint checkpoint, but also had zero yield/updates and failed
acceptance. **Launch readiness is false.** No freeze or successful main preview
exists; all four dry runs refused eligibility. No main job was submitted.

Cumulative consumption is 1.091111111 of 24 GPU-hours, with no outstanding
reservation, active job or watcher. Do not repeat the unchanged zero-yield pilot
or reinterpret empty-workload timings as training throughput. A concrete shared
format clarification is in `docs/task/07-shared-diff-format-decision.md`; user
authorization is pending because Task 07 preserves prompts. No prompt/parser,
sampling, reward, model or frozen-data change has been applied. A new study
identity must not reset the existing cumulative allowance. Read
`docs/task_completion/07-implemented-yale-execution-blocked-readiness.md` and the
launch runbook for exact evidence and the remaining scientific decision.

## Historical checkpoints (superseded operational rules; inspect live state)

The Task 07 rules above take precedence over mechanism and hardware restrictions
in every checkpoint below. Job states and accounting are historical snapshots.

## Task 03 historical checkpoint (2026-10-22)

- Normal Duo MFA established a control session for `yale-hpc-cpsc4880`.
  `cpsc4880_ts2236` has Slurm account `cpsc4880`, with `interactive,normal`
  QOS and a blank partition restriction. Read-only resource checks accepted
  the intended H100, H200 and RTX Pro 6000 Blackwell two-GPU requests. The
  scheduler still decides when a job can start.
- The writable personal course project and scratch directories are available.
  Group quota at the Oct 21 snapshot was 979/4096 GiB project and 515/10240
  GiB scratch. The user had no active jobs; two jobs from other `cpsc4880`
  users were running. The implementation commit `213647a` and validated data
  manifest `44fe6caffc05...` were synchronized to the user's project space.
- Task 03 built and converted a Linux AMD64 worker SIF successfully (job
  30217522; SHA-256 `76a43eef3424...`). Worker probes 30217591 and 30218129
  failed before candidate startup because Apptainer could not apply cgroup
  limits. The second node had cgroups v2 but no `/run/user/35094/bus`; normal
  user-bus discovery could not supply `DBUS_SESSION_BUS_ADDRESS`. This requires
  a YCRC-supported execution setup; do not remove limits or probes to proceed.
- Readiness used zero GPU-hours (approximately 0.148 allocated CPU-hours).
  No trainer SIF, GPU backend proof, base evaluation, pilot or frozen study has
  been established. Main A–D have not started. Local free storage was about
  22 GiB, below the 60 GiB trainer-build preflight minimum.
- `cpsc4710` and `sds6190` noninteractive access was rejected by MFA. A Duo
  prompt for `cpsc4710` timed out, so their permissions and quotas remain unknown.

Read `docs/task_completion/03-implemented-experiment-readiness.md` for the exact
state, fetched diagnostics, remaining gates and a YCRC support-request draft.
Re-check live state before new work. Trusted `backend-check` and
`hpc prepare-study --stage backend` do not require a worker or target execution
validation. Task 05 authorizes bounded GPU readiness after CPU/hosted CI. Backend evidence
uses `compatibility/backend-v2`; legacy evidence cannot satisfy the new contract.
Resolve the supported worker execution setup before generated-code work.
Trainer exports support `hpc images build --build-root PATH --builder NAME`;
the Docker builder also needs sufficient cache storage. Do not provision
another host. Task 05 explicitly authorizes contacting support.
The support request and proposed Yale managed-service design are in
`docs/yale-worker-support.md`.

## Task 05 checkpoint (2026-10-22)

- Yale-native Dockerfile-derived trainer build is verified. CPU job 30275129
  published an 11.02-GiB SIF, SHA-256
  `fbe2737a6815644ecaad01df40cfa7883b05e5554d518fd0352129b1f6fd69e2`,
  from source `2d17d4d`. Base Python 3.12.11, pinned packages/imports and CLI
  passed. Build mode is `hpc images build --kind trainer --build-mode yale`;
  no large local archive is required. CPU attempts consumed 11.6556 CPU-hours.
- The first H100 reference job 30278347 failed after 78 seconds with a nearly
  full device but only about 7 GiB attributed to this process. Allocation mapping
  was not proven, so this does not authorize H200. Source `cf1e644` now resolves
  Slurm GPU UUIDs, verifies actual container UUIDs/initial occupancy and rejects
  ambiguous or occupied devices. Its hosted CI passed 151 tests. Live acceptance
  of this correction is pending.
- The first reference attempt cost 0.021667 GPU-hours of the authorized 24. No main
  A–D runs have started. All production readiness gates remain false. Normal Duo
  reauthentication succeeded. Job 30287970 failed after 14 seconds because
  NVIDIA's CSV query rejects `minor_number`; source `eebd4c0` now reads/join XML
  minor numbers by UUID and passed hosted CI (151 tests). Total charged readiness
  cost is now 0.025556 GPU-hours (92 seconds).
- Corrected one-H100 reference job 30289141 / run `db32746277268abab0265c7e`
  was submitted from `eebd4c0`, last seen PENDING. It has a one-hour cap and
  reserves one GPU-hour. There is no watcher or automatic follow-up. Inspect this
  existing job before further preparation; do not duplicate it. After it passes,
  synchronize the clean committed checkout and resume backend-only preparation.
  If it fails, inspect/repair before another attempt. No two-GPU job is submitted.
- YCRC ticket 79403 / Gmail thread `1a1640b41863817b` contains the authorized
  request. Only an automated acknowledgement is available. No equivalent worker
  probe was repeated; require a documented supported setup change before another.
  Follow up in that thread after three business days (2026-10-27) if unanswered.

Read `docs/task_completion/05-implemented-yale-experiment-readiness.md` and
`docs/yale-launch-runbook.md` for exact artifacts, CI evidence, remaining gates
and continuation commands. The Task 03 absence of a trainer and zero GPU cost
above are historical and have been superseded by this checkpoint.

## Task 06 acceptance in progress

The user requires actual submission readiness, not implementation completion.
Job 30289141 failed before model load after 16 seconds; reconciled cumulative
readiness consumption is 0.03 GPU-hours. No H100 memory-fit failure is proven.
The old pending-job description above is superseded. No equivalent worker
acceptance probe may run until a documented supported setup change.

Task 06 adds `hpc worker-diagnose` (CPU only) and five-minute
`hpc allocation-check --profile single_gpu_80gb|dual_gpu_80gb` operations.
GPU ownership must be proven through the actual Slurm device cgroup and container
UUIDs; NVIDIA index/minor equality is not required. Diagnostics cannot establish
backend compatibility or worker safety. Runtime evidence includes the allocation
code and documented worker setup identity. See the launch runbook for commands.

Continue through all eight gates and four valid previews before recording Task 06
as complete. Existing approved Yale services are the only worker fallback;
new infrastructure, main training and exceeding 24 GPU-hours remain unauthorized.

Task 06 live checkpoint: commit `8bb364b` passed hosted CI (169 tests).
CPU diagnostic 30293989 proved missing user bus, non-writable cgroup controls and
no `pids` delegation below slurmstepd.scope; YCRC updates are recorded in the
support document. Normal-queue diagnostic 30295248 was cancelled before start
(zero GPU time). Corrected diagnostic 30296835 ran for 20 seconds and proved
kernel/container UUID ownership, then rejected an already occupied GPU on
`a1127u05n01` (VLLM worker PID 572771, 73,814 MiB). Reconciled GPU cost is
0.035556 hours. No model was loaded and no H100 memory-fit failure is established.
No controller is currently watching that failed run. The next code revision
adds an optional `scavenge_gpu` route for stateless trusted diagnostics only,
explicit node exclusions and `--no-requeue`; verify its CI before GPU use.
Continue with `hpc prepare-study --stage backend --diagnostic-partition scavenge_gpu
--exclude-node a1127u05n01 --gpu-hours-limit 24 --watch`. Inspect existing ledgers
first. Main jobs retain normal partitions and remain unauthorized.

Task 06 update 2026-10-27: YCRC replied to ticket 79403 (thread
`1a1640b41863817b`, message `1a17e27e17b5f49f`). Users cannot configure cgroups
on Bouchet; YCRC recommends Slurm job arrays/steps for per-task CPU/memory and
Apptainer for filesystem/network isolation. Slurm has no process-count limit,
so this guidance alone does not satisfy the existing per-worker 32-process
contract. Do not execute candidate code yet. A concise follow-up asks whether
Apptainer subordinate UID/GID mappings are available and allowed, to evaluate a
unique mapped UID plus kernel-enforced `RLIMIT_NPROC=32` inside a no-GRES Slurm
step. See `docs/yale-worker-support.md` and the updated Task 06 plan. This path
requires adversarial escape/cleanup checks and real Bouchet acceptance; it is
only a proposal.

Normal SSH now requires interactive Duo MFA. The previous `scavenge_gpu`
watcher exited when its SSH control connection failed; job 30299146 has not
been reconciled since then. Reauthenticate with `uv run selfplay hpc login`,
inspect that job and accounting, then resume monitoring without duplicating it.
The existing 24 GPU-hour cap and H100-first/OOM-only H200 rule still apply.
The user has not answered the separate H200 exception question. No main A–D
jobs have been submitted; Task 06 remains incomplete.
