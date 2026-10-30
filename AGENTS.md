# Project and operating instructions

## Standing submission authorization — 2026-10-30

The user authorizes the submissions needed to complete the corrected study,
including main GPU allocations and necessary continuations, without asking for
permission again. Start corrected Experiment A with one H200 allocation through
iteration 20; review scientific health at iterations 5, 10 and 20 and verify the
joint checkpoint before deciding on continuation. Reconcile live scheduler and
ledger state before submissions and never duplicate an uncertain job. This
authorization does not waive scientific identity, isolation, accounting,
checkpoint, failure-handling or held-out-data safeguards. Historical statements
below that main training requires a new authorization record earlier decisions
and are superseded for work needed to complete the corrected study.

Corrected study live checkpoint: A run `747bd2e097258445de647e47` started as
job `31364215`, completed iteration 20 cleanly in 1:34:28, and continued from
the controller-verified checkpoint `000020` in job `31372772`. Its development
result improved from 0/41 to 14/41; 34/320 proposals were valid and 36/136
episodes succeeded. Iteration-5 and -10 health reviews passed, and W&B cloud
history was verified through iteration 20. Fresh corrected B run
`f3271727f5fd25c07466e50d` completed its iteration-20 stop in job `31372774`:
13/41 development successes versus 0/41 initially, 30/320 valid proposals and
20/120 successful episodes. Its iteration-5, -10 and -20 health and cloud
delivery reviews were performed. Controller-verified checkpoint `000020`
continued in job `31381376`. Inspect these existing jobs and ledgers before
any action; do not duplicate them. Monitor development and telemetry for
regression. A's later development results were 12/41 at iteration 40, 11/41
at 60, 10/41 at 80, 11/41 at both 100 and 120, 10/41 at 140, and 11/41
at both 160 and 180,
following 14/41 at 20.
B was 13/41 at both 20 and 40, 12/41 at 60, 11/41 at 80, 9/41 at 100,
and 10/41 at both 120 and 140.
A's iterations 98–117
produced only 6 valid proposals out of 320 attempts, with 131 duplicates;
monitor the declining proposal yield. A has not repeated the earlier collapse
to zero. A's W&B delivery became intermittent after iteration 142 with repeated
`OSError` warnings, while training and local scientific records continued; cloud
history later reached 181, but the dashboard can incorrectly show finished. B
also reported delivery errors; its cloud history reached at least 133. After each
trainer writer has stopped, reconcile and cloud-verify checkpoint-backed aggregates
as necessary with the single-writer recovery mechanism. These job
states are a snapshot, not live state.

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

**Authorized initial A run:** the user explicitly authorized one H200 allocation
up to 23 GPU-hours, capped at iteration 20, with no automatic continuation and no
B/C/D. Controller run `a87f3772b284d35a375cf69e`, Slurm job `31219895`, was
submitted with frozen execution `54f087fdc5cf5170800d825bf1ee92b89a43c8ed`.
Inspect this existing run and scheduler state before any action; never duplicate
it. Review live W&B and scientific health at iterations 5, 10 and 20. A clean
checkpoint at 20 requires a new authorization for continuation.


Task 10 online successor is `repair-selfplay-2026-main-online-v1`, settings
`configs/task10-online.yaml`, frozen execution
`54f087fdc5cf5170800d825bf1ee92b89a43c8ed` (CPU freeze 31206522).
Frozen-source CI 40784025848 passed 272 tests. Allocated-node W&B cloud delivery
and same-run cross-process resume passed in CPU job 31208617; authorized entity
is `tatsu432s`, username `tatsu432`. All four actual online main previews passed;
no main training job was submitted, and main training remains unauthorized.
Use `docs/task10-launch.md` and the Task 10 completion record. Credentials remain
outside Git in private files. Login uses isolated W&B 0.22.3 for new-format keys;
the frozen trainer SDK/image/lock remain unchanged and online delivery is proven.
The user accepted the measured proposer numerical sensitivity for a bounded
initial A experiment. Main GPU-hour authorization is still required. Student total
counterfactual gradient cosines were about 0.9986, but the actual-D proposer
counterfactual Adam-update cosine was 0.58968. No estimator/science correction
was adopted. Replays 31208618 and 31212941 completed; the shared locked ledger
is reconciled at **7.79/24 allocated GPU-hours**, zero reservation, no active
Slurm job/watcher. Do not repeat probes to seek a different outcome or treat
controller eligibility as scientific risk acceptance. Task 09 and Task 08
freezes below remain separate preserved baselines.

Task 09 v2 is launch ready with caveats on `feat/09-efficiency-learnability`,
`configs/task09-diagnostics.yaml`, study `repair-selfplay-2026-efficiency-v2`.
Freeze 31197007 and all four actual previews passed; frozen execution is
`93a8533beeae26e54bb1510caa12991fb2471337`. CI 40748915396 passed. Main A–D
training remains unauthorized. The Task 08 frozen study is a separate valid
baseline. See the Task 09 completion record and launch runbook for evidence,
numerical caveats, costs and exact commands. Reconcile live jobs and budget
before operations; last reconciled consumption was 7.478055556/24 GPU-hours,
with no outstanding reservation.

Task 09 freeze 31149887 is incomplete for launch planning: it contains no
resource plan, and controller previews use a 484-second stop lead and one
continuation. The controller reports eligibility, but do not submit this study
until a workload-based resource plan is bound to a correctly frozen study and
the remaining performance validation is complete. Preserve the freeze for audit.

The follow-on mismatch instrumentation changed scientific code identity and
passed its own dependency-scoped validation. Its frozen measured plan uses stop
leads 4616/4540/4717/5211 seconds and 5/5/5/6 continuations for A/B/C/D.
The earlier v1 freeze above remains unsuitable for launch.

The last accepted readiness task is `docs/task/08-task-learnability-probe-spec.md`; its plan is
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

## Task 08 verified launch checkpoint — 2026-10-28

Launch readiness is **true**: all ten gates, immutable freeze 31082462 and four
actual A–D controller previews passed. **No main job was submitted; main training
remains unauthorized.** See `docs/task_completion/08-implemented-task-learnability-probe.md`
and the launch runbook for exact reports, hashes, costs and commands. Do not repeat
completed probes/pilots to obtain another stochastic outcome.

Use `configs/task08.yaml`, study `repair-selfplay-2026-protocol-v2`, immutable
`data/protocol-v2`, one H200 (`single_gpu_80gb`), seven trainer CPUs/128 GiB plus
one worker CPU/1 GiB. Seed 2026, 200 main iterations and all mathematical budgets,
objectives, rewards and sampling are unchanged. Main payloads pin frozen execution
`983a6cf57ce5caf5c9aa7ddf53f5ad7f313b4e73`; controller metadata fix `25b07f2`
reports the correct 200-iteration budget. Final settings checkpoint every iteration.
Frozen-source CI 40639028520 passed 242 tests; controller CI 40640124480 passed
244 tests (one skipped, three deselected), including Linux/Docker acceptance.

Prompt hash: `07968cec687ca52d8ce02c4a6d52eaca529f830855690cab1caff71c096f4322`.
Manifest file SHA: `934412b2181ead4377d383130a19ad0d58eaab6bb9f74c5d8005416d09412fd2`;
canonical manifest: `d9140367e20587051b328dda20a7d93916ee5cadc7e72c2564523cf6ab6beade`.
Original parents/tasks/oracles/splits/coverage are byte-identical; all 306 revised
admissions/target validations passed. `data/prepared` remains intact.
`configs/task08-source.yaml` is historical probe/revision setup, not main settings.

Worker 31033392 (19 checks) and both SIFs were reused with provenance. Probe
31069728 passed its fixed 32 proposals / five valid tasks / 20 episodes. Revision
31069732, runtime 31071377 and backend 31071410 passed. Corrected **greedy** base
31072777 completed 0/41; the older 0/41 remains historical sampled evaluation.
D jobs 31075857/31077825 completed ten iterations with verified resume at two,
18 valid tasks, 72 episodes, ten steps per optimizer and 18 alignment scores.
A/B/C 31080094/31080096/31080097 each completed one three-iteration window,
six valid tasks, 24 episodes and three actual steps per optimizer. C scored six
alignments. Fixtures are labelled, state-restored and separate from real yield.

Shared authorization `cpsc4880-readiness-24h-v1`, ledger
`state/budget-readiness-cpsc4880.json`: 3.234722222/24 GPU-hours consumed,
zero outstanding reservation, 31 reconciled records. Shared pilot use is
1.025000/12 hours. No active job or HPC watcher remains. Reconcile live state
before future operations; new study IDs never reset either allowance.

Whole-device measured peak was about 71.81 GiB of 140.40 GiB; host phase RSS
reached 28.45 GiB across actor/descendants. Samples are lower-bound peaks and
include rollout startup. Conditional total main training+monitoring cost is
70.24 hours at observed yield, 304.85 hours with all proposals valid at
observed lengths, or 329.45 planning hours. These are scenarios, not
future guarantees. Final evaluations/diagnostics are additional. Segments are
23 hours with measured stop lead 4,503–4,874 s and 5/5/6/5 clean continuations.
No representation/model/sampling/reward/budget change is authorized by readiness.
Initial probe/base B policy-cache files were archived outside the main namespace;
all four main directories and submission ledgers are absent. Their scientific
records and the freeze remain unchanged; see the closeout record for provenance.

## Task 07 historical checkpoint — 2026-10-27

This section records the 2026-10-27 snapshot. Task 08 above supersedes its
readiness status and pending shared-format authorization.

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

The current Task 08 instructions and retained Task 07 worker/resource rules take
precedence over mechanism and hardware restrictions below. Job states and
accounting here are historical snapshots.

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

## Task 10 initial main A stage — 2026-10-28

The user authorized one bounded Experiment A H200 allocation through iteration 20. Job 31219895 completed successfully, stopped at iteration 20 and consumed 1.527778 allocated GPU-hours; no continuation or B–D job was submitted. The latest checkpoint is `000020`, with complete manifest and resumable metadata; the controller must verify file checksums and identity before any future resume. W&B run: https://wandb.ai/tatsu432s/self-play-opsd-coding-agent/runs/9c125e660c942541076d5d50. Valid proposals 30/320, successful solver episodes 40/120, development evaluation 12/41 after training versus 0/41 initially. See `docs/task_completion/10-implemented-main-experiment.md` for scientific health, limits and accounting. Further main allocation requires authorization; do not automatically continue.

## Task 10 Experiment A completed — 2026-10-29

The user authorized continuation to the frozen 200-iteration target. Initial job 31219895 and continuation 31232768 used 7.7025 allocated H200 GPU-hours in total. Slurm recorded the continuation `COMPLETED` (exit 0); trainer health is `complete` at iteration 200 and the latest checkpoint pointer is `000200`. No B–D jobs have been submitted. Scientific events report 173/3,200 valid proposals, 162/692 successful episodes, and 5/173 mixed-outcome groups. The 41-task development success was 12 at iterations 20 and 40, then declined to 0 at iteration 100 and remained 0 through iteration 200. Final held-out evaluations have not been run. W&B delivery failed after iteration 139 (86 recorded delivery errors); the cloud dashboard is stale, while local events and checkpoints are preserved. A full checkpoint-inclusive fetch/reconciliation remains outstanding; see the Task 10 completion record and `docs/task10-launch.md`. Do not interpret cloud finished status at 139 as a training stop. Do not submit B–D or new training without authorization.
