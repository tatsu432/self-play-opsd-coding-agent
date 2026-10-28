# 07 — Establish verified Yale launch readiness

Source: `docs/task/07-fix-ycrc-issue-spec.md`. Main A–D training is excluded.

1. Reconcile existing jobs, immutable ledgers, image/evidence identities and the
   cumulative 24 allocated GPU-hour allowance before replacements. Preserve
   uncertain reservations and equivalent-submission exclusion.
2. Investigate single-process execution first on a CPU node with trusted
   fixtures: install an irreversible seccomp policy before candidate compilation
   or imports; deny process/thread creation and alternate ABI/execution paths.
   Combine it with independent Slurm step CPU/memory enforcement, private
   Apptainer filesystem/network namespaces, bounded temporary storage/output,
   wall time and verified cancellation cleanup. Start at concurrency one.
   Publish actual mechanisms, never a fictional pids.max/32-process cap.
3. Fix CPU cpuset detection, independent operation output/slot identities,
   trusted backend independence, dependency-scoped compatibility and image reuse.
   Permit justified entitled H200 without requiring H100 OOM. Support a measured
   single-GPU target without a redundant distributed comparison.
4. Run focused regressions and hosted CI before expensive acceptance. Complete
   worker/deployed-data validation, backend acceptance, distributed agreement
   only if selected, base evaluation, real ten-iteration D pilot stopping and
   resuming at iteration two, then fresh one-iteration A/B/C acceptance.
5. Freeze only after valid evidence; require live main eligibility and four
   successful submission previews. Record commit/CI/live artifacts, measured
   memory/runtime, cumulative cost, projected main cost and exact launch,
   monitoring and recovery commands. Do not submit main runs.

Preserve all scientific objectives, model/tokenizer/prompts/data boundaries,
seed 2026, 200 main iterations, mathematical budgets, fresh main initialization,
joint checkpoints and offline W&B. Reuse relevant compatible evidence; retain
full provenance. Diagnose failures before repetition. If a concrete external
capability blocks one path, finish independent work and identify the smallest
required action. No subordinate-UID allocator or service investigation until the
simpler route is proven unsuitable. Task 07 supersedes conflicting Tasks 03–06
mechanism requirements and H200 policy.

## Live continuation checkpoint — 2026-10-27

- Current execution implementation is `fa29cd9` (after `ea40906`). Hosted CI for
  `ea40906` passed; the new worker-only OOM changes have local regression evidence
  and live Bouchet acceptance, with their hosted CI in progress.
- Worker job 31033392 / run `639345f9a3d7a5c3953ece55` passed all 19 checks,
  including actual candidate OOM followed by correct-program recovery, bounded
  storage/output/time, native single-process enforcement and Slurm step KILL.
  Five observed contained-step processes were gone/not running after cancellation.
  Slurm's intentional CPU OOM diagnostic message is expected: the broker survived
  and wrote a successful attestation; `--oom-kill-step=0` retains the 1-GiB kernel
  bound while preventing the candidate OOM from killing its whole worker step.
  The active OOM witness is CPU acceptance only, never GPU job preflight.
- Runtime job 31030205 verified all 306 frozen tasks (14 reference, 41 monitoring,
  20 diagnostic, 115 final-ID, 116 external). Effective-limit reporting is now
  shared with scientific configuration; unused legacy pids fields are normalized
  only for the explicit single-process contract. Refresh target runtime evidence
  after the final worker changes; preserve the same dataset and tests.
- H200 job 31030407 passed actual ownership/container UUID/occupancy checks, then
  failed on an import in summary publication. `hpc recover-allocation` verified
  immutable source and raw evidence and published the summary without another
  GPU allocation; the failed Slurm state and original raw report remain intact.
- Backend job 31033410 / run `439186dc49ffec5d599f8356` is active on one H200 in
  `gpu_h200`. Inspect it before replacement. No model-memory result is claimed.
  The scavenge H200 request was cancelled while pending because its estimated
  start was about seven hours away; normal `gpu_h200` has higher priority tier
  and successfully started the short ownership check. All cancellations have
  durable `state/task07-cancel-JOB.json` records and zero GPU allocation.
- Reconciled consumption before this backend run was 0.0475 allocated GPU-hours;
  its one-hour reservation counts until accounting establishes actual usage.
  No main runs have been submitted. Task 07 is not complete.

### Backend initialization repair

Job 31033410 loaded the real pinned 4B model and stopped at student/proposer
initial-version equality. CPU BF16 reproduction showed equal values represented
as FP32 student masters versus BF16 proposer tensors. `e21e04d` promotes the
proposer to the student master dtype before copying, with exact version and
independent-storage regression coverage. No hash check was weakened and no
scientific objective, sampling budget, seed or dataset changed. New backend
acceptance is required; cumulative accounted GPU time is 0.071388889 hours.
Runtime job 31035577 / `f1bd1e3f13671d50990efe81` refreshes the matching source
identity. Image preparation was exercised and reused both old pinned SIFs with
full original provenance; no local Docker daemon or rebuild was needed.

### Rollout startup repair

Backend job 31036051 (`b47b43d30e33fe3089e969f8`) passed identical initialization,
objective-gradient probes and the first optimizer update, then failed at vLLM
architecture inspection: pinned vLLM 0.11.0 interprets CUDA_VISIBLE_DEVICES as
integer indices, while the launcher supplied UUIDs. The launcher now uses the
allocated NVIDIA indices only after an actual container CUDA UUID and occupancy
probe verifies that mask against kernel-proven Slurm ownership. A mismatch still
fails before model loading; no index/minor equality is assumed. Startup exceptions
now travel over the bounded RPC channel, and EOF names the child log.
The failed attempt consumed 73 GPU-seconds; cumulative accounted cost is
0.091666667 GPU-hours. No active GPU job or successful backend proof remains.
CPU runtime job 31035577 completed successfully but needs refresh after this
backend source change. Main A–D remain unauthorized and unsubmitted.

Hardware evidence comparison now normalizes only the default allocated GPU count
(None means used count), allowing CPU freeze to verify GPU-produced reports.
It continues checking the full effective hardware settings, including CPU and
rollout memory configuration, rather than merely the profile name.

### Isolated rollout rendezvous repair

Commit `e0b2336` passed hosted CI 40604870399 (211 passed, one skipped).
CPU job 31037352 (`3f4c23e8e1d923417e87b1c1`) revalidated all 306 tasks.
GPU allocation job 31038198 passed the numeric CUDA mask check (actual index 0,
minor 1, exact allocated UUID and initial occupancy). That proof remains reusable.
Backend job 31038874 (`2eddfa162616da98b977e779`) passed architecture inspection
but stalled initializing its independent process group. Its frontend retained
`TORCHELASTIC_USE_AGENT_STORE=True` from torchrun. A real CPU Gloo reproduction
fails connecting to a nonexistent agent store with that flag, then passes when
cleared. The rollout environment now removes all TORCHELASTIC_* state along with
trainer ranks/rendezvous variables; the GPU mask and cache settings are retained.
A CPU process-group regression verifies independent rank-zero store creation.
The stalled trusted diagnostic was stopped after recording its identity/reason in
`state/task07-stop-31038874.json`; no scientific checkpoint is resumed from it.
Slurm charged 371 GPU-seconds for that attempt and 65 for the successful allocation
check. Cumulative consumption is 0.212777778 GPU-hours, with no active GPU job.
The controller watcher has stopped. Worker evidence is still compatible.

### Backend accepted; fit the scheduler's CPU-per-GPU limit

Hosted CI 40607054798 passed 212 tests. Runtime job 31040084 passed all 306 tasks.
Backend job 31040984 (`d3a9ac3571ccc56dc379385b`), source 6768da2 with the same
e0f4374 scientific code, completed in 222 GPU-seconds. It passed all A–D numerical
probes, separate adapters, two real vLLM weight synchronization/scoring cycles,
sleep/wake transitions and joint policy/optimizer checkpoint restoration.
The canonical report appeared on the login node shortly after Slurm completion;
preparation was resumed from the valid report without another GPU allocation.
Cumulative consumption is 0.274444444 GPU-hours.

The first base submission was rejected before allocation: gpu_h200 allows at most
eight CPUs per GPU; eight trainer CPUs plus the worker CPU requested nine. Set
the Yale trainer to seven CPUs, leaving an independently bounded worker CPU and
1-GiB memory step. This changes operational resources only. Backend, worker and
deployed-data evidence remain compatible; subsequent real pilot measurements will
use this hardware configuration. `doctor --gpu-type h200` can now inspect the
selected target before freeze; default doctor still honors a frozen GPU choice.
No base job, main training or additional GPU allocation was created by the rejection.

### Reclaim a definitively rejected submission slot

Commit a8b8599 passed hosted CI 40608844739 (214 tests). Its corrected base preview
requests eight CPUs and 129 GiB total, retaining a seven-CPU/128-GiB trainer and
independent worker. Live doctor with `--gpu-type h200` recognizes worker, runtime,
single and target as passed. The next submission encountered the old rejected
request's slot: its ledger has no attempts or pending intent and records Slurm's
CPU-policy rejection, but the controller treated every empty-attempt slot as
unknown. Release the slot on definitive sbatch rejection and recognize this exact
legacy state. Empty ledgers without a rejection, active jobs and pending intents
remain blocking; the regression tests distinguish these states. This controller
repair changes no scientific, backend, worker or deployed-data identity. No GPU
was allocated by either rejected/blocked request. Consumption remains 0.274444444
GPU-hours; no main run has been submitted.

### Base no-op guard on sharded weights

Source 43f3a78 passed hosted CI 40609893327 (214 tests). Base job 31044199
(`9f6166b230d98f01927cecaa`) was accepted with eight CPUs total and passed real
worker preflight on a GPU compute host, then failed before repair generation:
PyTorch DTensor has no count_nonzero sharding strategy. Base evaluation now
counts zero/nonzero LoRA B coordinates on local tensors and all-reduces the
count for multiple ranks. The no-op requirement is unchanged. A real two-rank
CPU Gloo/DTensor regression accepts zero weights and makes both ranks reject
a nonzero B weight present only on rank one; nonzero A weights remain valid.
The test uses explicit loopback rendezvous for workstation portability.
Numerical backend identity remains 9a35161e745f6dc007ca4d7761fa096c9af5f21a935fcf6a38f9d9e6cb80fa78;
reuse the successful H200 backend proof. Refresh runtime evidence for the new
evaluation source identity, without changing any frozen data or sampling rules.
The failed base attempt cost 64 GPU-seconds; cumulative cost is 0.292222222
GPU-hours. No scientific training or main job ran. The failed base had no repair
rows and no partial optimizer state to resume; the next base starts fresh.

### Preserve pilot GPU-hour ceilings on the selected one-GPU profile

Source 4a393df passed hosted CI 40611635030 (215 tests) and CPU runtime job
31045342 (`6ab6345e274dd617c2f0d9da`), all 306 tasks. Base job 31046386
(`46ec4ac904e9611f551f8713`) is running on one H200, has cleared the real FSDP
no-op guard and begun repair generation. Inspect it before another base request.
The initial three episodes averaged about 66 seconds; these are partial real
evaluation observations, not pilot throughput or a final baseline score.

The old two/four wall-hour pilot segments assumed two GPUs (four/eight allocated
GPU-hours). Scale their wall limits by allocated count: the selected single GPU
gets four hours before the iteration-two stop and eight for its continuation.
The 12 pilot GPU-hour and 24 cumulative GPU-hour caps, ten iterations and required
resume at two remain unchanged. This avoids carrying a two-GPU wall allowance
into a one-GPU workload. A resource-budget regression checks these ceilings for
all supported single-node counts. No scientific or evidence dependency changes.
The local watcher was stopped while base continues, so inspect its actual result
before resuming preparation after CI; do not repeat an unmodified failed base.

### Real baseline and zero-yield D pilot

Commit 9964937 passed hosted CI 40613870205 (219 tests). Base 31046386 completed
all 41 monitoring tasks, with zero successful repairs, in 1205 allocated GPU
seconds. D pilot 31049338 stopped cleanly at iteration two; the controller
verified checkpoint 000002 and submitted continuation 31050754. The continuation
restored the joint checkpoint, completed iteration ten and correctly failed its
acceptance gate: 0/160 valid proposals, zero student episodes, zero student or
proposer updates, and no alignment scores. Counts: 154 required-header errors,
one hunk-count error and five JSON errors. The pilot consumed 406+513 GPU-seconds.
No successful pilot report, freeze or main eligibility is claimed. Its timings
and memory describe an empty-workload proposal run, not real training throughput
or learning memory fit. The frozen baseline and failed pilot remain intact.

Captured outputs were diagnosed without changing rewards, prompts or parser.
Of the first 112 captured proposals, 13 had Git metadata preambles; removing
only that metadata in an offline diagnostic still left context, hunk-count or
line-format errors, with no valid mutation. No diagnostic output was substituted
into training. A concrete common format clarification is proposed in
`docs/task/07-shared-diff-format-decision.md`; its synthetic example passes the
current validator. User authorization is pending because Task 07 explicitly
preserves prompts. Do not repeat the unchanged zero-yield D pilot.

Independent A/B/C launch 31052304 failed before model load: the common launcher
passed --config while method-smoke declares a positional method. The launcher now
translates that argument, and honors explicit A/B/C selection; D continues to
request the full suite. CLI/scientific source, data and numerical compatibility
are unchanged. An integration regression checks preserved settings/hardware
arguments and separate method launches (88 focused tests passed). The failed
launch cost 46 GPU-seconds. Cumulative consumption is 0.895 GPU-hours before
corrected A/B/C checks. No GPU job is currently active; main training remains
unauthorized.

### Authorized work checkpoint: launch readiness blocked

Execution source 9980c78 passed CI 40617600830 (220 tests). Corrected A/B/C jobs
31054795 / `3a3595c3ba3ddf9273d7d7a8`, 31055413 /
`86c10ae147544ad5343f0edc`, and 31056122 / `953d8f64c795c5016f07e36e` each
committed one iteration and a joint checkpoint, but each had 0/16 valid tasks,
zero episodes and zero updates. C also had no alignment scores. Their runtime
was 173/170/363 GPU-seconds. The launcher now passes the required positional
method and permits independent B/C checks after A fails, without changing the
scientific CLI, package identity or accepted data/backend evidence.

The final doctor has worker/runtime/single/target/base true and
pilot/methods/freeze false. All four real main dry runs refused eligibility;
zero main jobs were submitted. Reconciled consumption is 1.091111111 GPU-hours,
with zero outstanding reservations and no active job/watcher. The proposed
shared prompt-format clarification is pending user authorization. The original
prompts/parser and every scientific setting remain unchanged. Completion and
the concrete external decision are recorded in
`docs/task_completion/07-implemented-yale-execution-blocked-readiness.md`.
