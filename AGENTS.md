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

The current task is `docs/task/03-experiment-readiness-spec.md`; its plan is
`docs/plan/03-experiment-readiness-plan.md`. Task 02 established basic operations.
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

1. Start with `uv run selfplay hpc doctor` and `uv run selfplay hpc inspect`.
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
4. An approved isolated code worker is still required. The current Yale config
   points to an authenticated loopback remote worker. The tunnel must originate
   on the allocated compute host and the worker must pass real isolation
   preflight; alternatively validate a local Apptainer worker. Keep worker
   credentials in a user-owned mode-600 file outside Git if needed.
5. Use `hpc status`, `hpc logs` and `hpc fetch` to inspect a submitted run.
   `hpc watch` can resume a previously authorized training run after a confirmed
   PREEMPTED, NODE_FAIL or BOOT_FAIL state, at most twice, from the latest
   verified joint checkpoint. It reconciles the durable ledger with Slurm first.
   If a submission response or scheduler state is uncertain, stop and inspect;
   never create a duplicate. Scientific failures, OOM, authentication, quota
   and unauthorized resource requests are not automatic retries.

## Current observed state (2026-10-22)

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
Re-check live state before new work. Resolve the supported worker execution
setup and trainer-build storage before advancing readiness. Do not provision
another host or contact support without the user's explicit instruction.
