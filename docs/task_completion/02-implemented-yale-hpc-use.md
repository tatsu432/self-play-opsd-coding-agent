# 02 — Yale HPC operations implementation

Source: `docs/task/02-yale-hpc-use-spec.md`; plan:
`docs/plan/02-yale-hpc-use-plan.md`. Observed state: 2026-10-22.

## Completed

- Added `selfplay hpc` with MFA login, inspection, readiness reporting,
  resource validation, immutable Git and prepared-data sync, submission,
  status, bounded logs, artifact retrieval and training recovery. The Yale
  SSH/Slurm adapter is separate from the experiment code, leaving room for
  an AWS adapter. `AGENTS.md` and `docs/operations.md` carry the procedure
  across sessions; `README.md` remains brief.
- The standard-library remote controller uses a private, user-specific
  workspace under the account's **writable personal directory** within
  course project storage. It uses the corresponding course scratch directory
  for caches. The shared fileset root is not writable by this account. A
  clean committed Git tree, executed dataset validation, commit-specific
  checkout and independently SHA-256-named data directory precede a run.
- `scripts/yale.slurm` runs the pinned trainer SIF with read-only code/data
  mounts and writable durable runs/cache mounts. It checks the Git checkout
  and SIF digest on the allocated compute node. A pre-timeout signal requests
  a safe checkpoint. Operational mount paths do not alter the scientific
  configuration hash.
- Submission checks the live Slurm association, typed GPU, profile VRAM,
  advertised single-node CPU/RAM/GPU fit and bounded wall time. A durable
  intent is recorded before `sbatch`; uncertain replies are reconciled
  against the scheduler instead of being resubmitted. Recovery requires the
  latest checksummed joint checkpoint with matching scientific and dataset
  identity, accepts only PREEMPTED/NODE_FAIL/BOOT_FAIL, and reserves at most
  two retries. Scientific failures and unknown scheduler states stop retries.
- Added controller and wrapper regression tests, including rejected resource
  requests, ambiguous submission, retry bounds, checkpoint verification,
  personal storage discovery, stable scientific hash and pre-timeout signal.

## Live Yale evidence

- Normal Duo MFA established an SSH control connection to
  `yale-hpc-cpsc4880`; login node was `login2.bouchet.ycrc.yale.edu`, user
  `cpsc4880_ts2236`. Association was
  `cpsc4880||interactive,normal|` (no partition restriction in that row).
  Read-only checks accepted proposed two-GPU H100 (`gpu_h100`), H200
  (`gpu_h200`) and RTX Pro 6000 Blackwell (`gpu_rtx6000`) shapes. This is
  permission and node-shape evidence, not a reservation or backend proof.
- Group quota data (oldest sample 2026-10-21) showed project 979/4096 GiB
  and scratch 515/10240 GiB. Scratch has a 30-day purge policy. The user had
  no active jobs at inspection; two other `cpsc4880` jobs were running.
  Apptainer, Git, rsync and Slurm tools were present.
- After fixing personal-directory discovery, `hpc sync` completed for commit
  `213647ab03e70ad94cc196cf4e1bb1a6f66c9f9c` and dataset manifest SHA-256
  `44fe6caffc0548bebb01bf39208f8b2f5de10ab89ee8487dca20dd436c0154f5`.
  The checkout is under
  `/nfs/roberts/project/cpsc4880/cpsc4880_ts2236/selfplay-opsd-cpsc4880_ts2236/checkouts/`;
  the matching data and run directories are under that user's project space.
- `hpc doctor` then reported local Git clean, prepared data and validation
  ready, a Yale checkout and dataset present, and no GPU backend proof.
- The other course aliases rejected noninteractive access due to MFA. A Duo
  prompt for `cpsc4710` timed out; neither additional identity's permissions
  or limits were established. Do not assume three independent GPU pools.

## Verification and remaining gates

The final CPU suite passed with 64 tests and 11 skips. Ruff checks, format
checks, mypy and Bash syntax checks passed. No GPU was allocated or training
launched.

A trainer SIF with a verified SHA-256, an approved isolated code worker and
its compute-node connectivity, and actual target GPU backend acceptance are
still unverified. Next, build/distribute and pin the trainer SIF, validate the
worker, then run a small scheduled backend acceptance job. After that, collect
pilot memory/time/GPU-hour measurements, evaluate the base model, and freeze
the study before A–D training. Re-run `hpc doctor` and `hpc sync` after each
new clean commit.
