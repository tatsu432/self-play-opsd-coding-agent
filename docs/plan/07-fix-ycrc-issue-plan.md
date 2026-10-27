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
