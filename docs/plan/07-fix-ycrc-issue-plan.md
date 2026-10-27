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
