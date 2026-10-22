# 02 — Yale HPC operations plan

Source: `docs/task/02-yale-hpc-use-spec.md`. The default account is
`yale-hpc-cpsc4880`; other course identities are inspection only until their
authorization and shared limits are verified.

1. Discover live SSH, Slurm associations/QOS/partitions, jobs, quotas, storage and
   software capabilities using lightweight login-node commands. Record unknowns
   rather than inferring access or available GPUs from documentation.
2. Add a local HPC controller with separate SSH, Slurm and run-state concerns.
   Synchronize a clean Git commit to an immutable Yale checkout and copy the
   independently hashed prepared dataset into durable project storage. Require a
   pinned trainer SIF, verified settings, a valid resource association and exact
   provenance before job submission.
3. Run Yale jobs from the SIF without a host Python environment. Provide status,
   logs and artifact retrieval, plus bounded recovery from verified joint
   checkpoints. Reconcile Slurm and durable state before any retry so uncertain
   submissions never create duplicate jobs.
4. Preserve scientific comparison gates and run only lightweight diagnostics now.
   Later GPU acceptance, isolation, baseline evaluation, pilot and frozen study
   remain explicit readiness steps; no expensive GPU allocation or full training
   is part of this task.
5. Test controller failure cases and the Slurm wrapper, keep README brief, track
   AGENTS and operational docs in Git, and record completed work and blockers in
   `docs/task_completion/02-implemented-yale-hpc-use.md`.
