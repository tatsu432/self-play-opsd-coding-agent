Review and repair the current `main` branch of `tatsu432/self-play-opsd-coding-agent`, starting from commit `e70df4ad9d100d2cbe12794e046a0b527b13c3b4`.

The current implementation made progress on Yale HPC orchestration but is blocked by rootless Apptainer worker isolation. Before attempting more infrastructure changes, fix the confirmed CI regression and simplify the critical execution path.

**Task 1 — Fix CI**

GitHub Actions run `39730979158` failed five tests in `tests/test_trainer.py`.

The traceback points to `SelfPlayTrainer.checkpoint()` calling `execution_identity(config)`, which attempts to reopen `data/prepared/manifest.json` rather than using the already supplied validated `TaskRepository`.

Investigate the correct identity contract. Fix the fixture/configuration mismatch and avoid redundant filesystem dependencies where appropriate. Keep manifest checks strict; do not suppress exceptions or fabricate identity values.

Run the exact GitHub Actions test suite and ensure CI passes before proceeding.

**Task 2 — Decouple readiness gates**

Inspect `scripts/hpc_remote.py`, `scripts/hpc_job.py`, `src/selfplay_opsd/hpc_readiness.py`, and `src/selfplay_opsd/backend_check.py`.

The controller currently requires worker isolation before allowing any GPU operation. Separate trusted GPU-only backend diagnostics from operations that execute generated Python.

Allow independently validated model loading, gradient checks, vLLM/verl synchronization, and checkpoint roundtrips without requiring a working generated-code worker. Preserve full sandbox isolation requirements for self-play, data verification, pilot training, and evaluation.

**Task 3 — Resolve the infrastructure plan**

Read `docs/task_completion/03-implemented-experiment-readiness.md`.

The last Yale worker probe failed with `cannot use cgroups - DBUS_SESSION_BUS_ADDRESS is not set`.

Treat this as a potential site-level cgroup delegation limitation. Do not invent D-Bus sockets, weaken isolation, bypass resource limits, or repeatedly submit equivalent failing jobs.

Prepare a concise support request containing the observed node, job IDs, cgroup configuration, and required security properties. Document an approved alternative worker architecture if Yale cannot support nested unprivileged container limits. Do not contact YCRC without my approval.

**Task 4 — Simplify operations**

Review the custom controller for unnecessary coupling, redundant identity computations, fragile filesystem assumptions, and excessive synchronization. Preserve immutable scientific configurations, checkpoint integrity, bounded budgets, and duplicate-job prevention.

Make trainer image build storage configurable, rather than requiring large local artifacts under the repository filesystem.

Do not refactor unrelated scientific objectives.

**Task 5 — Verification and reporting**

Run CPU tests, linting, type checks, and controller regression tests. Add tests proving trusted backend checks do not require a worker, while generated-code operations cannot bypass isolation.

Do not spend GPU-hours or submit full experiments during this repair.

Produce a final report distinguishing fixed issues, remaining Yale infrastructure blockers, the exact next executable command, and the readiness status for the four-method A–D study.

Work in a dedicated branch, keep changes focused, and preserve the original scientific experiment and existing Yale artifacts.