# 04 — Repair Yale launch readiness

Source: `docs/task/04-fix-launch-issue-yale-hpc-spec.md`. Start from `e70df4a`
on `fix/04-yale-launch-readiness`. Preserve scientific configurations and Yale
artifacts; use zero GPU-hours and submit no experiments.

1. Fix checkpoint identity using the supplied validated TaskRepository, enforce
   configuration/root agreement, cache trainer identity, correct fixtures and
   add identity/resume regressions. Run the exact correctness workflow locally,
   push the branch, and require GitHub Actions to pass before infrastructure edits.
2. Introduce versioned backend evidence independent of worker and dataset
   execution. Update report writers/readers, reference comparisons, remote
   submission and allocation gates. Keep generated-code execution isolated.
   Add `prepare-study --stage backend`; default `all` retains full readiness.
3. Add configurable image build root (flag, environment, existing default) and
   existing Buildx builder selection. Check the actual artifact filesystem;
   document separate builder-cache capacity. Reuse command-local identity and
   context while retaining fresh scheduler/accounting checks and strict hashes.
4. Document the observed cgroup/D-Bus failure and an unsent YCRC request. Specify
   a proposed Yale managed worker service retaining all security properties;
   require site support and deployment approval. Do not repeat failing probes.
5. Test backend independence, generated-code gate failures, evidence compatibility,
   storage selection/integrity and existing controller safety behavior. Repeat
   workflow checks, require final hosted CI, and write the task completion record
   with remaining blockers, next commands and A–D readiness.

Decisions: publish the dedicated branch for CI; keep main unchanged. Document the
Yale managed service fallback without deploying it or contacting support. Actual
GPU validation requires later authorization. Existing backend evidence must be
regenerated for the new contract; old artifacts remain preserved.
