# Task 17 implementation and partial research completion — 2026-11-10

## Status

**Local implementation and retrospective CPU analysis are complete. Required
empirical Phase 1 remains incomplete because accepted execution is inaccessible.**
No Task 17 job was submitted, no generated program ran on the host, and newly
allocated GPU-hours are **0**. No Phase 2 control, calibration result, training
result or C/D authorization was fabricated.

Yale inspection failed with SSH exit 255: `Could not resolve hostname
bouchet.ycrc.yale.edu`. Local Docker's daemon socket is absent. Live scheduler,
account and checkpoint availability must be reconciled when access returns; the
historical completed jobs must not be relaunched.

## Delivered

- Exact unique text replacement with multiline edits, size/path/source/API/import
  validation and atomic rejection; unchanged strict-diff control behavior.
- Backward-compatible stored tool observations and interface/budget-specific new
  episode identity. Original B2026 checkpoint provenance is explicitly pinned;
  import restores weights with fresh optimizers, not a historical training resume.
- Oracle-free metadata recovery and deterministic action taxonomy; hashed private
  replay candidates and parent/length/mutation/complexity diagnostics.
- Deterministic disjoint training-only reservations: 40 interface, 40 curriculum,
  eight calibration, 16 development parents, from 113 eligible families.
- Isolated pool preparation/replay, four-cell frozen inference, paired analysis,
  and conditional static/calibrated student execution through existing machinery.
  These routes are CPU-tested but not verified on Yale in Task 17.
- Separate locked Phase 1/2 authorization ledgers, preserving historical budget
  state and charging failed/uncertain allocations through existing accounting.
- [Implementation specification](../task/17-repair-interface-spec.md),
  [execution plan](../plan/17-repair-interface-plan.md),
  [evidence/decision report](../investigations/task17-repair-interface-results.md),
  and a visually inspected aggregate taxonomy PNG/PDF. README, methods,
  operations and current agent instructions reflect the actual status.

## Actual evidence

All 1,386 locally present trained final trajectories reproduce exactly under
historical static parsing/source/API admission: 9,708 actions. External evaluations
have 4,301 rejected and 13 accepted patches. There are 1,132 unique-span metadata
recoveries among those rejected actions, of which **1,107/4,301 (25.7%)** pass
static source/API checks. ID has 837/1,836 (45.6%) statically admitted recoveries.
Across both distributions, 1,944 recovered actions pass static checks.

These are repeated action-level observations, not independent task successes.
None of the reconstructed programs has been semantically verified in Task 17.
The existing 231 common-base shards are absent at their recorded local paths;
coverage is explicitly 1,386/1,617 and historical base aggregates are not replaced.
H1–H5 remain empirical questions. No gating superiority or new transfer result
is established. Full C/D remains unjustified on existing evidence.

Authoritative private output is `runs/task17/recovery-v4/`. The earlier v3 and
final v4 summary and candidate hashes agree exactly; v4 additionally hashes all
analysis dependencies. SHA-256:

| Artifact | SHA-256 |
|---|---|
| summary.json | `68b35ec1948e2caaaa26fe2385654c6e8d5073de4c1a201a6827aa9125702fb4` |
| input-manifest.json | `2b06ed4cdcf1f7516c61f9a31af451f7c0260d12eae206099dea042cfd60ff0e` |
| replay-candidates.json | `820d8068810a3ecee5e01ccfaed8d43af81efde58939f8110a02a054a96569a6` |

## Validation

- Full local CPU selection: **403 passed, 34 skipped, four deselected**. The
  deselections are GPU/download/online-W&B tests; skips include unavailable local
  Docker-backed execution. This is not Yale isolation/backend acceptance.
- After final pool-integrity/calibration updates: **64 focused tests passed**,
  including interface, recovery, experiment controls and existing HPC checks.
- Ruff lint, Ruff format checks, mypy and `git diff --check` passed.
- CLI help entrypoints load; the independent retrospective rerun reproduced
  summary/candidate hashes. The aggregate figure was visually inspected.
- Hosted CI and Git integration are recorded below after publication.

## Remaining required work

Restore normal Yale DNS/access, inspect live ledgers, verify current containment
and changed runtime compatibility, prepare isolated tasks, run metadata semantic
replay and all four matched frozen-model cells. Make the Phase 1 decision from
verified repairs, not acceptance. Only then select any limited adaptation or
conditional objective training within the separate authorized budgets. Produce
actual outcome/efficiency figures and update the scientific decision. See the
execution plan for commands and expected private artifacts.
