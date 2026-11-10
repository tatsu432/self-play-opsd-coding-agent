# Task 17 implementation and research execution — in progress

The [research specification](../task/17-generalization-spec.md) is being executed
on `feat/17-repair-interface-generalization`. Phase 1 implementation and isolated
mechanical replay are complete; matched model evaluation is active. Phase 2 is
conditional on those results. No full C/D training is planned.

## Delivered

- Exact unique multiline text replacement with bounded input/output, allowlisted
  source path, syntax/source/API/import checks and atomic rejection. Historical
  strict diff is unchanged. Stored tool observations include the final turn;
  new episode identities bind interface and budgets.
- Oracle-free metadata-only reconstruction and full historical source replay;
  checksummed private inputs, deterministic taxonomy and feature-stratified
  analysis. All 1,617 final/base trajectories reproduce.
- Isolated semantic replay: all 1,991 admitted reconstructions, 449 distinct
  task/program verifier calls. Of 1,107 trained external reconstructions, 755
  pass; **242 originally failed episodes contain a correct reconstructed state**.
  This is not measured counterfactual agent accuracy.
- Frozen family-disjoint pool: 40 interface originals, 16 presentation variants,
  40 curriculum, eight calibration and 16 independent development tasks. Admission
  and replacement use verifiers only. Both tool protocols pass tokenizer
  admission on all 120 rows. Pool SHA-256:
  `602eaf506f8ac2cf17924596dae7c0ff3891ef5034a546d4d3751b9803412c83`.
- Four-cell frozen inference and paired family-bootstrap analysis; conditional
  fixed-curriculum/calibrated student execution through the existing trainer,
  with separate locked 10/24 H200-hour ledgers.
- Aggregate PNG/PDF figures, implementation specification, current results,
  methods and operational documentation. Raw source/oracle artifacts remain
  ignored under `runs/`.

## Execution records

| Operation | Run | Slurm job | State |
|---|---|---:|---|
| Initial runtime | `15b407b13adfc94780658de9` | 32792142 | Completed |
| Runner-matching runtime | `850c2d13f87782005d2de7b4` | 32793248 | Completed |
| Semantic replay | `9d9776357a4e23ae82cd34dc` | 32794598 | Completed |
| Eight-attempt preparation | `6be54fe0c6ab8d8a6c492675` | 32794599 | Failed; archived |
| Expanded preparation | `0522e9eae55f829aadc11294` | 32797537 | Originals verified; variant failure; archived |
| Variant diagnosis | `f0b3c13f22f1fdbd0aa7e20b` | 32801929 | Resource-fragile mutant found; archived |
| Final preparation | `86e2df49bb04a9321f7d755d` | 32804290 | Completed |
| Base strict diff | `9f12c84dfddf9b37f42dd4f1` | 32805621 | Submitted |

The original eight-CPU GPU submission `dc2a6394ef94ef1bd7768b1e` was rejected
before allocation. The corrected profile uses five trainer CPUs plus one worker
CPU, meeting the H200 partition limit. Failed preparation evidence, original
ledger bytes and archive histories are preserved. Current private status is
`runs/task17/execution-progress.json`; inspect live Slurm/ledgers before action.

## Validation

Focused interface, recovery, control and admission tests pass. Latest full local
CPU selection: **405 passed, 36 skipped, two deselected** (GPU/download unavailable;
Docker-backed isolation is checked in hosted CI). Ruff lint/format and mypy pass.
Hosted CI for the initial implementation passed 436 tests including Docker
worker checks. A later analysis-only typing failure was corrected; current CI
must pass before final integration. Yale preparation performs current runtime
validation using the already accepted isolated worker; generated programs were
never executed on the host.

## Remaining work

Complete the four matched frozen-model cells, analyze verified repairs and costs,
make the Phase 1 decision, and conduct an informative bounded objective pilot
when justified. Retain strict diff if interface evidence is inconclusive. Finish
outcome/efficiency figures and scientific conclusions, reconcile allocations and
telemetry, update this record and README, then merge to local main and push.
See the [results report](../investigations/task17-repair-interface-results.md) and
[execution plan](../plan/17-repair-interface-plan.md). Historical Task 15 remains
immutable; existing final splits are retrospective/exploratory development inputs.
