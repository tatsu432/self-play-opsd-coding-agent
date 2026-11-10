# Task 17 implementation and research execution — unfinished

The [research specification](../task/17-generalization-spec.md) is being executed
on `feat/17-repair-interface-generalization`. Implementation, historical analysis,
isolated semantic replay and the four primary GPU allocations are complete.
Final text-result retrieval, paired analysis and conditional Phase 2 remain
pending because normal Yale VPN/SSH access expired. This is an execution handoff,
not a claim that the research task is finished.

## Delivered implementation and evidence

- Exact unique multiline replacement with bounded arguments/source, allowlisted
  path, syntax/source/API/import checks and atomic rejection. Historical strict
  diff remains the control. Final-turn observations persist and episode identity
  binds interface/budgets.
- All 1,617 historical trajectories reproduce. Oracle-free metadata recovery
  leaves added/removed content unchanged and rejects ambiguous context. Isolated
  replay tested 1,991 admitted actions using 449 distinct task/program calls.
  Of 1,107 trained external reconstructed actions, 755 pass; 242 originally
  failed episodes contain a correct reconstructed state. This is not new agent
  accuracy or a counterfactual continuation.
- Frozen, verified pool: 40 primary interface families, 16 presentation variants,
  40 curriculum, eight calibration and 16 independent development tasks. All
  120 rows pass both protocols' tokenizer admission. Pool SHA-256:
  `602eaf506f8ac2cf17924596dae7c0ff3891ef5034a546d4d3751b9803412c83`.
- Frozen four-cell runner and family-aware paired analysis, including action
  validity, edit application, verification, length/structure strata, presentation
  effects, turns/tokens, errors and costs. The retrieval collector never submits
  jobs and preserves valid status files on authentication failure.
- Conditional fixed-curriculum GRPO/gated/ungated/calibrated-ungated execution,
  initial-gradient calibration, base-weight verification, control-checksum-bound
  checkpoint continuation, and separate locked 10/24 H200-hour ledgers. These
  paths are implemented; no objective calibration, pilot or training has run.
- Optional normalized diff retains strict prompts and observations, and labels
  metadata recovery separately. It is locally tested, unselected and unexecuted.
  Current changed code requires one matching executed runtime check before new
  GPU work. Dependency-matching runtime reuse retains the original raw proof
  and hash instead of repeating identical verifier audits for each objective.
- Public aggregate PNG/PDF taxonomy and replay figures, deterministic analysis
  scripts and research documentation. Raw task/source/oracle artifacts stay
  ignored under `runs/`.

## Executed operations

| Operation | Run | Slurm job | Result |
|---|---|---:|---|
| Initial runtime | `15b407b13adfc94780658de9` | 32792142 | Completed |
| Runner-matching runtime | `850c2d13f87782005d2de7b4` | 32793248 | Completed |
| Isolated semantic replay | `9d9776357a4e23ae82cd34dc` | 32794598 | Completed |
| Eight-attempt preparation | `6be54fe0c6ab8d8a6c492675` | 32794599 | Admission deficit; archived |
| Expanded preparation | `0522e9eae55f829aadc11294` | 32797537 | Originals verified; variant failed; archived |
| Variant diagnosis | `f0b3c13f22f1fdbd0aa7e20b` | 32801929 | Resource-fragile mutant; archived |
| Final preparation | `86e2df49bb04a9321f7d755d` | 32804290 | All 120 admitted |
| Base strict diff A | `9f12c84dfddf9b37f42dd4f1` | 32805621 | 6/56 repairs; 5/40 originals |
| Trained strict diff C | `c10d7452835f81c14c0726df` | 32808848 | 36/56 repairs; 27/40 originals |
| Text runtime | `a0cc876dae3dcb9af79dfe1e` | 32809208 | Completed |
| Base exact text B | `f7c617670830bcf229a93b0e` | 32811833 | Remote completion observed, 26/56; fetch pending |
| Trained exact text D | `819fec9d88332bf03f874d1e` | 32812080 | Remote completion observed, 31/56; fetch pending |

GPU cells ran sequentially A→C→B→D and used 1,234/1,121/928/857 allocated
H200-seconds: **4,140 seconds = 1.15 H200 GPU-hours**. B/D counts and elapsed
allocations were observed in the last successful remote inspection; final
artifact/ledger accounting must still be persisted locally. Their 40-original
breakdowns are unknown locally. Do not substitute the correlated 56-presentation
endpoint for the primary analysis. No other Task 17 GPU allocations were made.
The rejected eight-CPU submission `dc2a6394ef94ef1bd7768b1e` had no scheduler job;
corrected resources use five trainer plus one worker CPU. Preallocation failures
and failed preparation ledgers remain auditable. Historical Task 15 is immutable.

## Validation

Latest local CPU selection: **426 passed, 20 skipped, 18 deselected**, using
`uv run pytest -q -m 'not gpu and not download and not sandbox and not wandb_online'`.
Ruff lint/format and mypy (51 source files) pass. This selection excludes actual
GPU, download and sandbox execution. Hosted CI at `54867ec` passed
([run 42888954364](https://github.com/tatsu432/self-play-opsd-coding-agent/actions/runs/42888954364));
latest optional-interface/collector changes require their own CI result.
The executed Yale checks and preparation used the accepted isolated worker.
Generated programs were never executed on the host.

## Recovery and remaining work

Restore normal Yale VPN access and run:

```bash
uv run selfplay hpc login
uv run python scripts/collect_task17_interfaces.py
```

This retrieves existing cells, reconciles statuses and actual Slurm allocation,
then performs paired analysis in `runs/task17/interface-analysis-v1`. No task is
resubmitted. Inspect original-task effects, uncertainty, failure modes and costs;
finish outcome/efficiency figures and make the Phase 1 decision. Only then freeze
an evidence-backed objective control, calibrate, run the bounded pilot and
continue informative Phase 2 arms within the separate 24-hour ceiling. No full
C/D. Reconcile telemetry/checkpoints and final costs; complete the scientific
report and merge the finished branch to local main and push. The current branch
is preserved on GitHub for recovery; it is not merged as completed work.

See the [results](../investigations/task17-repair-interface-results.md),
[current decision](../investigations/task17-research-decision.md) and
[execution plan](../plan/17-repair-interface-plan.md).
