# Task 17 implementation and research execution — unfinished

The [research specification](../task/17-generalization-spec.md) is being executed
on `feat/17-repair-interface-generalization`. Implementation, historical analysis,
isolated semantic replay and the four primary GPU allocations are complete.
Text-result retrieval and paired analysis are complete. Exact text is selected
for a fresh-base 80-episode objective study. Calibration failed on a missing backend metric; its corrected replacement,
the bounded objective pilots and final scientific decision remain pending.

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
  The current code passed a matching executed CPU runtime check before new
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
| Base exact text B | `f7c617670830bcf229a93b0e` | 32811833 | Completed and fetched; 26/56, 19/40 originals |
| Trained exact text D | `819fec9d88332bf03f874d1e` | 32812080 | Completed and fetched; 31/56, 21/40 originals |
| Current text runtime | `cc5ec4e1db0226c7f6dcbe29` | 32872199 | Completed exit 0 |
| Objective calibration | `11610fb16bd9af1f197221b1` | 32872812 | Failed `update_norm`, 265 H200-seconds; archived |
| Corrected runtime | `472ee2f34cce416bd7294356` | 32876383 | Completed exit 0 |
| Corrected calibration | `06982710d021357786a605aa` | 32877712 | Completed, 32 rollouts / eight groups, 556 H200-seconds |
| GRPO pilot | `8bfce0676900d87de4a8e700` | 32880497 | Completed stop at iteration 4; 1,851 H200-seconds |
| Gated pilot | `1768acda0784dab259fb68ea` | 32887853 | Completed stop at iteration 4; 1,904 H200-seconds |
| GRPO continuation | `8bfce0676900d87de4a8e700` | 32898057 | Submitted from verified checkpoint 000004 to 80 episodes |

GPU cells ran sequentially A→C→B→D and used 1,234/1,121/928/857 allocated
H200-seconds: **4,140 seconds = 1.15 H200 GPU-hours**. All four artifacts and
allocation records are fetched and analyzed. Base strict/text original-task
repairs were 5/40 and 19/40; B2026 strict/text were 27/40 and 21/40. The
correlated 56-presentation endpoint is exploratory. Calibration failed after four saved rollouts because the backend does not
provide `update_norm`. The allocation used 265 H200-seconds (0.07361 hours).
Its artifacts and original ledger were archived; a corrected implementation
measures actual adapter displacement and restores adapters and optimizers.
Corrected runtime validation passed, and the replacement completed exit 0.
The eight-group RMS matched coefficient is `0.02958647361953192`; the
calibration JSON SHA-256 is
`704e1fad2b53daa360dce14a53ba359689bdc50a21bb5870f8c8230f5d0280fc`.
The 32 training-only rollouts had 25 verified repairs. These are calibration
data, not independent evaluation results. Failed and successful calibration
allocations total 821 H200-seconds (0.22806 GPU-hours).
The rejected eight-CPU submission `dc2a6394ef94ef1bd7768b1e` had no scheduler job;
corrected resources use five trainer plus one worker CPU. Preallocation failures
and failed preparation ledgers remain auditable. Historical Task 15 is immutable.

## Validation

Latest local CPU selection: **426 passed, 20 skipped, 18 deselected**, using
`uv run pytest -q -m 'not gpu and not download and not sandbox and not wandb_online'`.
Ruff lint/format and mypy (51 source files) pass. This selection excludes actual
GPU, download and sandbox execution. Hosted CI at `54867ec` passed
([run 42888954364](https://github.com/tatsu432/self-play-opsd-coding-agent/actions/runs/42888954364));
the optional-interface/collector changes also passed hosted CI
([run 42898769683](https://github.com/tatsu432/self-play-opsd-coding-agent/actions/runs/42898769683)).
The executed Yale checks and preparation used the accepted isolated worker.
Generated programs were never executed on the host.

## Remaining work

The corrected calibration is complete. Freeze its coefficient/checksum in the
final control, then run and assess bounded GRPO/gated pilots. Continue
informative arms within the separate 24-hour ceiling. Reconcile telemetry,
checkpoints and final costs; complete the scientific report, update README,
then merge the finished branch into local main and push. Full C/D remains deferred.

See the [results](../investigations/task17-repair-interface-results.md),
[current decision](../investigations/task17-research-decision.md) and
[execution plan](../plan/17-repair-interface-plan.md).

## GRPO pilot review

The seed-2026 fresh-base text-interface pilot completed 32 attempted episodes:
26 succeeded, five of eight four-rollout groups were mixed, and three finite
nonzero optimizer updates occurred. Baseline monitoring and independent
development were 21/41 and 7/16 respectively; there is no post-training
development result at the four-iteration stop. Maximum behavior/rescoring
log-probability difference was 1.1153, with warnings in all four iterations;
the behavior denominator is preserved, and a causal impact is unverified.
Checkpoint `000004` metadata and three file checksums were verified on the
shared filesystem, with pool, control, model and scientific identities matching.
This single pilot supports running the matched gated pilot; it is not an
objective-comparison result. The three Phase 2 allocations (failed calibration,
successful calibration, and GRPO pilot) used 2,672 H200-seconds (0.74222 hours).

## Gated pilot review

The matched seed-2026 gated pilot completed 32 episodes: 27 passed, three of
eight four-rollout groups were mixed, and four finite optimizer updates occurred.
Monitoring/development baselines matched GRPO task-by-task (21/41 and 7/16).
Gated's maximum behavior/rescoring difference was 0.8137, with warnings in all
four iterations. Its iteration-4 joint checkpoint's metadata and all file
checksums were verified, with scientific identity matching. Both pilots have
meaningful learning signal and support proceeding to matched 80-episode
continuations; no post-training comparison exists yet. Total Phase 2 allocation
through both pilots and calibration was 4,576 H200-seconds (1.27111 hours),
including the failed calibration.
