# Task 18 implementation and execution record — unfinished

The [specification](../task/18-interface-signla-bottleneck-fix-spec.md) is being
executed under a hard 12 new allocated H200-hour cap. The
[current evidence/status](../investigations/task18-status.md) and
[provisional decision](../investigations/task18-research-decision.md) distinguish
completed results from three queued operations. No new student-training arm is
enabled. Do not merge this task as complete yet.

## Delivered implementation

| Area | Entry points | Motivation and compatibility |
|---|---|---|
| Numerical diagnostics | `task18_numerical.saved_token_report`, `audit_updates`, `cached_scoring_probe`, `frozen_rollout_replay` | Exact saved IDs, clipping changes, restored Adam displacement, identical-operation repeats, cache/full scoring and bounded fresh rollout reproduction. Denominator substitutions are diagnostic only; no historical checkpoint is modified. |
| Tool protocol | `task18_protocol.Task18EpisodeConfig`, `NativePolicy`, `parse_episode_action` | Qwen schemas enter the pinned tokenizer template; one call delegates to the existing restricted editor. Plain JSON templates/actions remain behaviorally identical. Native targets/EOS retain their sampled IDs. |
| Configuration | `task18.Task18Config`, `configure_operation` | Separate study, two paired training seeds, control-checksum/serialization identity, disabled training gate, no historical configuration-hash changes. |
| Allocated runner | `scripts/task18_experiment.py`, `Task18Trainer` | Reuse verified pool, B2026 weights-only importer, static curriculum and transactional trainer. Conditional pilots stop at 32 episodes; target/evaluations remain frozen at 80 with 0/32/40/80 evaluations. |
| Controller | `hpc.run_payload`, `hpc_remote` budget/dependency/verifier helpers, `hpc_job.task13_command_args` | One locked 12-hour allowance across stages/seeds/readiness/failures/continuations, actual accounting, no historical budget import, explicit sequential interface dependencies. Unchanged executed verifier dependencies can be reused without re-auditing 306 tasks for each loss/interface. |
| Analysis | `analyze_task18.py`, `analyze_task18_tokens.py`, `analyze_task18_rewards.py`, `plot_task18_*` | Controller-bound paired outcomes, family uncertainty, saved-array alignment, logical hidden-reward bounds, privacy-safe aggregate PNG/PDF figures. Raw trajectories/oracles remain ignored. |

The shared environment delegates prompt/action selection to the Task 18 protocol
helper; historical default behavior is checked, while changed executable code
has a new code/prompt identity. Historical experiments are never resumed under
these changes. The accepted sandbox, path/import/API validation, worker resource
contract and binary student/proposer rewards are unchanged.

## Executed evidence

- CPU runtime job **33574512** completed exit 0, validating the 306 frozen tasks
  in the accepted isolated worker; unchanged containment evidence was reused.
- Saved-data analysis reconciled 120 committed Task 17 batches, verified 11
  recovered training-shard checksums remotely, reproduced the 5.4899 discrepancy,
  and checked all 495 calibrated-2026 turn prefixes without model execution.
- Numerical GPU replay **33575206** completed exit 0 in **473 H200-seconds**.
  A mixed batch showed 21.4% conditional Adam-displacement sensitivity versus
  0.4% repeat variation; uniform controls were within repeat variation.
- Normalized base/trained jobs **33575526 / 33575845** completed exit 0 in
  **925 / 599 H200-seconds**, with **6/40 / 28/40** verified repairs.
- Hidden-training fractions/bounds recovered from saved trusted counts give
  8/20 guaranteed informative groups versus binary's 7/20. No program execution,
  new feedback, or learning result is implied.

Completed new GPU allocation is **1,997 seconds = 0.554722 H200-hours**. Native
jobs **33576276 / 33576446** and cache/frozen-rollout job **33580028** remain
queued; exact controller IDs and pending-wall adjustments are in the status
record. Their outcomes, final accounting and research decision remain required.

## Validation

The full selected local suite passed **450 tests**, with 20 skipped and 18
deselected; Ruff lint/format and mypy for 55 source files passed. Focused tests
cover historical prompt/configuration parity, native duplicate/extra fields and
path restrictions, real Adam restoration after injected failure, cached prefix/
EOS and scoring-mode handling, token-divergence probability comparisons,
controller/runner identity, shared budget exhaustion, evidence reuse boundaries,
and interface-only scheduler dependencies. GPU/download/sandbox/online-W&B
markers were excluded locally; actual Yale worker and GPU results above are
reported separately.

## Remaining work

Reconcile and fetch the existing pending jobs without duplication. Complete the
backend diagnosis and native expansion gate, keep new RL disabled until its
numerical/substantive-change conditions are satisfied, then finalize the
research decision, results/accounting, README and completion record. Merge the
genuinely completed branch to local main and push, preserving user changes and
all historical/private artifacts.
