# Task 18 completion: interface and learning-signal bottlenecks

The bounded phase is complete. Native exact-text calling repaired **27/40 base
and 31/40 trained** tasks, versus plain JSON's 19/40 and 21/40. A restored-state
numerical diagnostic found material update sensitivity; new RL therefore
remains disabled. This satisfies the conditional training decision in the
[specification](../task/18-interface-signla-bottleneck-fix-spec.md), rather than
claiming a new objective study was run. Total new allocation was **4,589
H200-seconds / 1.274722 hours** within the hard 12-hour cap.

See the [results and exposure table](../investigations/task18-results.md),
[research decision](../investigations/task18-research-decision.md), and
[immutable executions / reproduction](../investigations/task18-status.md).

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

Full native evaluations additionally reuse eight checksum-bound completed
probe episodes and generate only 32 new episodes. `load_native_probe` validates
model/snapshot/seed/protocol/coverage, file hashes and completion. Analysis binds
the source controller allocation, checks copied records field by field, and
requires the exact 32/8 exposure split. Probe and expansion results remain
separate; no completed task was regenerated for the full comparison.

## Executed evidence and decisions

- All twelve Task 17 objective arms / 120 committed batches and four historical
  strict/text cells were reused. No historical training or external evaluation
  was duplicated. Only one originally reserved eligible parent remains unused;
  all new interface families were previously inspected development data.
- Two real CPU runtime allocations verified all 306 tasks in the accepted
  isolated worker. The second followed the native-parser dependency change;
  later runner-only changes reused compatible evidence. No GPU was allocated
  to either CPU audit, and original proof bytes were archived.
- Complete-state replay covered sixteen saved episodes / 3,842 targets.
  Diagnostic denominator substitution changed a mixed batch's Adam update by
  21.374%, versus 0.3993% repeat variation; uniform controls were within repeat
  variation. Six actual diagnostic Adam steps were restored and discarded.
  There are zero new training steps/endpoints. Original behavior denominators
  were never replaced in production or historical data.
- The cache/fresh-rollout probe generated 189 targets across four original
  unprivileged turns, without executing any generated action. Full/gradient
  scores match; incremental/full differences and fresh-vLLM probability
  differences demonstrate execution sensitivity without isolating every cause.
  The retired worst checkpoint prevents a causal result for its 5.4899 maximum.
- Normalized diff repaired 6/40 base and 28/40 trained. The first native probe
  exposed a real compatibility defect: its parser rejected permitted surrounding
  assistant content. That base diagnostic cost 351 seconds; its pending trained
  counterpart was cancelled before allocation. Corrected 7/8 probes for both
  snapshots passed the predeclared expansion screen; full native reached
  27/40 and 31/40. Historical/default behavior has parity tests and explicit
  distinct executable identities.
- Partial hidden-reward bounds add one guaranteed informative group (8/20
  versus binary's 7/20). This is signal evidence, not measured learning
  efficiency. Binary rewards and proposer difficulty remain unchanged.

The nine GPU parents and two CPU parents completed exit 0. One obsolete pending
native job was cancelled without allocation. All GPU reservations reconcile to
4,589 seconds; no job remains active for the project user. Every run ID,
checksum, accounting entry and resource adjustment, including rejected running
wall-extension attempts, remains recorded. Other users' account jobs were not
modified. Local artifacts are authoritative; no independent new W&B cloud-history
verification is claimed.

## Validation and limits

The selected full local suite passed **452 tests**, with 20 skipped and 18
deselected; Ruff lint/format (172 files) and mypy (55 source files) passed.
Focused tests cover legacy prompt/configuration parity, native surrounding
content and duplicate/multiple-call/path restrictions, actual tiny-model Adam
restoration after injected failure, cached prefix/EOS/mode handling, divergence
comparisons, shared budget exhaustion, runtime reuse/dependency boundaries,
controller/runner identity and checksum-bound native reuse. Actual full-model
GPU and isolated-worker results are reported separately from local tests.
PNG/PDF figures were rendered and visually inspected.

Native is selected on observed repairs and robust edit application. Its trained
three-task advantage over normalized is uncertain, and base native costs more
generated tokens than plain text. Family intervals are development resampling,
not seed uncertainty or untouched confirmation; remaining-32 results are shown
separately. No new OOD transfer claim is made. Numerical integrity has a valid
negative continuation decision, not a claim that the discrepancy is harmless
or that the historical measured repair outcomes are invalid.

Full C/D, new student RL, partial-reward training, adapter transfer, SFT, model
scaling, additional editors and changed observations were not supported in this
phase. The next proposed work is a narrowly frozen precision/execution-parity
ablation before a new paired native-interface objective cohort. Preserve all
Task 15–17 identities and private artifacts; do not relaunch completed work.
