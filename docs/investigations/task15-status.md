# Task 15 status and next action

Task 15 is in implementation/validation. Main A/B training and final evaluation are **not complete**. Worktree: `/private/tmp/selfplay-task15`, branch `feat/15-stabilize-ab`; baseline `1c8e190`. No Task 15 jobs have been submitted, and no watcher is running.

**External blocker:** read-only SSH to `yale-hpc-cpsc4880` returns `Permission denied (keyboard-interactive)`. Complete `uv run selfplay hpc login` with normal Yale MFA. Once access works, reconcile Slurm, durable ledgers and account limits before any submission. Historical queue-empty statements are not live verification. There is no new GPU authorization request.

## Verified state and reused evidence

- Task 14's [decision](task14-research-decision.md) and [completion](../task_completion/14-implemented-informative-selfplay.md) record all nine static controls and both bounded online arms. Private summaries exist under the original checkout's `runs/task14/`; source jobs and checkpoints remain unchanged.
- The local prepared manifest has 186 training parents, 41 monitoring repairs, 115 final in-domain repairs and 116 external repairs. Static AST inspection finds 1,487 legal structured actions before semantic filtering; this does not establish that all are valid or appropriately difficult.
- Current code confirms that old scoring computed but discarded frozen unprivileged logps, while SD used behavior logps. Legacy numerical replays coupled denominator and SD-gap changes. This is a confirmed identification limitation; its end-to-end training effect remains unresolved.
- Current code confirms permanent historical duplicate rejection while prompts omit that blacklist. This is a confirmed admission/information mismatch. Replay removes that rule; improvement in meaningful training exposure still requires the prospective test.
- Existing evidence supports retaining the pinned 4B, structured proposals and unified-diff student. Static learnability and negative body-replacement results need no replication. Singleton-valid reward normalization remains a structural limitation, not an outstanding correctness gate for A/B.

## Implemented candidate, not yet frozen

The successor stores three probability streams; preserves true behavior PPO ratios; versions same-backend signed gaps; independently replays denominator/gap interventions and no-privilege scoring; supports bounded refill, semantic caching and explicit task replay; persists supply/exposure/milestone state; and launches/resumes by episode target. The frozen primary comparison remains A ungated versus B sigmoid-gated at the same lambda 0.1.

Defaults, limits, statistical interpretation and execution commands are in [the plan](../plan/15-stabilize-training-complete-ab-plan.md) and [operations](../operations.md#task-15-online-ab-successor). A narrow scientific freeze requires actual allocated evidence and measured runtime planning. Its absence must not be papered over with readiness flags.

## Validation snapshot

The integrated local CPU suite passed **321 tests, 34 skipped, 4 deselected** before the final payload regression was added. Focused Task 15 tests also exercise real tiny-model counterfactual optimizer updates and checksum-verified supply restoration. Ruff lint/format and mypy passed. Local Docker is not running, so worker integration is explicitly unverified locally; hosted CI must exercise it. No H200 numerical, memory, runtime or resume result is claimed.

## Next concrete action

Finish local/hosted validation, then authenticate and inspect live Yale state. Synchronize the committed candidate; reuse unchanged worker evidence; run only changed runtime/backend acceptance and the saved-rollout Task 15 factorial audit. Run the separately identified 128-episode stabilization arm and one dependency-relevant interruption/resume check. Review numerical effects, teacher discrimination, actual task diversity/mixed groups, and phase cost. Repair only a localized defect if needed, freeze one common route, then submit and monitor the seed-2026 A/B pair toward 4,096 episodes each and complete seeds 2027–2028.

Keep this file current with exact jobs, ledgers, checks and the next command. Never restart the diagnosis from its beginning after interruption.
