# Task 15 status and next action

Task 15 is in implementation/validation. Main A/B training and final evaluation are **not complete**. Worktree: `/private/tmp/selfplay-task15`, branch `feat/15-stabilize-ab`; baseline `1c8e190`. Three Task 15 acceptance jobs were submitted once from `f11de62a2a848e0c8819f50e313de32892c0a51c`: main runtime `31665807` / `d417ec2cb88d51d18c70e710`, H200 backend `31665819` / `24b2aa8b8ef849d0fe0779c1`, and stabilization runtime `31665837` / `70c594198915015caf75a97d`. Inspect their ledgers and Slurm before action; never duplicate. Main runtime and backend completed exit 0; stabilization runtime was active at this snapshot. Saved-rollout audit job `31665993` / run `d1c5f8d2cfc62a5ffa4ee3a5` was submitted once from `aa01dcf5781c5f6932eb51bbefc573fd72ca2c67`; inspect its ledger and never duplicate. Stabilization job `31666009` / run `efb2e712a0484dc3d39e52ff` was submitted once from `1413994815ffc954a0153480b40b24d2c0ba17e8`, with a 128-episode target and a two-iteration stop. Never duplicate. A local controller watcher is running at `runs/task15/stabilization-watch.log`, without automatic clean continuation; review its ledger and logs before resuming.

Yale MFA succeeded. The live queue was empty before the first submission, prior Task 14 A/B, runtime, and backend jobs completed exit 0, and unchanged worker evidence was reused for both Task 15 studies. The three new acceptance jobs above were running at the latest check. Historical queue snapshots are not live verification; recheck before every submission. No new GPU authorization is needed.

## Verified state and reused evidence

- Task 14's [decision](task14-research-decision.md) and [completion](../task_completion/14-implemented-informative-selfplay.md) record all nine static controls and both bounded online arms. Private summaries exist under the original checkout's `runs/task14/`; source jobs and checkpoints remain unchanged.
- The local prepared manifest has 186 training parents, 41 monitoring repairs, 115 final in-domain repairs and 116 external repairs. Static AST inspection finds 1,487 legal structured actions on 128 parents before semantic filtering. The other 58 training parents have no legal structured site and are deterministically excluded from proposer sampling. This does not establish that all listed actions are valid or appropriately difficult.
- Current code confirms that old scoring computed but discarded frozen unprivileged logps, while SD used behavior logps. Legacy numerical replays coupled denominator and SD-gap changes. This is a confirmed identification limitation; its end-to-end training effect remains unresolved.
- Current code confirms permanent historical duplicate rejection while prompts omit that blacklist. This is a confirmed admission/information mismatch. Replay removes that rule; improvement in meaningful training exposure still requires the prospective test.
- Existing evidence supports retaining the pinned 4B, structured proposals and unified-diff student. Static learnability and negative body-replacement results need no replication. Singleton-valid reward normalization remains a structural limitation, not an outstanding correctness gate for A/B.

## Implemented candidate, not yet frozen

The successor stores three probability streams; preserves true behavior PPO ratios; versions same-backend signed gaps; independently replays denominator/gap interventions and no-privilege scoring; supports bounded refill, semantic caching and explicit task replay; persists supply/exposure/milestone state; and launches/resumes by episode target. The frozen primary comparison remains A ungated versus B sigmoid-gated at the same lambda 0.1.

Defaults, limits, statistical interpretation and execution commands are in [the plan](../plan/15-stabilize-training-complete-ab-plan.md) and [operations](../operations.md#task-15-online-ab-successor). A narrow scientific freeze requires actual allocated evidence and measured runtime planning. Its absence must not be papered over with readiness flags.

## Validation snapshot

Implementation commit `234d443` passed the frozen local CPU suite: **326 passed, 34 skipped, 4 deselected**. Ruff lint and formatting, mypy (47 source files), and `git diff --check` passed. Focused Task 15 checks include real tiny-model counterfactual optimizer updates, independent gradient-mode scoring, checksum-verified supply restoration, fallback credit/caps, final evaluation guards, and timeout recovery.

The controller's local snapshot check passed with prepared/executed data validation, clean Git, and manifest SHA-256 `4be6da81532938b3e03bf5e7e839c200335e2b3728b5bcbfbc7c7161c39288cf`. The ignored `data/protocol-v3` link points to the original immutable prepared dataset; no private data was copied into Git.

Local Docker is not running, so sandbox integration remains explicitly unverified locally. [Hosted CI 41367241355](https://github.com/tatsu432/self-play-opsd-coding-agent/actions/runs/41367241355) checks the exact implementation commit and is running at this handoff. Superseded CI attempts were cancelled to avoid redundant work; they are not passing evidence. No H200 numerical, memory, runtime, resume, stabilization, or main result is claimed.

## Next concrete action

Finish hosted validation of implementation `234d443`, authenticate, and inspect live Yale state. Exact restart commands:

```bash
cd /private/tmp/selfplay-task15
uv run selfplay hpc login
uv run selfplay hpc inspect
gh run view 41367241355 --log-failed
```

The login command requires the user's normal Yale MFA; it is not a request for a new scientific or GPU authorization. Synchronize the committed candidate; reuse unchanged worker evidence; run only changed runtime/backend acceptance and the saved-rollout Task 15 factorial audit. Run the separately identified 128-episode stabilization arm and one dependency-relevant interruption/resume check. Review numerical effects, teacher discrimination, actual task diversity/mixed groups, and phase cost. Repair only a localized defect if needed, freeze one common route, then submit and monitor the seed-2026 A/B pair toward 4,096 episodes each and complete seeds 2027–2028.

Keep this file current with exact jobs, ledgers, checks and the next command. Never restart the diagnosis from its beginning after interruption.
