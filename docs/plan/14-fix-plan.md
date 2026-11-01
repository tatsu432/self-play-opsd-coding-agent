# Task 14: recover an informative self-play experiment

The task specification is `docs/task/14-fix-spec.md`. Use the completed Task 13 evidence and preserve its identities and allocation records. The paired base-model Task 14 proposer probe prospectively selects 40 training parents outside the Task 13 sample by sorting `digest([2026, "task14", parent_id])`. Four proposals per parent use the same generation and verification settings for diff and structured representations. Subsequent phases compare both empty and checkpoint novelty history, student reward variation, and operator/location diversity.

Implement and test structured online proposals and a function-body repair protocol. Compare the latter with unified diffs on the same training-only tasks, snapshots, seeds and budgets; adopt it across new arms only if paired evidence supports improved repair. Keep historical protocols and checkpoints intact.

Quantify validity-only, difficulty, RL alignment and combined alignment on the same proposals and rollouts, including normalized advantages, gradients and restored-state Adam updates. Compare a separately identified mean-centered ablation when standardization erases differences. Use training-only reference anchors and measure their stability.

Run fixed-task GRPO, gated OPSD and training-calibrated matched-strength ungated OPSD from matched initial states and schedules at 80 and 160 attempted episode checkpoints for seeds 2026, 2027 and 2028. Report optimizer steps, tokens, compute, uncertainty and development outcomes separately. The final held-out sets remain closed during selection.

Start bounded online experiments only after informative proposal and student behavior is established. Preserve original contrasts where possible; defer C/D if richer rewards do not produce distinguishable optimizer updates. Review at iterations 5, 10 and 20 and stop scientifically uninformative runs.

Before each allocation reconcile live scheduler state and ledgers; use the accepted isolated worker and checksum-verified immutable sources. Account for failed attempts. Commit a reproducible report and Task 14 completion record only after tests, bounded jobs, monitoring and scientific analysis are complete.

## Completion

The specified probes, three-seed matched controls, bounded online A/B runs, numerical and reward counterfactuals, and final analysis are complete. See the [research decision](../investigations/task14-research-decision.md) and [completion record](../task_completion/14-implemented-informative-selfplay.md). All Task 14 allocations, including failed attempts, are reconciled; the final held-out set remains closed.
