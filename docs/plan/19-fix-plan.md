# Task 19 implementation and execution plan

Task 19 studies how gated on-policy self-distillation and gradient-based
proposer rewards affect coding-agent self-play. All five arms use
`Qwen/Qwen3.5-4B` at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`,
the same seeded initialization, one shared LoRA and one joint Dr. GRPO update.
The Control, A and B use difficulty rewards with zero, ungated and gated
distillation, respectively. C and D use gated distillation with RL-only and
combined student-gradient alignment, respectively.

The Task 19-only runtime and controller stages preserve the executable setups
for earlier experiments. The native protocol follows the pinned Qwen3.5
template and captures actual sampling probabilities. Teacher scoring receives
the same assistant targets and histories with additional training privilege.
Semantic mutations and repaired source execute through the previously
accepted Yale worker. The training-reference partition remains separate from
development and final evaluation.

1. Build the pinned Yale trainer and qualify the native tool path, real update,
   restored-state numerical behavior and joint loss scales on training data.
2. Evaluate the shared initialization on development, final ID and external
   repairs using the qualified sampling and budget.
3. Run 32 attempted episodes in each arm. Reconcile the five pilot checkpoints
   and use their actual allocated costs to freeze the common endpoint with a
   contingency and final evaluation costs reserved.
4. Continue each arm's verified checkpoint to the frozen endpoint and resumable
   final evaluations. Record terminal allocation accounting and W&B delivery.
5. Verify task coverage and checkpoint lineage. Report the four scientific
   contrasts, family-clustered intervals conditional on the single seed,
   aggregate results and five figures. Publish the completion record and merge
   the validated branch.

See the [specification](../task/19-fix-spec.md) and
[implementation status and commands](../investigations/task19-status.md).
