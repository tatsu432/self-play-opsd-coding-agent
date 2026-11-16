# Task 18 research decision — provisional

The [execution status and evidence](task18-status.md) are authoritative. Numerical
replay and normalized-diff cells are complete; the cached/frozen-rollout check
and native serialization probes are still queued. Task 18 is unfinished.

1. **Interface.** Exact text remains the strongest evaluated fresh-base option:
   19/40, against strict 5/40 and normalized 6/40. Trained B2026 scores are
   strict 27/40, normalized 28/40, text 21/40; the small normalized gain over
   strict is uncertain. Native eight-family probes cannot establish a full
   interface decision. Their expansion gate is lower invalid-call rate without
   fewer paired repairs, with common-40 evaluation required before selection.
2. **Application versus semantics.** Base text applies edits on 23/40 tasks,
   strict on 6/40 and normalized on 7/40. This supports an interface bottleneck
   beyond hunk metadata alone. Successful application still does not guarantee
   repair. Conditional success describes selected subsets and cannot allocate
   causal shares of unfamiliar-task failure. No new external/generalization
   confirmation was collected; almost no unused eligible family reserve remains.
3. **Numerical trust.** Exact-snapshot replay preserves actual behavior
   denominators and reproduces a material conditional update effect: diagnostic
   denominator substitution changes one Adam displacement by 21.4%, against
   0.4% repeat variation. Gradient-mode/frozen scoring agree on all tested
   targets, so that mode difference does not explain this sample. The numerical
   effect is not evidence that rescored denominators would be correct or improve
   repair. Cached scoring and fresh rollout reproduction must distinguish
   backend arithmetic from state/synchronization causes. New RL remains off.
4. **Gated OPSD.** Reuse the completed Task 17 three-seed evidence. It does not
   establish a credible robust gated advantage over GRPO or nominal ungated.
   Small fixed development sets, repeated families and numerical sensitivity
   limit the inference; no equivalence claim is warranted. Do not duplicate
   that study if the selected fresh-base interface and learning implementation
   remain unchanged.
5. **Partial rewards.** Saved hidden-training bounds guarantee informative
   variation in 8/20 groups versus binary's 7/20, with one additional all-binary-
   failure group. This is modest signal evidence from one historical run,
   not a learning-efficiency result. Missing initial-mutant baselines remain
   unknown. Keep student binary rewards and proposer binary-solve difficulty
   unchanged; prospective reward training is secondary to numerical integrity.
6. **Proposer C/D.** Existing distinct-reward candidate groups remain sparse,
   and no new proposer discrimination evidence was collected. Full C/D and
   adapter transfer remain unjustified in this phase. Validity alone is not
   proposer learning.
7. **Next step.** Fetch and verify the existing queued jobs, update the numerical
   diagnosis, and finish the native gate. If native qualifies, evaluate remaining
   primary families while reusing the completed probe records. Otherwise retain
   the supported snapshot-specific interface choices. A new objective pilot
   requires a substantive intervention, credible numerical evidence and budget
   for all paired arms. Do not tune until a favored method wins.

The current production backend explicitly selects BF16 for both FSDP forward
parameters and vLLM. A future FP32 precision ablation must first make those
execution paths honor the intended precision; changing the model-loading dtype
alone would not establish an FP32 production result. No such result is claimed.
