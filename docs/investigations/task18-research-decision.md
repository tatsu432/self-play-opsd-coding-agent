# Task 18 research decision

The bounded phase is complete. See the [results](task18-results.md),
[execution and reproduction record](task18-status.md) and
[completion record](../task_completion/18-implemented-interface-signal-bottlenecks.md).
New allocation was 1.274722 H200-hours of the hard 12-hour ceiling. New RL was
not supported by the numerical evidence; budget availability was not the blocker.

1. **Interface:** select Qwen-native single-call serialization with exact-text
   replacement for this benchmark. Base/trained verified repairs are 27/40 and
   31/40, against plain text's 19/40 and 21/40; valid-schema calls are 216/217
   and 189/189. Normalized diff scored 6/40 and 28/40. The trained native-minus-
   normalized gain is only three tasks, with family interval [−5, +20] points;
   normalized remains a lower-allocation-cost fallback. Native trained uses
   only 1.15% more generated tokens than normalized. This is an engineering
   choice on inspected development families, not universal interface superiority.
2. **Application versus semantics:** native base edits 37/40 tasks, versus plain
   text's 23/40 and strict's 6/40. Ten native base and seven trained episodes
   still fail after an edit. Applied subsets differ, so their conditional
   success rates cannot identify causal shares of failure. The strict trained-
   versus-base score gap of 22 tasks shrinks to four under native calling;
   that is descriptive conditioning dependence, not proof of a specific learned
   mechanism. No new HumanEval or untouched generalization confirmation was run.
3. **Numerical trust:** the backend is not cleared for another objective cohort.
   At an exact retained snapshot, diagnostic denominator substitution changes
   one restored Adam displacement by 21.4%, versus 0.4% repeat variation.
   Full/gradient scoring agrees, while incremental/full scoring and fresh vLLM
   execution differ on identical prefixes. Arithmetic contributes; every
   kernel/batching/state cause and downstream learning effect are not isolated.
   The worst historical checkpoint was retired. Keep actual behavior denominators;
   neither harmlessness nor historical training invalidity is established.
4. **Gated OPSD:** the completed Task 17 three-seed study does not establish
   robust superiority over GRPO or nominal ungated OPSD. Reuse its measured
   curves and endpoints, with the numerical limitation added. No equivalence
   claim or duplicated objective study is warranted.
5. **Partial rewards:** hidden-training fractions guarantee informative variation
   in 8/20 saved groups versus binary's 7/20, with nine possible given unresolved
   bounds. One extra all-binary-failure group is modest evidence, not a learning-
   efficiency result. Keep binary student rewards and binary-solve proposer
   difficulty unchanged; missing initial-mutant baselines remain unknown.
6. **Proposer C/D:** insufficient distinct-reward evidence remains. Better native
   tool use does not establish useful proposer gradients. Full C/D, transfer,
   SFT, new editors, observation changes and model scaling remain deferred.
7. **Next experiment:** freeze a narrow precision/execution-parity ablation on
   retained mixed and uniform states, with correct behavior probabilities and
   repeat-controlled real Adam displacements. Production FSDP and vLLM currently
   hardcode BF16; changing model-loading dtype alone is not an FP32 experiment.
   Only after numerical credibility should a separately identified native-interface
   GRPO/gated/nominal cohort proceed with two paired seeds, common 32-episode
   pilots and a predeclared common endpoint. Do not continue arms based on
   development wins or silently spend a historical budget in a new study.

The eight-family native screen was deterministic and its completed episodes
were reused unchanged. Full-40 family intervals include those screening tasks;
remaining-32 results are reported separately. All families were previously
inspected, intervals are not training-seed uncertainty, and historical controls
cross executable revisions with locally tested default prompt/action parity.
The remaining-32 base interval includes zero. These limits accompany the
interface decision rather than being hidden behind the full-set point scores.
