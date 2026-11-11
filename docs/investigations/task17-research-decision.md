# Task 17 research decision

The matched repair-interface and student-objective studies are complete.
All 12 objective endpoints were reconciled and verified. The detailed
[interface evidence](task17-repair-interface-results.md),
[objective results](task17-objective-results.md) and
[execution record](../task_completion/17-implemented-repair-interface.md) retain
run identities, methods, reproducible analyses, resource accounting and limits.

## Generalization and interface

The strongest supported explanation of the observed HumanEval+-derived repair
failure is a combination of edit-encoding failures and limited semantic
correctness. Across six trained snapshots, 4,301 patch actions were rejected
and 13 accepted. Isolated replay of oracle-free mechanically reconstructed
edits finds a correct saved state in 242 originally failed episodes; those
records are neither independent problems nor a counterfactual accuracy measure.
Only 118/234 distinct admitted trained external task/program reconstructions
passed, so syntax recovery does not explain every failure.

On 40 training-only families, fresh-base strict diff versus exact text applied
an edit on 6/40 versus 23/40 tasks and verified 5/40 versus 19/40 repairs.
The paired repair difference was +35 points, family-bootstrap 95% interval
[+17.5, +50]. For the trained B2026 snapshot the corresponding edits applied
were 33/40 versus 28/40, and repairs were 27/40 versus 21/40 (difference −15
points, interval [−35, +5]). Thus simplified text improves the base model's
observed acceptance and verified repair, while effects on the trained snapshot
are uncertain and may depend on its conditioning. Exploratory length strata
are small and do not establish a causal length interaction. Both interface and
semantic capability matter; the evidence does not quantify their causal shares
or establish solved transfer to genuinely unseen tasks.

The hypothesis-level assessment is: H1 (edit acceptance) and H2 (verified
repair gain) are supported for the fresh base on the training-only sample, but
not established for the trained snapshot. H3 (length or structure interaction)
is inconclusive in the small exploratory strata. H4 (less wasted tool activity)
has descriptive support: base rejected edits fell from 101 to 51 and total turns
from 308 to 275; action counts are correlated and do not isolate a mechanism.
H5 (failures are solely semantic) is contradicted for the 242 recorded failed
episodes with a verified reconstructed state, but semantic failures also occur.
These observations are not a new end-to-end estimate on unseen HumanEval+.

## Student objectives

The fixed-supply study uses the fresh 4B base model, exact text, 80 attempted
episodes per arm, equal task order, and independent 16-family development plus
41 monitoring tasks at 0/40/80. All three seeds of each of the four student objectives completed. At 80 episodes, independent development repairs out of 16 were:

| Objective | 2026 | 2027 | 2028 |
|---|---:|---:|---:|
| GRPO | 10 | 8 | 9 |
| Gated OPSD | 10 | 8 | 10 |
| Ungated OPSD, coefficient 0.1 | 11 | 11 | 10 |
| Ungated OPSD, matched initial gradient scale | 9 | 9 | 9 |

Gated does not show a robust independent-development advantage over GRPO in
these three seeds: paired differences are 0, 0 and +1 task. At the same
endpoint nominal ungated exceeded gated by one and three tasks in the first two
seeds, and tied it in the third (with two different tasks solved by each). Calibrated
ungated differs from gated by −1, +1 and −1 task. Matching initial auxiliary
gradient magnitude does not match later directions or optimizer states.
Monitoring results vary by seed and can favor ungated, but these are the same
tasks across arms and seeds and are not untouched confirmation. Within-seed
family bootstrap quantifies task sampling conditional on a training run; it is
not an interval over training seeds. The small fixed independent set and three
seeds do not support equivalence, reliable superiority or a strong efficiency
claim. Equal episode exposure does not imply equal generated tokens, optimizer
steps or GPU time; see the paired curves and accounting in the results.

The original behavior-policy denominators and `same_scorer_v2` were preserved,
with finite recorded gradients. Rollout/rescoring mismatch remains a material
numerical limitation: the largest observed value is 5.4899 in calibrated 2026,
with 0.1793% of 3,346 sampled tokens exceeding 0.2 in that iteration. Its
causal impact is unresolved. No post-hoc denominator change was made.

## Proposer and overall decision

The current proposer has too few informative multi-valid candidate groups for
a justified full C/D study. Historical difficulty rewards discriminated in
about 5.1–7.7% of attempts; a prior RL-versus-combined update cosine near 0.992
was close to identical-reward repeat variation near 0.995. A future C/D pilot
needs prospectively collected diverse, distinct-reward valid candidates and
repeat-controlled gradient/update separation. No full proposer C/D was launched.

The strongest supported conclusion is that repair-interface encoding materially
obscured useful proposed edits in saved trajectories and improved fresh-base
repair when simplified, while objective effects are small and uncertain on the
independent fixed set and depend on seed and comparison. The present evidence
does not isolate a general gating advantage or establish unseen generalization.
