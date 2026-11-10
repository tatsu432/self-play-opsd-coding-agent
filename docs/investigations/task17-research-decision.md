# Task 17 research decision — provisional, retrieval blocked

## Supported conclusion

Strict edit encoding hides substantial demonstrated repair ability in the saved
HumanEval+-derived repair trajectories. The six trained evaluations accepted only
13 edits and rejected 4,301. Oracle-free exact-context recovery followed by
isolated verification finds a correct reconstructed state in **242 originally
failed episodes**, covering 96/116 parent problems when all passing reconstructed
states are considered. At least these recorded failures cannot be explained
solely by an inability to propose semantically useful edits. These observations
span repeated snapshots/actions; they are not 242 independent problems or a new
agent's measured accuracy.

There are semantic limits too: only 118/234 distinct admitted trained external
task/program reconstructions pass. Recoverable encoding and semantic correctness
are different endpoints. Task distribution, source presentation and debugging
behavior remain plausible additional causes. No claim that HumanEval transfer is
solved is justified.

## Interface evidence and hypotheses

All four matched greedy allocations finished for 40 independent training-only
families plus 16 correlated presentation variants. They used the same 4B
snapshots, tasks, oracles, budgets, isolation and hardware. Only strict-control
artifacts are completely retrieved. Last remote aggregate inspection found
base strict/text repairs 6/56 versus 26/56, and trained strict/text 36/56 versus
31/56. These exploratory totals suggest snapshot-dependent conditioning effects;
they do not establish the prespecified primary effects or their uncertainty.

H1 (edit acceptance), H2 (verified repair gain), H3 (length/structure interaction)
and H4 (fewer wasted actions) await paired original-task analysis. H5, interpreted
as semantic inability being the sole explanation of the historical failures, is
contradicted by verified correct reconstructed states. The relative causal shares
of syntax and semantics remain unmeasured. Keep the primary strict control
unchanged; no new training interface has been selected.

Trained strict diff solved 5/8 originals versus 1/8 source-comment variants,
while specification repetition preserved 8/8. These small exploratory results
support source-presentation sensitivity, without separating comment content from
length. The exact-text counterparts are still needed to assess an interface
interaction. A locally tested metadata-only diff mode can distinguish tool-schema
conditioning from metadata application if primary results justify that one
additional diagnostic; no normalized model run has occurred.

## Student objectives

No new calibration or objective training has run. Consequently Task 17 does not
yet answer whether gating beats GRPO, ungated OPSD or initial-gradient-matched
OPSD, whether effects replicate across seeds, or which method is more efficient.
Historical Task 15's MBPP+-derived B advantage remains descriptive evidence from
evolving proposer/student runs, not an isolated causal gating result.

After retrieving the primary artifacts, select the simplest supported interface,
freeze common curriculum/exposure and initial-gradient calibration, and run a
bounded GRPO/gated pilot. Extend only if finite learning signal and measured
cost make an informative matched comparison feasible within the additional
24 H200-hour ceiling. Preserve actual PPO behavior denominators and
`same_scorer_v2`; calibration matches initial gradient strength, not all later
updates. An access dependency is not a negative training result.

## Proposer objectives

Full C/D remains no-go under this task. Historical candidate groups discriminate
difficulty rewards in only about 5.1–7.7% of attempts. Prior RL-versus-combined
update cosine near 0.992 was close to identical-reward repeat variation near
0.995, with few informative multi-valid groups. A further C/D pilot needs a
prospectively collected training-only set with diverse, distinct-reward valid
candidates and repeat-controlled gradient/update separation. No new evidence
currently supplies that justification.

## Next action and costs

Retrieve existing outputs after restoring Yale VPN/SSH authentication, run
`scripts/collect_task17_interfaces.py`, inspect paired effects and complete the
Phase 1 decision. Do not relaunch finished cells. The four allocations consumed
**1.15 H200 GPU-hours**, with final all-cell accounting persistence pending.
Phase 2 consumed zero. The branch remains unfinished and unmerged. See the
[results](task17-repair-interface-results.md) and
[execution record](../task_completion/17-implemented-repair-interface.md) for
identities, limitations and exact recovery commands.
