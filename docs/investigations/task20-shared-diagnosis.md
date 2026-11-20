# Task 20 shared-policy viability diagnosis

The seed-2026 shared-control pilot failed its prespecified viability gate after
64 attempted solver episodes. It produced one effective solver RL update in
two batches and no within-group reward variation among valid proposer outputs.
These observations do not establish that the shared gradient interfered with
solver learning; they show that this small pilot did not provide the required
optimization and proposer-discrimination evidence.

The pilot reused the solver control's initial adapter, worker, data and
scientific execution identity. Its 41-task initial development evaluation was
25/41, matching the solver-only baseline task by task. Its frozen proposer scale
was 0.028920942507191107, calibrated on a separate initial batch. Endpoint
development was 26/41: two tasks gained and one lost relative to baseline.
This small, previously monitored set is exploratory. Slurm charged 8,749
H200-seconds for this pilot; total Task 20 allocation was 63,218 seconds
(17.560556 H200-hours).

| Batch | Solver successes per group | Mixed groups | Solver gradient norm | Proposer valid/attempted | Groups with distinct valid rewards |
|---|---|---:|---:|---:|---:|
| 1 | 2, 2, 8, 8 | 2/4 | 0.0088883 | 7/8 | 0/2 |
| 2 | 0, 8, 8, 8 | 0/4 | 0 | 6/8 | 0/3 |

The [checkpoint-verified signal figure](../figures/task20/shared-pilot-signal.png)
shows both the solver reward groups and the absence of valid-task ranking.

In batch 1 the unweighted proposer gradient norm was 0.1536654; its weighted
norm was 0.0044441 and its cosine with the solver gradient was 0.00261. The
joint update norm was 0.05670. In batch 2 the corresponding unweighted and
weighted proposer norms were 0.0201306 and 0.0005822; the joint update norm was
0.03779. The second cosine is undefined because the solver gradient was zero.
Both optimizer steps were finite, and same-policy rescoring recorded zero
support exits. Batch 2's step therefore shows a proposer contribution, not a
solver RL update.

Inspection of the private batch records shows that each proposal group gave
all its valid members the same reward. Groups containing only valid proposals
therefore had zero centered advantages. The nonzero proposer gradient came from
contrasting invalid proposals (reward -1) with valid proposals, not ranking
valid candidates by usefulness. Duplicate candidates and the small number of
parent groups further limit the pilot: only three of the five groups contained
more than one unique valid task, and none of those three had different valid
rewards. A near-zero cosine in one nonzero batch
cannot establish either harmful interference or its absence over training.
The collector stops at four unique tasks per batch and limits each new proposal
group to the remaining task slots. Consequently the second batch's groups had
four, three and one proposal; a one-proposal group cannot have a centered
ranking signal. A larger unique-task allowance per update is a concrete
training-only intervention to investigate, but its cost, reward diversity and
solver signal would need fresh qualification under a separate identity.
These results are conditional on one seed and a short pilot, and do not prove
that shared self-play or OPSD cannot work under a different distribution.

The [solver-only anchor](task20-positive-control.md) showed repeatable small
training-set gains over eight informative updates but no clear improvement on
its separate training probe. It supports optimizer learnability under a fixed,
screened curriculum; it does not establish held-out generalization. The main
Control/A/B comparison and C/D alignment arms should not be presented as
completed or launched automatically from the failed gate. A new intervention
would need a specific training-only difficulty or valid-reward discrimination
hypothesis, separate identity and a fresh qualification, rather than selecting
on final-test outcomes.
