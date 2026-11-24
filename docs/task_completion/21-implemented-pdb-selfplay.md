# Task 21: PDB self-play study completed

Implemented the pinned PDB adapter, independent scorer parity checks,
parent-disjoint partitions, training/evaluation pipeline, checkpoint and
allocation verification, monitoring, and guarded interrupted-job recovery.
Matched Control, ungated A, and gated B completed 512 attempted episodes in
each of two seeds; all 12 frozen baseline and endpoint cells completed and
were verified. The final primary PDB single scores were Control/A/B
22/26/28 and 21/21/28 of 64 by seed. Within-seed task-bootstrap intervals
include zero for the primary B-versus-Control comparisons, and 19–21 optimizer
steps do not establish convergence. Small transfer cells limit generalization
claims. Total reconciled allocation was 217.155 H200-hours, including failures
and continuations. See [results](../investigations/task21-results.md) and
[reproduction and execution history](../investigations/task21-status.md).
All scientific records and verified endpoint checkpoints remain preserved;
private data and artifacts remain ignored.
