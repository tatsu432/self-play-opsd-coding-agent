# Task 22 completion: PDB gradient alignment

The study qualified and executed the C and D proposer objectives on the same
PDB configuration as Task 21. Both seed-2026 runs reached 512 attempted
student episodes, with checkpoint-backed primary and transfer evaluations.
C and D verified 33/64 and 25/64 primary tasks, respectively; the paired
D-minus-C difference was −12.5 percentage points, with a task-family bootstrap
interval of [−25, 0] conditional on the two runs. This does not establish a
population-level winner with one seed. The results, transfer counts, mechanism
and limitations are detailed in [Task 22 results](../investigations/task22-results.md).

The implementation supplies the PDB reference gradient and paired
RL-only/combined gradient diagnostics without changing the task curriculum,
model, optimizer, shared adapter or qualified execution path. It adds
provenance-preserving continuation support, bounded accounting, aggregate
analysis and reproducible publication figures. The original Task 21 study and
its frozen records remain intact. The matched qualification found C/D near-zero
candidate gradients on 5/14 and 0/14 tasks; trajectory diagnostics give
additional descriptive evidence but do not establish an accuracy effect.
See [qualification](../investigations/task22-alignment-qualification.md),
[mechanistic analysis](../investigations/task22-mechanistic-analysis.md), and
[execution and reproduction](../investigations/task22-status.md).

All Task 22 allocations, including timeouts and continuations, total 196,387
H200-seconds (54.5519 hours) under the 95-hour cap. C/D transfer results were
1/8 and 1/8 harder PDB, 1/1 and 1/1 cross-source, and 5/9 and 6/9 mutants.
These small transfer sets support descriptive comparisons only. The optional
matched 768-episode Control/A/B extension was deferred because the estimated
training cost plus evaluation and contingency did not fit reliably in the
remaining budget; see the results document. No Task 22 jobs were active at
final Slurm reconciliation. Completed jobs must not be duplicated.

Private datasets, task records, checksums, receipts and accounting remain in
ignored paths. The analysis consumes verified receipts and fetched summaries;
the execution document identifies runs and the reproducible command. Retain
checkpoints and scientific records for future authorized experiments.
