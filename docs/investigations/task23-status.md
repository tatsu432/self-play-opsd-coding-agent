# Task 23 execution and reconciliation

The Tasks 21 and 22 records reproduce all 12 primary scores, all 60 development
measurements, and 322.530 historical H200-hours. See the [report-ready results](../report/task23-results-and-discussion.md)
and the [five-minute presentation](../report/task23-presentation.md).

Yale access resumed via Duo. At the first live check, the project filesystem had
138 GB available and the account had no active jobs. Task 23 uses separate
contents-addressed evaluation manifests and a locked Slurm submission ledger;
original checkpoints and historical results remain in place. All new results
must match the original model, adapter, execution runtime and verifier.

The budget sensitivity experiment varies the per-turn generation cap from
4,096 to 6,144 tokens. The original eight-turn and 12,288-token episode limits,
task seeds and tool templates are retained. The original 64 tasks are sorted
by the stable family/task hash before model execution. An eight-task timing
tranche uses the same manifest and resumes from cached task records.

Independent source and complexity cohorts use pinned fixture inputs, prior
parent and structural-family exclusion, and model-independent selection.
The accepted worker must verify correct and buggy source before model
evaluation. The original small transfer sets remain separately reported.
