# Task 23 execution and reconciliation

The Tasks 21 and 22 records reproduce all 12 primary scores, all 60 development
measurements, and 322.530 historical H200-hours. See the [report-ready results](../report/task23-results-and-discussion.md)
and the [five-minute presentation](../report/task23-presentation.md).

Yale access resumed via Duo. At the first live check, the project filesystem had
138 GB available and the account had no active jobs. Task 23 uses separate
content-addressed evaluation manifests and a locked Slurm submission ledger;
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


| Stage | Slurm job | Scope |
| --- | ---: | --- |
| Relaxed-budget timing | 34871390 | Seed 2026 B. Eight-task prefix of the frozen 64-task ordering. |
| Transfer fixture validation | 34871763 | CPU validation of separate source and complexity candidate pools. |

The structural inventory covered 151 previously used families. The strict
adapter found one unused PDB-Multi family with directly compatible fixtures.
The expanded candidate freeze has 65 source and 113 complexity families
before execution filtering. BigCodeBench adaptation uses existing assertions
where supported and captures reproducible canonical outputs on pinned upstream
test inputs otherwise. The complexity pool prioritizes PDB-Multi and supplements
it with pairs of independent AST-site changes in held-out LiveCodeBench-derived
parents. Each single change and the joint change must fail intended checks.
These structural exclusions do not guarantee semantic disjointness.

The baseline [consolidated W&B run](https://wandb.ai/tatsu432s/self-play-opsd-coding-agent/runs/task23-final-aggregates)
has independently verified cloud history and five verified aggregate tables.

## Relaxed-cap comparison launched

The B timing job completed exit 0 with eight verified task records in 1,218
H200-seconds. The expansion reads those same eight task files in the same
frozen namespace before generating any new records. Based on same-task timing
and historical per-cell episode runtime, the full 64-task comparison was
priced at approximately 33 H200-hours, with 43.8 hours of original evaluation
wall reservations before continuation. Those forecasts are separate from actual
charges. The scheduler currently enforces six simultaneous H200 jobs on the
account.

| Seed | Initial | Control | A | B | C | D |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026 | 34872118 | 34872121 | 34872127 | 34872238 | 34872130 | 34872132 |
| 2027 | 34872134 | 34872135 | 34872136 | 34872138 | 34872139 | 34872143 |

Use the local private evaluation ledgers and Slurm terminal state before any
continuation. A complete cell's manifest and task records remain authoritative.
With Yale SSH available, fetch completion markers and actual accounting with:

```bash
uv run --frozen python scripts/fetch_task23.py
uv run --frozen python scripts/analyze_task23.py --output docs/figures/task23
```

## Independent transfer freeze

CPU job 34871763 completed exit 0 in 1,222 elapsed seconds and used zero GPUs.
It accepted 17 BigCodeBench-derived source-transfer tasks and 32 independent
LiveCodeBench-derived complexity tasks. All 49 are from distinct structural
families. All source tasks use stable canonical observations from the pinned
upstream tests. The compatible native PDB-Multi candidate did not qualify, so
the 32 complexity tasks are verified two-site mutations. The original PDB-Multi
cell stays separate.

The full accepted cohort is frozen before model evaluation. The seed-2026 B
eight-task timing job (34872707) is queued under the same isolated worker and
original inference budget. Its task records belong to the final OOD cell.

The 512-episode strength-matched A′ arm is deferred. The exact matched
initialization replay establishes the scale difference, while a useful new
training arm would require representative multi-snapshot calibration and
roughly 21–28 H200-hours for one additional seed. The remaining experimental
priority is completing the two checkpoint evaluation comparisons within the
six-job scheduler limit. Additional training seeds and longer continuations
also remain deferred.

The OOD timing job completed exit 0 in 371 H200-seconds. Its eight tasks
averaged 39.20 seconds of generation, tool and verifier time. The final B
cell will reuse all eight records. The other cells were forecast using the
older primary and harder-transfer timings, for about 23.7 H200-hours total
and 34.0 hours of initial full-cell wall reservations. The queue briefly
prioritized the OOD timing job ahead of pending sensitivity D/2027 using a
25-point nice adjustment to the sensitivity job. The original priority
was restored after the timing job started.

| Seed | Initial | Control | A | B | C | D |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026 | 34882862 | 34882860 | 34882855 | 34882857 | 34882858 | 34882861 |
| 2027 | 34882869 | 34882867 | 34882864 | 34882865 | 34882866 | 34882868 |


All 12 relaxed-cap cells completed exit 0 with the full 64 task records per
cell. Task 23 charged 118,105 H200-seconds to the sensitivity comparison,
including its timing tranche. The final comparisons are in the [results
and discussion](../report/task23-results-and-discussion.md#per-turn-generation-budget-sensitivity).
