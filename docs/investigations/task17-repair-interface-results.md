# Task 17: repair encoding and controlled objective study

## Current decision (2026-11-10, execution in progress)

Mechanical replay establishes a substantial encoding bottleneck in the saved
trajectories. It does not measure a new agent's end-to-end accuracy. The required
matched frozen-model comparison is next; exact replacement has not yet been
selected for training. Full C/D remains deferred. No Task 17 GPU allocation has
yet started. The first submission was rejected before allocation because H200
permits at most six CPUs per GPU. Task 17 now requests five trainer CPUs plus
one isolated-worker CPU; science and runtime compatibility are unchanged.

## Retrospective evidence

The checksummed Task 16 inputs reproduce **all 1,617** final/base trajectories,
11,516 actions. The six trained snapshots contribute 1,386 episodes. Reconstruct
only complete, uniquely matching old hunk context; ignore coordinates/counts,
leave added/removed content unchanged, and reject ambiguous/overlapping/reordered
hunks or unsafe headers. Reconstruction never consults the correct solution.
Syntax/source/API admission precedes isolated semantic verification. Each edit
is tested at its recorded source state; later saved actions are not a valid
counterfactual continuation.

| Six trained snapshots | MBPP+-derived repair | HumanEval+-derived repair |
|---|---:|---:|
| Episodes | 690 | 696 |
| Accepted / rejected patches | 613 / 1,836 | 13 / 4,301 |
| Unique-context metadata recoveries | 864 | 1,132 |
| Recoveries passing static admission | 837 | 1,107 |
| Reconstructed actions passing full verifier | 500 | 755 |
| Distinct passing task/program pairs | 80 | 118 |
| Episodes with a passing reconstructed state | 193 | 249 |
| Originally failed episodes with such a state | — | **242** |

CPU job `32794598`, run `9d9776357a4e23ae82cd34dc`, verified all 1,991
admitted trained/base actions using **449 distinct task/program verifier calls**.
Identical programs reuse the recorded outcome. The base contributes 13/14
passing ID and 25/33 passing external reconstructions, spanning six and ten
episodes respectively. Across trained external snapshots, correct reconstructed
states cover 96/116 parent problems; repeated actions and snapshots are not
independent tasks. No oracle solution is used to choose or reconstruct an edit.

| First deterministic rejection | Trained MBPP+ | Trained HumanEval+ |
|---|---:|---:|
| Context mismatch | 661 | 2,913 |
| Hunk counts | 997 | 1,141 |
| Hunk offsets | 7 | 0 |
| File headers | 115 | 116 |
| Forbidden paths / extra file content | 56 | 131 |

Action acceptance falls with source length within MBPP+: 396/694 edits (57.1%)
for tasks at/below 35 source tokens versus 217/1,755 (12.4%) above that median.
HumanEval+ is near its floor in both strata: 13/2,230 versus 0/2,084 at a
173-token median. All 13 accepted external patches also lie below the median
specification length (115.5 tokens). These are confounded action-level
associations, not causal length effects. MBPP+ tasks with at most one branch
accept 488/1,181 edits versus 125/1,268 with more branches. The reproducible
private stratified table also covers mutation type/sites, public-case count,
preceding turn, method and each checkpoint/seed; source and specification
features overlap. Reproduce with `scripts/analyze_task17_strata.py` against
`runs/task17/recovery-full-v1/`, using a new output directory.

The static parser reproduces errors even where historical final-turn tool
observations were absent. Categories describe the first rejection and can mask
later defects. Another 50 ID and 215 external actions fail JSON/schema parsing.
Repeated unchanged-source/error-category attempts beyond the first total 1,153 ID
and 3,226 external. This is broader than identical-action repetition.

![Rejected patch taxonomy](../figures/task17/rejection-taxonomy.png)

![Mechanical recovery verification](../figures/task17/mechanical-recovery-verification.png)

These results reject an explanation based solely on semantic inability for at
least 242 recorded external failures. They do not quantify a causal improvement
under a changed tool prompt, prove all failures are syntactic, or establish new
HumanEval generalization. H1–H4 await the matched comparison; a mixed syntactic
and semantic explanation remains possible.

## Frozen-model comparison and admission

The [implementation specification](../task/17-repair-interface-spec.md) defines
base and checksum-verified B2026-final snapshots × strict diff and exact text,
identical greedy budgets/oracles, 40 primary training-only families and 16
exploratory presentation variants. Curriculum, calibration and independent
development reserve 40/8/16 other families. No model outcome selects parents.

Initial preparation `32794599` / `6be54fe0c6ab8d8a6c492675` lacked admissible
mutations for seven reserved parents. Revised preparation uses 32 bounded fixed
attempts, caches identical mutants and refills from unused families. Job
`32797537` / `0522e9eae55f829aadc11294` verified all 104 originals, then failed
variant verification. Diagnostic job `32801929` /
`f0b3c13f22f1fdbd0aa7e20b` found that `Mbpp/599`'s mutant exhausted the fixed
allowance in a comment variant, original recheck and variant repeat: 90 runtime
failures and zero semantic failures each. Inputs reach 100,000,007; the cause of
timing variation from initial admission is unproven. Exclude it before inference
and use the next reserved family, retaining the other 103 admissions.

Replacement preparation `32804290` / `86e2df49bb04a9321f7d755d` completed exit 0 from
execution `8654b20`. Failed preparation directories and original ledger bytes
remain archived with updated artifact locations and archive histories. No
historical Task 15 task, oracle, checkpoint or result was changed.

Normal Yale VPN/Duo access is restored. CPU runtime checks `32792142` and
`32793248` completed successfully; preparation now performs current runtime
validation itself. The accepted existing isolated worker is reused. Earlier DNS
and missing-base-shard issues are resolved; partial local recovery versions are
historical, not current evidence.

## Reproduction

```bash
uv run python scripts/analyze_task17.py --recovered runs/task16/recovered-analysis-verified --data data/protocol-v3 --output runs/task17/recovery-full-v1 --figures docs/figures/task17
uv run python scripts/analyze_task17_replay.py --audit runs/task17/recovery-full-v1 --replay runs/hpc/yale-hpc-cpsc4880/9d9776357a4e23ae82cd34dc --output runs/task17/semantic-replay-analysis --figures docs/figures/task17
```

Use a fresh directory for each rerun. Full coverage, input checksums, original
outcomes and final-source reconstruction are enforced. Private task/source/case
artifacts stay ignored under `runs/`; public plots contain aggregates only.
Replay candidates SHA-256:
`37f01a4e83c7522e97c4936f44ec1a1cd56411ef62cad430896593c5bbf65f8f`.
See the [execution plan](../plan/17-repair-interface-plan.md) and
[completion record](../task_completion/17-implemented-repair-interface.md).

Prepared pool SHA-256: `602eaf506f8ac2cf17924596dae7c0ff3891ef5034a546d4d3751b9803412c83`.
All 120 rows passed tokenizer admission under both protocols. Rejected GPU
submission `dc2a6394ef94ef1bd7768b1e` has no scheduler job or charged allocation.

Base strict-diff job `32805621` / `9f12c84dfddf9b37f42dd4f1` is queued from
`e6e7412715169090add451737f91d01c55283eea`. Its pending wall limit was reduced
from two hours to one for backfill; the ledger preserves the original immutable
specification and an explicit pending-resource adjustment, while retaining the
conservative two-hour budget reservation. No task/model/decoding parameter
changed. Actual allocation will be reconciled from Slurm. The next strict cell
must use the same prepared pool and frozen model budgets.
