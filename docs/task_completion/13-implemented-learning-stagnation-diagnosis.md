# Task 13: learning-stagnation diagnosis implemented

Completed 2026-10-31. The CPU diagnosis, tested calculations, private quantitative artifacts, offline health dashboard, research report, and follow-up decision plan are implemented. No GPU job or new training study was submitted. GPU experiments are planned follow-ups with explicitly identified runner/controller dependencies, not completed experiments.

## Deliverables

- `scripts/diagnose_learning.py`: CPU-only observer over active committed event lineage and referenced raw shards. It reuses production parsing, objective normalization, OPSD coefficients, storage lineage, and the existing scientific-health analysis. It never executes generated Python, loads models, or writes historical W&B runs.
- `tests/test_diagnose_learning.py`: 12 focused tests for epsilon-sensitive reward invariance, uniform rewards, undefined cosine, duplicate missingness, overlapping diff failures, actual gate order, uncertainty-sensitive group-size projections, rollback/conflicting commits, missing/escaping shard paths, scheduler allocation integration, recurring-parent comparisons, and cumulative-history observation deduplication.
- [Research diagnosis](../investigations/learning-stagnation-diagnosis.md): Q1–Q8, measured funnels and time windows, proposer discrimination, student/OPSD behavior, scientific identifiability, and ranked interventions.
- [Execution and follow-up plan](../plan/13-diagnosis-plan.md): fixed-parent representation/model probe first, static student controls second, C/D counterfactual third; sampling, splits, costs, decisions, and proposed interfaces. GPU interfaces are explicitly marked as not yet implemented.
- `AGENTS.md` and the Task 12 completion record now record successful C reconciliation through 100, avoiding a duplicate recovery job.

The user-supplied untracked `docs/task/13-diagnosis-spec.md` remains unchanged. There are no changes to `src/`, scientific configuration, datasets, dependencies, checkpoints, or original experiment events. Relative to frozen execution, the existing source difference is the previously implemented controller continuation-stop change; the observer did not introduce it.

## Evidence reused and reconciled

Corrected runs A `747bd2e097258445de647e47`, B `f3271727f5fd25c07466e50d`, and C `35bfbd1bc074a8963e916f77` supplied 500 committed iterations, all 500 proposal shards, and all 500 episode shards. C's missing raw 41–100 trajectories were fetched with the existing Yale transport by targeting its trajectories directory, avoiding a checkpoint download. Existing historical A regression reports remain separate.

Live controller status, Slurm accounting, remote main directories, and corrected submission ledgers were inspected. A/B completed 200, C stopped at 100, no active job exists, and D has no corrected main directory or train ledger. Existing CPU job 31528917 completed with exit 0 in 29 seconds. Its normal reconciliation path verifies the latest checkpoint; cloud verification reached 100 with canonical revision `07e12f64924f324c994551ebb4a8f98037da9561849c9bf2796557cc9971a5e6`. No verification or reconciliation was repeated.

Primary quantitative results:

| Metric | A | B | C |
|---|---:|---:|---:|
| Accepted / attempted | 161/3,200 | 162/3,200 | 116/1,600 |
| Diff-application first rejection | 1,632 | 1,632 | 854 |
| Duplicate rejection | 1,204 | 1,140 | 492 |
| Groups with distinct rewards among accepted candidates | 4/800 | 5/800 | 7/400 |
| Advantages within L2 1e-5 of validity-only | 796/800 | 795/800 | 393/400 |
| Mixed student groups | 16/161 | 19/162 | 18/116 |

The 49 recurring A/B parents show declining acceptance and increasing duplication despite increased syntax/structural eligibility. This supports a novelty/concentration bottleneck rather than general formatting deterioration, without proving exhaustive mutation-space depletion. Capacity, matched-strength OPSD effects, static-GRPO learnability, and C/D update differences still require controlled probes.

## Reproduction and generated artifacts

Run from the repository root:

```bash
uv run --frozen python scripts/diagnose_learning.py \
  --run A=runs/hpc/yale-hpc-cpsc4880/747bd2e097258445de647e47 \
  --run B=runs/hpc/yale-hpc-cpsc4880/f3271727f5fd25c07466e50d \
  --run C=runs/hpc/yale-hpc-cpsc4880/35bfbd1bc074a8963e916f77 \
  --accounting runs/task13/accounting.json --output runs/task13/diagnosis
```

The private accounting JSON maps method names to scheduler jobs with `job_id`, UTC `start_epoch`, `elapsed_seconds`, and `gpus`. It was constructed from `TZ=UTC sacct -X -n -P --format=JobID,State,ExitCode,Start,End,ElapsedRaw` for the seven actual main allocations. Summed totals agree with the existing A/B/C records. Evaluation-time costs integrate completed and currently allocated attempts up to the event timestamp; they include startup and evaluation, unlike event-only training timers.

Private outputs:

- `runs/task13/diagnosis/{A,B,C}/diagnosis.json`: aggregate findings, source hashes, code/lock hashes, coverage, logical funnel with unknowns, actual ordered funnel, temporal windows, group statistics, OPSD action/outcome strata, exposure, and parent bootstrap contrasts.
- Per-method `proposals.json`, `groups.json`, and `parents.json`: auditable row-level diagnosis, actual/counterfactual advantages, per-parent diversity and sites, without duplicating raw source in the report.
- Per-method `funnel.csv`, `recorded_pipeline.csv`, `exposure.csv`, `group_size_projection.csv`, and `health.csv`.
- Per-method `health/`: existing regression report, windows, task transitions, and summary, reused rather than reimplemented.
- `runs/task13/diagnosis/summary.md` and `learning-efficiency.png/.pdf`: compact offline dashboard. The PNG was visually inspected.
- `runs/task13/*-status.json`, `inspection.json`, `corrected-ledgers.json`, `remote-inventory.json`, `accounting*.txt/.json`, and `verification-accounting.txt`: operational provenance.
- `runs/task13/validation.json`: unchanged input hashes, matching observer hashes, coverage/totals checks, current scientific source hash, and zero GPU-hours allocated by this task.

The new per-iteration observability export identifies groups with 2+ accepted tasks, reward-discriminating groups, and advantage distance from validity-only. These are the proposed minimal additions to a new study's W&B logger; this task makes no historical cloud mutation or raw/private-data upload.

## Validation

```bash
uv run --frozen pytest -q tests/test_diagnose_learning.py tests/test_scientific_health.py
uv run --frozen pytest -q -m 'not gpu and not download and not wandb_online'
uv run --frozen ruff check src tests scripts
uv run --frozen ruff format --check src tests scripts
uv run --frozen mypy src/selfplay_opsd
git diff --check
```

Results: **16 focused tests passed**; full CPU suite **269 passed, 33 skipped, 4 deselected**. Ruff and formatting checks passed for all 85 Python files, mypy passed for 43 source files, and diff whitespace checks passed. Four dependency warnings concern existing Typer/Click and W&B/Pydantic usage. Final artifact checks reconciled all 8,000 proposals, 439 within-run unique admitted tasks, and 1,756 episodes, and verified every recorded input hash was unchanged. CPU tests do not claim new GPU compatibility or generated-code isolation evidence.

## Remaining uncertainty and next action

No existing-data blocker remains for the CPU diagnosis. Generation stop reasons, full policy entropy, complete historical proposer ratios, per-step adapter displacement, and dense RL/OPSD gradient decomposition were not recorded; they are not reconstructed from aggregate proxies. G=8 projections are prior-sensitive scenarios, not observed eight-rollout performance. One-seed development differences do not establish causal superiority.

Implement the scoped inference runner/controller route in the follow-up plan, freeze its manifest, validate its preview, and start the 4B paired representation probe. Initial planning caps are four H200-hours for staged inference, six for static controls, and two for the C/D counterfactual, with expansion based on information gained. The user authorized efficient GPU spending without another budget question. No full D, larger-model training, or final held-out evaluation follows automatically from this completion record.
