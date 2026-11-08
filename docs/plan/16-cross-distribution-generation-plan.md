# Task 16 implementation and execution plan

Analyze the completed Task 15 A/B study without retraining, changing the frozen data/protocol, or editing historical outputs. The accepted plan authorizes a conditional earlier-checkpoint pilot within eight new H200-hours; development-only evidence determines whether it is useful.

## Evidence and implementation

1. Work on `feat/16-generalization-checkpoint-analysis`, preserving the user-supplied specification unchanged. Reuse the private Task 15 manifest, six run event streams, common base, frozen data and allocation ledgers. Hash all consumed inputs and record unavailable dependencies.
2. Validate active committed lineage and complete evaluation attempts, including exact task/parent identities. Audit partition/family boundaries, frozen mutation regeneration, original validation evidence, source complexity, public/hidden case counts, and specification style. Never execute generated code locally.
3. Implement a reusable analysis module and `scripts/analyze_task16.py`. Export private dataset features, final-outcome diagnostics, training-health windows, checkpoint selection, matched-budget checkpoint availability, paired effects and resampling uncertainty.
4. Fetch only event-referenced final-evaluation and training/proposer trajectories using a validated relative-path list. The selective route must exclude checkpoints and model weights. A run-root override supports fetched copies without modifying the historical manifest.
5. Preserve primary terminal categories and overlapping diagnostic flags separately. The primary precedence is success, confirmed infrastructure, public-pass/hidden-fail, recorded budget exhaustion, no-edit rejected-patch/parse/other failure, edited semantic/runtime failure, unknown. Missing trajectories must not create zero action counts. Recover an action's observation only from the next saved history; the final observation can be absent.
6. Select maximum monitoring-development parent-macro accuracy, earliest measured checkpoint on ties. Bootstrap families jointly across checkpoints, rerunning selection and inspecting out-of-bootstrap differences. Never select using test outcomes. Compare earlier/final checkpoints using paired tasks and preserve small-set uncertainty.
7. Integrate scheduler allocation intervals including initialization, evaluation, interrupted work and retries. Compare observed milestones and the best measured checkpoint available within supported common episode/token/step/hour budgets; label the estimand explicitly. Do not treat interpolated or carried checkpoint availability as a measured equal-time model evaluation.
8. Aggregate committed proposal/student supply, repetition, outcome mixtures, action counts, scoring costs and recorded gradient diagnostics. Use actual milestone crossings for early/late windows. Do not infer group counts from a per-candidate nonzero-advantage counter, or zero total updates from uniform rewards.

## Scientific decisions and outputs

- All six locally reconstructed parent-macro selectors choose the final checkpoint. Verify this in implementation; absent contrary corrected evidence, skip the GPU pilot and preserve the eight-hour allowance unused. No new secondary-evaluation route is needed without a selected earlier checkpoint.
- Rank transfer explanations by measured support and counterevidence. Prefer descriptive stratification over an unstable unregularized regression with near-zero external successes. Treat final-test analyses as exploratory.
- Produce six aggregate PNG/PDF figures, the public investigation and completion record. Keep sources, hidden tests, raw trajectories and task-level outcomes in ignored storage. If Yale is unavailable, complete independent work and state exact recovery commands and incomplete conclusions.

## Validation and delivery

Test lineage conflicts, incomplete/duplicate/mismatched outcomes, taxonomy precedence, missing trajectories/checkpoints, selection/ties, allocation accounting, exposure overshoot, clustered statistics and selective-fetch path validation. Run the CPU suite, Ruff lint/format, mypy and hosted CI where accessible. Check deterministic numerical output and visually inspect the figures. Commit task-owned changes, merge into local `main`, and push after validation, preserving unrelated changes.
