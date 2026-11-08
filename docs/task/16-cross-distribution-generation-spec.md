# Task 16 — Diagnose Cross-Distribution Generalization, Overtraining, and Compute Efficiency Using Completed A/B Experiments

**Repository:** https://github.com/tatsu432/self-play-opsd-coding-agent

## 1. Mission

Implement and execute a rigorous, reproducible post-hoc scientific analysis of the completed Task 15 self-play experiments.

We have already completed six expensive A/B training runs. Do not retrain them.

Your primary responsibilities are:

1. Determine why trained coding agents generalize strongly to held-out MBPP+-derived repair tasks but fail to improve substantially on HumanEval+-derived repair tasks.
2. Determine whether the models show evidence of overtraining or validation performance regression, and whether selecting an earlier checkpoint using development data would improve the result.
3. Compare A/B sample efficiency and compute efficiency, not just final accuracy at matched student-episode exposure.
4. Quantify proposer validity, student action validity, repair correctness, and genuinely informative learning exposure.
5. Use the findings to provide a concrete, evidence-based decision on the next research experiment, including whether C/D training is worthwhile.

**This is an implementation and analysis task, not another planning exercise.** Inspect the code and existing artifacts, implement the required analysis, run everything that can be completed using existing data, validate the results, and produce a finished research report with figures.

Reuse previous infrastructure and evidence. Do not restart the project, repeat previous GPU diagnostics, or introduce unnecessary scientific changes.

## 2. Current research context

The project investigates gated on-policy self-distillation (OPSD) and gradient-guided proposer rewards in coding-agent self-play.

The system consists of:

- A shared frozen Qwen3-4B-Instruct-2507 backbone.
- An independently trainable proposer LoRA adapter.
- An independently trainable student LoRA adapter.
- A proposer that generates semantic bugs in correct Python programs using structured AST mutations.
- A student that repairs buggy programs using a multi-turn tool interface.
- A deterministic verifier that assigns binary correctness rewards.
- A privileged self-teacher that supplies token-level guidance to the student.

The student has four available actions: `read_file`, `apply_patch`, `run_tests`, and `finish`.

The student can inspect a specification, one Python source file, and public tests. It may execute up to eight tool turns per episode. Correctness is determined using the final verifier, including hidden tests.

The original four methods are:

| Method | Proposer reward | Student objective |
|---|---|---|
| A | Difficulty | GRPO + ungated signed-gap OPSD |
| B | Difficulty | GRPO + sigmoid-gated signed-gap OPSD |
| C | RL-gradient alignment | GRPO + gated OPSD |
| D | RL+OPSD-gradient alignment | GRPO + gated OPSD |

Methods A/B are complete. C/D are deferred and must not be launched by this task.

The latest completed A/B study is:

- Study ID: `repair-selfplay-2026-task15-ab-v2`
- Scientific execution commit: `0ac77cde4bc5f70ba2b3eb9e457749e3106a2bc5`
- Configuration: `configs/task15-v2.yaml`
- Model: `Qwen/Qwen3-4B-Instruct-2507`
- Training seeds: 2026, 2027, 2028
- Exposure: 4,096 student episodes per arm, except B-2028 with 4,100
- Four student rollouts per task group
- One H200 GPU per training allocation
- Three-seed A/B comparison completed November 4, 2026

All six training endpoints and final held-out evaluations were checksum-verified. They must be treated as immutable historical results.

### Existing final results

| Seed | A MBPP+ | B MBPP+ | A HumanEval+ | B HumanEval+ |
|---|---:|---:|---:|---:|
| 2026 | 61/115 | 70/115 | 2/116 | 3/116 |
| 2027 | 57/115 | 68/115 | 1/116 | 0/116 |
| 2028 | 64/115 | 71/115 | 1/116 | 0/116 |

The common untrained base scored 7/115 and 2/116, respectively.

Main Task 15 A/B training cost 145.724 allocated H200 GPU-hours across all six runs.

### Research cautions

- Strong performance on held-out MBPP+-derived repairs establishes generalization within that repair distribution, not general software-engineering capability.
- Poor HumanEval+ performance might reflect distribution shift, intrinsic difficulty, public-test differences, patch-protocol failures, or narrow training. The cause has not been established.
- A/B uses independently evolving proposers and students. The performance difference does not isolate token gating on identical trajectories.
- Same nominal OPSD lambda does not mean identical effective distillation strength.
- There is no matched Task 15 GRPO-only full-run baseline.
- Training-seed uncertainty and evaluation-task uncertainty are different.
- HumanEval+ has already been observed as a final test set. Any additional retrospective analyses or checkpoint evaluations must be disclosed as exploratory rather than new independent confirmatory evidence.

## 3. Inspect existing implementation and evidence first

Start by inspecting:

- `AGENTS.md`
- `README.md`
- `docs/methods.md`
- `docs/operations.md`
- `docs/investigations/task15-final-results.md`
- `docs/investigations/task15-status.md`
- `docs/investigations/task14-research-decision.md`
- `docs/investigations/learning-stagnation-diagnosis.md`
- `docs/task_completion/15-implemented-stabilize-training-complete-ab.md`
- `scripts/analyze_task15.py`
- `scripts/analyze_task15_health.py`
- `src/selfplay_opsd/analysis.py`
- `src/selfplay_opsd/evaluation.py`
- `src/selfplay_opsd/environment.py`
- `src/selfplay_opsd/protocol.py`
- `src/selfplay_opsd/data.py`
- `src/selfplay_opsd/mutations.py`
- `src/selfplay_opsd/structured_mutations.py`
- `src/selfplay_opsd/trainer.py`
- `src/selfplay_opsd/task15.py`
- `src/selfplay_opsd/storage.py`

Check the current Git HEAD, working tree, and available private analysis data.

Use an appropriately isolated branch or worktree for new code, such as `feat/16-generalization-checkpoint-analysis`.

Do not overwrite any existing scientific results.

The previous analysis already produced:

- `learning-curves.csv`
- `health.csv`
- `final-outcomes.csv`
- `summary.json`
- Per-iteration committed events
- Per-task evaluation records
- Private student trajectories
- Training allocation ledgers
- Verified joint checkpoints

Find and reuse these rather than recreating them.

The standard analysis entry point was:

`uv run python scripts/analyze_task15.py --manifest runs/task15/manifest.json --data data/protocol-v3 --output runs/task15/analysis`

The private analysis directory was recorded under `/private/tmp/selfplay-task15-v2/runs/task15/analysis/` on the original worktree. Verify its actual location rather than assuming the path still exists.

If the raw artifacts are available on Yale but not in the local checkout, use the existing authenticated HPC inspection/fetch interfaces. Transfer only what is necessary. Do not bulk-copy model checkpoints unless an actual additional evaluation requires them.

Do not repeat completed training or duplicate existing Slurm jobs.

## 4. Task A — Audit dataset comparability and evaluation validity

Before analyzing failures, verify that the datasets and evaluation conditions are comparable.

The relevant splits are:

- Training: 186 reserved MBPP+ parent programs, 127 usable in Task 15's structured proposer schedule.
- Reference: 14 MBPP+-derived repair tasks.
- Monitoring development: 41 MBPP+-derived repair tasks from 23 parents.
- Diagnostic development: 20 MBPP+-derived repair tasks.
- Final in-distribution evaluation: 115 MBPP+-derived repairs from 60 parents.
- External evaluation: 116 HumanEval+-derived repairs.

Inspect the actual frozen manifest and confirm these values.

Verify:

1. All evaluation task IDs and parent IDs are accounted for.
2. Training, development, reference, final ID, and external parents are disjoint according to the frozen split manifest and recorded family-grouping policy.
3. The original parent programs were split before mutation generation.
4. The student receives the same tool interface, action/token budgets, and decoding policy on both final splits.
5. Both final splits use the actual frozen task-generation method, including comparable mutation operators where supported.
6. The original correct implementation passes the evaluator, and the mutant fails at least one semantic verification case.
7. No external or final ID evaluation tasks appear in the training curriculum or gradient-reference data.
8. Every reported denominator is complete; infrastructure failures must not silently disappear from evaluation.

Report factual differences between MBPP+-derived and HumanEval+-derived repair tasks, including:

- Original program length (lines, tokens, AST nodes).
- Function size and structural complexity.
- Mutation type and location.
- Number and coverage of public examples.
- Number of hidden verification cases.
- Original task specification length and style.
- Whether the initial bug fails public tests, hidden tests, or both.
- Any measurable differences in generated repair difficulty.

Do not assume HumanEval+ is intrinsically harder merely because its performance is worse. Test measurable hypotheses.

Use task/parent/family-level aggregation as appropriate. Avoid treating multiple mutants from the same parent as fully independent.

Keep original program sources and hidden test details private.

## 5. Task B — Explain the HumanEval+ transfer failure

This is a central deliverable.

Use existing saved A/B final evaluations and base-model evaluations. Analyze all six trained runs, both test splits, and the common base.

Create a structured, reproducible per-task failure analysis.

### B1. Action and final-outcome taxonomy

Inspect the saved evaluation records and episode trajectories.

For each evaluated task, derive:

- Model, seed, split, parent ID, task ID, checkpoint.
- Final verified success/failure.
- Number of tool turns.
- Generated tokens.
- Number of public test calls.
- Valid versus invalid action counts.
- JSON/action parsing failures.
- Patch application failures, including malformed hunks, mismatched context, invalid paths, and syntax errors where identifiable.
- Whether at least one patch was successfully applied.
- Whether the final source differs from the initial buggy source.
- Whether public tests passed.
- Whether hidden tests passed.
- Whether the episode ended with `finish`, a budget limit, or another termination.
- Whether a real sandbox/infrastructure failure occurred.

Build two views:

**Mutually exclusive primary failure attribution:** Assign one clearly defined primary terminal failure category using a documented precedence rule.

**Overlapping diagnostic flags:** Preserve all observed action errors, test failures, and protocol problems, allowing a task to have multiple contributing factors.

Do not conflate patch validity with semantic correctness. Do not assume every unsuccessful `finish` action is premature without evidence.

If the stored trajectory lacks enough information to identify an exact root cause, report `unknown` rather than inventing a category.

If reproducing a tool failure requires executing generated code, use the previously accepted isolated worker only. Do not execute untrusted candidate code directly in an ordinary local Python process.

### B2. Compare splits and methods

Compute the failure distributions for:

- Base model on MBPP+.
- Base model on HumanEval+.
- A on MBPP+, for each seed.
- B on MBPP+, for each seed.
- A on HumanEval+, for each seed.
- B on HumanEval+, for each seed.

Explicitly answer:

1. Is the low HumanEval+ success rate mostly caused by rejected patches?
2. Do HumanEval+ episodes execute fewer valid edits or public tests?
3. Do they apply valid patches but fail hidden semantic verification?
4. Are they running out of turns or context more frequently?
5. Are the human-written specifications and public examples less useful for repair?
6. Are errors associated with particular mutation operators or program-complexity strata?
7. Does the model improve on MBPP+ at the expense of any measurable behavior it previously performed correctly on HumanEval+?
8. Is HumanEval+ performance close to the base-model floor primarily because the baseline already fails almost everything?

Examine paired per-task outcomes, not only aggregate accuracy.

When comparing B with A, identify tasks repaired by both, only A, only B, and neither, using identical task IDs within seed.

When comparing trained models with the base, identify regressions and newly solved tasks.

Do not infer catastrophic forgetting from a near-zero baseline without task-level evidence.

### B3. Controlled explanatory analysis

If feasible from the existing data, fit a small, interpretable descriptive model of repair success or failure category versus:

- Benchmark source.
- Program length/complexity.
- Public example availability.
- Mutation type.
- Agent action validity.
- Episode length and test usage.
- Method and seed.

Avoid treating post-outcome action variables as causal pre-task predictors. Separate pre-task covariates from post-treatment behavioral diagnostics.

Because HumanEval+ has very few successful cases, watch for complete separation and unstable estimates. Prefer descriptive stratification and penalized or appropriately regularized models when needed.

Report uncertainty with parent/family-aware resampling, and keep training-seed variability separate. Do not make strong causal claims from observational comparisons.

The final answer must provide a **ranked list of plausible root causes**, with direct supporting evidence, counterevidence, and remaining uncertainties. Clearly separate confirmed mechanisms from hypotheses.

## 6. Task C — Check validation-selected checkpoints and overtraining

Use existing training histories and saved milestone evaluations.

The current study selected final checkpoints at a prespecified exposure endpoint. There was no validation-based early stopping.

Determine whether earlier development performance was superior to final development performance.

### C1. Reconstruct the learning curves

For every A/B seed, reconstruct:

- Development task-macro repair accuracy.
- Development parent-macro repair accuracy.
- Cumulative student episodes.
- Outer iterations.
- Student and proposer optimizer steps.
- Cumulative generated student tokens.
- Cumulative allocated H200 GPU-hours.
- Training repair success.
- Training mixed-group fraction.
- Proposal validity and diversity, where available.

Use exact recorded evaluation checkpoints; don't fabricate intermediate evaluations.

The intended Task 15 exposure milestones were 0, 256, 1,024, 2,048, and 4,096 episodes, subject to actual completed-group overshoot. Verify the actual observations.

### C2. Define a transparent validation checkpoint selector

For each method and seed:

1. Use **only the frozen monitoring-development split**, never final MBPP+ or HumanEval+ outcomes.
2. Select the checkpoint with maximum development parent-macro accuracy.
3. Break ties by the earliest measured checkpoint.
4. Preserve the primary final checkpoint unchanged.
5. Report the selected checkpoint's actual episode count, training iteration, GPU-hours, and development accuracy.
6. Identify whether the selected checkpoint exists and is checksum-verifiable.
7. Compare the selected and final checkpoints on development accuracy and actual compute exposure.

Label this a retrospective secondary selection policy, not the original prespecified Task 15 endpoint.

Do not claim to find the global optimal checkpoint from only a few evaluated milestones.

Since the monitoring set has 41 tasks from 23 parents, explicitly account for checkpoint-selection noise and the winner's curse. Bootstrap parents/families where appropriate. Report how often a selected checkpoint differs from the final checkpoint and the uncertainty in its apparent advantage.

### C3. Assess whether there is evidence of overtraining

Determine:

- Does development performance increase throughout training?
- Does it plateau?
- Does it peak and then significantly decline?
- Are the declines consistent across seeds or driven by a few tasks?
- Is the apparent decrease explainable by evaluation sampling variability?
- Do training rewards improve while fixed development performance deteriorates?
- Are repeated or narrow tasks associated with poorer development learning?

Distinguish:

- Overtraining on a narrow curriculum.
- Model forgetting.
- Random fluctuations on a small development set.
- Genuine evidence of a validation-performance decline.

Do not diagnose overfitting simply because HumanEval+ transfer is weak.

### C4. Evaluate selected checkpoints only when justified

First use existing saved evaluations and checkpoints.

If the selected checkpoints differ from the final checkpoints and the development evidence suggests that evaluating them would materially clarify the question, implement the ability to evaluate those existing frozen checkpoints under the unchanged evaluation protocol.

Prioritize a bounded pilot involving one paired A/B seed, with both final splits if justified.

Do not perform new training.

Preserve the original final evaluation lineage and publish secondary evaluations under distinct identities.

The user wishes to minimize GPU usage. Aim for CPU-only analysis first. If additional checkpoint evaluations are genuinely necessary, keep them bounded to at most 8 newly allocated H200 GPU-hours for this feature. Do not exceed that ceiling or run broader evaluations automatically.

Before submitting, inspect live Slurm, account limits, existing ledgers, and checkpoint integrity. Never duplicate an uncertain or completed evaluation job. Use the accepted sandbox/isolation and existing evaluation infrastructure.

If authentication or required private data is unavailable, still complete all independent analysis and implementation work, then report the exact missing dependency and executable next action.

Because final test outcomes are already known to the research team, any new selected-checkpoint test comparison is exploratory. Do not present it as a newly untouched holdout estimate.

## 7. Task D — Compare A and B at matched compute

The fixed-episode experiment has already shown B's superior final in-distribution repair success.

Now compare **compute efficiency**.

Use actual scheduler-integrated allocation costs, not just trainer phase timing.

Produce paired per-seed learning curves against:

1. Student episodes.
2. Generated student tokens.
3. Student optimizer steps (descriptive, not a fair budget by itself).
4. Allocated H200 GPU-hours.

Determine whether B reaches comparable development accuracy using fewer GPU-hours or tokens than A.

Compare only supported common budget ranges.

Useful summaries include:

- Accuracy at actually observed common exposure milestones.
- Accuracy at comparable observed GPU-hour budgets.
- Earliest observed cost to reach predeclared development-performance thresholds.
- Development accuracy integrated over a common observed compute interval, if adequately supported.
- Final episode-matched performance.
- Total compute cost of each completed arm.
- Additional cost of the distillation scoring and proposer phases, where recorded.

If curves require interpolation, clearly distinguish interpolation from measured observations. Do not extrapolate beyond the observed range or label interpolated values as actual evaluations.

Distinguish **sample efficiency** from **compute efficiency**.

Report whether the evidence supports:

- B is better at matched student exposure.
- B is better at matched generated-token budgets.
- B is better at matched H200 GPU-hour budgets.
- B yields a better validation-selected endpoint.
- Neither method has a demonstrated advantage under a particular budget definition.

Do not assume these are equivalent.

Retain the fixed-episode primary result even if a secondary compute analysis favors a different conclusion.

## 8. Task E — Quantify training-signal and proposer behavior

Use the saved Task 15 per-iteration events to produce exact aggregate statistics for all six runs.

Report separately:

### Proposer

- Attempted proposals.
- Valid proposals.
- Unique task identities where observable.
- Previously seen versus new mutations.
- Mutation operator distribution.
- Parent/site coverage.
- Proposer optimizer steps.
- Fraction of attempted proposer groups with nonzero advantages.
- Fraction of groups whose valid candidates receive distinguishable difficulty rewards.

### Student

- Total attempted episodes.
- Verified successful episodes.
- Valid versus invalid tool actions, by action type.
- Successful versus rejected patches.
- All-failed, mixed, and all-successful four-rollout groups.
- Student optimizer steps.
- Episodes per student optimizer update.
- Generated tokens per successful repair.
- Mixed-group yield per allocated H200 GPU-hour.
- Public-pass/hidden-fail frequency.
- OPSD gradient diagnostics, where available.

Do not substitute mixed-group fraction for overall repair-success fraction.

Distinguish zero GRPO advantage from zero total student update. OPSD may still contribute a nonzero gradient.

Compare early versus late training intervals, preferably by actual episode exposure rather than arbitrary iteration indices.

The goal is to determine whether the proposer learned to provide increasingly informative repair tasks or mainly learned to produce syntactically valid/repeated mutations.

Do not claim proposer learning has caused student improvement without a controlled comparison.

A frozen-versus-trained proposer snapshot diagnostic may be recommended as a later bounded experiment if the existing data does not identify the effect. Do not launch it automatically.

## 9. Task F — Decide the next research experiment

After the analysis, make a concrete recommendation among these directions:

- Stop training and write up completed A/B with both ID and OOD results.
- Perform a narrow repair-interface investigation.
- Improve task diversity or the environment representation in a separate future study.
- Perform a matched GRPO-only or matched-effective-distillation-strength control.
- Conduct a bounded C/D proposer-reward identifiability pilot.
- Proceed with full C/D training only after a successful and interpretable pilot.

Do not recommend full C/D merely because it was in the original proposal.

The previous Task 14 alignment diagnostics used very few groups with multiple valid candidates. Its RL-versus-combined proposer optimizer-update cosine was approximately 0.992, close to an identical-reward repeat.

The current structured proposer has much higher validity, but it has not been established that this translates into sufficiently informative C/D gradient-alignment reward variation.

If recommending C/D, specify an efficient preliminary design using matched frozen student/proposer snapshots, identical proposed tasks, repeated numerical counterfactuals, and disjoint reference-gradient estimates.

Explain how the pilot would distinguish a real C/D update-direction difference from gradient noise, reward normalization, and optimizer-state effects.

Do not start full C/D runs in this task.

## 10. Research and implementation standards

Maintain the existing research engineering practices.

- Preserve the exact Task 15 scientific configuration and its checkpoints.
- Do not modify historical metrics or committed W&B runs.
- Do not silently alter dataset splits, oracle definitions, reward functions, or evaluation protocols.
- Do not use held-out test outcomes for model selection or hyperparameter tuning.
- Use deterministic, tested analysis code.
- Reuse existing utilities such as `active_events`, `TaskRepository`, checkpoint validators, and current paired-analysis functions.
- Preserve event lineage and rollback semantics.
- Treat missing records as missing, not as zero or failure unless the frozen evaluation protocol explicitly defines them that way.
- Avoid double-counting repeated or superseded events.
- Distinguish training, validation, and test task identifiers and parent families.
- Separate parent/family uncertainty and training-seed uncertainty.
- Avoid excessive multiple testing or unqualified significance claims.
- Avoid adding a new training framework, general orchestration system, or unnecessary dependencies.
- Keep generated code and test oracles in private, ignored storage.
- Publish only privacy-safe aggregate results and sanitized examples.
- Never expose W&B credentials or private environment information.

Write focused tests for missing or duplicate evaluation records, conflicting event lineage, incorrect task matching, failure-taxonomy precedence, checkpoint selection, ties, missing checkpoints, compute accounting, and clustered statistics.

Run relevant pytest, Ruff, formatting, and mypy checks. Reuse hosted CI where applicable. Tests with synthetic fixtures establish analysis correctness, not scientific results.

Do not make actual GPU evaluation dependent on repeating unrelated training readiness checks. Only validate the dependencies necessary for the new read-only evaluation job.

## 11. Deliverables

Implement a maintainable Task 16 analysis command using the existing repository conventions, for example:

`scripts/analyze_task16.py`

You may split the logic into a small reusable module if it materially improves testing and maintainability.

Produce the following outputs where supported by existing evidence:

### Private analysis artifacts

Under an ignored run directory such as `runs/task16/analysis/`:

- `dataset-audit.json`
- `evaluation-failure-taxonomy.csv`
- `transfer-analysis.json`
- `training-health.csv`
- `checkpoint-selection.csv`
- `compute-efficiency.csv`
- `summary.json`
- A manifest of exact input artifact hashes and run identities

Preserve original private task-level evidence outside version control.

### Figures

Create publication-quality figures suitable for a class final report:

1. A/B development learning curves against episodes.
2. A/B development learning curves against allocated H200 GPU-hours.
3. Best-development versus final checkpoint comparison.
4. MBPP+ versus HumanEval+ failure-category comparison.
5. Per-seed A/B improvements and uncertainty on the final held-out MBPP+ set.
6. A figure comparing proposer validity, repeated tasks, student mixed-group yield, and verified training success, if the data supports it.

Each figure must use honest axis labels, units, and denominators. Explain when a point is an observed evaluation versus an estimate.

### Public research report

Create:

`docs/investigations/task16-generalization-checkpoint-decision.md`

The report must answer:

1. What is the best-supported explanation of HumanEval+ transfer failure?
2. Which failure categories dominate each evaluation split?
3. Does training improve the model's ability to produce valid actions and successful repairs?
4. Are there signs of validation overtraining?
5. Which checkpoints would a development-based selection rule choose?
6. Do earlier development checkpoints outperform the final ones, and how certain is that?
7. Does B outperform A at equal allocated compute, rather than only at equal episode exposure?
8. Is there evidence that the proposer itself learns a useful curriculum?
9. What evidence is missing, and why does it matter?
10. Which next experiment should be run, if any?

Use exact numerical results, confidence intervals, effect sizes, and caveats. Avoid speculative conclusions.

Create a Task 16 completion record with exact execution status, input hashes, validation evidence, remaining limitations, and reproduction commands.

## 12. Execution priorities and completion criteria

Proceed in this order:

**Phase 1 — Reconcile and inspect.** Confirm all six completed A/B runs and their immutable artifact identities. Identify which private data is already available.

**Phase 2 — CPU-only analysis.** Implement and execute dataset comparison, training-health aggregation, evaluation failure attribution, checkpoint selection, and compute-efficiency analysis.

**Phase 3 — Evaluate evidence gaps.** Determine whether the existing data already answers the research questions. Do not schedule GPU work just to reproduce a result already present.

**Phase 4 — Optional bounded checkpoint evaluation.** Only if a development-selected earlier checkpoint offers a meaningful unanswered question, perform the smallest useful read-only evaluation using the verified frozen model and task protocol, respecting the GPU budget above.

**Phase 5 — Final report and next-experiment decision.** Produce completed tables, figures, interpretation, and an explicit recommendation.

Do not stop after generating a plan. Execute the actual analysis using existing real artifacts.

If a specific private dependency or Yale authentication action blocks one subtask, complete all other work independently. Report exactly which artifact or permission is missing, what was completed, and the shortest supported action needed to finish.

Do not claim a result that has not been measured.

Do not run any new A/B training, C/D training, model-size sweep, or data-generation study under this task.

Commit the new analysis implementation, focused tests, and privacy-safe documentation. Preserve unrelated work and keep the scientific training state unchanged.

In your final response, present the main quantitative findings, the strongest supported explanations, checkpoint-selection result, compute-efficiency comparison, recommended next experiment, validation status, and exact paths to the deliverables.

**Success means that we can explain what we learned from the completed A/B study, why external transfer is weak, whether early stopping would have helped, and what research experiment is actually worth doing next—without repeating expensive completed training.**