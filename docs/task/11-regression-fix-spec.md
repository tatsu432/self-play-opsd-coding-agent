# Task: Diagnose Experiment A Regression and Prepare Reliable Experiments B–D

## 1. Context and objective

Repository: https://github.com/tatsu432/self-play-opsd-coding-agent

We are conducting a research project for Yale's CPSC 5880: Advances in Frontier AI Models, investigating self-play reinforcement learning and on-policy self-distillation (OPSD) for coding agents.

Experiment A has completed all 200 iterations but experienced a severe performance regression after initially improving. Before running the remaining configurations B–D, I want you to conduct a rigorous, evidence-based investigation, determine the most likely causes, implement necessary fixes or diagnostics, and prepare a reliable next-experiment plan.

**Your primary objective is to explain why Experiment A initially improved and subsequently collapsed, distinguish genuine learning instability from implementation defects, and determine the minimum scientifically justified changes needed before B–D.**

Do substantive engineering work, not merely write a plan. Use the actual source code, complete training records, saved trajectories, optimizer diagnostics, and checkpoints. Prioritize scientific correctness, efficient experimentation, and actionable results.

Do not get stuck introducing unnecessary readiness requirements, additional infrastructure abstractions, or speculative safety gates. Existing HPC execution, isolation, checkpointing, and training infrastructure have already undergone extensive verification.

### Research design

Our frozen study compares four configurations:

| Method | Proposer reward | Student objective |
|---|---|---|
| A | Difficulty-based | GRPO + ungated OPSD |
| B | Difficulty-based | GRPO + gated OPSD |
| C | RL-gradient alignment | GRPO + gated OPSD |
| D | Combined RL+gated-OPSD gradient alignment | GRPO + gated OPSD |

Research questions:

1. Does gating OPSD improve the stability and performance of self-play coding-agent training compared with ungated OPSD?
2. Does proposer gradient alignment improve learning compared with difficulty-based curriculum rewards?
3. When OPSD contributes to the student's objective, should the proposer optimize alignment with the RL gradient alone or the combined RL+OPSD gradient?

These contrasts are central to the project. Do not silently alter them or invalidate the A–D comparison.

The relevant research includes Absolute Zero Reasoner, CompassPlay, Self-Distilled Reasoner, SDAR, RLSD, SIPO, and GradAlign. Consult the project's existing specifications and literature documentation rather than redesigning the methods from scratch.

## 2. Observed Experiment A results

Experiment A completed all 200 iterations on one Yale H200 GPU.

The authoritative execution record is:

`docs/task_completion/10-implemented-main-experiment.md`

Additional documentation:

- `docs/task10-launch.md`
- `docs/methods.md`
- `docs/task/10-main-experiment-spec.md`
- `configs/task10-online.yaml`

W&B run:

https://forge.coreweave.com/wandb/tatsu432s/self-play-opsd-coding-agent/runs/9c125e660c942541076d5d50?nw=nwusertatsu432

W&B telemetry stopped being delivered at iteration 139, but training continued to iteration 200. The local scientific event store and final checkpoint are authoritative.

### Development evaluation results

The fixed greedy development evaluation contains 41 repair tasks.

| Iteration | Successful repairs |
|---|---|
| 0 | 0/41 |
| 20 | 12/41 |
| 40 | 12/41 |
| 60 | 10/41 |
| 80 | 9/41 |
| 100 | 0/41 |
| 120 | 0/41 |
| 140 | 0/41 |
| 160 | 0/41 |
| 180 | 0/41 |
| 200 | 0/41 |

Thus, Experiment A peaked at 29.3% success and subsequently returned to zero, with no recovery.

This is a serious regression from the intermediate peak. Because the initial model also scored zero, distinguish collapse from the best checkpoint from degradation below the initial baseline.

These numbers are development results, not final held-out test scores.

### Training statistics

The repository documents:

- 3,200 attempted proposals.
- 173 valid proposals (5.40625%).
- 692 solver episodes.
- 162 successful solver episodes (23.4104%).
- 130 all-failure groups.
- 38 all-success groups.
- Only five mixed-reward groups.
- 111 student optimizer steps.
- 111 proposer optimizer steps.

Major proposal rejection categories:

- 1,103 duplicate mutations.
- 730 hunk-count mismatches.
- 627 invalid hunk lines or extra files.
- 317 patch-context mismatches.

The proposer validity rate declined from approximately 9.38% over the first 20 iterations to approximately 4.97% over the remaining 180.

The training configuration uses four sampled solver episodes per valid task. GRPO computes group-relative advantages, which are zero for all-failure and all-success groups.

Consequently, 168 of 173 task groups (97.1%) offered zero group-relative RL advantage.

Given only five mixed-reward groups, at most five outer iterations could include nonzero group-relative RL gradients. Therefore, at least 106 of the 111 student optimizer steps were driven solely by OPSD, assuming the documented objective and update implementation.

**This is the most important observation to investigate.**

It suggests that the nominal GRPO+OPSD method effectively became predominantly self-distillation training because the environment rarely supplied informative RL advantages.

This is an arithmetic consequence of the available group statistics, but whether it *caused* the development regression is not yet established.

## 3. Understand the actual implementation before diagnosing it

Inspect at least:

- `src/selfplay_opsd/trainer.py`
- `src/selfplay_opsd/objectives.py`
- `src/selfplay_opsd/policy.py`
- `src/selfplay_opsd/verl_backend.py`
- `src/selfplay_opsd/environment.py`
- `src/selfplay_opsd/mutations.py`
- `src/selfplay_opsd/protocol.py`
- `src/selfplay_opsd/evaluation.py`
- `src/selfplay_opsd/data.py`
- `src/selfplay_opsd/scientific_health.py`
- `src/selfplay_opsd/telemetry.py`
- `src/selfplay_opsd/observability.py`

Also inspect the existing tests and historical numerical-audit reports.

In particular, verify the implementation details of:

### Student objective

The current code uses a combined loss approximately of the form:

`L_student = L_GRPO + 0.1 * L_OPSD`

The OPSD signal is derived from teacher-versus-behavior-policy token log probabilities.

For each token:

`delta = teacher_logp - old_logp`

Then:

- A uses an ungated coefficient derived from clipped delta.
- B/C/D apply a sigmoid gate to delta.
- The teacher and old log probabilities are detached.
- The student learns on its own sampled trajectories.

Check the precise actual loss, signs, normalization, masks, clipping, optimization and accumulation semantics. Do not rely on the approximate equation above if the implementation differs.

Teacher privileged information includes the verified correct source and inverse initial patch.

The teacher scores the student's actual action tokens under the actual interaction history.

### Proposer objective

A/B use the difficulty reward:

`r = 4 * p * (1 - p)`

where `p` is the sampled solver success fraction for a generated valid task.

Invalid proposals receive a fixed negative reward.

Proposer advantages are normalized within the four proposals for each parent.

The proposer generates unified-diff patches that mutate a verified correct Python implementation into a repair problem. Validation applies the patch, checks mutation constraints, constructs the task, and verifies admission.

### Important implementation details

The proposer and student use distinct LoRA adapters initialized identically, with independent optimizers.

The system uses vLLM for rollouts and a separate trainer scoring path.

Training uses verifiable final repair rewards and frozen development tasks.

Check these facts against the actual current code. Report discrepancies if the implementation has changed.

Do not assume that an optimizer step implies useful RL learning, or that a finite loss/gradient implies a meaningful improvement in coding ability.

## 4. Investigation A: Explain the development performance collapse

This is the highest-priority investigation.

Use the actual stored development-evaluation trajectories.

The most informative checkpoints are iterations 20, 40, 60, 80, 100, and 200.

For every task solved at iteration 20, track whether it remains solved at later evaluations.

Build a task-level transition analysis, including:

- Success at each evaluation checkpoint.
- First checkpoint at which an initially successful task fails.
- Whether any task recovers.
- Parent ID and task ID.
- Number and type of tool calls.
- Generated token counts.
- Number of patch attempts.
- Whether the agent invokes `finish`.
- Final public-test and hidden-test outcomes.
- Termination reason.
- Tool protocol and parser errors.
- Patch-application failures.
- Final-source differences when relevant.

Identify what happened behaviorally.

Possible failure categories include:

1. Invalid JSON or malformed tool calls.
2. Valid tool calls but repeated unsuccessful patch application.
3. Repeated reading or test execution without useful edits.
4. Incorrect code modifications.
5. Successful public tests but failing hidden tests.
6. Premature `finish`.
7. Token/turn/context exhaustion.
8. Degenerate repetition or unusually long responses.
9. Teacher-privilege leakage into unprivileged student behavior.
10. Other empirically observed failure modes.

Define a defensible classification rubric with an unknown/other category. Distinguish primary failure reasons from downstream symptoms.

Provide representative, compact examples comparing the same task at its successful and failed checkpoints. Keep private task content in trusted local artifacts; the main report should use task IDs, bounded descriptions, and redacted excerpts where necessary.

Important: do not confuse training-task success with development-task success.

Also inspect whether the performance collapse is gradual, abrupt, or associated with a change in action distribution.

Determine whether the primary failure is behavioral/protocol degradation, semantic repair degradation, or both.

## 5. Investigation B: Quantify RL versus OPSD dominance

The key hypothesis is:

**Because almost all solver groups have uniform outcome rewards, GRPO provides little or no gradient while ungated OPSD continues to update the student, potentially driving the policy away from successful repair behavior.**

Test this hypothesis quantitatively.

### Required longitudinal statistics

Analyze windows such as:

- 1–20.
- 21–40.
- 41–80.
- 81–100.
- 101–140.
- 141–200.

For each window, compute:

- Attempted and valid proposals.
- Number of valid solver task groups.
- All-zero/mixed/all-success group counts.
- Student optimizer steps.
- Steps with nonzero RL gradient.
- Steps with OPSD-only gradient.
- Available RL, lambda-scaled OPSD, and combined gradient norms.
- Gradient cosine where measured.
- OPSD teacher-student log-probability gap statistics.
- Positive versus negative gap fractions.
- Gap clipping frequency.
- Gradient clipping statistics.
- Training repair reward.
- Development repair performance.
- Action-format and termination statistics.

Use existing recorded metrics wherever sufficient.

If exact RL-versus-OPSD decomposition is not already recorded, explicitly report coverage and missingness. Perform targeted checkpoint-matched replay only where supported and useful.

Do not silently substitute a later checkpoint for the pre-update snapshot that produced earlier trajectories. Check model/adapter versions and optimizer state.

### OPSD reliability analysis

Investigate whether the privileged teacher provides useful supervision on:

- Successful episodes.
- Failed episodes.
- Valid repair actions.
- Invalid actions.
- Early versus late interaction turns.

Assess whether OPSD gradients may be reinforcing unsuccessful trajectories or penalizing desirable ones.

Do not equate a positive teacher-student gap with verified correctness. Teacher preference is not automatically a ground-truth reward.

Where feasible, compare gradient directions with an environment-reward-grounded or supervised diagnostic direction, clearly stating the limitations.

Inspect whether token-level signals disproportionately emphasize JSON syntax, repeated tool calls, EOS, or other stylistic behavior instead of meaningful repair edits.

Remember that the distillation objective optimizes token likelihoods on on-policy trajectories. It does not directly optimize the final binary repair reward.

### Additional hypothesis: policy drift without a fixed-base KL

Inspect whether the lack of an explicit fixed-reference KL penalty plausibly allows excessive policy drift.

Measure policy displacement or divergence using existing supported diagnostics, if feasible.

Do not automatically add a KL penalty to the original experiment. Such a change would constitute a distinct scientific treatment.

## 6. Investigation C: Proposer collapse and curriculum quality

The proposer generates very few valid tasks. Analyze why.

### Validity analysis

Break down invalid proposals by iteration, parent, rejection type, output length, mutation family, and duplicate history.

Determine whether invalidity is due primarily to:

- Inability to generate strict unified diffs.
- Incorrect hunk counts.
- Invalid patch headers or line markers.
- Context mismatch.
- Truncated proposal generation.
- JSON formatting.
- Semantically invalid mutations.
- Excessively restrictive validation.
- Repetition of previously accepted mutations.
- Genuine exhaustion of simple mutations on frequently sampled parents.

Examine real failed outputs rather than inferring causes from error labels alone.

Check the cumulative deduplication mechanism carefully.

A high duplicate rate may reflect proposer policy collapse, accumulated history on a limited pool of parent programs, or both.

Analyze the evolution of unique valid mutations and diversity. Distinguish novelty exhaustion from deteriorating model behavior.

### Curriculum quality

For valid tasks, examine:

- Solver success fraction.
- Task difficulty distribution.
- Public versus hidden outcomes.
- Mutation complexity and bug families.
- Parent coverage.
- Diversity across iterations.
- Whether valid tasks differ meaningfully from earlier tasks.
- Proposer difficulty reward.
- Proposer advantage distribution.

Determine whether self-play is producing useful intermediate-difficulty tasks or mostly trivial/impossible tasks.

Because the difficulty reward is `4*p*(1-p)`, only mixed-outcome groups receive positive difficulty reward. Quantify how frequently this actually guides the proposer.

Assess whether invalid-versus-valid discrimination dominates proposer learning.

### Potential fixes

Consider implementation options only after diagnosing the actual failure.

For example, model-generated modified source followed by deterministic patch construction might improve format validity, but it changes the generation protocol and potentially the task distribution.

Do not introduce such a change into B–D while preserving A as the baseline.

For every proposed intervention, state whether it is:

- A correctness bug fix.
- An operational or observability improvement.
- A scientific intervention changing the experiment.

Prefer minimal changes first.

## 7. Investigation D: Numerical consistency and optimizer correctness

The project previously identified nontrivial differences between vLLM behavior-policy log probabilities and trainer-rescored log probabilities.

The historical numerical audit documented substantial proposer-gradient sensitivity in saved counterfactuals, even though no indexing or serialization defect was proven.

This issue should be investigated proportionately because it may influence both A/B training and the gradient-alignment methods C/D.

Review historical reports and available diagnostic outputs before generating new measurements.

### Targeted audit

Using actual A data where matching saved snapshots exist, inspect:

- Exact token IDs.
- Token-position alignment.
- Causal log-probability indexing.
- Prompt padding and position IDs.
- EOS handling.
- LoRA version identity.
- Behavior-policy versus trainer log probabilities.
- Teacher versus student prefix construction.
- Importance ratios and clipping effects.
- GRPO gradient differences.
- OPSD coefficient and gradient sensitivity.
- Proposer gradient differences.
- Hypothetical optimizer-step effects where warranted.

Compare saved behavior probabilities with trainer-rescored probabilities, but **do not replace the original behavior denominator merely to make pre-update ratios equal one**.

That could change the mathematical estimator rather than correct it.

For any hypothetical update, restore the exact original model, optimizer, and RNG state afterward. Never promote a diagnostic model state into the scientific run.

Separate:

1. Proven implementation bugs.
2. Numerical precision/backend differences.
3. Measured but unresolved estimator sensitivity.
4. Hypothetical consequences not yet established empirically.

Avoid excessive numerical investigation once enough evidence exists to make a decision.

## 8. Investigation E: W&B recovery and observability

The existing W&B run only shows cloud-delivered metrics through approximately iteration 139, although A completed all 200 iterations.

The authoritative local events are intact.

Inspect:

- `docs/task10-launch.md`
- `src/selfplay_opsd/observability.py`
- `src/selfplay_opsd/telemetry.py`
- `src/selfplay_opsd/wandb_tools.py`

Determine why upload delivery failed or stopped, to the extent supported by available logs.

Recover the complete canonical metrics through iteration 200 using the established `wandb-reconcile` mechanism.

Requirements:

- Verify the completed iteration-200 checkpoint metadata and source identity.
- Ensure the original W&B writer has stopped.
- Preserve the logical run identity.
- Use the single-writer recovery mechanism.
- Reconcile checkpoint-backed aggregate events.
- Avoid duplicate, stale, or misleading canonical rows.
- Do not overwrite original scientific events.
- Do not upload raw private trajectories, hidden tests, oracle solutions, unrestricted source content, model checkpoints, or credentials.

Prefer reconciliation directly from the trusted Yale filesystem to avoid transferring multi-gigabyte checkpoints unnecessarily.

If cloud connectivity is unavailable, produce a safe local aggregate report and documented reconciliation command.

Do not allow a W&B problem to block scientific analysis or require rerunning training.

At completion, distinguish cloud-verified metric recovery from merely successful local enqueueing.

## 9. Additional diagnostic comparisons

After investigating the original run, propose the smallest useful experiments to disambiguate causes.

### Primary next comparison: staged B

B changes the OPSD gate while retaining the difficulty-based proposer reward.

Run B first, rather than launching B/C/D simultaneously.

Recommend an initial bounded stage through approximately iteration 20, with an optional extension to 40 after reviewing development performance and training health.

Keep all original B scientific settings unchanged.

Compare B with A at matched iterations and, where possible, matched GPU-hours.

Look particularly at:

- Development performance.
- Proposal validity.
- Mixed-reward group frequency.
- OPSD gradient behavior.
- Student action validity.
- Proposer diversity.
- Training stability.

If B improves, do not immediately conclude that gating is sufficient; check whether the improvement persists past A's regression point.

### Strongly recommended diagnostic control: GRPO-only

The repository already supports a `GRPO` method without OPSD.

Consider a short GRPO-only control with the same model, seed, data, proposer difficulty reward, sampling budget, and evaluation protocol.

This control can distinguish harmful OPSD effects from problems shared by all training methods.

Keep this control separate from the four main research configurations.

Interpretation:

- A collapses, B improves, GRPO-only stable: supports the gating hypothesis.
- A/B collapse, GRPO-only stable: suggests OPSD-related problems remain.
- All collapse: prioritize shared proposer/curriculum or training defects.
- GRPO-only remains near baseline, B improves: suggests beneficial dense supervision.

These are diagnostic interpretations, not definitive conclusions from single-seed results.

Do not add the control blindly if the existing evidence already establishes a more urgent correctness defect.

### Important constraints

Do not automatically launch full B–D runs.

Do not run expensive diagnostics or consume additional GPU allocations without an applicable existing authorization or explicit new authorization.

Do complete all useful read-only analysis, CPU checks, diagnostic code, recovery preparation, and experiment planning autonomously.

If limited GPU work is needed, provide a concrete bounded command and compute budget. This should be a concise decision request, not an excuse to stop independent work.

## 10. Scientific comparability and implementation rules

Preserve the original completed A result and its evidence.

Do not modify:

- Original A checkpoints.
- Original A event records.
- Historical manifests.
- Frozen dataset splits.
- Hidden final test sets.
- The original scientific configuration.
- Existing study identities or scientific hashes.

Work in a new feature branch, such as:

`feat/11-experiment-a-regression-diagnosis`

If you identify a genuine correctness bug, implement a tested correction in the new branch and document how it affects each method.

Determine whether affected comparisons need to be rerun under a new immutable study.

A change to the proposer generation format, reward function, student loss, optimizer, sampling distribution, or task-generation policy is a scientific intervention, not merely an infrastructure fix.

Do not silently introduce such changes.

Diagnostic instrumentation, read-only reports, and W&B recovery should be implemented with minimal disruption.

Reuse existing abstractions and utilities rather than duplicating them.

Follow the repository's established coding style, typing, deterministic data ordering, and test conventions.

Do not weaken necessary generated-code isolation, hidden-test boundaries, or checkpoint integrity. But do not invent new HPC process-count requirements, sandbox policies, or unrelated mandatory readiness cycles.

This task is primarily a scientific diagnosis, not another infrastructure-readiness project.

## 11. Implementation deliverables

### Deliverable 1: Comprehensive regression report

Create:

`docs/investigations/experiment-a-regression.md`

Include:

1. Executive summary.
2. Verified experiment identity and source artifacts.
3. Reproduced Experiment A results.
4. Longitudinal training analysis.
5. Evaluation trajectory comparisons.
6. RL-versus-OPSD dominance.
7. Proposer curriculum analysis.
8. Numerical audit findings.
9. Root-cause ranking.
10. Correctness issues versus design limitations.
11. Recommended changes.
12. Implications for B–D.
13. Remaining uncertainties and the smallest ways to resolve them.

Use actual quantitative evidence and code references.

Mark every significant conclusion as confirmed, strongly supported, plausible, or unverified.

A numerical observation must not automatically become a causal claim.

### Deliverable 2: Reproducible analysis artifacts

Implement a reusable, preferably CPU-only report generator using the existing scientific event store and raw shards.

It should support:

- Per-iteration/window aggregation.
- Proposal validity and rejection analysis.
- Mixed-reward and zero-RL-signal analysis.
- Evaluation checkpoint comparisons.
- Failure-category distributions.
- OPSD diagnostics where recorded.
- Missing-data provenance.
- Machine-readable JSON/CSV output.
- Human-readable report tables and optional charts.

Do not unnecessarily create a new analytics framework. Extend the existing scientific-health tooling where practical.

Make the output deterministic.

### Deliverable 3: Necessary tests

Add focused tests for any new analysis or corrected implementation.

Prioritize:

- Correct counting of mixed versus uniform groups.
- RL-gradient-zero classification.
- Episode/task/iteration aggregation.
- Evaluation task matching across checkpoints.
- Missing checkpoint and missing shard handling.
- Rejection-category aggregation.
- Correct handling of rollback or superseded events.
- W&B canonical reconciliation if changed.
- Any concrete bug discovered.

Run the relevant existing CPU test suite, linting and type checking.

Do not require an unrelated full HPC acceptance campaign merely to publish a diagnostic report.

### Deliverable 4: Actual fixes where justified

Implement clearly proven low-risk bugs and useful diagnostic/telemetry improvements.

For larger scientific interventions, provide an implementation proposal and a new-study plan rather than silently altering the original methods.

If the primary cause appears to be a legitimate weakness of ungated OPSD rather than a correctness bug, preserve that negative result.

### Deliverable 5: Next-experiment runbook

Create:

`docs/investigations/next-experiment-decision.md`

Include:

- Whether the original A result is scientifically usable.
- Whether B should use the existing frozen protocol.
- Whether a GRPO-only control is necessary.
- Whether C/D gradient alignment is sufficiently numerically trustworthy.
- The proposed staged launch order.
- Exact commands using the current launcher.
- Expected GPU resources and conservative budget.
- Metrics and checkpoints to inspect.
- What evidence would justify continuing, debugging, or stopping.
- A response if B also regresses.
- Which changes require a new study and rerun of A.

Use already-supported launch mechanisms. Do not create new scheduling infrastructure.

## 12. Acceptance criteria

The task is complete when:

- [ ] A's 200 committed iterations and final checkpoint are verified from authoritative records.
- [ ] The recorded development trajectory is reproduced.
- [ ] The model's actual behavioral failure modes after its initial improvement are identified.
- [ ] The exact prevalence of OPSD-only updates and lack of GRPO signal is quantified.
- [ ] The proposer validity and duplicate problem is explained using real outputs.
- [ ] Relevant numerical issues are evaluated without changing the original estimator.
- [ ] W&B is reconciled if safely possible, or precise recovery instructions and preserved aggregate data are available.
- [ ] Any confirmed low-risk implementation bugs are fixed with tests.
- [ ] Proposed scientific interventions are explicitly separated from implementation fixes.
- [ ] A clear GO / CONDITIONAL GO / NO-GO recommendation for staged B is provided.
- [ ] C/D-specific risks and required next checks are identified.
- [ ] Existing scientific outputs, frozen configurations, and final test boundaries remain untouched.

## 13. Execution strategy

Please proceed autonomously.

Start by inspecting the repository, authoritative A artifacts, existing reports, and available Yale compute/storage access.

Avoid repeated broad reconnaissance if existing documentation already contains verified results.

Prioritize the investigations in this order:

1. Reconstruct the regression and actual evaluation failures.
2. Quantify GRPO/OPSD imbalance.
3. Diagnose proposer invalidity and curriculum degeneration.
4. Audit material numerical uncertainty.
5. Recover W&B aggregates.
6. Implement targeted diagnostics and justified fixes.
7. Prepare staged B and the GRPO-only control decision.

Work on independent analysis even if access to one dependency is unavailable.

If an artifact or historical checkpoint does not exist, explain the limitation and analyze what is available. Do not require reconstruction of every historical model state to reach a useful conclusion.

Do not run new main experiments without authorization, and do not modify scientific settings without establishing that the comparison needs to change.

Avoid unnecessary refactoring, speculative safety mechanisms, lengthy readiness checklists, excessive GPU diagnostics, and unrelated optimization experiments.

Keep the scope tight and finish with substantive deliverables rather than simply proposing future work.

## 14. Final response expected from Codex

When finished, provide a concise but evidence-rich summary containing:

**A. What happened**

Explain the development regression and actual observed failure behavior.

**B. Why it happened**

Rank root causes, with confidence, supporting evidence, and key uncertainties.

**C. What you changed**

Summarize code changes, diagnostics, tests, W&B recovery, and generated artifacts. Include commit references.

**D. What remains unresolved**

State only material unresolved issues, without arbitrary blockers.

**E. What we should do next**

Give one primary recommendation:

1. Continue to staged B under the original protocol.
2. Run the bounded GRPO-only control first.
3. Fix a verified scientific defect and begin a corrected study.

Explain why that option is best.

**F. Exact next commands**

Provide the concrete next action or launch commands, with resource estimates and applicable authorization requirements.

The overarching goal is to maximize research insight per GPU-hour and produce scientifically credible A–D comparisons. We do not need perfect infrastructure or zero uncertainty; we need to understand the observed regression, correct genuine defects, and proceed efficiently without compromising the experimental design.
