# Task 20 — Rescue the Coding-Agent Self-Play Experiment and Deliver Final Results by December 2, 2026

## 0. Mission and operating instructions

You are Codex, acting as the lead research engineer responsible for **diagnosing, repairing, executing, and analyzing** an existing reinforcement-learning research project.

**Repository:** https://github.com/tatsu432/self-play-opsd-coding-agent

**Current date:** November 19, 2026  
**Hard final deadline:** December 2, 2026  
**Primary objective:** Obtain scientifically credible, reproducible evidence on whether gated on-policy self-distillation (OPSD) improves learning in coding-agent self-play, compared with ungated OPSD and verifier-driven Dr. GRPO.

This is an execution task, not merely a planning exercise.

You should inspect the repository, independently verify the hypotheses below, make necessary code changes, run diagnostics, prepare and execute appropriately bounded experiments using authorized resources, analyze the results, and produce final research-ready figures and documentation.

Do not blindly follow the previous implementation's assumptions or redesign the entire infrastructure. Find the smallest set of changes that produces reliable learning and interpretable comparisons.

### Guiding principles

1. **Scientific validity over the number of experiments completed.** Two reliably executed comparisons are better than five uninformative runs.
2. **Prove that learning works before scaling.** Do not repeat the mistake of running A–D before demonstrating that the underlying RL training is effective.
3. **Diagnose root causes, not symptoms.** A finite gradient and a completed optimizer step are necessary but do not establish useful learning.
4. **Prioritize efficient computation.** GPU time, student rollouts, expensive teacher scoring and evaluation must be allocated according to expected scientific value.
5. **Use a staged go/no-go design.** Continue only when the preceding phase provides sufficient evidence, while avoiding overly rigid criteria based on noisy, tiny samples.
6. **Preserve scientific comparability.** Control model initialization, data exposure, hyperparameters, inference interface, evaluation and computational cost.
7. **Avoid unnecessary engineering blockers.** Do not demand new security capabilities, infrastructure features, or numerical guarantees beyond what is necessary and supported by the existing validated setup.
8. **Deliver by December 2.** Protect sufficient time and compute for analysis, figures and final documentation. Do not allow endless debugging to consume the deadline.
9. **Take action.** Do not stop after writing a plan. Implement, test, execute and analyze what is feasible. If an external permission, GPU allocation or credential is genuinely unavailable, document the precise blocker and finish every independent task.
10. **Never fabricate results, convergence, statistical significance or benchmark comparability.** Preserve negative findings honestly.

Historical experiments, scientific configurations, checkpoints, datasets and artifacts must remain unchanged. Create a new Task 20 study namespace and branch.

---

## 1. Existing research and experimental context

### 1.1 Research question

The project studies a coding agent that learns through self-play.

A proposer generates executable Python bugs or repair tasks. A student attempts to fix them with tools, reasoning and code edits. Verifiable tests provide rewards for successful repairs.

The proposer and student share trainable model parameters. The student may additionally learn through a privileged self-teacher, which receives information unavailable to the student at inference time and provides token-level guidance.

Our original experimental configurations were:

| Method | Proposer objective | Student objective |
|---|---|---|
| Control | Difficulty reward | Dr. GRPO |
| A | Difficulty reward | Dr. GRPO + ungated OPSD |
| B | Difficulty reward | Dr. GRPO + gated OPSD |
| C | RL-gradient alignment | Dr. GRPO + gated OPSD |
| D | RL+OPSD-gradient alignment | Dr. GRPO + gated OPSD |

The central research questions are:

- Does adding OPSD to RL improve student learning and sample efficiency in self-play?
- Does gated OPSD improve upon ungated OPSD by selectively weighting useful teacher guidance?
- Does gradient-based proposer reward improve task generation over difficulty-only rewards?
- Does including the OPSD component in the candidate gradient improve gradient-guided proposal?
- How do these differences affect transfer to unseen programming problems?

Do not change these scientific questions without a documented reason.

### 1.2 Current model and architecture

Task 19 uses:

- `Qwen/Qwen3.5-4B`
- Revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- One shared proposer/student LoRA adapter
- FP32 training path
- Native Qwen3.5 thinking and tool-call serialization
- Exact-text code replacement
- One shared AdamW optimizer
- Dr. GRPO without group-standard-deviation normalization
- A privileged teacher that scores the student's sampled trajectories
- Cached rollout execution qualified numerically on Yale H200

Preserve this model and interface initially. Do not switch back to Qwen3 or increase model size as the first response to poor learning.

### 1.3 Completed Task 19 results

Read these before making changes:

- `README.md`
- `AGENTS.md`
- `docs/investigations/task19-results.md`
- `docs/investigations/task19-status.md`
- `docs/task_completion/19-implemented-qwen35-shared-selfplay.md`
- `docs/investigations/task18-results.md`
- `docs/investigations/task17-objective-results.md`
- `docs/investigations/task15-final-results.md`
- `docs/methods.md`

The completed single-seed Task 19 results were:

| Method | ID repair success | External repair success |
|---|---:|---:|
| Base | 88/115 | 93/116 |
| Control | 87/115 | 92/116 |
| A | 88/115 | 92/116 |
| B | 90/115 | 92/116 |
| C | 88/115 | 88/116 |
| D | 85/115 | 92/116 |

None of the trained configurations establishes a reliable improvement over the base model.

But this is **not evidence of convergence or equivalence**.

All trained conditions reached only 96 attempted student episodes. The Dr. GRPO control performed six optimizer steps, and A–D performed seven each.

The endpoint was frozen using the available compute budget, not a scientific convergence criterion.

The initial Qwen3.5 model already performed strongly on this specialized repair benchmark.

### 1.4 Critical observed failure signals

Task 19's qualification previously found 29 student rollout groups:

- 23 all-success groups
- 5 all-failure groups
- 1 mixed-reward group

For standard Dr. GRPO using centered binary rewards, unanimous groups provide zero solver RL advantage.

The completed main study also had very limited proposer reward discrimination among valid tasks:

| Method | Proposer groups with distinct valid rewards |
|---|---:|
| Control | 1/18 |
| A | 1/18 |
| B | 2/18 |
| C | 3/18 |
| D | 8/18 |

Configuration C additionally recorded 29 near-zero alignment gradients, compared with zero for D.

Difficulty-based proposer updates may have been driven primarily by distinguishing valid from invalid proposals rather than by identifying useful differences among valid tasks.

Finally, the existing Dr. GRPO control jointly optimizes the solver and proposer. It is **not** a pure solver-only Dr. GRPO baseline.

These observations are hypotheses about insufficient effective learning signal and gradient interference. Verify the mechanisms using the original private artifacts before deciding what to change.

### 1.5 Earlier findings

Task 15, using Qwen3-4B, achieved substantial in-distribution gains after approximately 4,096 training episodes, but almost no external generalization.

Task 18 showed that switching from the old strict patch interface to native tool calling materially improved repair success on previously inspected tasks.

The older Qwen3 model and interface results cannot be used as a controlled comparison against Qwen3.5.

Task 19 also resolved a meaningful FP32 numerical-qualification issue. Its recorded 64 comparison checks were successful, with very small gradient and restored optimizer-update discrepancies.

Do not redo these costly qualifications without evidence that a relevant execution component changed or that the existing qualification no longer applies.

The existing W&B runs and Task 19 checkpoints should remain archived and immutable.

---

## 2. Deliverables and priority

The project has three tiers of deliverables.

### Mandatory — P0

By the deadline, deliver:

1. A rigorous diagnosis of why Task 19 did not show learning.
2. A trustworthy, sufficiently challenging and reproducible training/evaluation setup.
3. A functioning pure solver-only Dr. GRPO positive control, or decisive evidence explaining why it cannot learn under the tested conditions.
4. A matched shared-policy Dr. GRPO control.
5. Experimental results comparing ungated OPSD (A) and gated OPSD (B), if the validated training pipeline supports meaningful comparisons.
6. Training learning curves, checkpoint analysis, effective-gradient diagnostics, compute accounting and final metrics.
7. A final report explaining what worked, what failed, and what the evidence actually supports.

If the solver-only positive control fails despite a reasonably bounded diagnostic effort, report the validated failure mechanism rather than burning the remaining budget on A–D.

### Important — P1

If feasible, also deliver:

- PDB-based harder Python debugging experiments.
- Clean parent-disjoint evaluation.
- Multiple training seeds for the main comparison.
- Student versus proposer gradient-interference analysis.
- A harder multi-line debugging generalization set.
- A cost/performance frontier and sample-efficiency comparison.

### Stretch — P2

Only after the main study is functioning:

- Configuration C: RL-gradient-aligned proposer.
- Configuration D: RL+OPSD-gradient-aligned proposer.
- An ablation comparing gradient-alignment definitions.
- An additional real-world bug-repair benchmark.

Do not sacrifice P0 completion to run P2.

---

## 3. Phase 1 — Audit existing evidence before using more GPUs

### 3.1 Inspect the actual scientific pipeline

Inspect at least:

- `src/selfplay_opsd/task19.py`
- `src/selfplay_opsd/task19_train.py`
- `src/selfplay_opsd/task19_policy.py`
- `src/selfplay_opsd/task19_rollout.py`
- `src/selfplay_opsd/task19_protocol.py`
- `src/selfplay_opsd/objectives.py`
- `src/selfplay_opsd/data.py`
- `src/selfplay_opsd/evaluation.py`
- `scripts/analyze_task19.py`
- `configs/task19-control.json`
- `configs/task19-yale.yaml`

Follow imports and actual runtime call paths rather than assuming a documented configuration field is active.

Use the existing private Task 19 training batches, checkpoints, logs and W&B aggregate records where available.

Do not rerun completed experiments to regenerate information that has already been recorded.

### 3.2 Effective RL learning signal

For each configuration and training iteration, report:

- Attempted and valid proposed tasks.
- Unique task counts and task families.
- Four-rollout student success histograms: 0/4, 1/4, 2/4, 3/4 and 4/4.
- Fraction of mixed-reward groups.
- Fraction of groups giving a nonzero solver RL gradient.
- Nonzero student optimizer steps.
- Student gradient norm and optimizer displacement.
- Effective generated tokens contributing to nonzero RL gradients.
- Public-test success and hidden-test success.
- Invalid actions, failed edits and truncation.
- Task difficulty distribution and how it changes during training.

Separate:

1. Total rollouts.
2. Rollouts belonging to informative groups.
3. Number of optimizer updates.
4. Nonzero RL-gradient updates.
5. Updates driven only by OPSD or proposer gradients.

A high aggregate reward is not sufficient evidence of useful RL training.

### 3.3 Inspect gradient scaling

The current Dr. GRPO implementation uses a fixed normalization constant related to `max_generated_tokens`, with group and task averaging.

Investigate whether the normalization is so large relative to actual generated trajectories that effective RL gradients are unnecessarily weak.

Specifically measure:

- Actual sampled response-token counts versus the normalization constant.
- RL loss gradient norms.
- OPSD loss gradient norms.
- Proposer loss gradient norms.
- Calibration coefficients and their actual contributions.
- Relative gradient directions.
- Learning-rate effects on the adapter.
- Optimizer update norms.
- Parameter drift from the initialization.
- Approximate KL/policy drift on fixed training-only probes.
- Actual clipping frequency and importance ratios.

Do not arbitrarily increase learning rates or change objective normalization.

If scaling is implicated, conduct a small controlled, training-only numerical and optimization comparison. Freeze the chosen recipe before main evaluation.

Keep the Dr. GRPO objective mathematically correct and document any implementation-level normalization choice.

### 3.4 Gradient interference

The shared update is conceptually:

\[
g_{\mathrm{joint}}
=
g_{\mathrm{RL}}
+\lambda g_{\mathrm{OPSD}}
+\alpha g_{\mathrm{proposer}}.
\]

Measure:

\[
\|g_{\mathrm{RL}}\|,\qquad
\|g_{\mathrm{OPSD}}\|,\qquad
\|g_{\mathrm{proposer}}\|
\]

and pairwise cosines:

\[
\cos(g_{\mathrm{RL}},g_{\mathrm{proposer}}),
\]

\[
\cos(g_{\mathrm{RL}},g_{\mathrm{OPSD}}),
\]

\[
\cos(g_{\mathrm{OPSD}},g_{\mathrm{proposer}}).
\]

Examine whether the proposer and OPSD gradients dominate, reinforce or interfere with solver gradients.

When possible, use the same saved batch and restored optimizer state to compare one-step student-only versus joint updates. Measure the effect on a separate training-only probe loss or expected reward.

Do not infer the causal impact of an optimizer direction solely from cosine similarity. A short paired counterfactual update can provide stronger evidence.

### 3.5 Sanity checks

Confirm:

- Sampled token IDs and log-probability targets align causally.
- Real rollout behavior probabilities remain PPO denominators.
- Teacher gradients are detached appropriately.
- Loss masks exclude prompts, tool observations, padding and forced control tokens.
- Gradients reach the intended shared LoRA parameters.
- The base is frozen and the intended LoRA parameters are trainable.
- Successful repairs receive the correct outcome reward.
- Uniform-reward groups yield zero centered RL advantages.
- The final evaluated checkpoint is different from initialization after real updates.
- Saving/restoring checkpoints preserves optimizer and RNG states.
- Model inference and evaluation use the intended Qwen3.5 native chat template.
- No hidden tests or reference solutions are exposed to the student.

Add tests for any confirmed defect. Avoid reconstructing an entire training framework merely to add redundant checks.

### Phase 1 output

Create:

`docs/investigations/task20-root-cause.md`

Include confirmed findings, rejected hypotheses, unresolved limitations, and recommended corrective actions.

Rank suspected causes by observed evidence and practical impact.

**Do not proceed directly to full A–D training.**

---

## 4. Phase 2 — Replace the overly easy benchmark, subject to a feasibility gate

### 4.1 Preferred benchmark: Precise Debugging Benchmark (PDB)

Investigate and prioritize:

**Paper:** Precise Debugging Benchmark: Is Your Model Debugging or Regenerating?

**Official project:**
https://precise-debugging-benchmark.github.io/

**Official implementation:**
https://github.com/Bill1235813/PDB

**Hugging Face dataset:**
https://huggingface.co/Precise-Debugging-Benchmarking

The benchmark was released in April 2026, after Qwen3.5-4B's March 2, 2026 release.

It provides Python debugging tasks with verifiable outputs and edit-level metrics.

Important dataset distinctions:

- PDB-Single-Full: approximately 7,589 examples.
- PDB-Single (the hard filtered subset, also previously named PDB-Single-Hard): 5,751 examples.
- PDB-Multi: 256 multi-line debugging tasks.
- PDB-Wild: a broader 484-example collection including repository-level tasks.

Verify exact release versions and avoid confusing these overlapping sets.

PDB includes original programming tasks from BigCodeBench and LiveCodeBench with injected semantic bugs. The tasks have fields such as `task_id`, `task_prompt`, `buggy_code`, `gt_solution`, `gt_diff` and `bug_count`.

The original correct code, edit annotations, and hidden tests are privileged artifacts, not information available to the student during evaluation.

### 4.2 Prior research to inspect

Also inspect:

**PreciseCoder: Faithful and Generalizable Debugging Beyond Unit-Test Rewards**

https://arxiv.org/abs/2610.18755

This October 2026 study is particularly relevant because it post-trains Qwen3.5-4B on PDB and reports meaningful improvements over the base model.

Its approximate experimental setup included:

- 5,274 training instances from PDB-Single-Full.
- 1,205 held-out PDB-Single test instances.
- Parent-problem-disjoint training and evaluation.
- GRPO and other RL objectives.
- Unit-test success, edit precision and recall.

Use this as evidence of potential learnability and as a reference for dataset construction and compute-efficient debugging.

Do not copy its proprietary assumptions or imply that our multi-turn agent results are directly comparable to its single-turn model results.

We are not trying to reimplement PreciseCoder. Our research question concerns gated OPSD and self-play.

### 4.3 Conduct a short feasibility test

**Time-box the PDB migration to November 22.**

Before implementing an entire pipeline:

1. Inspect PDB's dataset schema, licenses, pinned versions and official scoring code.
2. Select approximately 20–30 varied Python debugging examples using training-only data.
3. Verify that original buggy programs fail the intended tests and known corrected programs pass.
4. Determine whether the current isolated worker can execute these tasks without prohibited dependencies or unsupported runtime requirements.
5. Adapt the current student agent to inspect the buggy source, reason, make edits and test.
6. Confirm the final workspace's code can be evaluated independently.
7. Measure base-model success, error categories and runtime.
8. Estimate the integration and evaluation costs.

If the official PDB evaluation dependencies are heavy, evaluate whether a supported subset can be executed using its official handlers or an equivalence-tested adapter.

Do not silently replace complex test semantics with a weaker evaluator.

Avoid adding new network or container infrastructure if the existing Yale worker can run the selected tasks correctly.

### 4.4 Decision on migration

Proceed with PDB as the main benchmark only if:

- The selected subset executes reliably.
- Correctness is verifiable.
- The agent cannot access hidden answers.
- The benchmark provides more useful difficulty than the old task set.
- Task and family grouping can prevent leakage.
- The total projected cost permits meaningful training before December 2.

If these conditions are not met by November 22, **activate the fallback**.

### 4.5 Mandatory fallback

Keep the existing EvalPlus-derived repair environment and create a harder verified training-task distribution.

Expand task difficulty with:

- Multiple interacting semantic bugs.
- More challenging boundary conditions.
- Multi-line edits.
- Conditional and loop logic.
- Interactions between functions, where supported by the current verifier.
- Diverse mutation families.
- More difficult test cases with reliable oracles.

Every generated task must remain executable and verifiable.

Do not merely increase bug count if that produces overwhelmingly unsolvable tasks.

Preserve the existing trained and evaluation datasets, and give the new curriculum a separate versioned identity.

A timely, reliable harder curriculum using existing infrastructure is preferable to an incomplete PDB migration.

### 4.6 Dataset splitting and contamination controls

Whether using PDB or the fallback:

- Split by base programming problem or parent family, never by arbitrary corrupted-task row.
- Keep all variants of the same program in one split.
- Detect near duplicates, including across PDB-Single-Full, PDB-Single and PDB-Multi.
- Keep training-reference examples separate from development and final evaluation.
- Keep benchmark hidden tests and canonical fixes private during student inference.
- Do not use final-test outcomes for curriculum tuning, model selection, proposer reward construction or early stopping.
- Freeze dataset version, checksums and partition identities.
- Do not call a publicly released dataset guaranteed unseen merely because its benchmark was published after the model.
- Record original problem dates separately from dataset publication dates where available.

If the dataset has been inspected in prior research or earlier project work, describe it as a public held-out evaluation—not a guaranteed uncontaminated benchmark.

### 4.7 Suggested data sizes

Use these only as starting points, subject to measured runtime and eligible task counts:

- Training pool: several hundred or more parent-distinct examples, with enough diversity for online task generation.
- Training-only difficulty probe: approximately 20–50 tasks initially.
- Development monitoring: approximately 50–100 examples.
- Primary final evaluation: approximately 200–300 examples if affordable.
- Secondary harder evaluation: a small PDB-Multi subset.

Do not incur large GPU costs constructing huge datasets before establishing learner compatibility.

Prefer data quality, diversity, verifiability and difficulty calibration over raw task count.

---

## 5. Phase 3 — Build a useful difficulty distribution

This is central to the experiment.

### 5.1 Measure per-task difficulty

For each candidate training task \(x\), estimate:

\[
\hat p(x)
=
\frac{1}{K}
\sum_{i=1}^{K}
\mathbf{1}\{\text{repair}_i(x)\text{ passes verification}\}.
\]

Use a small initial K, such as 4, and selectively increase to 8 or more when the estimate is too uncertain for a decision.

Do not run eight expensive samples for every example automatically.

Use the current model and training-compatible agent interface.

Measure:

- Initial pass rate.
- Mixed-group probability.
- Mutation complexity.
- Source/specification length.
- Bug family.
- Public/hidden verification status.
- Generated tokens and runtime.
- Diversity across programs and mutations.

### 5.2 Prefer learnable tasks

Prioritize tasks with estimated solver success in a useful intermediate range, approximately 20–80%, while retaining diverse tasks and a meaningful fraction near 30–70%.

These are heuristic difficulty targets, not hard scientific boundaries.

Do not select tasks solely because they are difficult.

Tasks with zero observed successes may be useful for OPSD or curriculum learning, but they typically provide zero vanilla group-relative RL signal with binary rewards.

Tasks with all successes also provide zero Dr. GRPO advantage.

A good curriculum should produce enough variation to support learning while maintaining challenging examples and diversity.

### 5.3 Compare group size and curriculum changes

Investigate whether increasing group size from 4 to 8 gives meaningful additional informative groups relative to its extra cost.

Do this with a small paired diagnostic batch.

Prefer improving task selection when that is more efficient than repeatedly sampling easy tasks.

Do not automatically increase rollout group size across the main study without calibrating the cost and freezing the choice.

### 5.4 Preserve the self-play contribution

The proposer should learn to generate useful new challenges, not simply replay a fixed dataset forever.

However, for optimization diagnosis, a short fixed training-only curriculum is appropriate.

Implement a configurable option for:

- Frozen fixed-task curriculum.
- Shared self-play proposer.
- A diagnostic proposer with no gradient contribution.
- Difficulty-calibrated candidate selection.

Keep these modes explicitly labeled.

Avoid silently mixing fixed supervised training, generated tasks and self-play without reporting their proportions.

### 5.5 Proposer reward diversity

For the existing difficulty reward:

\[
r_P=4\hat p(1-\hat p),
\]

measure separately:

- Validity reward variation.
- Reward variation among valid tasks.
- Distinct valid-reward groups.
- Proposal uniqueness and family diversity.
- Difficulty distribution.
- Fraction of proposals that produce informative student trajectories.

Do not count a proposer update as evidence of meaningful curriculum learning merely because invalid proposals received lower rewards.

Consider separating task validity screening from the valid-task learnability reward.

Any modification to the proposer reward must be scientifically identified and compared under matched conditions. Do not silently redefine A/B while claiming to replicate Task 19.

---

## 6. Phase 4 — Prove solver-only Dr. GRPO can learn

**This phase is an essential gate before the main expensive comparison.**

### 6.1 Add a true solver-only control

Create a configuration where:

\[
\mathcal L
=
\mathcal L_{\mathrm{DrGRPO}}^{S}
\]

with:

\[
\lambda_{\mathrm{OPSD}}=0,
\qquad
\alpha_{\mathrm{proposer}}=0.
\]

There should be no hidden proposer gradient contribution.

Use training-only verified debugging tasks.

A fixed curriculum is appropriate for this first positive-control experiment.

Initialize from the same Qwen3.5-4B model and fresh LoRA used by the subsequent main study.

### 6.2 Verify one-step learning

On a small batch containing mixed-success groups:

1. Save model and optimizer state.
2. Collect authentic on-policy rollouts.
3. Verify rewards.
4. Compute the Dr. GRPO objective.
5. Apply a real optimizer step.
6. Confirm expected parameter changes.
7. Measure changes in action log-probabilities, training loss and, if practical, expected repair reward.
8. Restore and repeat to verify numerical consistency.

A policy-gradient objective is not guaranteed to improve empirical reward after every single step. Therefore, do not use one noisy outcome as proof of failure.

The test should establish that the mathematical objective, gradients and optimizer operate as intended.

### 6.3 Short positive-control training

Train on a small, intentionally learnable, verified training-only subset.

Evaluate the model before and after a bounded number of informative updates.

Measure whether it can acquire or improve the ability to repair tasks that were initially difficult.

It is acceptable and useful to demonstrate overfitting to a small training set as an optimizer sanity check.

But explicitly distinguish this from generalization.

Possible diagnostics:

- Expected training-set reward over repeated samples.
- Number of previously unsuccessful tasks that become successful.
- Reward on a separate training-only probe set.
- Policy changes toward successful actions.
- Loss and update trajectories.
- Failure categories.

If the model cannot improve even on a small learnable training set after several informative, numerically valid updates, investigate:

- Reward correctness.
- Data quality.
- Gradient flow.
- Objective normalization.
- Learning rate.
- LoRA configuration.
- Response masks and PPO ratios.
- Optimizer state.
- Sequence truncation.
- Agent interface.

Fix the identified problem before scaling.

### 6.4 Small, targeted hyperparameter diagnostics

Use only training-only probes for tuning.

Relevant parameters include:

- Effective Dr. GRPO normalization.
- Learning rate.
- Group size.
- Gradient clipping.
- Rollout temperature.
- Candidate difficulty selection.
- Episode token budgets.

Use a small, justified sweep or sequential testing strategy rather than a broad blind grid search.

Choose one robust recipe and freeze it before the main method comparison.

Do not optimize hyperparameters separately for A, B, C and D based on final-test outcomes.

### 6.5 Go/no-go criteria

Proceed when there is credible evidence of:

- Correct and reproducible task rewards.
- Nonzero RL gradients on informative groups.
- Real, stable optimizer updates.
- Learnability on the deliberately chosen training subset.
- A sufficiently challenging candidate distribution.
- Reasonable compute cost.

As an initial signal-coverage screening target, aim for approximately 20–30% or more mixed-reward groups. This is a heuristic, not a mandatory threshold if other evidence demonstrates effective learning.

If very few groups are informative, recalibrate difficulty before more training.

If the first few development evaluations are flat but the small training positive control works, do not cancel immediately. Continue for a prespecified minimum amount of informative training exposure, subject to compute constraints.

If the positive control fails, STOP the expensive main study and investigate.

OPSD can still provide a learning signal in all-failed groups. If the specific failure is proven to be RL reward sparsity, a narrowly bounded OPSD rescue pilot is scientifically justified. Do not assume that failed Dr. GRPO logically implies all OPSD methods must fail.

---

## 7. Phase 5 — Main experiments and scientific controls

### 7.1 Required experimental progression

Use the following order:

| Priority | Method | Purpose |
|---|---|---|
| Diagnostic | Solver-only Dr. GRPO | Verify solver learnability |
| Main control | Shared Dr. GRPO + difficulty proposer | Measure self-play baseline |
| Main A | Shared Dr. GRPO + ungated OPSD | Test added ungated teacher signal |
| Main B | Shared Dr. GRPO + gated OPSD | Test gating |
| Stretch C | Shared Dr. GRPO + gated OPSD + RL alignment | Test candidate gradient-based proposal |
| Stretch D | Shared Dr. GRPO + gated OPSD + RL/OPSD alignment | Test hybrid gradient-based proposal |

The core scientific comparison is Control versus A/B and A versus B.

If shared Dr. GRPO fails while solver-only Dr. GRPO succeeds, investigate shared gradient interference before launching full A/B training.

If shared Dr. GRPO has useful learning and OPSD is numerically qualified, proceed with A/B.

Do not require the Control to show statistically significant held-out improvements after only a few updates. The gate is meaningful, reproducible optimization and learning potential, not an arbitrarily early significance test.

### 7.2 Shared model implementation

Preserve:

- One shared trainable LoRA.
- One optimizer.
- Frozen-snapshot on-policy rollout collection.
- Correct behavior-policy denominators.
- Joint student/proposer gradient accumulation.
- Exactly one optimizer step per joint update.
- Detached privileged teacher signals.
- Versioned checkpoints.
- Correct native Qwen3.5 serialization.

Do not revert to separate trainable proposer/student adapters unless there is compelling evidence and a separately identified scientific reason.

Maintain independent gradient measurements for each role.

### 7.3 OPSD objectives

The ungated objective uses the teacher–student log-probability gap over student-sampled tokens.

The gated objective applies a smooth gate such as:

\[
w_t=\sigma(\Delta_t)
\]

to modulate the teacher guidance.

Review the existing Task 19 implementation rather than replacing it casually.

Confirm:

- Teacher and student score identical generated token IDs.
- Teacher-only privileged information is not exposed to the student.
- Frozen scoring is correct.
- Teacher guidance has the intended sign.
- Gates are detached where mathematically required.
- Loss scaling is consistent.
- The auxiliary gradient is neither silently zero nor overwhelming.
- The implementation is not accidentally dominated by style or forced-token supervision.

Measure gate distributions and signals separately for reasoning tokens, tool-call tokens and other sampled assistant tokens.

Use the same nominal OPSD coefficient in A and B for the principal comparison, with training-only calibration performed before the study.

Be clear that gating changes both selectivity and effective auxiliary gradient strength. If budget permits, add a small strength-matched diagnostic, but do not make it a prerequisite for main completion.

### 7.4 C/D alignment only after prerequisites

If Control/A/B have demonstrated informative learning, inspect whether gradient alignment is likely to provide meaningful proposer differentiation.

For C, investigate the near-zero RL-gradient problem.

For D, verify whether OPSD-derived gradients produce useful directional information or merely nonzero magnitudes.

Evaluate the mismatch between reference cross-entropy gradients and the actual student RL optimization gradient.

Do not assume that nonzero cosine values imply useful downstream improvements.

Possible cheap diagnostics:

- Distribution of candidate-gradient norms.
- Distribution of reference-gradient norms.
- Candidate/reference cosines.
- Proposer reward spread among valid tasks.
- Proposer reward correlation with small measured learning progress.
- One-step counterfactual validation-loss/reward change on training-only reference tasks.

If alignment cannot discriminate useful candidates or is not predictive of short-horizon benefit, do not launch expensive full C/D.

Report that result and preserve remaining resources for the core analysis.

### 7.5 Exposure, stopping and selection

Do not repeat Task 19's mistake of treating a short fixed-budget endpoint as convergence.

Instead:

- Track attempted episodes, informative episodes, informative groups, actual optimizer steps and tokens.
- Run development evaluations at predefined milestones.
- Set a minimum informative-training exposure before evaluating a plateau.
- Continue while meaningful learning progress and budget remain.
- Stop for repeated, credible numerical failures, clearly unproductive training or budget exhaustion.
- Use development data only for early stopping and model selection.
- Never select checkpoints using final-test success.
- Always preserve a primary matched-budget checkpoint comparison, even if secondary best-development checkpoints are reported.

Any early-stopping rules must be frozen before the main runs.

Compare primary arms at matched attempted training exposure or a predeclared matched compute budget. Report effective update counts separately; do not retrospectively choose whichever normalization favors one method.

---

## 8. Efficient execution and compute management

### 8.1 Existing compute environment

The repository already supports Yale HPC execution and isolated Python workers.

Use existing Yale H200 resources where authorized.

The completed Task 19 allocation of approximately 64.879 H200-hours belongs to the historical study. Do not assume that an unused portion of its old cap is automatically available for Task 20.

Before launching new GPU jobs, establish the current account allocation and an explicit new study budget using the existing project authorization and resource tools.

If the new budget is unknown or not authorized, finish CPU-side preparation, diagnostics and launch artifacts, and report the precise approval required. Never invent a GPU allowance.

### 8.2 Cost estimation before full training

Measure actual pilot:

- GPU-seconds per student rollout.
- GPU-seconds per optimizer update.
- GPU-seconds for privileged teacher scoring.
- GPU-seconds per proposer candidate.
- GPU-seconds for reference gradients.
- GPU-seconds per evaluation.
- GPU memory use.
- Student tokens per episode.
- Effective optimizer updates per hour.

Use these measurements to estimate how much meaningful training fits before the deadline.

Reserve compute for final evaluation and a contingency for interrupted jobs.

Do not allocate every available GPU-hour to training.

### 8.3 Optimize only measured bottlenecks

Profile the pipeline.

Potential improvements include:

- Avoid unnecessary repeated evaluation of the same immutable base checkpoint.
- Reuse valid frozen reference gradients within a policy snapshot.
- Avoid redundant teacher scoring.
- Reuse validated dataset and source-oracle artifacts.
- Reduce unnecessary full-checkpoint serialization.
- Use cheaper training-only difficulty screening.
- Batch model operations where numerical equivalence is supported.
- Minimize repeated Python execution unrelated to reward correctness.
- Keep evaluation and model initialization efficient.

Be cautious about changes to precision, rollout caching, score reconstruction or PPO probability semantics.

The qualified FP32 path is the initial trusted baseline.

A faster execution path is acceptable only after targeted numerical and behavior-parity checks establish that it preserves the intended computation sufficiently for the experiment.

Do not introduce a new vLLM/FSDP/precision stack merely because it might be faster.

### 8.4 Job reliability

Use the existing Slurm controller, checkpoint and resume mechanisms.

Before retrying a job, inspect its actual scheduler state and existing artifacts.

Never duplicate a running or completed experiment.

Preserve checkpoints and scientific identities when resuming.

Record:

- Job IDs.
- Model/data/config hashes.
- Actual GPU allocation.
- Interrupted allocations and retries.
- Checkpoint identities.
- Current execution health.

Avoid unnecessary readiness barriers. A confirmed working worker, valid resources and passing relevant tests are sufficient to proceed unless a concrete defect is found.

If infrastructure authentication or scheduling is temporarily unavailable, continue with independent local analysis and implementation instead of restarting work or waiting indefinitely.

---

## 9. Evaluation protocol and scientific reporting

### 9.1 Main evaluation metrics

Report at minimum:

**Primary**

- Verified repair success rate.
- Paired difference versus the relevant baseline.
- Accuracy as a function of training exposure and allocated GPU-hours.

**Secondary**

- Edit-level precision.
- Bug-level recall.
- Edit F1, where official metrics are available.
- Number of lines changed.
- Student generated tokens.
- Number of turns and tool calls.
- Truncation.
- Public-pass/hidden-fail.
- Runtime and GPU-hours per successful repair.
- Inference-budget-normalized success.

When using PDB, prefer its official precision/recall definitions and scoring implementation. Do not invent an incompatible precision metric while calling it an official PDB score.

If the agent's multi-turn tool protocol differs from the published PDB protocol, label results as **PDB-derived agentic repair**.

If feasible, additionally evaluate saved checkpoints using the original PDB inference protocol, with the same evaluation conditions for all models.

Do not claim official leaderboard comparability without matching the prescribed protocol.

### 9.2 Evaluation on harder and real-world cases

Use a clean parent-disjoint PDB-Single evaluation where possible.

Use PDB-Multi for a smaller, more challenging generalization study.

If time and execution allow, investigate:

https://goldset.dev/

Goldset contains actual historical Python bug fixes with executable fail-to-pass tests and commit timestamps.

A valuable secondary set would be function-level fixes committed after March 2, 2026, excluding known overlap.

However, the number of eligible and executable cases is unknown. Determine it empirically before committing resources.

Goldset is optional. Do not let dependencies or environment construction for real projects jeopardize the main results.

### 9.3 Statistical reporting

Use the same fixed evaluation tasks across methods.

Compute paired differences at the task level and use family/parent-clustered bootstrap intervals when tasks share source programs.

Where multiple training seeds exist, report seed variability separately from evaluation-task variation.

Do not treat bootstrapping tasks from one trained seed as uncertainty over independent training seeds.

Avoid interpreting overlapping confidence intervals as proof of equivalence.

If only one seed is affordable, make the limitation explicit.

Report both the endpoint and development-selected checkpoint results where permitted by the predeclared selection rules.

Do not discard negative results or change the target metric after seeing outcomes.

### 9.4 Required figures

Generate publication-quality figures:

1. **Learning curves:** Base, solver-only Dr. GRPO, shared control, A and B, with C/D only if completed.
2. **Compute efficiency:** Verified repair success versus actual GPU-hours and sampled training tokens.
3. **Training signal:** Mixed-reward fraction, zero/nonzero RL-gradient updates, optimizer norms and policy drift.
4. **OPSD analysis:** Teacher–student gaps, gating values and RL/OPSD gradient relationships.
5. **Proposer analysis:** Valid task rate, valid-reward diversity, task difficulty and alignment behavior.
6. **Generalization:** Primary debugging set, harder transfer set and failure categories.
7. **Failure analysis:** Truncation, failed hidden tests, wrong edits and tool/interface errors.

Do not fabricate missing milestone observations or interpolate training checkpoints as though they were real evaluations.

Preserve source data for every plot.

---

## 10. W&B observability and intervention

Maintain W&B observability for the new Task 20 study.

Log actual training metrics, not merely labels indicating that an SDK call was attempted.

A dashboard should enable an observer to answer:

- Is the model receiving informative RL rewards?
- Are model weights changing?
- Is optimization improving training-only performance?
- Is development accuracy improving?
- Is the proposer creating harder or more useful tasks?
- Is OPSD providing nontrivial gradients?
- Is the shared proposer interfering with solver gradients?
- Are invalid calls or truncation increasing?
- Is the run still within budget?

At minimum, log:

- `train/attempted_episodes`
- `train/mixed_reward_fraction`
- `train/effective_rl_updates`
- `train/verified_reward`
- `train/generated_tokens`
- `student/rl_gradient_norm`
- `student/opsd_gradient_norm`
- `proposer/gradient_norm`
- `shared/gradient_cosine`
- `shared/update_norm`
- `opsd/mean_gap`
- `opsd/gate_distribution`
- `proposer/valid_rate`
- `proposer/distinct_valid_reward_fraction`
- `evaluation/dev_success`
- `runtime/gpu_hours`
- `runtime/seconds_per_effective_update`

The actual names may follow the repository's existing schema.

Because Yale jobs previously required offline telemetry capture and replay, preserve the existing authenticated offline-sync workflow where needed. Verify cloud-history delivery rather than interpreting a logging mode label as proof.

Keep the authoritative metrics in local checkpoint-backed artifacts so that missing cloud telemetry does not invalidate completed scientific work.

---

## 11. Explicit go/no-go rules

Implement a simple experiment decision record with the following stages.

### Gate A — Benchmark feasibility

**Pass:** A usable, verified and sufficiently challenging data distribution exists.

**Fail:** Evaluation semantics are unreliable, task preparation is too expensive, or PDB migration cannot be finished on schedule.

**Action on failure:** Use the harder existing-environment fallback. Do not spend additional days attempting full benchmark integration.

### Gate B — Solver RL learnability

**Pass:** The pure solver-only Dr. GRPO positive control shows valid gradients, real updates and credible learning on a controlled training-only set.

**Fail:** No learning despite informative groups and a bounded, targeted hyperparameter diagnosis.

**Action on failure:** Investigate and fix the causal issue. Do not automatically launch shared Control/A–D.

### Gate C — Shared self-play viability

**Pass:** Shared training is stable, proposer validity is adequate, and the solver receives meaningful learning signals without overwhelming gradient interference.

**Fail:** Shared updates systematically undermine solver learning or the proposer only learns validity while providing no useful task discrimination.

**Action on failure:** Diagnose the shared gradient contribution; use the solver-only experiment as the comparison anchor.

### Gate D — Main A/B comparison

**Pass:** Both objectives run under scientifically matched conditions and reach a meaningful predetermined training exposure.

**Fail:** Numerical instability, invalid learning signals or insufficient remaining compute makes a defensible comparison impossible.

**Action on failure:** Preserve available results, document limitations and prioritize analysis instead of running inconclusive replacements.

### Gate E — C/D

Only advance when:

- Core A/B comparisons are complete or securely on track.
- Proposer rewards have useful variation.
- Gradient-alignment scores are numerically meaningful.
- The remaining budget and deadline support additional experiments.

Otherwise, defer C/D explicitly.

A failed gate does not always imply the entire project must stop. It means the expensive downstream work must be reconsidered using evidence, rather than automatically executed.

---

## 12. Deadline schedule

The following is the target plan, subject to actual resource availability.

| Date | Required milestone |
|---|---|
| Nov 19–20 | Audit Task 19 artifacts, gradient signals, optimization and benchmark requirements |
| Nov 21–22 | Complete PDB feasibility test or activate fallback; freeze new data partitions |
| Nov 22–23 | Run solver-only RL positive control and training-only optimization diagnostics |
| Nov 24 | Make documented training go/no-go decision; freeze main scientific configuration |
| Nov 25–28 | Execute matched shared Control/A/B study, with additional seeds if affordable |
| Nov 29 | Close main training and select checkpoints using development rules |
| Nov 29–30 | Final evaluation, confidence intervals, compute accounting and diagnostics |
| Dec 1 | Freeze figures, research conclusions and reproducibility artifacts |
| Dec 2 | Final report, presentation-ready figures and verified completion |

Do not treat this schedule as requiring every possible method.

If delayed, cut in this order:

1. Extra real-world transfer benchmarks.
2. C/D full training.
3. Extra gradient-alignment ablations.
4. Additional training seeds beyond the minimum viable comparison.
5. Nonessential visualizations and engineering refinements.

Do not cut:

- Reward and optimizer correctness.
- The solver-only positive control.
- Basic experiment comparability.
- Clean primary evaluation.
- Final analysis.
- Reproducibility and documentation.

Set an internal completion target earlier on December 2 than the formal submission deadline to leave room for verification and submission.

---

## 13. Research engineering and repository requirements

### 13.1 Branch and preservation

Create a new branch such as:

`feat/20-deadline-learning-rescue`

Keep all Task 15–19 artifacts, checkpoints, source configurations and historical results immutable.

Use a new scientific study namespace and explicit metadata identifying data, model, control and execution versions.

Do not overwrite previous result files with Task 20 metrics.

### 13.2 Implementation quality

Use:

- Modular, type-checked Python.
- Existing configuration classes and command-line interfaces where possible.
- Reproducible seeds and manifests.
- Deterministic dataset partitioning.
- Reliable checkpointing and recovery.
- Small, targeted regression tests.
- Shared utilities instead of duplicated research code.
- Existing HPC and observability integrations.

Do not create an entirely separate framework for PDB if a small dataset adapter and evaluator bridge can reuse the existing trainer.

Avoid speculative optimizations, excessive abstraction and unrelated refactoring.

### 13.3 Security and execution

Preserve the project's existing isolated Python-execution policy and Yale resource limits.

Reuse supported isolation and process controls.

Do not demand unavailable cgroups, privileged container features or unnecessary exact process-count enforcement when the existing approved single-process isolation provides the intended execution boundary.

Never execute untrusted generated code directly on a login node or outside the approved worker.

Use pragmatic safety checks proportional to the actual risks.

### 13.4 Mandatory documentation

Produce:

- `docs/investigations/task20-root-cause.md`
- `docs/investigations/task20-benchmark-decision.md`
- `docs/investigations/task20-positive-control.md`
- `docs/investigations/task20-main-results.md`
- `docs/investigations/task20-final-decision.md`
- `docs/task_completion/20-implemented-deadline-study.md`

You may consolidate documents if that improves clarity and avoids duplication.

Update `README.md` with a concise summary and links.

Keep all experiment commands, run IDs, data hashes, model IDs, W&B links, environment details and scientific-control identities reproducible.

Private or restricted data should not be committed publicly.

---

## 14. Final report requirements

The final report must explain:

### Scientific objective

What we wanted to learn about self-play, OPSD gating and gradient-guided proposer objectives.

### Why Task 19 failed to demonstrate improvement

Distinguish confirmed causes from hypotheses, including:

- Too few optimizer steps.
- Uniform-reward groups.
- Insufficient proposer reward discrimination.
- Possible gradient interference.
- Dataset difficulty.
- Evaluation ceiling and protocol limitations.

Do not say the model converged unless actual training evidence supports this.

### What changed in Task 20

Describe model, data, interface, optimization, reward, curriculum, and experimental-control changes precisely.

### Whether RL actually learned

Show the pure solver-only positive control and the shared-policy control.

Separate optimizer sanity, training-set improvement, development transfer and final generalization.

### Main results

Compare Control, A and B, plus C/D only if completed.

Show both accuracy and efficiency.

### Failure analysis

Analyze why particular tasks remain unsolved and whether the remaining failures are primarily due to reasoning, edit quality, tests, interface or inference budget.

### Limitations

Discuss training seeds, sample size, dataset overlap, benchmark protocol changes, compute constraints and incomplete conditions.

### Research conclusion

State the strongest defensible conclusion, even if the outcome is null.

A credible negative result with a verified learning baseline is more valuable than an apparently positive but confounded result.

The report should be suitable for conversion into the final CPSC 5880 project report and presentation.

---

## 15. Immediate execution instructions

Start now.

**Step 1:** Inspect the current repository and available private Task 19 artifacts.

**Step 2:** Produce an evidence-based, ranked root-cause summary. Identify which conclusions can be obtained without additional GPU time.

**Step 3:** Inspect official PDB implementation and test whether a small Python debugging subset can be imported and executed correctly.

**Step 4:** In parallel, prepare the existing-environment harder-task fallback so that PDB integration is never the sole path to completion.

**Step 5:** Implement and test a genuine solver-only Dr. GRPO control.

**Step 6:** Run the minimum necessary numerical and learning-signal diagnostics using authorized compute.

**Step 7:** Record a go/no-go decision. If training works, freeze the main study and execute the highest-priority matched experiments.

**Step 8:** Continue with guarded, checkpointed runs; monitor scientific metrics and actual costs.

**Step 9:** Freeze the study in time to perform proper evaluation, analysis and reporting.

**Step 10:** Produce the final documentation, figures, reproducibility artifacts, and a concise research conclusion.

After each meaningful milestone, update a compact project status document showing:

- Completed work and evidence.
- Current scientific findings.
- Confirmed blockers, if any.
- Remaining GPU budget.
- Next concrete action.
- Whether December 2 completion remains feasible.

Use that document as the source of truth instead of writing lengthy repeated readiness summaries.

Do not repeatedly seek approval for ordinary implementation decisions already covered by this specification. Make reasonable research-engineering choices, document them, and proceed.

Only require user intervention for genuinely unavailable access, external authorization, budget approval, or a material scientific decision not covered here.

**Your ultimate responsibility is not to make every experiment finish. It is to produce the strongest valid scientific answer possible by December 2, while avoiding unnecessary cost and another cycle of debugging without learning.**

Begin by examining the repository and executing Phase 1. Do not stop after producing a plan.