# Task: Diagnose Stagnant Learning in Coding-Agent Self-Play and Design an Efficient Experimental Recovery Plan

## 1. Role and objective

Act as a senior research scientist and research engineer specializing in LLM reinforcement learning, RLVR/GRPO, on-policy self-distillation (OPSD), self-play, automated curriculum generation, and coding agents.

**Repository:** https://github.com/tatsu432/self-play-opsd-coding-agent

**Current investigation branch:** `feat/11-experiment-a-regression-diagnosis`

Our project investigates how gated OPSD and gradient-aligned proposer rewards affect coding-agent self-play.

We have successfully implemented and executed substantial portions of the experiment, but the results are not particularly convincing. In particular:

- Proposer validity is extremely low, approximately 5% in completed A/B runs.
- Proposer validity appears to deteriorate late in training.
- Most student task groups have uniform verifier rewards, resulting in zero GRPO advantages.
- Gradient-alignment rewards in Experiment C are frequently neutral because candidate RL gradients are exactly zero.
- Development performance improves initially but then largely stagnates.
- It remains unclear whether the bottleneck is model capacity, task representation, overly restrictive validity rules, optimization, curriculum design, or a combination.

**Your mission is to identify the strongest causes using existing experimental evidence and inexpensive diagnostics, then produce a concrete, prioritized plan for obtaining meaningful research results efficiently.**

Do not assume our current experimental design is fundamentally correct. Critically evaluate whether the methodology can answer the intended research questions.

Do not prematurely rewrite the entire pipeline or launch expensive training runs.

I want actual investigation, measurements, and a decision-ready plan—not a generic list of possible explanations.

---

## 2. Research questions and experimental context

Our current four configurations are:

| Method | Proposer reward | Student objective |
|---|---|---|
| A | Difficulty | GRPO + ungated OPSD |
| B | Difficulty | GRPO + gated OPSD |
| C | RL-only gradient alignment | GRPO + gated OPSD |
| D | RL + gated OPSD gradient alignment | GRPO + gated OPSD |

Intended comparisons:

1. A vs. B: Does gated OPSD improve student learning and stability?
2. B vs. C: Does gradient-aligned task generation outperform difficulty-based generation?
3. C vs. D: Is it beneficial to align task generation with the student's combined RL+OPSD objective rather than RL alone?

The current implementation uses:

- Qwen3-4B-Instruct-2507.
- Separate student and proposer LoRA adapters over a frozen common base.
- Four parents per iteration.
- Four proposal attempts per parent.
- Four student rollouts per accepted task.
- An executable Python repair environment based on MBPP+-derived problems.
- Strict JSON and unified-diff patch protocols.
- Binary verifier rewards.
- Group-normalized GRPO.
- A privileged self-teacher conditioned on the correct source and inverse patch.
- A coding reference gradient constructed from canonical repair-edit actions.
- Approximately 200 iterations per main method.

Inspect the actual code and configuration to verify these details instead of relying on this summary.

Relevant files include:

- `AGENTS.md`
- `docs/methods.md`
- `docs/investigations/experiment-a-regression.md`
- `docs/investigations/next-experiment-decision.md`
- `docs/task_completion/11-implemented-experiment-a-regression.md`
- `docs/task_completion/12-implemented-corrected-study-readiness.md`
- `configs/task12-corrected.yaml`
- `src/selfplay_opsd/trainer.py`
- `src/selfplay_opsd/objectives.py`
- `src/selfplay_opsd/mutations.py`
- `src/selfplay_opsd/environment.py`
- `src/selfplay_opsd/data.py`
- `src/selfplay_opsd/policy.py`
- `src/selfplay_opsd/verl_backend.py`
- `src/selfplay_opsd/scientific_health.py`

Read any additional files necessary to understand the implementation.

### Existing experimental evidence

As of the latest known branch state:

**Corrected Experiment A:**
- Completed 200 iterations.
- 161 valid proposals out of 3,200.
- Only 17 valid proposals in iterations 141–200.
- 82 of 98 student optimizer steps had zero current group-relative RL signal.
- Final development success: 11/41.

**Corrected Experiment B:**
- Completed 200 iterations.
- 162 valid proposals out of 3,200.
- Only 21 valid proposals in iterations 141–200.
- 84 of 102 student optimizer steps had zero current group-relative RL signal.
- Final development success: 12/41.

**Corrected Experiment C:**
- Reached iteration 100 in a bounded continuation.
- 116 valid proposals out of 1,600.
- 98 of 116 candidate alignment scores were neutral due to zero candidate RL gradients.
- 18 mixed-outcome groups among 116 valid tasks.
- Development success: 11/41 at iteration 100.

**Experiment D:**
- Has prior pilot evidence, but verify whether a full corrected run has been submitted or completed before making any claims.

The original Experiment A suffered a catastrophic regression: the agent began double-escaping patch newlines and could no longer successfully apply edits. This behavior did not recur in the completed corrected A/B runs. Do not conflate the original regression with the current learning stagnation.

The baseline development success was 0/41. Thus, current runs show some observed improvement, but single-seed differences of one or two tasks do not demonstrate method superiority.

A previous numerical audit also found material sensitivity in the proposer update to behavior-versus-rescored token probabilities, including a counterfactual Adam-update cosine around 0.59. No specific implementation defect was established.

**First reconcile the latest branch state, saved evidence, and any existing jobs.** These observations may have been superseded.

---

# Part I — Establish the root causes using existing data

## 3. Investigate why proposer validity is so low

This is the highest priority.

I want a quantitative breakdown of the complete proposal validation pipeline.

### 3.1 Construct a proposal validity funnel

Using saved raw proposals and validation records, reconstruct the following stages:

1. Generation completed normally.
2. Valid JSON parsed.
3. Correct proposal schema.
4. Unified-diff headers accepted.
5. Patch hunks and line counts accepted.
6. Patch context matches source.
7. Patched Python parses.
8. AST, import, API, and structural constraints pass.
9. Proposed mutation produces an observable semantic failure.
10. Mutation passes execution/verifier requirements.
11. Mutation is sufficiently novel and not a duplicate.
12. Proposal becomes an accepted training task.

For each stage, report:

- Number and fraction passing.
- Number and fraction rejected.
- Conditional pass rate among proposals reaching that stage.
- First rejection reason.
- Multiple applicable failure reasons where these can be determined without changing the protocol.
- How these rates change over training.

Use the existing recorded outcomes wherever possible. Do not re-execute arbitrary generated code merely to reconstruct a formatting rejection.

When actual execution is necessary, use the existing accepted isolated worker and bounded verification machinery.

Distinguish **syntactic/protocol failure**, **semantic-invalid task**, **safety/unsupported-code failure**, and **duplicate-but-otherwise-valid task**.

Do not count duplicates as evidence that the model is incapable of generating valid code mutations.

### 3.2 Analyze temporal behavior

Compare early, middle, and late training windows, using matching iteration ranges across A/B/C where available.

For example:

- 1–20
- 21–40
- 41–80
- 81–100
- 101–140
- 141–200

Determine whether the declining validity rate is explained by:

- Worsening output-format compliance.
- Increasing duplicate generation.
- Changes in source-parent difficulty.
- Exhaustion of available valid mutations.
- Proposer policy concentration or mode collapse.
- Increasingly severe structural errors.
- Reduced exploration.
- Actual deterioration of learned code-generation ability.

In particular, compare validity on the same parent programs across checkpoints where data permit.

Do not confuse evolving novelty history with deterioration of the proposer policy itself.

### 3.3 Investigate duplicate proposals

Existing evidence suggests many duplicate proposals occur within the same iteration, even with distinct sampling seeds.

Determine:

- Duplicate rate within each four-proposal parent group.
- Duplicate rate relative to earlier accepted mutations.
- Duplicate rate across independently sampled outputs.
- Number of distinct proposed AST mutation sites per parent.
- Diversity of mutation operators and code locations.
- Whether repeated samples are identical token sequences or different patches producing the same normalized AST.
- Whether the proposer learns to concentrate on a few high-probability mutations.

Compare duplicate rates with proposer output entropy, response length, and sampling configuration where available.

Determine whether the model is repetitive or the available mutation space is genuinely small.

A useful optional CPU diagnostic is to enumerate a limited set of deterministic admissible AST mutations per parent and measure how many produce verified semantic bugs.

Do not claim that such an enumerator exhausts the full valid mutation space unless it actually does.

### 3.4 Evaluate the validity rules themselves

Inspect the implementation critically.

The existing validator enforces exact unified-diff syntax, context and hunk counts, preserved APIs/imports, small AST edits, safety constraints, and semantic failure.

Classify requirements into:

- Essential for correctness and verifiability.
- Essential for safe execution.
- Necessary for comparability with the current research protocol.
- Convenient but potentially unnecessarily restrictive.
- Accidental constraints arising from implementation details.

Pay special attention to whether requiring the LLM to generate a precise unified diff creates an unnecessarily difficult output-format problem.

Compare the following conceptual representations:

**Current representation:** The model produces a complete JSON-wrapped unified diff.

**Alternative representation:** The model identifies a target AST location and a permitted transformation, while deterministic code constructs the patch.

**Another possible representation:** The model proposes a modified code fragment or source, and deterministic code constructs and validates a diff.

Assess their trade-offs in expressivity, mutation diversity, correctness, implementation effort, and comparability.

Do not automatically weaken semantic validity or execution isolation.

Determine whether changing the representation is likely to be more effective than making the underlying model larger.

---

## 4. Audit whether the proposer reward actually affects learning

This is equally important scientifically.

### 4.1 Investigate group-normalized proposer advantages

The proposer currently uses GRPO advantages normalized within each four-proposal parent group.

Invalid proposals receive reward -1. Valid proposals receive a difficulty or alignment reward.

Consider a group such as:

`[-1, -1, -1, 0.8]`

Because of per-group standardization, the exact value 0.8 is almost irrelevant as long as it remains above -1 and is the only valid proposal.

Thus, the proposer may be learning primarily to distinguish valid from invalid outputs rather than learning task difficulty or gradient alignment.

Verify this mathematically and against the actual implementation, including normalization epsilon, PPO ratios, token weighting, and optimizer behavior.

For all saved proposal groups, calculate:

- Number of valid proposals per parent group: 0, 1, 2, 3, 4.
- Variance of proposer rewards.
- Variance of rewards among valid proposals only.
- Fraction of groups with nonzero advantages.
- Fraction with at least two valid proposals receiving distinguishable rewards.
- Fraction in which difficulty/alignment rewards affect normalized advantages beyond a binary validity signal.
- Effective number of meaningful proposer-gradient updates.

### 4.2 Compare real advantages against validity-only advantages

For each observed proposer group, construct a counterfactual reward vector:

- Invalid proposals: -1.
- Valid proposals: a fixed constant, such as 0.

Compute the resulting normalized advantages using the same estimator.

Compare these with the original advantages.

Report:

- Fraction of groups with exactly or approximately identical advantages.
- L2 difference between advantage vectors.
- Correlation/cosine where defined.
- Whether gradient-alignment scores meaningfully change the relative weighting of valid candidates.
- The number of groups in which C/D's more sophisticated proposer reward could influence the policy update.

Use actual saved proposal rewards and outputs.

Where saved parameters allow, a small non-mutating gradient diagnostic may help, but do not perform unnecessary full-model backward passes if advantage comparisons already establish that the reward cannot matter.

**Key question: Are we meaningfully optimizing learnability/alignment, or mostly training a model to produce validator-accepted JSON patches?**

If the latter, treat this as a central scientific limitation, not merely a hyperparameter problem.

### 4.3 Check proposer optimizer correctness

Inspect:

- GRPO loss implementation.
- Reward normalization and zero-variance handling.
- Behavior versus trainer-rescored log probabilities.
- Importance ratios and clipping.
- Gradient accumulation weights.
- Optimizer steps and skipping logic.
- Gradient norms and clipping.
- Adapter synchronization and parameter updates.
- Whether the proposer is receiving meaningful signals from the attempted candidates.

Incorporate the existing numerical audit, but do not assume its counterfactual discrepancy is a proven bug.

If a specific defect is demonstrated, provide a minimal reproducible test and proposed correction.

Do not silently replace behavior probabilities with rescored probabilities just to make the importance ratios closer to one.

---

## 5. Investigate why student learning stagnates

### 5.1 Quantify informative GRPO groups

For every admitted task, reconstruct the four rollout rewards.

Report:

- All-failure groups.
- Mixed groups.
- All-success groups.
- Fraction with nonzero group-relative advantages.
- Proportion of student updates with nonzero current RL gradients.
- Distribution of task success rates.
- Distribution of successful patches, failed patches, test calls, and finish actions.

Distinguish zero current GRPO gradients from nonzero total gradients caused by OPSD.

Distinguish current gradients from optimizer momentum carried over from earlier updates.

### 5.2 Analyze learning efficiency using useful exposure rather than just steps

Compare development performance against:

- Iteration count.
- Number of accepted training tasks.
- Number of unique training tasks.
- Number of mixed-reward groups.
- Number of nonzero-RL-gradient updates.
- Generated student tokens.
- Actual allocated GPU-hours.

Determine whether apparent stagnation reflects:

- Too little useful training data.
- Insufficient informative RL feedback.
- Saturation of the current mutation distribution.
- Forgetting and interference.
- Poor reward/optimization dynamics.
- Excessively strong or misleading OPSD updates.
- A limitation of the 4B model.

Do not presume additional iterations will help without examining the effective training exposure.

### 5.3 Investigate OPSD dynamics

For A and B, examine how much of the actual training update is attributable to:

- GRPO.
- OPSD.
- Their interaction.

Inspect the signed teacher-minus-student log-probability gaps, gate weights, clipping, and gradient norms.

Important concern: for B, the sigmoid gate is approximately 0.5 when the teacher-student gap is zero, whereas A has gate weight 1. Thus A versus B changes both selective gating and the effective distillation strength.

Assess how much this confounds our intended comparison.

Consider whether a later control with matched effective OPSD strength is necessary.

Also investigate whether teacher supervision on unsuccessful repair trajectories reinforces undesirable behaviors or merely provides useful token-level correction.

Where possible, stratify OPSD signals by action category: read, patch, test, finish.

Do not interpret a lower OPSD loss or larger teacher gap as proof of improved repair behavior.

### 5.4 Evaluate possible policy drift and format collapse

Compare recorded student behavior across checkpoints.

In particular:

- JSON validity.
- Patch applicability.
- Literal versus escaped newlines.
- Successful and rejected edits.
- Premature finish.
- Public-test usage.
- Length of actions and complete trajectories.
- Output entropy, where available.
- Estimated KL or parameter displacement from the initial policy, where available.

Investigate whether training improves task performance but degrades tool/protocol compliance, and whether such behavior is affected by OPSD gating.

The original catastrophic A regression should remain separate from corrected A/B evidence.

### 5.5 Test the initial-reward-drop hypothesis

Consider the hypothesis that RLVR initially decreases reward or validity before recovering as it learns formatting.

Do not treat this as a generic expectation of GRPO.

Examine actual curves and determine whether there is evidence of:

- Temporary exploration followed by recovery.
- Persistent policy collapse.
- Repeated oscillation.
- Curriculum novelty exhaustion.
- Reward sparsity preventing recovery.

Compare early and late fixed-probe metrics where available.

If existing evidence is insufficient, specify the smallest targeted experiment needed to distinguish the hypotheses.

---

# Part II — Evaluate alternatives without expensive training

## 6. Should proposer/student group sizes increase from 4 to 8?

Treat proposer and student group sizes separately.

### Student group size

For task success probability `p` and `G` independent rollouts, the probability of observing a mixed group is:

`P(mixed) = 1 - p^G - (1-p)^G`

Compare G = 4, 8, and optionally 16.

Use existing evidence to estimate task-difficulty distributions, including uncertainty from having only four rollouts per task.

Do not treat observed 0/4 or 4/4 as proof that the underlying success probability is exactly zero or one.

Determine:

- Expected increase in mixed-outcome groups.
- Expected increase in nonzero GRPO gradients.
- Additional generation and scoring cost.
- Whether the gain in informative groups per GPU-hour justifies the additional rollouts.
- Whether a better task-difficulty distribution would be more effective than increasing G.

If eight or more independent rollouts are necessary for reliable estimation, propose a bounded inference-only diagnostic rather than pretending existing four-rollout records establish the answer.

### Proposer group size

Compare G = 4 and G = 8 with respect to:

- Probability of at least one valid candidate.
- Probability of at least two valid candidates.
- Reward discrimination among valid candidates.
- Number of meaningful proposer updates.
- Duplicate correlation.
- Additional generation/validation cost.

Use empirical within-parent dependence where possible.

A simple independent-binomial approximation may be reported as an illustration, but label it clearly and do not mistake it for measured behavior.

Compare compute-matched variants as well as equal-iteration variants.

Recommend whether we should increase group size, improve candidate validity, use a different grouping approach, or change the proposer objective.

---

## 7. Evaluate whether model capacity is the bottleneck

I want a concrete proposal for a **small inference-only model comparison**, not an immediate full retraining campaign.

Candidate models:

1. Current Qwen3-4B-Instruct-2507.
2. Qwen2.5-Coder-7B-Instruct.
3. Qwen3-8B in an appropriate non-thinking output mode.
4. Optionally Qwen3.5-9B if compatible with available inference infrastructure.

These are examples; inspect current model support and recommend better candidates if justified.

Design a paired, frozen training-parent probe, potentially 40–64 parents and a small fixed number of proposals per parent.

Evaluate:

- Current 4B baseline versus trained 4B proposer checkpoints.
- Current prompt/format across model sizes.
- Optional structured mutation representation across model sizes.
- Unique semantic-valid tasks.
- Format and patch validity.
- Duplicate rate.
- Mutation diversity.
- Estimated GPU-hours per unique useful task.

Control generation temperature, sampling configuration, input parents, and random-seed methodology as closely as possible.

Document differences in tokenizers, chat templates, reasoning modes, and architecture that limit comparability.

Determine whether:

- A larger model solves the problem without training.
- A coding-specialized model performs better than a larger general model.
- A different representation benefits the existing 4B model enough to avoid scaling.
- Model size helps syntax but not semantic validity.
- Neither model size nor representation resolves the issue.

Do not spend significant effort making a new 9B architecture compatible with the existing trainer merely to run a quick diagnostic. Inference-only compatibility should be assessed separately from training compatibility.

**Do not launch a full 8–9B training run before the smaller comparison establishes its value.**

---

## 8. Evaluate a vanilla GRPO baseline on fixed training tasks

We need to determine whether the student can actually learn this repair environment without self-play or OPSD.

Design the smallest informative experiment using a frozen set of verified repair tasks derived strictly from the training split.

Possible comparisons:

| Condition | Task generation | Student training |
|---|---|---|
| Static-GRPO | Fixed verified tasks | Vanilla GRPO |
| Static-Gated | Same tasks | GRPO + gated OPSD |
| SelfPlay-GRPO | Learned proposer | Vanilla GRPO |
| SelfPlay-Gated | Learned proposer | GRPO + gated OPSD |

Prioritize the first two.

Include a tiny positive-control overfitting/optimization test if needed to establish that the student can learn correct edit behavior at all.

Consider preselecting tasks with a useful mixture of success and failure probabilities, rather than relying on randomly generated tasks whose gradients are almost always zero.

Assess whether using deterministic mutation generation in the existing repository is sufficient for such a baseline.

Check whether the `GRPO` objective is already implemented and merely absent from the Yale launcher/controller's accepted method list. Prefer a small launcher extension rather than rewriting the training system.

Recommend a bounded pilot of roughly 20–40 effective optimization steps as an initial learning-dynamics check, with expansion only when justified. Do not interpret a negative short pilot as definitive proof that GRPO cannot learn.

Compare conditions under matched validated-task exposure, optimizer updates, and approximate GPU budget.

Determine which outcome would support each hypothesis:

- Static GRPO learns, self-play GRPO fails: curriculum/proposer problem.
- Static GRPO fails and OPSD succeeds: sparse reward or optimization problem.
- Both fail: inspect agent protocol, model capability, verifier semantics, and training correctness.
- Both learn: proceed to isolate task-generation reward effects.

Provide estimated GPU cost and concrete success/failure diagnostics.

---

## 9. Investigate C versus D using a cheap gradient counterfactual

Our main proposed scientific contribution involves whether the proposer should align with the RL-only gradient or the combined RL+OPSD gradient.

C has many zero candidate gradients because task groups are uniform.

D may produce nonzero candidate gradients under the same groups.

Before running a full D experiment, determine whether the two reward definitions produce different meaningful proposer updates.

At a matched saved model snapshot:

1. Use the same verified candidate tasks and sampled student rollouts.
2. Compute the RL-only candidate gradient.
3. Compute the combined RL+gated OPSD candidate gradient.
4. Use the same fixed reference gradient.
5. Calculate candidate gradient norms, cosine scores, and proposer rewards.
6. Group the same candidate proposals according to the current proposer GRPO grouping.
7. Calculate the resulting normalized proposer advantages.
8. Compare actual candidate ranking and proposer update directions.

Use existing gradients or saved data wherever possible.

If new backward passes are required, design a small non-mutating GPU diagnostic with checkpoint restoration and bounded compute.

Do not run hundreds of training iterations merely to measure whether the two reward functions differ.

Specifically answer:

**Does D provide useful gradient information where C provides none?**

**Even if D has richer candidate scores, does the current proposer grouping let those scores affect actual optimization?**

---

## 10. Consider math versus code reference gradients as a lower-priority ablation

The CompassPlay author informed us that an unpublished small preliminary comparison ranked math, math+code, and code references in that order for both mathematical and coding evaluation, while keeping code proposals.

This is interesting but not conclusive.

Our current reference uses canonical repair-edit cross-entropy, which may emphasize tool-output formatting rather than general reasoning.

Eventually consider:

- Mathematical reference solutions.
- Canonical coding-agent repair actions.
- Richer coding-reasoning solutions or trajectories.
- Math+code mixtures.

However, do not prioritize this before fixing low candidate validity and reward-discrimination problems.

Changing reference gradients cannot make an exactly zero RL-only candidate gradient informative.

First determine whether reference changes affect normalized proposer advantages and actual update directions at fixed checkpoints. Only then recommend a training ablation.

Do not use held-out development or final evaluation problems as the training reference.

---

# Part III — Improve observability and deliver a concrete plan

## 11. Improve diagnostic observability

Identify which necessary metrics already exist and which are missing.

Do not duplicate current dashboards or add expensive full-model gradient instrumentation on every step.

I want a concise, easily interpretable scientific-health dashboard with the following categories.

### Proposer health

- Raw JSON validity.
- Patch syntax/applicability.
- AST/semantic eligibility.
- Semantic failure rate.
- Duplicate rate within iteration and relative to history.
- Unique accepted tasks per iteration.
- Validity by parent and mutation family.
- Fraction of groups with 0, 1, or 2+ valid candidates.
- Fraction of groups with informative difficulty/alignment differences.
- Effective proposer reward versus validity-only reward.
- Proposer entropy, behavior-policy ratio diagnostics, gradient norms, and displacement where measurable.

### Student health

- All-failure, mixed, and all-success group rates.
- Nonzero RL-gradient updates.
- OPSD and combined update norms.
- Successful versus rejected patches.
- Public-test calls.
- Premature finish.
- Token lengths and tool-format validity.
- Student success against actual training exposure and GPU-hours.

### Gradient-alignment health

- Candidate gradient zero/nonzero rate.
- Reference gradient norms.
- Cosine distributions.
- Neutral alignment fraction.
- Discriminative proposer groups.
- Changes in rewards and normalized advantages when using RL-only versus combined gradients.

### Model behavior

- Fixed-probe proposer validity across checkpoints.
- Fixed-probe student repair success across checkpoints.
- Known valid patch likelihood where useful.
- Repetition and diversity changes.
- Initial-policy displacement/KL estimates where reasonably available.

Add only the highest-value missing aggregates to W&B.

Prefer offline analysis from existing events when full gradient recomputation is unnecessary.

Do not upload private oracle source, hidden tests, secrets, or raw sensitive trajectories to W&B.

---

## 12. Implement inexpensive diagnostics now

You are authorized by this task to perform read-only investigation and modest CPU-only analysis work autonomously.

Do not stop after drafting a plan if the required evidence already exists in the repository or stored run artifacts.

Specifically:

1. Inspect the current implementation and scientific records.
2. Locate existing run data and reconcile active event lineage.
3. Reuse existing analysis utilities wherever possible.
4. Implement a focused CPU-only diagnostic script if necessary.
5. Generate quantitative tables and a compact report.
6. Add focused unit tests for any new calculations.
7. Run the relevant tests and static checks.
8. Document remaining uncertainties and propose bounded GPU experiments.

Suggested output artifacts:

- A proposer rejection-funnel report.
- A proposer group reward/advantage-discrimination report.
- A student mixed-group and effective-update report.
- A parent-level repetition/diversity report.
- A longitudinal learning-efficiency summary.
- A prioritized experimental decision plan.

If raw records are unavailable locally, use the existing supported Yale fetch mechanism. Avoid repeated expensive downloads if the artifacts already exist.

Distinguish:

- Proven from raw recorded evidence.
- Supported by aggregate records.
- Plausible hypothesis.
- Requires a new controlled experiment.

Do not fabricate missing outcomes or infer raw trajectory behavior from incomplete aggregates.

Use lightweight plots or tables only where they make the results easier to interpret.

---

## 13. Decision framework

After completing the analysis, provide a clear decision matrix.

For example:

| Observation | Likely implication | Recommended next action |
|---|---|---|
| Most rejections are unified-diff formatting errors | Output representation bottleneck | Structured mutation probe |
| 7B/8B substantially outperforms 4B with the same prompt | Model-capability contribution | Small larger-model pilot |
| Fixed-probe validity remains stable but live validity falls | Novelty/curriculum exhaustion or parent-distribution effects | Improve diversity/curriculum |
| Most proposer groups contain only one valid task | Difficulty/alignment rewards cannot meaningfully discriminate | Improve validity before changing reward |
| Student tasks remain mostly uniformly failed | Weak RLVR signal | Test better curriculum and G=8 |
| Static GRPO succeeds but self-play GRPO fails | Task generation bottleneck | Redesign proposer/curriculum |
| Static GRPO fails with informative rewards | Student objective/implementation concern | Focused optimizer/protocol audit |
| D produces distinct rewards but identical proposer advantages | Group-normalization bottleneck | Change grouping/reward estimator in a new study |
| D produces distinct meaningful advantages | Combined alignment is promising | Bounded D pilot before full training |

Adapt this table to actual observations. Do not mechanically recommend interventions whose premises are unsupported.

For every proposed intervention, estimate:

- Expected scientific information gained.
- GPU-hours or CPU cost.
- Implementation complexity.
- Scientific risk or confounding.
- Whether it requires a new immutable study.
- Which observed result would justify continuing.
- Which result would make us stop or pivot.

Prioritize **information gained per GPU-hour**, not simply higher benchmark scores.

Do not require arbitrary absolute success thresholds. Use diagnostic effect sizes, uncertainty, and feasible compute to justify decisions.

---

## 14. Preserve experimental integrity without unnecessary blockers

This project previously lost significant time to overly rigid readiness and infrastructure requirements. Avoid repeating that problem.

Rules:

- Preserve the historical and corrected A–D studies, checkpoints, and authoritative event records.
- Do not silently alter a frozen scientific protocol and resume old checkpoints under a new implementation.
- Use a new study identity when changing models, prompts, rewards, sampling budgets, mutation representation, or training objectives.
- Do not use final held-out evaluations to select interventions.
- Do not duplicate existing Slurm jobs or checkpoint reconciliation tasks.
- Do not launch a new full A–D run, 8–9B training run, or expensive multi-model benchmark automatically.
- Existing authorization to complete the corrected study should not be assumed to authorize unrelated new GPU studies.
- Provide concrete proposed GPU commands and cost caps for subsequent experiments, but complete available CPU-only work first.
- Reuse accepted sandbox isolation, checkpoint, and Yale execution infrastructure. Do not recreate the old process-count/cgroup blocker or repeat entire readiness suites for observational scripts.
- Do not sacrifice safe generated-code execution or verifier correctness for speed.
- Avoid broad refactors, new frameworks, redundant dashboards, or speculative hyperparameter searches.
- If an optional diagnostic cannot run because a resource is missing, continue other independent analyses and document the missing evidence.
- Do not ask me questions that can be answered from the repo or existing artifacts.

Treat real scientific-integrity failures differently from merely poor experimental performance. Stagnant learning is an outcome to investigate, not a reason to abort all work.

Prefer small, targeted, tested changes.

---

# Part IV — Required final deliverables

## 15. Written research diagnosis

Create a comprehensive report under `docs/investigations/`, with a suitable descriptive filename.

The report should answer:

**Q1. Why is the proposer validity rate only around 5%?**

Rank causes using actual measured evidence. Quantify the contribution of formatting, semantic validation, duplicate generation, model behavior, and policy learning wherever possible.

**Q2. Are the validation requirements scientifically necessary, or unnecessarily restrictive?**

Identify specific rules or representations that may be simplified without weakening safety or semantic verification.

**Q3. Is the 4B model a likely bottleneck?**

Give an evidence-based recommendation for whether 7B/8B inference-only testing should be our next GPU experiment.

**Q4. Is the proposer actually optimizing difficulty/gradient alignment, or mostly binary validity?**

Show how often the chosen reward changes actual normalized advantages.

**Q5. Why are student RL updates frequently uninformative?**

Quantify reward sparsity and identify whether increasing G from 4 to 8 is likely to help.

**Q6. What explains the stagnant development performance?**

Distinguish inadequate task exposure, ineffective gradients, policy drift, format problems, model limitations, and normal stochastic variation.

**Q7. Are our A–D comparisons scientifically identifiable with the current setup?**

Pay attention to proposer grouping, shared-versus-independent curricula, gating strength, and numerical sensitivity.

**Q8. What should we do next to maximize meaningful results before the final project deadline?**

Provide the shortest credible path to a defensible research result.

## 16. Concrete follow-up experiment plan

Create a separate plan under `docs/plan/` with:

- Ranked experiments.
- Exact hypotheses.
- Controlled variables.
- Datasets and splits.
- Model candidates.
- Sampling and group sizes.
- Metrics.
- Expected compute and resource requirements.
- Evaluation checkpoints.
- Interpretation of possible outcomes.
- Stop/continue conditions.
- Minimal implementation changes and dependencies.

Prefer a small number of high-information experiments over a broad grid search.

Distinguish:

1. Immediate CPU-only analysis.
2. Small inference-only diagnostic.
3. Small static-GRPO training sanity check.
4. Optional C/D gradient counterfactual.
5. Later full training experiments.

Make a specific recommendation about which experiment to run first after the CPU diagnosis.

## 17. Code and observability changes

Implement the minimum useful CPU-only analysis and observability improvements.

Provide:

- File paths changed.
- Commands run.
- Tests and results.
- Generated report paths.
- Existing evidence reused.
- Any unavailable evidence or remaining limitations.

Avoid changes to the active frozen scientific training code unless a confirmed defect requires a separately documented new study.

## 18. Final response format

When finished, report:

1. **Executive diagnosis:** The three most likely causes, with confidence and evidence.
2. **What is confirmed versus uncertain:** Include the key quantitative results.
3. **What should change:** Clearly distinguish necessary fixes, promising experiments, and unnecessary changes.
4. **Next three experiments:** Exact recommended order, estimated GPU cost, and decision criteria.
5. **Implementation completed:** Files, tests, and generated analysis.
6. **Remaining blockers:** Only genuine blockers, not hypothetical infrastructure concerns.

Make your conclusions scientifically rigorous, but do not be overly conservative to the point of preventing useful experimentation.

Our priority is to obtain meaningful and interpretable results efficiently—not to preserve a flawed experimental design merely because it was originally specified.

**Start by inspecting the repository and existing authoritative experiment records, then perform the highest-value analyses available without additional GPU training.**