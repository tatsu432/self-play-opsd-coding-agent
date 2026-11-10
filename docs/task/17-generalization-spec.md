# Task: Resolve Coding-Agent Generalization Bottlenecks and Conduct the Next Controlled Experiments

## 0. Your Role and Objective

You are the lead research engineer responsible for advancing my coding-agent self-play research project.

**Repository:** https://github.com/tatsu432/self-play-opsd-coding-agent

Your job is not simply to review the repository, propose a plan, or repeat previous debugging work. I want you to **investigate the current scientific bottlenecks, implement the necessary changes, conduct controlled experiments where feasible, analyze the results, and deliver a defensible research conclusion.**

Prioritize scientific validity, training stability, efficiency, and obtaining informative results over maximizing experiment complexity.

We have already spent substantial time and GPU compute debugging this project. Do not repeat previously resolved work without concrete evidence that the problem has returned.

You should make reasonable technical and experimental decisions autonomously. Document material choices and their reasoning, but do not stop to ask me about routine implementation details.

The principal research goal is to understand whether gated on-policy self-distillation (OPSD) helps coding agents learn through self-play, and under what conditions that improvement transfers to unfamiliar coding tasks.

The immediate goal is narrower:

**Determine why our trained coding agents perform well on MBPP+-derived repairs but almost completely fail on HumanEval+-derived repairs, test whether the repair interface is a major cause, and then conduct the most informative next comparison of student training objectives.**

Do not begin full Experiments C/D until the earlier problems have been meaningfully addressed.

---

## 1. Understand the Existing Research Before Making Changes

Read the current repository, including the latest main branch, relevant open branches, existing experiment code, and completed research documentation.

Start with:

- `README.md`
- `AGENTS.md`
- `docs/methods.md`
- `docs/investigations/task14-research-decision.md`
- `docs/investigations/task15-final-results.md`
- `docs/investigations/task16-generalization-checkpoint-decision.md`
- `docs/task_completion/15-implemented-stabilize-training-complete-ab.md`
- `docs/task_completion/16-implemented-generalization-checkpoint-analysis.md`
- `src/selfplay_opsd/environment.py`
- `src/selfplay_opsd/protocol.py`
- `src/selfplay_opsd/mutations.py`
- `src/selfplay_opsd/objectives.py`
- `src/selfplay_opsd/trainer.py`
- The existing rollout, evaluation, and analysis infrastructure.

Also examine the relevant configuration, test, and HPC execution files.

Do not assume that every historical document represents the current state. Use the latest verified execution records and completion reports as authoritative. Historical documents are useful for understanding previously attempted approaches and failures.

### 1.1 Project background

The project uses an LLM-based proposer, student, and privileged self-teacher to learn Python program repair through self-play.

The proposer generates executable bug-injection tasks using verified Python source programs.

The student attempts to repair the buggy program through a multi-turn coding-agent interface.

The environment verifies the resulting program with public and hidden tests.

The self-teacher receives privileged information, including the correct source code and inverse patch, and provides token-level feedback along the student's on-policy trajectory.

The original study defined:

| Configuration | Proposer reward | Student objective |
|---|---|---|
| A | Difficulty | GRPO + ungated OPSD |
| B | Difficulty | GRPO + gated OPSD |
| C | RL-gradient alignment | GRPO + gated OPSD |
| D | RL+OPSD gradient alignment | GRPO + gated OPSD |

The principal A/B study is now complete. Full C/D experiments have not been justified by the available evidence.

### 1.2 Verified A/B results

The completed Task 15 study uses three paired seeds: 2026, 2027, and 2028.

Results:

| Seed | A: MBPP+ /115 | B: MBPP+ /115 | A: HumanEval+ /116 | B: HumanEval+ /116 |
|---|---|---|---|---|
| 2026 | 61 | 70 | 2 | 3 |
| 2027 | 57 | 68 | 1 | 0 |
| 2028 | 64 | 71 | 1 | 0 |

The common base model achieved:

- MBPP+: 7/115
- HumanEval+: 2/116

B consistently exceeds A on MBPP+-derived repairs, with a three-seed mean paired parent-macro improvement of approximately 7.5 percentage points.

However, HumanEval+ performance is close to zero for both methods.

**These results do not prove that gating itself caused the improvement.** The proposer and student evolved together in separate stochastic training runs. The same nominal OPSD coefficient does not imply matched effective auxiliary gradients.

### 1.3 Important findings from Task 16

The most recent trajectory analysis recovered all 1,617 final/base evaluation trajectories and all 1,447 committed training/proposer trajectory pairs.

Across six trained HumanEval+ evaluations:

- 4,301 patch attempts were rejected.
- Only 13 patches were accepted.
- Context mismatches and malformed hunks dominated identifiable rejection details.
- Models made very few public test calls on HumanEval+.
- They often exhausted action budgets without successfully modifying the source.

In contrast, trained MBPP+ evaluation runs had approximately 93–111 accepted patches per run.

The HumanEval+ source and specification distributions also differ substantially:

- Mean source tokens: approximately 181.7 versus 48.5.
- Mean specification tokens: approximately 127.3 versus 20.0.
- Public test counts are similar.
- HumanEval+ contains longer documentation and more complex executable structures.

These differences are observational. Do not assume patch formatting is the only problem.

### 1.4 Additional scientific limitations

The proposer generates many valid tasks, but historical repetition remains common.

Only about 5.1–7.7% of attempted proposer groups contain distinct difficulty rewards among valid candidates.

Earlier RL-only versus RL+OPSD gradient-alignment comparisons produced extremely similar proposer optimizer updates, close to identical-reward repeat variation.

Consequently, the present evidence does not justify large C/D runs.

Task 14 also found that replacing entire function bodies improved edit validity but reduced verified repairs. **Do not simply repeat the same whole-function-body experiment.**

All six Task 15 arms selected their final checkpoints under the retrospective monitoring-development selector. There is no demonstrated terminal overtraining problem requiring another checkpoint search.

The existing 4B model has demonstrated meaningful learning on MBPP+-derived tasks. Do not increase the model size before evaluating the more immediate interface and distribution problems.

---

## 2. Overall Research Strategy

Execute the work in three phases:

**Phase 1 — Required:** Diagnose and experimentally test the repair-interface bottleneck.

**Phase 2 — Conditional but strongly preferred:** Run a controlled GRPO/OPSD comparison after the repair-interface result is understood.

**Phase 3 — Stretch:** Validate whether C/D gradient alignment has a sufficiently informative signal to justify subsequent training.

Phase 1 must be completed to the extent supported by available data and compute access. Phase 2 should proceed if Phase 1 yields an informative result and can be conducted within the budget. Phase 3 must not take resources away from completing the earlier work.

Do not launch new full-scale A/B/C/D runs as the default next step.

Create a new research branch and study identity. Preserve all historical experimental configurations, results, checkpoints, and immutable manifests.

For example:

`feat/17-repair-interface-generalization`

Use the repository's existing conventions where appropriate.

---

# PHASE 1 — Investigate and Improve the Repair Interface

## 3. Research Question

The principal hypothesis is:

**A significant fraction of the HumanEval+ generalization failure arises because the coding agent cannot reliably express otherwise potentially useful edits using the strict unified-diff tool protocol.**

We must distinguish three separate abilities:

1. Selecting a semantically useful repair.
2. Expressing the repair in a valid tool-action format.
3. Successfully applying and verifying the repair.

These are not equivalent.

A model can identify the correct change and fail to encode a patch. Conversely, the model can generate perfectly valid edits that are semantically incorrect.

Our objective is to determine whether simplifying the editing interface improves actual verified repair performance, not merely syntactic validity.

## 4. Task 1A — Audit the Existing Failure Mechanisms

First inspect the already recovered Task 16 trajectories and analysis outputs.

Reuse existing analysis rather than recreating it from scratch.

For rejected patches, distinguish:

- Invalid JSON or action schema.
- Invalid unified-diff headers.
- Incorrect hunk counts.
- Incorrect line offsets.
- Context mismatches.
- Invalid paths.
- Syntax-invalid proposed programs.
- API or import preservation failures.
- Repeated attempts that fail for the same reason.
- Valid edits that do not pass verification.
- Rejections with insufficient evidence to classify.

Identify the relative frequency of each failure type for MBPP+ and HumanEval+.

Examine failure patterns by:

- Source-token count.
- Specification-token count.
- Source complexity.
- Mutation operator.
- Available public cases.
- Number of preceding actions.
- Student checkpoint.
- Training method.

Most importantly, investigate whether many rejected patches appear to contain a **recoverable intended edit**.

For example, a diff may specify a valid old/new line replacement but contain an incorrect hunk offset.

A context mismatch may arise because the model reproduced the source incorrectly.

Do not infer that a proposed patch would have solved the task merely because its apparent intent looks reasonable.

For counterfactual replay:

- Use only source context legitimately available to the student at that point.
- Do not use the oracle solution to reconstruct a failed edit.
- Distinguish deterministic mechanical correction from semantic guesswork.
- Reject ambiguous reconstructions.
- Use the accepted isolated execution worker when testing generated code.
- Record both syntax validity and final semantic correctness.

The existing final evaluation data may be used for retrospective descriptive diagnosis, but not to tune a new method that is subsequently described as evaluated on an untouched final split.

Deliver an evidence-based failure taxonomy, not speculation.

## 5. Task 1B — Design a Better Editing Interface

Investigate at least two implementations.

### Interface 1: Existing strict unified diff

Preserve the historical interface and evaluator as the control.

The current implementation requires exact unified-diff syntax, including context lines, hunk counts, and source matching.

Do not weaken or silently modify this baseline.

### Interface 2: Exact unique text replacement

Implement a simple, deterministic editing action that avoids unified-diff hunk arithmetic.

Illustrative schema:

```json
{
  "action": "replace_text",
  "path": "src/solution.py",
  "old_text": "return x + 1",
  "new_text": "return x - 1"
}
```

The exact schema can be refined based on the existing architecture, but it should retain the following properties:

- Only an explicitly allowlisted source file is editable.
- `old_text` must match exactly one span of the current source.
- No match or multiple matches result in rejection.
- The action cannot modify tests or hidden verification data.
- The edited program must pass existing source-admission and API-preservation checks.
- Existing resource, syntax, execution-isolation, and file-access protections remain enforced.
- The action produces deterministic tool observations.
- No oracle solution is used to decide how to apply the edit.

Support multiline replacements safely.

Do not automatically fall back to the correct reference solution.

Do not allow arbitrary shell commands or unrestricted file writes.

The new interface should support localized fixes without requiring the model to regenerate the entire function body.

### Optional Interface 3: Deterministic diff normalization

If the failure analysis justifies it, consider a narrow correction layer for mechanically recoverable unified diffs.

This could address unambiguous offset or hunk metadata mistakes.

However:

- Do not perform fuzzy replacements when multiple plausible locations exist.
- Do not change the semantic content of added or removed lines.
- Do not silently replace malformed patches using an oracle.
- Label corrected patches separately from strict-parser successes.
- Do not merge this intervention with exact text replacement in the primary comparison.

Implement this only if it is likely to provide additional explanatory value.

### Important experimental distinction

A new tool schema changes model conditioning.

A model trained only on unified diffs may initially perform poorly when presented with a new editing tool.

Therefore, an unsuccessful zero-shot interface change does not establish that the new interface is inferior after training.

Design the investigation to separate:

- The existing model's ability to use a new tool immediately.
- Mechanical patch recoverability under unchanged model outputs.
- Performance after a limited, controlled adaptation to the new interface, if needed.

Preserve these distinctions in the analysis.

## 6. Task 1C — Test Interface Correctness

Add targeted automated tests for:

1. Unique single-line replacement.
2. Unique multiline replacement.
3. Missing source span.
4. Ambiguous source span.
5. Empty or malformed arguments.
6. Path traversal and forbidden paths.
7. Test-file modification attempts.
8. Syntax-invalid replacements.
9. API-changing replacements.
10. Import and source-policy violations.
11. Multiple sequential valid edits.
12. Rejected edits leaving source unchanged.
13. Deterministic application.
14. Tool observation and serialization consistency.
15. Existing historical unified-diff behavior remaining unchanged.

Test the tool independently from model generation.

Use small synthetic programs for unit tests.

Run relevant existing tests and integration checks. Repair actual failures, but do not create an excessively broad readiness process unrelated to the new interface.

## 7. Task 1D — Run a Matched Frozen-Model Interface Experiment

This is the most important empirical part of Phase 1.

Use available frozen student snapshots, including the base model and a representative trained B checkpoint if practical.

Use training-only tasks that do not overlap the frozen final evaluation or tuning splits.

Construct a task set that is sufficiently diverse to expose interface problems.

Include examples spanning:

- Short and long Python programs.
- Short and long specifications.
- Arithmetic and comparison mutations.
- Branching and loop-related programs.
- Single-function and helper-function programs where admitted.
- Tasks with different public-test informativeness.

Where possible, use paired controlled perturbations of source presentation or specification length while keeping the underlying bug and verification oracle unchanged.

Do not invent semantic requirements when constructing these perturbations.

### Experimental controls

Compare the interfaces with:

- Identical model checkpoints.
- Identical task identities and initial source states.
- Identical available tests and oracles.
- Identical turn and token budgets.
- Identical decoding configurations.
- Matched evaluation seeds where stochastic sampling is used.
- Identical execution isolation.
- No modification to the underlying correct program or hidden tests.

Greedy evaluation should be the primary comparison if consistent with the established evaluation protocol.

Sampled rollouts may be a secondary diagnostic.

The model's generated outputs will differ when prompts/tool schemas differ; matching RNG seeds does not make those trajectories identical. Do not describe this as a same-trajectory counterfactual.

A practical starting point is 40–80 diverse training-only tasks, using a balanced design that avoids treating repeated mutations from one parent as independent.

Adjust sample size based on measured uncertainty and compute availability. Prefer a smaller complete comparison to a large incomplete one.

### Metrics

Report:

- Action JSON validity.
- Edit-action acceptance.
- Distinct tasks with at least one successful edit application.
- Final verified repair success.
- Public test call count.
- Public-pass/hidden-fail behavior.
- Number of rejected edits per task.
- Repeated failed-edit patterns.
- Number of turns to first successful edit.
- Total turns and generated tokens.
- Context/turn/token termination.
- Mean and distribution of runtime.
- Actual allocated GPU-hours.

Do not use patch acceptance as a substitute for repair success.

Use paired task-level comparisons with parent/family-aware uncertainty where appropriate.

Show concrete, privacy-safe examples of failure and success modes using synthetic or appropriately sanitized source code.

### Phase 1 hypothesis test

Determine whether:

**H1:** The new interface produces more accepted edits.

**H2:** Increased edit acceptance results in more verified repairs.

**H3:** The benefit is larger for longer or structurally different tasks.

**H4:** Simplifying the edit interface reduces wasted turns and tool errors.

**H5:** The observed failure is mainly semantic rather than syntactic.

Treat these as empirical questions rather than conclusions.

---

## 8. Phase 1 Decision

After the interface experiment, make a clear research decision.

### Case A: Edit acceptance and verified repairs both improve

Select the simplest effective interface for the next controlled training experiment.

Do not over-optimize small improvements in parser permissiveness.

### Case B: Edit acceptance improves, but verified repairs do not

Investigate semantic repair quality, debugging behavior, specification comprehension, and task-distribution differences.

Do not claim that the generalization problem has been solved.

A small additional diagnostic may be justified, but avoid repeatedly cycling through minor prompt changes.

### Case C: The new interface does not improve acceptance

Check whether the failure is attributable to unfamiliar tool conditioning rather than inherently poor interface design.

If justified, perform one limited training-only adaptation study.

Do not immediately launch a complete self-play retraining.

### Case D: Neither interface nor limited adaptation helps

Treat this as an informative negative result.

Focus subsequent analysis on model capability, task-distribution shift, and training-data diversity.

Do not continue redesigning the editing interface indefinitely without a new falsifiable hypothesis.

Document the result and continue to the most informative feasible research question.

---

# PHASE 2 — Isolate the Effect of Gated OPSD

## 9. Research Question

The existing full A/B results suggest that gated OPSD may improve performance on narrow repair tasks.

However, we have not established whether this comes from:

- The gating mechanism itself.
- A smaller effective OPSD update.
- Different curriculum trajectories.
- Different numbers or distributions of informative student updates.
- Other stochastic training differences.

We want a cleaner comparison.

**Does gated OPSD improve verified coding-agent repair relative to GRPO alone and ungated OPSD when initial models, curriculum, optimization budgets, and execution conditions are controlled?**

Phase 2 should use the interface decision from Phase 1, provided that decision is sufficiently supported.

If Phase 1 is inconclusive, do not silently switch interfaces. Preserve the current interface and explicitly state the limitation.

## 10. Task 2A — Construct a Matched Student-Training Study

Use a frozen or pre-generated training curriculum.

The objective is to isolate the student loss without allowing separately evolving proposers to become an uncontrolled source of variation.

Use the existing valid task-generation infrastructure to create a training-only curriculum.

Requirements:

- No final evaluation tasks in training.
- Common task pool and task schedule.
- Common initial student weights.
- Common optimizer initialization.
- Common model and tokenizer.
- Common valid action interface.
- Common episode and generation budgets.
- Same verifiers and reward semantics.
- Same initial seeds, with clear accounting for subsequent stochastic trajectory divergence.

Do not use final MBPP+ or HumanEval+ outcomes to select hyperparameters.

Start from the existing validated student-training implementation. Do not create a second independent RL framework unless absolutely necessary.

## 11. Task 2B — Compare the Objectives

The preferred comparison is:

| Method | Objective |
|---|---|
| G | GRPO only |
| U | GRPO + ungated OPSD |
| B | GRPO + gated OPSD |
| U-calibrated | GRPO + ungated OPSD with initially matched auxiliary-gradient strength |

The exact historical student objective matters.

Inspect `docs/methods.md` and the executed Task 15 implementation.

The current stabilized variant uses an immutable rollout behavior denominator for PPO and a same-scorer privileged-versus-unprivileged log-probability contrast.

The ungated method uses the signed, clipped contrast without sigmoid weighting; the gated method applies the sigmoid gate.

These are sampled-token auxiliary policy-gradient-style terms, not necessarily exact full-vocabulary KL distillation.

Do not casually replace the implemented objective with textbook OPSD or KL and still call the result a controlled comparison.

Preserve the intended reward direction, stop-gradient behavior, token masking, and teacher-scoring semantics.

### Gradient calibration

For U-calibrated:

- Estimate initial auxiliary gradient strength on separate training-only calibration tasks.
- Choose the ungated coefficient to approximately match the initial aggregate gated auxiliary-gradient magnitude.
- Hold the chosen coefficient fixed during the comparison.
- Do not recalibrate using development or final evaluation outcomes.
- Record gradient norms and optimizer displacement separately.

Task 14 previously investigated a similar calibration. Reuse that knowledge rather than repeating unnecessary calibration work.

A previously reported calibrated coefficient is not automatically transferable to a new interface or curriculum. Verify the calibration under the new conditions.

### Efficiency

If compute resources are constrained, prioritize:

1. GRPO.
2. Gated OPSD.
3. Ungated OPSD.
4. Calibrated ungated OPSD.

Do not silently drop the GRPO-only baseline.

A useful complete smaller comparison is preferable to a larger incomplete factorial study.

## 12. Task 2C — Measure Learning and Optimization Health

For each training configuration, monitor:

**Task performance**

- Training verified repair success.
- Fixed-development verified repair success.
- Parent-macro accuracy.
- Public and hidden verification.
- Accepted and rejected edits.
- Public test execution.
- Repair quality by task complexity.

**RL learning signal**

- Mixed-reward groups.
- Uniform-success groups.
- Uniform-failure groups.
- Nonzero group advantages.
- Student optimizer steps.
- Approximate and observed KL statistics.
- PPO clipping fraction.
- Gradient norms.
- Nonfinite values and skipped updates.

**OPSD learning signal**

- Privileged/unprivileged log-probability gap.
- Fraction of positive and negative gaps.
- Sigmoid gate distribution.
- Effective auxiliary gradient norm.
- Auxiliary-to-RL gradient ratio.
- Gradient alignment between RL and auxiliary objectives.
- Auxiliary optimizer displacement.
- Teacher-scoring overhead.

**Efficiency**

- Attempted episodes.
- Generated tokens.
- Student updates.
- Allocated GPU-hours.
- Training throughput.
- Development accuracy versus episode and compute budget.

Do not interpret a negative auxiliary loss or larger KL as inherently better or worse without considering its mathematical definition.

Do not treat zero group-relative RL advantage as proof of zero total optimizer update; the auxiliary loss and optimizer state can still contribute.

### Required comparisons

Answer:

1. Does GRPO alone achieve comparable results?
2. Does gating help relative to ungated OPSD?
3. Does the advantage remain after initial gradient-strength calibration?
4. Does gating improve repair success or merely action validity?
5. Is its benefit concentrated on hard tasks or particular error categories?
6. Is it more sample-efficient?
7. Is it more GPU-compute-efficient?
8. Does OPSD create instability or harmful privileged-information dependence?
9. Do improvements persist on an independent development distribution?

Do not make causal statements beyond what the controlled design supports.

## 13. Task 2D — Training Scale and Extension

Begin with a bounded pilot.

Prefer three matched seeds where practical, but use a smaller pilot to verify that the objectives are implemented correctly and can learn.

Use the existing four-rollout grouping, optimizer, backend, and resource orchestration unless an actual defect requires a change.

Extend training only when:

- Execution and numerical checks pass.
- The model produces a meaningful learning signal.
- The pilot reveals an informative comparison.
- GPU allocation is available.
- The extension fits the budget.

If the first pilot is inconclusive because it is too short, estimate whether a longer controlled run is likely to resolve the uncertainty before proceeding.

Do not declare a method superior based on a single favorable seed.

Conversely, do not require statistically significant superiority before conducting an informative pilot.

---

# PHASE 3 — Conditional Investigation of Proposer Gradient Alignment

## 14. Research Question

The original C/D comparison asks:

Does the proposer generate more useful tasks when its reward aligns with the student's actual RL+gated-OPSD update instead of the RL-only update?

Before training C/D, establish whether those alignment rewards distinguish candidate tasks reliably.

## 15. Task 3A — Fixed-Snapshot Gradient Study

Using training-only candidate groups:

1. Freeze a student snapshot.
2. Sample candidate tasks from a frozen proposer.
3. Execute student rollouts with verified rewards.
4. Compute candidate gradients for RL only.
5. Compute candidate gradients for RL + gated OPSD.
6. Compute gradients for an appropriate disjoint training-guidance reference set.
7. Measure candidate/reference cosine alignment.
8. Compare within-group reward variation.
9. Compare resulting proposer advantages.
10. Compare actual proposer optimizer updates with identical initial model and optimizer states.

Use independent resampling or repeated fixed-snapshot probes to assess variance.

Preserve exact behavior-probability denominators, gradient masks, optimizer restoration, parameter ordering, and distributed reduction semantics.

The reference set must not contain final evaluation tasks.

Do not use the actual held-out test set as a gradient reference.

### Necessary measurements

- Fraction of valid candidates.
- Fraction of neutral alignment rewards.
- Fraction of proposer groups with distinct valid rewards.
- Gradient norm distributions.
- RL versus RL+OPSD gradient cosine.
- Proposer update cosine under C versus D.
- Identical-reward repeat variability.
- Reference-gradient split-half stability.
- Task diversity and historical repetition.
- Measured runtime and GPU cost.

The important quantity is not whether the two gradient formulas differ numerically.

It is whether their differences provide a reproducible and useful optimization signal.

## 16. Phase 3 Decision

Proceed to a bounded C/D training pilot only if:

- Alignment rewards are nondegenerate on a meaningful fraction of candidate groups.
- Differences between C and D exceed numerical and repeated-sampling variation.
- The reference-gradient direction is sufficiently stable.
- The resulting proposer updates are meaningfully different.
- The proposed comparison can be run at reasonable compute cost.

If these conditions fail, do not force C/D training.

Instead, document why the gradient-guided proposer signal is currently uninformative.

If necessary, recommend a future study on proposal diversity or task learnability without implementing an entirely new proposer system now.

---

# ENGINEERING AND EXECUTION REQUIREMENTS

## 17. Preserve Existing Scientific Evidence

The historical Task 15 study identity is:

`repair-selfplay-2026-task15-ab-v2`

The frozen scientific execution was:

`0ac77cde4bc5f70ba2b3eb9e457749e3106a2bc5`

Do not mutate this study or its completed results.

Use distinct study identities, configuration hashes, output directories, checkpoint metadata, and W&B runs for new experiments.

Historical outputs must remain reproducible.

Do not mix trajectories generated under different tool interfaces or loss definitions into a single purportedly homogeneous training run.

Maintain explicit provenance for any reused dataset, checkpoint, or runtime.

## 18. Data and Evaluation Integrity

Keep training, development, reference, and final evaluation data separated.

Important: We have already examined aggregate and some per-task results for the existing MBPP+ and HumanEval+ final splits. They are no longer genuinely unseen to the research-development process.

Therefore:

- Do not tune the new interface against those final outcomes.
- Use training-only data for initial interface experiments.
- Use a separate development set for model selection.
- Preserve parent/family disjointness.
- Do not introduce task-oracle information into student-visible context.
- If a new confirmatory evaluation is necessary, construct and freeze a genuinely separate evaluation set before examining its outcomes.
- If the current final splits are evaluated again after development, explicitly label the results as retrospective or exploratory, not untouched confirmatory testing.

Distinguish MBPP+-derived repair from official MBPP+ code generation, and similarly for HumanEval+.

## 19. Yale HPC and Compute Efficiency

Use the established Yale Bouchet execution infrastructure.

The repository already supports isolated generated-code execution, Slurm orchestration, GPU training, checkpoints, and telemetry.

Reuse these capabilities.

Do not reintroduce the old requirement that the user must be able to configure a specific 32-process cgroup limit on Bouchet. The existing project previously resolved its execution route using accepted isolation and resource controls.

However, do not weaken verified isolation, source restrictions, test secrecy boundaries, or cluster policies merely to make a run start.

Use the already accepted isolated worker and resource setup.

### Compute budget

Treat this prompt as approval to perform the following bounded research work, subject to Yale policies and available account allocations:

- Phase 1: up to **10 newly allocated H200 GPU-hours**.
- Phase 2: up to **24 additional H200 GPU-hours**, only after an informative Phase 1 result and successful small pilot.
- Phase 3: diagnostic CPU work and small existing-snapshot calculations where possible; no full C/D training under this authorization.

These are ceilings, not spending targets.

Count allocated GPU time, including failed or interrupted allocations. Avoid duplicating existing results.

If the experiment can answer its question for less compute, use less.

If the estimated experiment cannot fit the budget, scale the design down responsibly while preserving the most important controls.

Do not consume unused GPU budget simply because it remains available.

Before launching:

- Check existing live jobs and ledgers.
- Avoid duplicate submissions.
- Confirm checkpoint/runtime compatibility.
- Reuse known working resource profiles.
- Verify that the new implementation passes focused tests.

Use inexpensive CPU analysis whenever possible.

Do not launch expensive parallel jobs merely to reduce wall-clock time when the scientific result can be obtained more efficiently.

If Yale access is temporarily unavailable, finish all feasible local implementation, tests, and analysis. Report the precise external dependency and a ready-to-run command rather than repeatedly attempting the same inaccessible operation.

## 20. Avoid Excessive Debugging and Readiness Work

We have repeatedly lost time because the project became stuck on readiness checks and increasingly elaborate safety gates.

Use the following principle:

**Only an actual blocker to correctness, scientific validity, safe execution, or reproducibility should prevent a bounded experiment.**

Examples of genuine blockers:

- Unsafe execution of generated code.
- Incorrect reward computation.
- Incorrect probability denominators.
- Data or oracle leakage.
- Nonfinite gradients that invalidate training.
- Broken checkpoint restoration.
- Corrupt scientific identity.
- Incompatible runtime or resource configuration.
- Failed essential functionality tests.

Examples that are not sufficient reasons to indefinitely delay progress:

- A possible theoretical failure without supporting evidence.
- An unimplemented optional diagnostic.
- Missing decorative dashboards.
- A warning already investigated and determined not to invalidate the run.
- Desire for another redundant infrastructure audit.
- A slightly imperfect experiment that can still provide an interpretable bounded result.

When a diagnostic contradicts assumptions, investigate the smallest plausible cause.

Do not keep adding new experimental requirements as a substitute for running the experiment.

After resolving a bug, run the relevant regression tests and continue.

Avoid broad refactors unless directly necessary.

## 21. Logging and Observability

Use W&B and existing event/analysis infrastructure.

For each experiment, record:

- Study and run identity.
- Code revision.
- Scientific configuration hash.
- Model/tokenizer revision.
- Dataset/split identity.
- Seed.
- Interface version.
- Student objective.
- Proposer configuration.
- Episodes and optimizer steps.
- Training success and development accuracy.
- Edit acceptance/rejection statistics.
- Gradient and update diagnostics when applicable.
- GPU-hours and throughput.
- Terminal execution status.

W&B is for live monitoring and convenient comparison.

Local committed scientific events and verified checkpoints remain authoritative if W&B delivery fails.

Do not upload raw private oracles, training secrets, or sensitive source artifacts unnecessarily.

Do not let a transient W&B delivery failure invalidate an otherwise correct scientific run.

## 22. Reproducibility and Testing

Follow the repository's established Python tooling.

Typical entry points include:

```bash
uv sync --frozen
uv run selfplay doctor --settings configs/local.yaml --isolation
uv run selfplay smoke --output runs/local-smoke
```

Inspect the current CLI before using these commands; adapt if the repository has changed.

Also run appropriate:

- Unit tests.
- Focused integration tests.
- Static type checks.
- Ruff lint/format checks.
- Existing CI checks.
- Relevant GPU/backend checks, only where required.

Do not execute generated or otherwise untrusted Python in an ordinary unsandboxed host process.

Tests should distinguish implementation correctness from scientific effectiveness.

A syntactically valid interface is not evidence of improved model capability.

A successful smoke test is not evidence of training improvement.

## 23. Efficient Implementation Practices

Prefer the smallest architecture change that supports the required experiment.

In particular:

- Reuse the existing `RepairEnvironment`.
- Reuse source validation and task-oracle execution.
- Reuse the existing rollout and training pipeline.
- Add new edit protocols through a clear configurable interface.
- Keep old protocol behavior backward compatible.
- Reuse existing parent/family-aware analysis tools.
- Avoid duplicated model-loading and GPU orchestration logic.
- Maintain strict typed interfaces and deterministic serialization.
- Keep generated experiment artifacts out of Git unless they are privacy-safe summaries.

Do not rewrite the project around a new agent framework.

Do not introduce an external model or teacher unless specifically justified as a separate study.

Do not increase the student model size in the primary experiment.

## 24. Scientific Analysis Requirements

For each completed comparison, provide:

1. The primary hypothesis.
2. The precise intervention.
3. What was held fixed.
4. What changed.
5. The number of independent tasks and parents.
6. The number of training seeds.
7. The measured outcomes.
8. Paired effect estimates.
9. Appropriate uncertainty estimates.
10. GPU-compute and sample-efficiency comparisons.
11. Failure-mode analysis.
12. Limitations and alternative explanations.
13. A clear go/no-go decision.

Use parent/family-aware resampling for repeated mutations.

Separate task-sampling uncertainty from training-seed variation.

Avoid treating multiple actions or rollouts within one task as independent evaluation samples.

When reporting percentages, include numerators and denominators.

Do not invent missing observations.

Do not present preliminary single-seed findings as conclusive.

If an experiment shows no improvement, report that result without searching for a favorable checkpoint or changing the comparison post hoc.

---

# REQUIRED OUTPUTS

## 25. Code and Documentation

Create appropriate Task 17 documentation, including:

`docs/task/17-repair-interface-spec.md`

`docs/plan/17-repair-interface-plan.md`

`docs/investigations/task17-repair-interface-results.md`

`docs/task_completion/17-implemented-repair-interface.md`

These are suggested filenames; follow the repository's conventions if they have evolved.

For Phase 2, create a separate study specification and results document if performed.

The results documentation must include actual evidence, not placeholder tables.

Add scripts for deterministic analysis and figure generation.

Produce at least:

- Interface acceptance by task distribution.
- Verified repair success by interface.
- Failure taxonomy by interface and task characteristics.
- Repair success versus source/specification length where supported.
- Student learning curves for the controlled objective comparison, if run.
- Sample-efficiency and GPU-hour comparisons, if run.

Include commands to reproduce the analysis from saved manifests and results.

## 26. Deliver a Research Decision

At completion, write a concise scientific decision document answering the following questions.

### Generalization

- What is the strongest supported cause of poor HumanEval+ transfer?
- How much of the observed failure is associated with patch rejection?
- Did simplifying the interface improve acceptance?
- Did it improve verified repair success?
- Does the evidence support an interface-specific problem, a semantic capability problem, or both?

### Student optimization

- Does gated OPSD outperform GRPO alone?
- Does it outperform ungated OPSD?
- Is the improvement robust to matching initial auxiliary-gradient strength?
- Is the result consistent across seeds?
- Is gating more efficient at matched episodes, tokens, or GPU-hours?

### Proposer optimization

- Does the current proposer produce sufficiently diverse and informative tasks?
- Is gradient alignment reliably distinguishable from difficulty-based proposer updates?
- Is a C/D pilot scientifically justified?
- If not, what specific improvement in candidate quality or gradient measurement would change that decision?

### Overall contribution

State the strongest defensible scientific conclusion supported by the experiments.

It is acceptable for the conclusion to be negative or mixed.

Do not exaggerate transfer, causal effects, or statistical certainty.

## 27. Final Handoff

In your final response to me, include:

- A short summary of implemented changes.
- A table of completed experiments.
- The principal quantitative results.
- Which hypotheses were supported or rejected.
- Any unresolved limitations.
- Actual new GPU-hours consumed.
- Commit hashes and relevant branch/PR links.
- Paths to research reports and figures.
- Whether the next action should be further training, redesign, or final report writing.

If compute or access prevents completing an experiment, distinguish clearly between completed implementation, completed CPU analysis, successful pilot, and unexecuted training.

Do not say an experiment was completed unless actual verified results exist.

---

# PRIORITY AND EXECUTION POLICY

Follow this order:

1. Inspect the current repository and recover authoritative existing evidence.
2. Perform the focused repair-failure analysis.
3. Implement the new exact-replacement interface and necessary tests.
4. Run the matched frozen-model comparison.
5. Analyze whether acceptance improvements translate to actual repair improvements.
6. Make and document the Phase 1 interface decision.
7. If justified, construct and execute the bounded GRPO/OPSD comparison.
8. Analyze gating, effective auxiliary-gradient strength, and efficiency.
9. Only if resources remain and the signal is informative, investigate C/D through fixed-snapshot counterfactuals.
10. Produce the final research decision and reproducible artifacts.

**Do not complete only steps 1–3 and stop because additional work might be complicated.**

When a bounded experiment is scientifically useful and the execution environment is ready, run it.

When it is not useful, explain the evidence and move to the next meaningful research question instead of endlessly debugging the same issue.

Do not ask for confirmation at every intermediate step.

Do not spend the majority of the time building infrastructure.

Do not create new scientific hard blockers without evidence.

Do not change multiple scientific variables simultaneously and then claim that the observed difference is attributable to one of them.

**Success means that we obtain an evidence-backed explanation of the current generalization failure, a tested repair-interface intervention, and—if supported by the Phase 1 outcome—a meaningful controlled comparison of GRPO and gated/ungated OPSD.**

The objective is not to make every experiment look positive. It is to make the research scientifically defensible, reproducible, and informative while using our remaining time and compute efficiently.

Begin working on the repository now. Implement and execute the plan rather than only returning another proposal.
