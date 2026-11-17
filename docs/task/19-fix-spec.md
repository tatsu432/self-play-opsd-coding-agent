# CODEX TASK: Complete the Qwen3.5 Shared-Parameter Self-Play Research Experiment

## 0. Your mission

You are the primary research engineer responsible for completing my Yale CPSC 5880 *Advances in Frontier AI Models* final project.

Repository:

https://github.com/tatsu432/self-play-opsd-coding-agent

**Your responsibility is to implement, validate, execute, analyze, and document the experiment—not merely prepare a plan or produce another readiness report.**

The central research question is:

> When does gated on-policy self-distillation improve agentic coding self-play, and should the proposer be rewarded according to the student's RL gradient alone or its combined RL-plus-distillation gradient?

We will conduct five experimental configurations (one essential control and A–D) with a single training seed. Additional seeds are optional and should not delay completion of all five configurations.

We have already spent considerable time on training instability, interface problems, and numerical diagnostics. You must carefully understand that history, avoid repeating completed work, fix remaining fundamental problems, and prioritize obtaining scientifically interpretable experimental results.

The new experiment must use:

- **Model:** `Qwen/Qwen3.5-4B`
- **Optimization:** Dr. GRPO, replacing conventional GRPO
- **Parameter sharing:** One shared LoRA adapter for proposer and student
- **Student inference:** Thinking mode enabled
- **Proposer:** More expressive, verifiable task generation
- **Student environment:** Native tool calling with robust editing
- **Training seeds:** One seed, initially `2026`
- **Methods:** Dr. GRPO-only control, A, B, C, and D
- **Compute:** Existing authorized Yale HPC resources, with measurable and bounded allocations
- **Observability:** Weights & Biases plus complete local experiment records

Do not run additional Qwen3-versus-Qwen3.5 model-selection experiments. We have selected Qwen3.5 as the new backbone.

Do not turn this task into an open-ended architecture search.

Make implementation decisions based on correctness, established research methods, measured bottlenecks, and efficiency. Document significant decisions without unnecessarily asking for confirmation.

## 1. Understand the existing repository before modifying it

First, inspect:

- The current Git branch and commit.
- `AGENTS.md` and applicable repository instructions.
- `README.md`, `docs/methods.md`, and `docs/operations.md`.
- Existing training, evaluation, proposer, student, environment, and HPC infrastructure.
- Existing scientific configurations and tests.
- Historical experiment results and recorded failure diagnoses.

At the time of this specification, `main` is at:

`c7345b3b9899c3697bdd79d2350760a8087e8c3f`

Check whether more recent changes exist.

Particularly important files include:

- `docs/investigations/task15-final-results.md`
- `docs/investigations/task16-generalization-checkpoint-decision.md`
- `docs/investigations/task17-research-decision.md`
- `docs/investigations/task17-objective-results.md`
- `docs/investigations/task18-results.md`
- `docs/investigations/task18-research-decision.md`
- `docs/task_completion/18-implemented-interface-signal-bottlenecks.md`
- `src/selfplay_opsd/objectives.py`
- `src/selfplay_opsd/trainer.py`
- `src/selfplay_opsd/policy.py`
- `src/selfplay_opsd/verl_backend.py`
- `src/selfplay_opsd/vllm_worker.py`
- `src/selfplay_opsd/task18_protocol.py`
- `src/selfplay_opsd/structured_mutations.py`
- `src/selfplay_opsd/environment.py`
- `src/selfplay_opsd/evaluation.py`

These paths are starting points. Search the repository for other implementation dependencies.

### Historical findings you must account for

**Task 15 — Original A/B experiment**

Three seeds per configuration were completed using Qwen3-4B-Instruct-2507, independent proposer/student LoRA adapters, and the original editing interface.

The base achieved:

- MBPP+-derived ID repairs: 7/115
- HumanEval+-derived external repairs: 2/116

A achieved 57–64/115 ID repairs, and B achieved 68–71/115. External repairs remained approximately 0–3/116.

Therefore, substantial ID learning occurred, but meaningful external generalization was not demonstrated.

**Task 16 — Generalization diagnosis**

The strict unified-diff interface rejected many model-generated patches, especially on HumanEval+-derived tasks.

The HumanEval+-derived tasks also differ from the MBPP+-derived training distribution in source length, specification length, structural complexity, and other properties.

Do not assume all OOD failures were caused by the interface alone.

**Task 17 — Controlled student-objective comparison**

An 80-episode, fixed-curriculum study compared GRPO-only, gated OPSD, nominal ungated OPSD, and calibrated ungated OPSD over three seeds.

The small development set did not establish a robust gated-OPSD advantage over GRPO-only. Nominal ungated OPSD performed competitively.

These historical controls already exist. Do not duplicate them under the old experimental setup.

**Task 18 — Native interface and numerical investigation**

Qwen-native exact-text tool calling improved repairs substantially on previously inspected development tasks:

- Native base: 27/40
- Native trained: 31/40
- Plain-text JSON base/trained: 19/40 and 21/40

Native calling is therefore the preferred starting interface.

However, no new HumanEval+-derived generalization result was established using this interface.

Task 18 also found material optimizer-update sensitivity when comparing the true rollout behavior probabilities against a hypothetical substituted scoring denominator.

One restored Adam displacement changed by approximately 21.4% under denominator substitution, compared with approximately 0.4% variation between repeated identical operations.

**Important:** This does not justify replacing the actual behavior probabilities. Original behavior-policy probabilities must remain the PPO importance-sampling denominator.

The remaining issue is to understand and qualify differences between rollout and scoring implementations, particularly where they materially affect clipping, gradients, and optimizer updates.

### Repository handling

Create a dedicated new feature branch for this study.

Preserve historical code, configurations, checkpoints, results, and experiment identities. Do not retroactively modify old runs.

Implement the new study cleanly, reusing functioning components rather than rewriting the repository from scratch.

Use small, logically separated changes and meaningful automated tests.

## 2. Freeze the new scientific experiment

The new experiment consists of:

| Arm | Proposer reward | Student objective |
|---|---|---|
| Control | Difficulty-based | Dr. GRPO only |
| A | Difficulty-based | Dr. GRPO + ungated OPSD |
| B | Difficulty-based | Dr. GRPO + gated OPSD |
| C | RL-gradient alignment | Dr. GRPO + gated OPSD |
| D | RL+OPSD-gradient alignment | Dr. GRPO + gated OPSD |

These five configurations are required.

The scientific comparisons are:

1. **Control vs. A/B:** Does additional self-distillation improve over verifier-driven RL alone?
2. **A vs. B:** Does token-level gating improve ungated OPSD?
3. **B vs. C:** Does rewarding the proposer using RL-gradient alignment outperform difficulty-based rewards?
4. **C vs. D:** Does incorporating the OPSD objective into the proposer's gradient-alignment reward improve task generation and downstream student learning?

All five configurations must use the same:

- Qwen3.5 checkpoint and tokenizer revision.
- Shared-parameter architecture.
- LoRA rank and target modules.
- Dr. GRPO implementation.
- Initial model and adapter weights.
- Training data and eligible parent population.
- Student and proposer interface definitions.
- Thinking-mode configuration by role.
- Evaluation data and rules.
- Training seed.
- Initial training budget.
- Proposer/student loss-scaling policy.
- Student rollout group size and episode budget.

Different online proposer trajectories are expected. Do not force the evolving curricula to be identical between configurations.

Avoid varying other scientific factors between methods.

Use one seed initially. Preserve the ability to execute additional seeds later without implementing an entirely separate workflow.

For the final statistical analysis, clearly distinguish training-seed uncertainty, task/family uncertainty, and within-run learning variation.

## 3. Migrate the model to Qwen3.5-4B

Use the instruction-tuned model:

https://huggingface.co/Qwen/Qwen3.5-4B

Do not accidentally use `Qwen3.5-4B-Base` or the older Qwen3-4B-Instruct model.

Pin the exact model and tokenizer revisions, and store their identities in the experiment manifests.

Review the official model card and current compatible implementations of:

- Hugging Face Transformers
- PEFT
- vLLM
- PyTorch
- FSDP
- Verl, if retained
- Relevant kernels and attention implementations

Qwen3.5 has a different architecture from Qwen3, including Gated DeltaNet components. The current repository contains architecture-specific assumptions, such as Qwen3 decoder-layer wrapping and particular LoRA targets.

Audit these assumptions rather than simply replacing the model name.

### Model compatibility requirements

Ensure:

1. Correct text-only forward and generation behavior.
2. Correct native tool-call serialization.
3. Correct thinking-mode behavior.
4. Compatible and verified LoRA target modules.
5. Working forward and backward passes.
6. Stable FSDP or alternative supported distributed training.
7. Correct adapter loading and synchronization with the rollout engine.
8. Consistent token IDs, attention masks, position handling, and causal log-probability scoring.
9. Full checkpoint save/load and exact scientific-state restoration.
10. Finite gradients and actual optimizer updates on target hardware.

Prefer retaining the existing stack if it works reliably.

If the pinned Verl/FSDP/vLLM implementation cannot support Qwen3.5 correctly without extensive modification, choose a simpler supported training implementation, such as a direct PyTorch/PEFT/Accelerate path, provided it supports the required algorithms.

Do not spend the entire research budget repairing an unnecessarily complicated framework.

A clean, correct, moderately slower implementation is preferable to an unreliable high-throughput implementation.

However, do not abandon efficiency without measuring it.

Do not silently fall back to Qwen3-4B.

### LoRA

Use one shared trainable LoRA adapter attached to the Qwen3.5 backbone.

Start from the historical rank-32, alpha-64 design where architecture-compatible, but inspect actual available modules and parameter counts.

Target meaningful attention and feed-forward or DeltaNet projections that are compatible with stable PEFT training.

Do not copy an incompatible target-module list without verification.

Keep the backbone frozen unless a small, explicitly justified compatibility change is necessary.

The model should not be duplicated in memory simply to implement student, proposer, or teacher roles.

## 4. Enable thinking mode correctly

### Student: thinking ON

Enable Qwen3.5 native thinking mode during student training and evaluation.

Use the official chat-template interface, including `enable_thinking=True` where supported.

Do not use old Qwen3 `/think` or `/no_think` prompt commands.

The student should be able to reason about the bug, inspect code, select tools, run tests, and make a repair.

The thinking trace is part of the model-generated trajectory. Preserve sampled token IDs and corresponding log-probabilities for training.

Separate the following token types for observability:

- Reasoning tokens.
- Tool-call tokens.
- Other sampled assistant tokens.
- Forced template/control tokens.

Do not train on tool observations, externally supplied prompts, padding, or forced tokens.

Train on appropriate model-sampled assistant tokens, including reasoning where present, using a clearly defined loss mask.

### Important Qwen3.5 history behavior

The official model guidance recommends omitting previous thinking content from subsequent conversational history.

Implement the official conversation serialization consistently.

Preserve complete raw sampled trajectories in private training records, but construct subsequent prompt histories according to the supported Qwen3.5 chat template.

Ensure that the training scorer reconstructs exactly the model-conditioning context used at rollout time.

Do not accidentally score tokens against a different history serialization.

### Proposer: configurable thinking

For proposer generation, start with non-thinking mode to reduce unnecessary token costs, since its initial task may be expressible as a short structured tool call.

However, because we are expanding the proposer to generate more complicated bugs, support thinking mode for the proposer as a configuration option.

During a small training-only compatibility pilot, determine whether non-thinking proposals produce adequate valid-task diversity.

If non-thinking mode is clearly unsuitable, enable proposer thinking **before freezing the main experiment**, and then use the same setting in all five arms.

Do not switch proposer thinking on or off midway through the main study.

### Generation parameters

Use the model developer's recommended sampling settings as the starting point.

For thinking-based precise coding, the official recommendation includes:

- Temperature 0.6.
- Top-p 0.95.
- Top-k 20.
- No minimum-p truncation.
- No presence penalty.
- Repetition penalty 1.0.

The settings must be supported by the actual backend and recorded in every run.

For evaluation, choose one fixed, reproducible decoding protocol before examining final outcomes. Do not assume greedy decoding is necessarily optimal for a thinking model.

Preserve the chosen setting across all checkpoints and experimental arms.

### Token budgets

Thinking mode increases token consumption. The historical per-turn and per-episode limits may be insufficient.

Measure the actual distributions of thinking lengths, tool-call lengths, episode turns, truncation, and successful repairs during the initial pilot.

Choose feasible budgets that allow reasoning and tool use without excessively wasting GPU time.

Use identical budgets for all main experimental arms.

Record truncation separately from semantic failure.

Do not interpret an unfinished thinking trace as evidence that the model cannot repair the task.

## 5. Implement a single shared proposer–student policy

This is a required change from the original architecture.

The previous implementation used two independent LoRA adapters over a frozen shared backbone.

The new implementation must use:

\[
\pi_\theta^{P}(\cdot\mid z)
\]

and

\[
\pi_\theta^{S}(\cdot\mid x)
\]

as different role-conditioned distributions of the **same trainable parameters** \(\theta\).

The proposer, student, and privileged self-teacher are different conditioning contexts, not independently trained adapters.

### Joint optimization

Define the student objective:

\[
\mathcal L_S(\theta)
=
\mathcal L_{\mathrm{DrGRPO}}^S(\theta)
+
\lambda_{\mathrm{SD}}\mathcal L_{\mathrm{OPSD}}(\theta).
\]

For the control, \(\lambda_{\mathrm{SD}}=0\).

Define the proposer objective:

\[
\mathcal L_P(\theta)
=
\mathcal L_{\mathrm{DrGRPO}}^P(\theta).
\]

The joint objective is:

\[
\boxed{
\mathcal L_{\mathrm{joint}}(\theta)
=
\mathcal L_S(\theta)+
\alpha\mathcal L_P(\theta)
}
\]

where \(\alpha\) is a fixed proposer-loss coefficient.

At every training iteration:

1. Freeze the current policy snapshot for rollout collection.
2. Generate proposer candidates from that snapshot.
3. Validate proposed tasks.
4. Generate student repair trajectories from the same snapshot.
5. Compute verifiable environment rewards.
6. Compute privileged self-teacher scores on the student trajectories using that snapshot.
7. Compute proposer rewards, including reference/candidate gradients for C/D.
8. Construct the student and proposer training losses.
9. Accumulate both losses against the same parameter state.
10. Apply exactly one joint optimizer update.
11. Commit the resulting state transactionally.

Do not perform an intermediate student optimizer update before computing the proposer objective.

Do not let gradient-based proposer reward calculation accidentally update parameters.

Use a single optimizer and one set of optimizer states.

The gradients of the student and proposer losses should be independently measurable for diagnostics, but their final update must be joint.

### Relative gradient scaling

A shared model introduces the possibility that the student and proposer gradients have very different magnitudes.

Measure their norms on a small training-only calibration batch.

Select and freeze a reasonable \(\alpha\) before main training. Similarly, choose the common OPSD coefficient using a transparent, fixed calibration protocol.

Do not tune these coefficients independently for A–D based on development or final evaluation outcomes.

Log:

\[
\|g_S\|,\quad
\|g_P\|,\quad
\cos(g_S,g_P)
\]

and the actual combined-update norm.

This will help determine whether the two roles show gradient synergy, orthogonality, or interference.

Gradient conflict is a hypothesis to measure, not a reason to revert automatically to separate adapters.

## 6. Replace GRPO entirely with Dr. GRPO

Implement Dr. GRPO for the student and proposer RL objectives.

Reference:

https://arxiv.org/abs/2503.20783

Do not use conventional group-standard-deviation normalization in the new experiments.

For a task with \(G\) sampled trajectories and rewards \(r_1,\ldots,r_G\):

\[
A_i=r_i-\frac{1}{G}\sum_{j=1}^{G}r_j.
\]

No division by group reward standard deviation.

Let \(\rho_{it}\) denote the PPO importance ratio for sampled token \(t\) of trajectory \(i\).

Use the clipped policy-gradient surrogate with a fixed token normalization constant \(C\):

\[
\mathcal L_{\mathrm{DrGRPO}}
=
-\frac{1}{G}
\sum_{i=1}^{G}
\frac{1}{C}
\sum_{t\in\mathcal T_i}
\min
\left[
\rho_{it}A_i,\,
\operatorname{clip}(\rho_{it},1-\epsilon,1+\epsilon)A_i
\right].
\]

Here, \(\mathcal T_i\) contains the appropriate sampled assistant-token positions.

For the multi-turn student, apply the fixed normalization across the **complete episode**, not separately to each turn.

Choose a fixed \(C\) based on the configured episode token budget and apply it consistently across methods and iterations.

Do not substitute the actual response length as the denominator.

An empty trajectory should contribute zero while retaining the appropriate group accounting.

All-equal reward groups should correctly produce zero centered RL advantages.

The proposer should use the same Dr. GRPO principle over complete proposal generations, with its own appropriately defined fixed proposal-token normalization.

Maintain correct task/episode/proposer-group weighting.

### Essential implementation tests

Implement independent reference tests proving:

- Equal rewards yield zero RL advantages.
- Mixed binary rewards have correctly centered advantages.
- Scaling does not depend on group standard deviation.
- Short and long responses use the same fixed normalization.
- Multi-turn episodes are normalized as complete episodes.
- Padding, observations, and forced tokens receive no gradients.
- PPO clipping behaves correctly for positive and negative advantages.
- Gradients remain finite.
- The implemented objective matches an independently written small-tensor reference.
- Gradient accumulation produces the intended joint objective.

Do not rely on framework defaults; verify the actual mathematical loss.

## 7. Keep gated and ungated OPSD scientifically comparable

The self-teacher is the same shared policy snapshot conditioned on privileged, verifier-grounded information.

For a student rollout token \(y_t\), define:

\[
\Delta_t=
\operatorname{stopgrad}
\left[
\log\pi_T(y_t\mid x,c,y_{<t})
-
\log\pi_S(y_t\mid x,y_{<t})
\right].
\]

Here, \(c\) is training-only privileged context, such as the original correct implementation and verified initial edit information.

The teacher scores the **same student-generated target tokens**, without generating a separate teacher trajectory.

The teacher must receive the same actual student history, except for its additional privileged prefix.

For ungated OPSD:

\[
w_t=1.
\]

For gated OPSD:

\[
w_t=\sigma(\beta\Delta_t).
\]

Use a bounded signed teacher–student gap, for example:

\[
\widetilde{\Delta}_t
=
\operatorname{clip}(\Delta_t,-\delta,\delta).
\]

A sampled-token auxiliary objective can then take the form:

\[
\mathcal L_{\mathrm{OPSD}}
=
-\frac{1}{C_{\mathrm{SD}}}
\sum_t
\operatorname{stopgrad}
(w_t\widetilde{\Delta}_t)
\log\pi_\theta(y_t\mid x,y_{<t}).
\]

Implement a clearly defined, consistent aggregation over episodes and tasks.

Do not let response-specific length normalization silently reintroduce the length bias that Dr. GRPO is intended to remove. If the auxiliary objective uses a different normalization for a scientifically justified reason, document it, calibrate its scale, and retain it identically across all arms.

The ungated and gated variants must differ only in their gating function, not in arbitrary coefficients, data processing, or loss masking.

### Privileged-information leakage

Because the self-teacher knows information that is unavailable to the student during evaluation, explicitly monitor:

- Teacher/student log-probability gaps.
- Positive versus negative gap fractions.
- Gate activation distribution.
- OPSD gradient norms.
- RL versus OPSD gradient cosine similarity.
- Relative influence of OPSD on actual updates.
- Divergence across reasoning and tool-call tokens.
- Signs of privileged-context leakage into unprivileged generations.

Do not assume dense teacher supervision is always correct.

Do not allow the privileged context into student observations, test-time prompts, or the model's external tools.

## 8. Expand proposer expressiveness without sacrificing verification

The current structured proposer mostly selects a predefined AST mutation site and operator.

This is robust but potentially too narrow for generalizable coding-agent learning.

Implement a more expressive, executable proposal interface.

### Supported proposal types

At minimum, support:

**A. Structured AST mutation**

Retain the existing deterministic structured-mutation implementation as a reliable option.

**B. Flexible source edit**

Allow the proposer to specify a bounded modification to the source program, preferably using a native tool call with exact-text replacement or another reliably applicable structured representation.

For example, the model can choose an exact current source span and its replacement, while deterministic code handles locating and applying the edit.

Do not require the proposer to generate unified-diff headers, hunk counts, line numbers, JSON escapes, and other predictable formatting when the harness can produce them.

### Greater semantic diversity

The expanded representation should be capable of expressing a meaningful variety of repair tasks, including:

- Arithmetic and comparison mistakes.
- Boundary-condition errors.
- Indexing or slicing errors.
- Missing or incorrect conditional logic.
- Incorrect variable or expression use.
- Loop-condition mistakes.
- Small multi-line semantic errors.

These are examples of allowable bug families, not a requirement to hard-code a new catalog of fixed substitutions.

The LLM should make the substantive semantic decision. The harness should automate deterministic formatting, execution, and verification.

Use explicit bounds on patch size and structural change to prevent unlimited corruption, but do not retain unnecessarily narrow restrictions that make almost every proposal a one-token operator change.

Prefer changes that preserve a meaningful, solvable repair problem.

### Verifier contract

For each candidate:

1. Parse and validate the proposed operation.
2. Apply it to the original source.
3. Confirm the mutation is syntactically valid.
4. Confirm required APIs and function signatures remain usable.
5. Verify that the original program passes the trusted tests.
6. Verify that the mutated program exhibits an actual semantic failure.
7. Confirm the mutation does not violate the defined task constraints.
8. Record the mutation type, size, changed lines/nodes, and deduplication identity.
9. Admit the candidate only if it creates a valid verifiable repair task.

Use the existing isolated execution worker.

Do not execute model-generated code directly on a login node or in the trainer process.

Do not permit generated code to read private test data, modify tests, access external resources, or escape the accepted execution boundary.

Existing sandbox and worker safety guarantees must be preserved, but do not invent additional infrastructure requirements without a demonstrated need.

### Proposer feedback

Keep invalid proposals in the appropriate attempted-proposal accounting, including a clearly defined penalty.

Track:

- Valid proposal rate.
- Duplicate rate.
- Unique valid tasks.
- Mutation-family diversity.
- Source and specification complexity.
- Student solve-rate distribution.
- Informative proposer groups.
- Generated tokens and validation cost.
- Proposer reward variance.
- Curriculum change over training.

Avoid an overly aggressive rejection procedure that spends substantial GPU time generating candidates that never reach student training.

### Fairness

Freeze the same proposer action space, validator, prompting rules, and task constraints for all five experimental arms.

The distinction between B, C, and D must be the proposer reward, not differences in allowed bug types.

Do not make the proposer stronger only in C or D.

## 9. Implement gradient-based proposer rewards C and D

For configurations C and D, replace the difficulty proposer reward with a reward based on gradient alignment.

The idea is inspired by CompassPlay, but the exact comparison in this project differs because student training includes gated OPSD.

### Reference gradient

Define a small, trusted, fixed reference set of repair tasks sampled only from the authorized training-reference partition.

The reference must not use hidden evaluation labels, held-out final tasks, or HumanEval+-derived external test information.

Construct reference examples in the **new Qwen3.5 native-tool format**, including realistic valid repair actions.

Do not blindly reuse the old Qwen3 canonical JSON reference strings if their format no longer matches the student environment.

The reference objective can be cross-entropy on verified reference repair actions, provided it is defined consistently and documented.

Let:

\[
g_{\mathrm{ref}}(\theta)
=
\nabla_\theta\mathcal L_{\mathrm{ref}}(\theta).
\]

Compute the reference gradient at the current shared policy snapshot.

The reference gradient is guidance for task generation, not direct validation accuracy.

Its alignment is a first-order proxy; do not claim it directly measures downstream improvement.

### C: RL-only candidate gradient

For proposed task \(x\):

\[
g_C(x)
=
\nabla_\theta
\mathcal L_{\mathrm{DrGRPO}}^S(x).
\]

Then:

\[
R_C(x)
=
\frac{1+\cos(g_C(x),g_{\mathrm{ref}})}{2}
\]

for a valid, well-defined nonzero gradient.

### D: Combined student-objective gradient

For proposed task \(x\):

\[
g_D(x)
=
\nabla_\theta
\left[
\mathcal L_{\mathrm{DrGRPO}}^S(x)
+
\lambda_{\mathrm{SD}}\mathcal L_{\mathrm{gated\ OPSD}}(x)
\right].
\]

Then:

\[
R_D(x)
=
\frac{1+\cos(g_D(x),g_{\mathrm{ref}})}{2}
\]

for a valid, well-defined nonzero gradient.

The candidate gradients must correspond to the **actual new Dr. GRPO student objective**, rather than a legacy GRPO approximation.

### Undefined gradient cases

Handle zero or near-zero gradients explicitly.

An all-failure Dr. GRPO group can yield zero RL gradient even when the gated OPSD component is nonzero.

For undefined cosine rewards, use a prespecified low-information reward or exclusion convention that does not artificially reward zero gradients over informative candidates.

Record zero-gradient cases separately.

Do not silently convert undefined cosine similarity into a meaningful positive alignment value.

Use the same candidate normalization, reference set, optimizer snapshot, and reward handling for C and D.

### Preserve gradient integrity

Reference and candidate gradient calculations must not mutate parameters, optimizer states, or accumulated production gradients.

Check:

- Correct parameter-coordinate ordering.
- Full trainable LoRA parameter coverage.
- Correct sharded-gradient aggregation.
- FP32 or higher precision for gradient statistics where appropriate.
- Correct near-zero handling.
- No accidental optimizer steps.
- Correct restoration after exceptions.

Measure the additional compute cost per scored candidate.

Reuse the reference gradient while the policy snapshot is unchanged.

### Critical proposer-signal diagnostic

Task 17 observed that the historical proposer often had insufficient distinct rewards.

Before expensive C/D training, verify that the expanded proposer creates enough valid and distinct candidate tasks to make the alignment objective meaningful.

Measure reward variance and the proportion of proposer groups with multiple valid, distinct-reward candidates.

If C/D still have very weak reward discrimination, diagnose the mechanism and make a bounded, principled fix that is applied consistently where necessary.

Do not conduct an endless search for a proposer representation that happens to produce favorable development results.

If meaningful gradient differentiation cannot be established within a reasonable bounded investigation, still complete the available controlled C/D pilots and report their limitations. Do not misrepresent a degenerate proposer update as successful gradient-guided learning.

## 10. Repair and qualify numerical reliability

The previous study identified nontrivial differences between rollout-time and scoring-time token probabilities.

This must be addressed before interpreting the new objective comparisons.

However, do not require bitwise-identical floating-point execution across fundamentally different GPU kernels.

The scientific question is whether discrepancies are large enough to materially change optimization.

### Required investigation

Using small saved/frozen training batches:

1. Reconstruct the exact prompt token IDs.
2. Confirm generated target token IDs.
3. Verify causal log-probability alignment.
4. Check multi-turn position and attention-mask handling.
5. Verify EOS and tool-call boundaries.
6. Compare training versus inference model states.
7. Compare the effects of precision, batching, caching, and execution kernels.
8. Confirm identical LoRA parameter values during rollout and scoring.
9. Check the actual PPO clipping decisions.
10. Compare resulting gradients and real optimizer displacements.

Maintain the actual rollout behavior log-probabilities as the old-policy denominator:

\[
\rho_t=
\exp\left(
\log\pi_{\theta}(y_t\mid h_t)
-
\log\pi_{\theta_{\mathrm{behavior}}}(y_t\mid h_t)
\right).
\]

Do not replace the denominator with newly rescored probabilities merely to eliminate an observed difference.

Remember that the Task 18 21.4% result came from a **diagnostic denominator substitution**, not a proven 21.4% error in every historical training update.

### Correctness qualification

Use repeat-controlled experiments that compare:

- Same-operation numerical variation.
- Cross-backend rollout/scoring differences.
- Actual clipping changes.
- Gradient cosine similarity.
- Relative gradient/update differences.
- Restored AdamW parameter displacement.
- Stability after a small number of genuine optimizer updates.

Set defensible numerical acceptance criteria in advance, informed by repeat variation and the effect on optimizer steps.

A small harmless log-probability difference should not become a permanent launch blocker.

A material unexplained change in optimization direction must not be ignored.

If the existing vLLM/HF execution path is the source of irreducible material discrepancies, consider a single-backend rollout/scoring implementation for the new experiment.

Prioritize reliable science over theoretical throughput, but measure the efficiency tradeoff.

Do not modify production optimization mathematics based solely on speculative explanations.

### Other training-health requirements

Verify:

- Finite logits, log-probabilities, losses, and gradients.
- Correct gradient clipping.
- Correct learning-rate behavior.
- Nonzero optimizer steps for informative batches.
- No unintended update on zero-gradient batches.
- Stable model/optimizer checkpoint restoration.
- Correct shared-parameter update semantics.
- Bounded KL or policy drift, as a monitoring quantity.
- No excessive invalid-action collapse.
- No pathological output-length growth.
- No silent failure in proposer task generation.

Tests should cover the actual Qwen3.5 execution path, not only a toy model.

## 11. Establish valid base and OOD evaluations

Before launching long training runs, evaluate the selected Qwen3.5 base checkpoint using the new final student interface.

The previous base model's low score could reflect both semantic ability and interface compatibility.

We need a trustworthy new baseline.

### Benchmark distinction

The repository uses:

- MBPP+-derived **program-repair** tasks for ID evaluation.
- HumanEval+-derived **program-repair** tasks for external evaluation.

These are not the same metrics as official MBPP+ or HumanEval+ code-generation pass@1.

Report them clearly as repair success rates.

If a small standalone code-generation check is necessary to answer why the model appears weak relative to published benchmarks, do it only if inexpensive. Do not turn this into another major project.

### Evaluation protocol

Use a consistent Qwen-native tool interface for all snapshots.

Measure:

- Verified repair success.
- Tool-call validity.
- Actual edit application.
- Public-test success.
- Hidden-test success.
- Public-pass/hidden-fail outcomes.
- Truncation and budget exhaustion.
- No-edit failures.
- Semantic/runtime failures after applied edits.
- Total turns.
- Generated thinking and tool tokens.
- Inference and verification time.
- Cost per evaluated repair.

Analyze results by task length, specification length, structural complexity, and mutation type.

### Establish OOD transfer

Evaluate on the previously defined HumanEval+-derived external repair set using the corrected interface.

Determine whether improvement in native tool calling meaningfully raises external repair success.

The existing external set has been inspected in prior diagnostic work. Treat it as an established exploratory external benchmark, not a completely untouched confirmation set.

Do not use its hidden solutions for proposer training, reference-gradient construction, prompt tuning, or loss optimization.

If another genuinely held-out set is already available through the authorized dataset pipeline, reserve it before training for final confirmation.

Do not spend disproportionate effort constructing a new benchmark from scratch.

### Training-distribution mismatch

Inspect whether the training sources contain sufficient semantic and structural diversity for transfer to HumanEval+-derived tasks.

The old structured-mutant distribution is potentially too narrow.

The richer proposer should address mutation diversity, but increased mutation diversity alone does not guarantee that short MBPP-derived programs represent longer HumanEval-derived programs.

If a substantial source-complexity gap remains, consider a limited, verified expansion of the training-only parent pool, from appropriately licensed and disjoint sources if already available.

Any such expansion must happen before the five-arm study and be applied identically to every method.

Do not contaminate OOD evaluation by training on its target programs.

Do not keep changing the training distribution in response to final-test results.

### If OOD remains poor

Do not automatically abandon the experiment.

Clearly distinguish:

- In-distribution learning.
- Valid tool-use capability.
- Semantic repair capability.
- Transfer to longer or different source programs.

A negative OOD result is scientifically informative if the implementation and evaluation are valid.

Complete the planned method comparisons whenever technically feasible, rather than repeatedly redesigning the study until OOD becomes positive.

## 12. Efficient experiment execution

The project runs on Yale HPC, using Slurm and the existing verified worker infrastructure.

Prefer efficient use of the currently authorized GPUs.

Do not assume the earlier cluster process-limit problem remains unresolved if the current accepted worker already satisfies the actual execution contract.

Check current cluster and job state before submitting anything.

### Do not repeat completed work

Do not rerun:

- Task 15's historical three-seed A/B experiments.
- Task 17's historical 12 student-objective controls.
- Task 18's completed interface comparisons.
- Existing unchanged dataset/verifier audits whose provenance can be reused.

The new Qwen3.5 model requires new scientific validation, but unchanged verified dataset and sandbox evidence should be reused where compatible.

### Main efficiency bottlenecks to measure

Measure actual wall-clock time spent in:

- Proposer generation.
- Candidate validation.
- Student generation.
- Tool execution.
- Verifier execution.
- Teacher scoring.
- Gradient-alignment scoring.
- Backward passes.
- Optimizer steps.
- Model/adapter synchronization.
- Checkpointing.
- Development evaluation.

Optimize the dominant contributors, not presumed bottlenecks.

### Preferred optimizations

Consider, where compatible:

- Batching independent proposer generations.
- Batching student rollouts from a frozen model snapshot.
- Overlapping CPU-based verifier work with suitable GPU operations.
- Reusing identical deterministic verification artifacts.
- Caching tokenized static prompts and stable prefixes.
- Deterministic generation of schema fields.
- Avoiding unnecessary model-generated JSON formatting.
- Reusing a reference gradient within one unchanged snapshot.
- Avoiding repeated model loads.
- Efficient shared-LoRA synchronization.
- Streaming or microbatching long trajectories.
- Correctly accounting for GPU residency and memory usage.

Do not enable caching or batching if it changes the intended model conditioning or produces material, unexplained probability differences.

### Resource planning

After the short hardware pilot, estimate:

\[
\text{GPU-hours per 1,000 attempted episodes}
\]

for the student-objective arms and gradient-alignment arms separately.

Use actual allocated GPU time, not only modeled FLOPs or observed execution time.

Choose a realistic common primary endpoint before examining treatment-effect results.

A reasonable starting target is approximately **1,024 attempted student episodes per arm**, provided the measured cost fits available resources. A smaller 512-episode endpoint or a larger endpoint may be justified by a prespecified resource assessment.

Do not automatically run 4,096 episodes per arm merely because that was the previous study's length.

Do not use development-result wins to decide which arms receive more training.

Keep a running study-wide resource ledger, including failed jobs, resumed segments, training, diagnostics, and evaluations.

An initial target is to keep the new study within approximately 80 allocated H200 GPU-hours if achievable. This is a planning target, not authorization to exceed actual available cluster quota or policy.

If measured costs indicate that this budget will not support the intended design, choose the largest feasible common endpoint or a documented equal-compute allocation and prioritize completing all five arms over overtraining one or two.

Do not provision paid AWS resources or incur new external costs without explicit authorization.

## 13. Run the experiments in the correct order

The following is an execution sequence, not an invitation to stop after planning.

### Phase 1 — New model and objective implementation

Deliver:

- Functional Qwen3.5 model backend.
- Native thinking/tool interface.
- Shared LoRA and optimizer.
- Dr. GRPO objective.
- Ungated and gated OPSD.
- Expanded proposer representation.
- Gradient-alignment proposer rewards.
- Complete unit and integration tests.

Run real small GPU checks.

Confirm that actual generated code is executed only through the accepted worker.

### Phase 2 — Baseline and numerical qualification

Run:

- Qwen3.5 base repair evaluation.
- Native tool-call compatibility checks.
- OOD evaluation.
- Frozen-batch numerical diagnostics.
- Shared-parameter gradient/update checks.

Resolve material scientific bugs.

Do not require unrelated infrastructure redesign.

### Phase 3 — Matched short pilots

Run a matched pilot for:

- Control.
- A.
- B.
- C.
- D.

Start with approximately 32–80 attempted episodes per arm, depending on the measured minimum useful batch size.

Confirm:

- Correct common initialization.
- Correct model/optimizer state.
- Correct reward computation.
- Correct thinking-token accounting.
- Valid student and proposer actions.
- Real shared-weight updates.
- Nondegenerate proposer reward measurements.
- Finite losses and gradients.
- Successful checkpoint recovery.
- Measured cost.

A pilot does **not** need to demonstrate positive downstream improvement before continuation. A short pilot may be too small to detect meaningful accuracy changes.

Continue if the system is numerically credible and produces the intended learning signals.

### Phase 4 — Main one-seed training

Freeze the scientific configurations and common endpoint.

Run all five configurations:

1. Control.
2. A.
3. B.
4. C.
5. D.

All use seed `2026`.

Use reproducible initial snapshots, shared data splits, identical evaluation, and a consistent budget.

Use resumable Slurm jobs and verified checkpoints.

If a job stops due to allocation limits, resume the same scientific run from its verified checkpoint rather than restarting or counting a duplicate run.

Never silently retry already completed jobs.

Do not spend the entire available compute budget on Control/A/B while leaving C/D unexecuted.

If resources become constrained, prioritize equal, interpretable exposure across all five arms.

### Phase 5 — Evaluation and analysis

Evaluate the frozen checkpoints at the prespecified milestones.

Report all arms, including negative or inconclusive results.

Do not select whichever checkpoint happens to produce the most favorable final-test result.

Use the primary common-budget endpoint for the main comparison.

A separate best-development-checkpoint analysis is acceptable if the selection rule was frozen in advance and does not use final outcomes.

## 14. W&B and local experiment observability

Provide live W&B dashboards with a consistent metric namespace across the five arms.

At minimum, monitor the following.

### Student performance

- Training repair success.
- Development ID repair success.
- External repair success at prespecified evaluations.
- Public test pass rate.
- Hidden test pass rate.
- Mixed-reward group fraction.
- All-success and all-failure group fractions.
- Valid edit rate.
- Tool-call errors.
- Budget/truncation failures.
- Average turns per episode.
- Tokens per episode.

### Thinking and action behavior

- Thinking-token count.
- Tool-call-token count.
- Reasoning/tool token ratio.
- Truncated-thinking fraction.
- Number of read/edit/test calls.
- Actual accepted edits.
- Successful repairs after accepted edits.

### Dr. GRPO

- RL loss.
- Mean and variance of centered advantages.
- Gradient norm.
- PPO clipping fraction.
- Importance-ratio distribution.
- Approximate KL or policy drift.
- Actual optimizer updates.
- Update norm.

### OPSD

- Auxiliary loss.
- Teacher–student log-probability gaps.
- Signed-gap distribution.
- Gate distribution.
- Positive/negative gap fractions.
- Gradient norm ratio, OPSD to RL.
- RL/OPSD gradient cosine.
- Reasoning-token versus tool-token auxiliary signal.

### Proposer

- Valid task fraction.
- Unique candidate count.
- Duplicate fraction.
- Mutation-family distribution.
- Student solve-rate distribution.
- Difficulty reward distribution.
- Gradient-alignment distribution.
- Zero/near-zero candidate-gradient fraction.
- Reference-gradient norm.
- Candidate-gradient norm.
- Groups with distinct valid rewards.
- Proposer update norm.

### Shared-parameter behavior

- Student gradient norm.
- Proposer gradient norm.
- Student/proposer gradient cosine.
- Joint update norm.
- Student/proposer gradient-scale ratio.
- Actual parameter displacement.

### Efficiency and reliability

- Allocated GPU-hours.
- GPU memory use.
- Tokens/second.
- Episodes/hour.
- Verified repairs/GPU-hour.
- Proposal-validation time.
- Teacher-scoring time.
- Gradient-alignment overhead.
- Worker failures.
- Rollout/scoring mismatch statistics.
- Checkpoint duration.
- Resume count.

Ensure the data is sufficient to diagnose whether poor results arise from model capability, training signal, proposal quality, tool-use failures, or implementation instability.

Do not rely exclusively on W&B cloud history. Local logs, manifests, checkpoints, and event records must be authoritative.

Raw benchmark solutions, private oracles, and credentials must not be uploaded to public artifacts.

## 15. Required scientific analyses

Produce publication-quality figures and tables.

### Table 1 — Main benchmark results

Report the base model and Control/A–D:

- MBPP+-derived ID repair rate.
- HumanEval+-derived external repair rate.
- Appropriate confidence intervals over independent task families.
- Evaluation tokens and execution cost.

Do not treat multiple mutations from the same source program as statistically independent.

### Figure 1 — Learning curves

Show ID development success against:

- Attempted student episodes.
- Generated student tokens.
- Allocated GPU-hours.

These are complementary views. Equal episodes do not mean equal token or compute expenditure.

### Figure 2 — Generalization

Show ID versus external repair performance and their changes from the same base checkpoint.

Do not equate an OOD score near the floor with proof of catastrophic forgetting.

### Figure 3 — OPSD analysis

Compare Control, A, and B using:

- Performance and sample efficiency.
- RL/OPSD gradient interaction.
- Teacher–student gap behavior.
- Gating behavior.
- Changes in reasoning/tool-call token allocation.

Test whether gating improves credit assignment rather than assuming it does.

### Figure 4 — Proposer analysis

Compare B, C, and D using:

- Proposer reward diversity.
- Generated-task difficulty.
- Validity and semantic diversity.
- Gradient alignment.
- Student learning progress.
- Verified repairs per GPU-hour.

Investigate whether a higher alignment score predicts useful training progress.

### Figure 5 — Failure taxonomy

For ID and external repair tasks, break down:

- Invalid tool output.
- Rejected edit.
- Applied but semantically wrong edit.
- Public-pass/hidden-fail.
- Truncation.
- No-edit termination.
- Verified successful repair.

Distinguish observed categories from causal explanations.

### Statistical interpretation

With only one training seed, do not claim robust superiority across random initializations or training trajectories.

Use paired task-level comparisons and parent/family-clustered bootstrap intervals where appropriate.

Describe them as uncertainty conditional on the observed training run.

Report absolute differences in percentage points and denominators, not only relative percentage gains.

Do not compute inappropriate significance estimates by treating generated episodes as independent.

Historical three-seed A/B evidence may be discussed separately, but it uses a different experimental system.

### Interpret negative results

If gated OPSD does not outperform Dr. GRPO-only, explain why that finding is informative.

If C and D do not outperform difficulty rewards, analyze whether gradient-alignment signals were uninformative, noisy, poorly aligned with the target, or simply not beneficial.

If ID learning occurs without external transfer, use the failure taxonomy and dataset characteristics to investigate.

A scientifically credible negative result is better than an unsupported success claim.

## 16. Research engineering and testing requirements

Follow established software and research-engineering practices.

Prefer:

- Typed configuration objects.
- Modular policies, environments, objectives, and trainers.
- Minimal coupling between scientific methods and HPC orchestration.
- Deterministic seed management.
- Explicit model/tokenizer/data fingerprints.
- Versioned scientific configurations.
- Transactional checkpointing.
- Reusable evaluation modules.
- Unit tests for mathematics.
- Integration tests for actual model-tool behavior.
- Clear exceptions and failure classifications.
- No silent fallbacks.
- Reproducible analysis scripts.

Run the applicable Ruff, mypy, and pytest checks.

Avoid unnecessarily strict tests that require irrelevant exact GPU-kernel equality or prohibit valid execution strategies.

Conversely, do not weaken tests that protect the correctness of rewards, gradient computation, data isolation, or checkpoint recovery.

### Key invariants to test

1. Shared LoRA really is shared across roles.
2. One optimizer updates both role-conditioned objectives.
3. Student and proposer gradients are accumulated at the same snapshot.
4. Teacher logits are detached.
5. Gradient-alignment diagnostics do not mutate weights.
6. Dr. GRPO does not standardize rewards or normalize by variable trajectory length.
7. Thinking tokens are counted and scored consistently.
8. Tool-call arguments execute correctly.
9. Invalid proposals are penalized without corrupting task groups.
10. Hidden tests and reference answers never leak to the student.
11. Train, development, and final task identities remain correctly separated.
12. Saved behavior probabilities remain the actual PPO denominator.
13. Restored runs reproduce the intended scientific state.
14. All five arms share the same fixed scientific configuration except for their specified objectives.

## 17. Avoid previous failure patterns

This requirement is important.

We previously lost substantial time to repeated debugging and readiness work.

Do not repeat the following mistakes:

**Mistake 1: Endless readiness checks**

Do not keep adding new mandatory validation stages without identifying what concrete scientific or operational failure they prevent.

Run targeted checks and proceed when the necessary evidence is sufficient.

**Mistake 2: Repeating completed experiments**

Reuse verified historical results and artifacts when they remain relevant.

Do not rerun old experiments just to satisfy a newly introduced documentation rule.

**Mistake 3: Optimizing infrastructure instead of the experiment**

Do not build a large infrastructure abstraction before demonstrating that the model can train and the environment can execute tasks correctly.

Prioritize a thin, correct implementation.

**Mistake 4: Ignoring fundamental numerical errors**

Do not silence mismatch warnings or replace actual behavior probabilities with convenient substitutes.

Investigate whether differences materially affect optimization.

**Mistake 5: Changing too many factors between arms**

The new architecture changes multiple historical design choices, but within this new study the five arms must differ only in their scientific objectives.

**Mistake 6: Spending compute on uninformative data**

Monitor task validity, mixed-reward groups, proposer reward diversity, and actual update magnitudes.

Avoid training configurations dominated by invalid proposals or empty gradients.

**Mistake 7: Overly restrictive proposer interfaces**

Keep verification strict while allowing sufficiently expressive semantic mutations.

Do not mistake a syntactically restrictive mutation catalog for a general coding-task curriculum.

**Mistake 8: Overclaiming OOD performance**

Do not equate a successful edit application, correct intermediate state, or public-test pass with full verified task success.

**Mistake 9: Stopping after a successful smoke test**

A smoke test is not the final deliverable.

The objective is a completed and analyzed five-arm scientific study.

## 18. Deliverables

Create a concise, navigable set of research artifacts in the repository, such as:

- A new scientific implementation specification.
- A frozen five-arm experiment manifest.
- Qwen3.5 migration and compatibility evidence.
- Dr. GRPO mathematical validation tests.
- Shared-LoRA joint-update implementation and tests.
- Expanded proposer and native student tool interfaces.
- Numerical qualification results.
- New base-model and external evaluation results.
- One-seed Control/A/B/C/D run records.
- W&B run links.
- Learning and efficiency figures.
- Proposer-gradient and OPSD diagnostics.
- A main-results table.
- A final research decision and limitations document.
- Clear reproduction commands.

Prefer reusing repository conventions instead of creating redundant parallel systems.

Update `README.md` to distinguish the historical Qwen3 study from the new Qwen3.5 study.

Use descriptive run names such as:

- `qwen35-shared-drgrpo-control-s2026`
- `qwen35-shared-drgrpo-A-s2026`
- `qwen35-shared-drgrpo-B-s2026`
- `qwen35-shared-drgrpo-C-s2026`
- `qwen35-shared-drgrpo-D-s2026`

Use immutable run IDs and record exact commit, checkpoint, tokenizer, and configuration hashes.

## 19. Required execution behavior

You are authorized by this task to work in the repository, execute local tests, and use already authorized Yale HPC resources for bounded research jobs, subject to the actual permissions and resource policies available in your environment.

Do not provision paid external compute or modify unrelated accounts.

Inspect current Slurm jobs before creating or resuming any job.

Do not duplicate jobs that are pending, running, or already complete.

If an existing job can be resumed safely, resume it instead of launching a new scientific run.

If a genuinely blocking issue occurs:

1. Identify the specific failure and evidence.
2. Determine whether it is scientific, numerical, environmental, or administrative.
3. Implement the smallest reliable fix.
4. Test the fix.
5. Continue the experiment when the issue is resolved.

Do not repeatedly stop for user confirmation on ordinary implementation decisions.

If external authentication, inaccessible hardware, quota restrictions, or a material unresolved scientific-integrity problem truly prevents execution, document the blocker precisely and provide the runnable commands and verified artifacts completed so far.

Never claim a job ran, an experiment completed, or a result was observed unless supported by actual execution records.

Commit and push the work to the dedicated feature branch when repository access permits, preserving unrelated changes. Do not merge into `main` before validation.

## 20. Definition of success

This task is fully complete when:

- Qwen3.5-4B is operational in the new training and evaluation pipeline.
- Student thinking mode works correctly.
- The proposer and student share one trainable LoRA.
- Dr. GRPO is correctly implemented for both roles.
- Native student tool calling works reliably.
- The proposer can generate diverse, verified semantic bugs.
- Gated and ungated OPSD are implemented consistently.
- C and D calculate the correct student-objective gradient alignment.
- Numerical behavior is credible and documented.
- The corrected base and external evaluations are available.
- All five one-seed configurations complete the common training endpoint.
- Evaluation results and efficiency curves are generated.
- The final report directly answers the four scientific comparisons.
- Historical results and new results are cleanly separated.
- The code, configurations, checkpoints, and analyses are reproducible.

If the results show no benefit for gated OPSD or gradient-guided proposing, that is acceptable as a research outcome.

The requirement is scientific validity and completion, not obtaining a positive result at all costs.

### Final instruction

**Begin by inspecting the current repository and its latest evidence, then implement the revised architecture and execute the experiments.**

Do not stop after writing a plan, revising a specification, or passing local tests.

Use the available time and compute efficiently. Resolve concrete failures, preserve the scientific controls, and prioritize completing Control and experiments A–D with one seed.

At the end, report exactly what was implemented, what actually ran, the observed outcomes, GPU-hours consumed, reproducibility paths, and any genuinely remaining limitations.