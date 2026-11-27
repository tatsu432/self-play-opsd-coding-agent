# Task 22: Complete the PDB Self-Play OPSD Study — Gradient-Aligned Proposers, Mechanistic Analysis, and Compute-Efficient Training

## 1. Overall objective

We are conducting a research project for the Yale Frontier AI Models course.

Repository:
https://github.com/tatsu432/self-play-opsd-coding-agent

Research title:
**Demystifying the Effect of Gated On-Policy Self-Distillation on Students and Proposers in Gradient-Guided Self-Play for Coding Agents**

The five-minute presentation is on **December 2, 2026** and is a soft milestone: present defensible findings available at that time, even if experiments are still running. The **hard deadline is December 12, 2026**, for the final report. The original planning date was November 24, 2026.

After many iterations of debugging, investigating learning-signal failures, changing datasets, improving the agent interface, and stabilizing the training pipeline, Task 21 finally produced promising results using Qwen3.5-4B, the Precise Debugging Benchmark (PDB), and shared proposer/student parameter updates.

We must now focus on answering the actual research questions rather than continuing open-ended engineering investigations.

Your responsibility is to inspect the existing implementation and experimental evidence, implement the minimum necessary changes, qualify the gradient-alignment proposer objectives on PDB, execute experiments C and D efficiently, analyze the results, and prepare publication-quality experimental outputs.

If sufficient time and compute remain, extend the existing Control/A/B runs.

**The overarching objective is to maximize scientifically useful information per GPU-hour and per day, not to maximize the number of tests, infrastructure features, training steps, or engineering safeguards.**

Do not stop after writing a plan. Implement, validate, launch, monitor, recover, analyze, and document the experiments using the existing authorized execution workflow.

---

## 2. Existing experimental evidence

### 2.1 Task 21 results

Task 21 successfully completed matched Control, A, and B training runs for seeds 2026 and 2027.

Each arm completed 512 attempted student training episodes, corresponding to approximately 19–21 optimizer steps.

The primary held-out PDB single-bug repair results are:

| Method | Seed 2026 | Seed 2027 | Mean accuracy |
|---|---:|---:|---:|
| Initial model | 21/64 | 20/64 | 32.03% |
| Control | 22/64 | 21/64 | 33.59% |
| A | 26/64 | 21/64 | 36.72% |
| B | 28/64 | 28/64 | 43.75% |

B improves over Control by approximately 10.16 percentage points and over A by approximately 7.03 percentage points on average.

The results are promising, but the task-bootstrap confidence intervals include zero, and two training seeds are insufficient to establish a reliable population-level winner.

Other relevant observations:

- The current PDB training configuration produces meaningful nonzero RL gradients substantially more frequently than the earlier MBPP+/EvalPlus-based configurations.
- B outperforms both A and Control on the primary benchmark in both seeds.
- A outperforms Control in seed 2026 and matches Control in seed 2027.
- The primary evaluation contains 64 single-bug tasks.
- The transfer evaluation includes eight PDB-Multi tasks, one adapted BigCodeBench-derived PDB task, and nine separately generated BigCodeBench mutants.
- The transfer sets are too small to support strong generalization claims.
- The current 19–21 optimization steps do not establish convergence.
- Approximately 29–38 of the 64 primary evaluation episodes per trained arm encountered truncation.
- A and B use the same nominal OPSD coefficient, but their effective distillation gradient magnitudes differ.
- The full Task 21 study consumed approximately 217.155 H200-hours, including pilots, failed jobs, continuations, and evaluations.
- Shared project storage exhaustion previously coincided with six failed endpoint evaluations.

Read the existing evidence before making decisions:

- `README.md`
- `docs/investigations/task21-results.md`
- `docs/investigations/task21-status.md`
- `docs/task_completion/21-implemented-pdb-selfplay.md`
- `docs/investigations/task19-results.md`
- `docs/investigations/task20-final-decision.md`
- `configs/task21-main.json`
- `configs/task21-main.yaml`
- `configs/task21-main-2027.json`
- `configs/task21-main-2027.yaml`

Existing W&B final aggregates:
https://wandb.ai/tatsu432s/self-play-opsd-coding-agent/runs/task21-final-aggregates

Existing learning curves:
https://wandb.ai/tatsu432s/self-play-opsd-coding-agent/runs/e3684c8e0122b912f97c238a

Treat the verified checkpoint-backed artifacts and reconciled allocation ledgers as the authoritative evidence.

Do not rerun completed Task 21 experiments or repeat their evaluations unnecessarily.

### 2.2 Earlier C/D evidence

Task 19 previously implemented and executed all five arms using Qwen3.5-4B and shared parameters, but the training exposure was only 96 attempted episodes, and the earlier dataset had limited useful RL signal.

No reliable method winner emerged.

However, one particularly interesting observation was:

- C encountered 29 near-zero alignment gradients.
- D encountered zero near-zero alignment gradients.

This suggests that incorporating OPSD into the candidate gradient may make gradient-based proposer rewards informative for tasks where sparse outcome rewards provide little or no learning signal.

We should investigate this mechanism using the better PDB experimental setting.

Do not treat Task 19 as evidence that C or D already works or fails on PDB.

---

## 3. Research questions and experimental configurations

The central research question is:

**How should the student and proposer objectives be designed when integrating gated on-policy self-distillation into self-play for coding agents?**

The five experimental configurations are:

| Arm | Proposer reward | Student objective |
|---|---|---|
| Control | Difficulty-based | Dr. GRPO |
| A | Difficulty-based | Dr. GRPO + ungated OPSD |
| B | Difficulty-based | Dr. GRPO + gated OPSD |
| C | RL-only gradient alignment | Dr. GRPO + gated OPSD |
| D | RL + gated OPSD gradient alignment | Dr. GRPO + gated OPSD |

The intended comparisons are:

1. A versus Control: Does ungated OPSD improve self-play?
2. B versus A: Does gated OPSD improve over ungated OPSD?
3. B versus Control: Does gated OPSD improve over verifier-only RL?
4. C versus B: Does gradient-aligned task generation improve over difficulty-based task generation when the student uses gated OPSD?
5. D versus C: Does accounting for the actual combined student objective improve gradient-based proposer rewards?
6. D versus B: Does the complete proposed combination outperform difficulty-based gated OPSD self-play?

The most novel remaining comparison is **D versus C**.

The existing literature has separately investigated gated OPSD for agentic RL and gradient-based proposer rewards for self-play. Our contribution is investigating their interaction.

Relevant literature:

- Absolute Zero Reasoner: https://arxiv.org/abs/2505.03335
- CompassPlay: https://arxiv.org/abs/2609.32228
- Self-Distilled Agentic Reinforcement Learning: https://arxiv.org/abs/2605.15155
- Self-Distilled RLVR: https://arxiv.org/abs/2604.03128
- On-Policy Self-Distillation: https://arxiv.org/abs/2601.18734
- GradAlign: https://arxiv.org/abs/2602.21492
- π-Play: https://arxiv.org/abs/2604.14054
- PreciseCoder: https://arxiv.org/abs/2610.18755

Do not introduce additional algorithms from these papers at this stage. They provide motivation and context for the existing experimental design.

---

## 4. Experimental priorities and decision hierarchy

Use the following priority ordering.

| Priority | Task | Requirement |
|---|---|---|
| P0 | Preserve and audit existing Task 21 results | Mandatory |
| P0 | Qualify PDB gradient-alignment rewards | Mandatory before C/D |
| P1 | Experiment C, seed 2026, 512 episodes | Primary goal |
| P1 | Experiment D, seed 2026, 512 episodes | Primary goal |
| P1 | Matched endpoint evaluations | Mandatory for completed arms |
| P1 | Mechanistic, efficiency, and failure-mode analyses | Mandatory |
| P2 | Longer Control/A/B training, seed 2026 | Conditional |
| P3 | Strength-matched ungated OPSD ablation | Stretch goal |
| P4 | Additional C/D training seeds | Stretch goal |

Do not restart Control, A, or B from scratch.

Do not prioritize additional seeds or substantially longer training over obtaining the first meaningful matched C/D results on PDB.

Seed 2026 is selected because it was the original primary seed, not because of its observed performance.

The desired main result table for the presentation is:

| Method | Seed 2026 | Seed 2027 |
|---|---|---|
| Initial | Completed | Completed |
| Control | Completed: 512 episodes | Completed: 512 episodes |
| A | Completed: 512 episodes | Completed: 512 episodes |
| B | Completed: 512 episodes | Completed: 512 episodes |
| C | Target: 512 episodes | Deferred |
| D | Target: 512 episodes | Deferred |

The single-seed C/D comparison must be explicitly described as exploratory.

---

## 5. Non-negotiable execution principles

We have spent too much time repeatedly debugging and preparing for experiments instead of conducting them.

Avoid repeating this pattern.

### Preserve the functioning pipeline

Do not change the following unless a concrete correctness failure makes a change unavoidable:

- Qwen3.5-4B model and pinned revision.
- PDB training curriculum and evaluation partitions.
- Native Qwen reasoning and tool-call protocol.
- Shared proposer/student LoRA parameterization.
- Qualified cached FP32 execution.
- Existing optimizer and learning rate.
- Four student rollouts per task.
- Existing proposal group construction.
- Existing proposer prompts and mutation constraints.
- Existing student and teacher context limits.
- Existing isolated execution worker and verifier.
- Existing objective coefficients.
- Existing checkpoint and recovery conventions.
- Existing authorized Yale HPC execution workflow.

Do not redesign the repository, migrate to a new RL framework, change model architecture, introduce another inference engine, or launch a new broad stabilization effort.

The currently working implementation is the baseline.

### Use pragmatic validation

Validation should answer concrete questions:

- Does the implementation compute the intended objective?
- Are the relevant gradients and rewards numerically valid?
- Can C/D produce task-specific proposer learning signals?
- Can training and evaluation execute reliably?
- Can the experiments finish within the remaining budget?

Do not require every conceivable edge case to be resolved before launching research experiments.

Do not require statistically significant pilot accuracy improvement.

Do not require a large number of diagnostic batches if existing evidence or a small focused test is sufficient.

Do not bypass actual correctness, execution-isolation, or resource-authorization requirements. However, distinguish necessary safety and correctness checks from optional engineering preferences.

### Work autonomously within the approved scope

Make reasonable engineering decisions without repeatedly asking me for confirmation.

Ask only when there is a genuinely essential decision involving scientific scope, unavailable credentials, resource authorization, or an unavoidable blocker that cannot be resolved within the specified scope.

Do not interpret this prompt as authorization to exceed actual Yale resource limits, use another person's allocation, or incur unapproved financial charges.

---

## 6. Phase 0: Audit current state and establish the compute budget

Before implementing or launching new experiments:

1. Inspect the current Git branch, commit, uncommitted changes, existing experiment documentation, and relevant code.
2. Reconcile the latest repository state with Task 21's recorded results.
3. Locate the verified initial and endpoint checkpoints for Control/A/B.
4. Inspect existing batch records, iteration metrics, evaluation outputs, optimizer states, and W&B histories.
5. Inspect current Yale Slurm jobs, available authorized GPU capacity, queue conditions, shared filesystem capacity, and allocation ledgers.
6. Identify any existing jobs or unfinished work that should be reused instead of duplicated.
7. Record the exact experimental configuration and data identities for seed 2026.

The relevant Task 21 private artifacts are expected to include locations such as:

- `runs/task21/matched-analysis/`
- `runs/task21/matched-analysis-manifest.json`
- `runs/task21/budget-matched-final.json`
- `data/task21/pdb-solver-v1/`

Verify their actual locations rather than assuming these paths are available in every environment.

The latest documented Task 21 training allocations for seed 2026 were approximately:

| Arm | H200-hours |
|---|---:|
| Control | 25.67 |
| A | 20.98 |
| B | 27.52 |

These are historical allocations, not guarantees of future runtime.

### Proposed additional compute budget

Use the following initial planning envelope:

| Activity | Estimated H200-hours |
|---|---:|
| C/D qualification | 3–5 |
| C training | 25–35 |
| D training | 25–35 |
| Endpoint evaluations | 6–8 |
| Contingency | 6–12 |
| Total | 65–95 |

Treat 95 additional H200-hours as the proposed upper limit for the primary follow-up study, subject to actual authorized capacity.

Refine this estimate using observed runtime and alignment-gradient overhead.

Include all relevant allocated compute, including failed jobs, resumed jobs, gradient diagnostics, development evaluations, and endpoint evaluations.

Do not silently exceed the budget.

A longer Control/A/B study is conditional on remaining authorized capacity. It is not automatically entitled to another 70–85 H200-hours beyond this budget.

Avoid spending a substantial fraction of the remaining GPU budget on readiness checks.

---

## 7. Phase 1: Enable gradient-alignment proposer rewards on PDB

### 7.1 Known implementation gap

The existing Task 19 implementation already contains shared-policy C/D training logic.

However, the Task 21 PDB pipeline appears to inherit reference-gradient construction that was originally designed for the earlier dataset and verifier.

Inspect at least:

- `src/selfplay_opsd/task21.py`
- `src/selfplay_opsd/task20.py`
- `src/selfplay_opsd/task19.py`
- `src/selfplay_opsd/task19_train.py`
- `src/selfplay_opsd/data.py`
- `src/selfplay_opsd/pdb_data.py`
- `src/selfplay_opsd/pdb_verifier.py`
- `src/selfplay_opsd/pdb_proposer.py`
- `src/selfplay_opsd/objectives.py`
- `src/selfplay_opsd/task19_policy.py`
- `scripts/task21_experiment.py`
- `scripts/freeze_task21.py`
- `scripts/hpc_remote.py`

Known issues identified from repository inspection:

1. `Task21Study` inherits the older `references()` implementation.
2. The inherited reference builder expects `TaskRepository.anchors("reference")`.
3. `PdbRepository` does not appear to override the legacy anchor-loading method.
4. The older reference builder verifies repairs through the legacy verifier rather than the PDB-specific verifier.
5. The documented PDB partitions do not appear to contain a dedicated reference partition.
6. The current Task 21 configurations have `alignment_gate_hash: null`.
7. The HPC controller explicitly requires separate alignment evidence before launching C or D.

Confirm these findings against the actual latest code.

Implement the smallest correct extension needed to support PDB-compatible reference gradients and verified C/D execution.

Do not replace the existing shared-policy training implementation.

### 7.2 Construct a fixed PDB reference set

Construct approximately 8–16 verified reference examples initially.

This is a compute-conscious starting point, not an immutable requirement. Use a smaller or larger set if justified by measured signal quality and computational overhead.

Reference examples must:

- Represent the target Python repair capabilities.
- Have a buggy initial source and a verified correct repair.
- Use the same native agent interface and edit protocol as the current study.
- Be independently executable and verifier-confirmed.
- Be frozen before examining C/D endpoint results.
- Be identical for C and D.
- Be disjoint from the development monitor, primary final evaluation, and all transfer evaluation families.

Prefer a small parent-disjoint reference partition drawn from eligible training-domain PDB examples, without modifying the existing proposer curriculum.

If constructing a new independent reference partition is unnecessarily expensive, reusing a fixed subset of existing training-only examples is acceptable, provided that the overlap is explicitly documented.

Never use evaluation tasks or their solutions to construct the reference gradient.

Prefer a versioned reference sidecar or equivalent immutable artifact rather than rebuilding or modifying the existing Task 21 dataset bundle. The existing B training data and parent schedule must remain unchanged.

Record:

- Reference task IDs.
- Parent and family IDs.
- Source benchmark and provenance.
- Dataset and reference hashes.
- Reference selection procedure.
- Verification evidence.
- Canonical edit construction.
- Any reference/training overlap.
- Token and context lengths.

Keep private solutions, reference records, hidden tests, and raw trajectories in authorized ignored storage.

### 7.3 Construct canonical reference repair actions

Adapt the existing reference CE mechanism to the native PDB agent interface.

For each reference task:

1. Construct the normal unprivileged student prompt.
2. Construct a canonical native `replace_text` tool action that repairs the verified buggy source.
3. Confirm that applying the edit produces the expected correct source.
4. Verify the repaired source using the PDB verifier and its appropriate oracle and fixture handling.
5. Tokenize the prompt and native repair action using the same pinned tokenizer and relevant native tool-call conventions.
6. Construct the appropriate loss mask over the intended reference repair-action tokens.

Preserve the existing reference-loss semantics wherever possible.

Do not invent reasoning traces or supervise arbitrary hidden thought tokens.

If a task cannot be represented cleanly by the existing canonical edit mechanism, exclude it using a deterministic, documented eligibility rule. Do not make the reference builder substantially more complex merely to accommodate every possible repair.

The correct source is available to the reference-loss construction, but must not be inserted into the unprivileged student prompt.

The reference gradient is:

g_ref = ∇θ L_CE(θ; D_ref)

where the CE loss is averaged across the frozen reference examples using the established reference-token masking and normalization.

### 7.4 Computational efficiency of reference gradients

Cache static artifacts such as:

- Verified reference examples.
- Canonical repair actions.
- Tokenized prompts.
- Target token IDs.
- Loss masks.

Do not rerun expensive verifier checks or regenerate static reference representations unnecessarily at every optimizer step.

Recompute the actual reference gradient at the current frozen shared-policy parameters when required by the algorithm.

Reuse one reference gradient across all eligible candidate tasks within the same training batch.

Do not reuse a stale reference gradient across different policy snapshots without explicitly introducing and documenting a separate approximation.

Prefer the mathematically correct current-snapshot implementation first. Only investigate an approximation if profiling demonstrates a real bottleneck.

---

## 8. Phase 1 continued: Preserve the intended learning objectives

The actual current Task 19–21 implementation is authoritative. Historical documentation may describe older objectives or parameterizations.

Preserve the existing Dr. GRPO implementation, including its reward centering, clipping, behavior-policy probabilities, token normalization, and treatment of uniform-reward groups.

### 8.1 Student objective

For a task x, define:

g_RL(x) = ∇θ L_RL(θ; x)

g_SD(x) = ∇θ L_gated_OPSD(θ; x)

Both C and D use the same student objective as B:

L_student = L_RL + λ L_gated_OPSD

The gate must be computed using the existing teacher-student scoring implementation.

Teacher targets, privileged-context signals, rollout rewards, old-policy probabilities, and gating weights must have the intended stop-gradient semantics.

Use the same privileged teacher information as B.

Preserve masking so that environment observations, prompt tokens, hidden tests, and other non-policy-generated tokens do not accidentally receive student policy loss.

Do not change the OPSD coefficient.

For seed 2026, the recorded main configuration includes:

- `sd_scale`: approximately 0.1717172194518303
- `proposer_scale`: approximately 0.30352814071666645
- `learning_rate`: 1e-5
- `grad_clip`: 1.0
- `group_size`: 4
- `unique_tasks`: 8

Verify these against the actual frozen configuration.

### 8.2 Experiment C: RL-only gradient alignment

For each proposed task x:

g_C(x) = g_RL(x)

The proposer reward uses the cosine alignment between the RL-only candidate gradient and the reference CE gradient:

r_C(x) = (1 + cos(g_C(x), g_ref)) / 2

Preserve the existing implementation's handling of undefined cosine similarities, zero gradients, invalid proposals, duplicates, and unscored-capacity proposals.

The currently inspected shared implementation assigns zero reward to candidates with undefined cosine similarity. Confirm and document the actual behavior.

Do not modify the student objective: the student still uses gated OPSD.

### 8.3 Experiment D: Combined-objective gradient alignment

For each proposed task x:

g_D(x) = g_RL(x) + λ g_SD(x)

The proposer reward is:

r_D(x) = (1 + cos(g_D(x), g_ref)) / 2

Use the same λ as the actual student update.

Do not use an unweighted OPSD gradient in D.

Do not introduce an independently tuned distillation coefficient solely for alignment scoring.

C and D must differ only in the candidate-gradient definition, apart from their resulting endogenous training trajectories.

### 8.4 Shared parameter updates

The shared-policy update has the conceptual form:

g_joint = g_RL + λ g_SD + α g_proposer

Preserve the existing optimizer, clipping, loss aggregation, and shared parameter update semantics.

Gradient alignment is a first-order proxy for the effect of a candidate student update on reference loss. It is not an exact prediction of the actual joint Adam update or downstream accuracy.

Do not differentiate through the computed alignment reward.

Do not introduce second-order gradients, nested solver optimization, or SOAR-style inner training loops.

Reuse the candidate student gradients already computed during training wherever possible.

---

## 9. Phase 2: Bounded alignment qualification

Before launching C/D, conduct a narrowly scoped qualification.

Target: approximately 2–4 representative training batches and no more than roughly 3–5 H200-hours unless there is a concrete reason to revise the budget.

Prefer reusing existing Task 21 rollout records together with their exact corresponding frozen policy checkpoints.

Only reuse a rollout for gradient analysis if its policy identity, behavior probabilities, teacher scoring context, and relevant token records are compatible with the checkpoint.

Do not use historical rollouts with mismatched current parameters and call the resulting gradients on-policy.

If the matching checkpoint is unavailable, generate a small fresh training-only batch.

### 9.1 Compare proposer rewards on identical tasks

For the same candidate proposals and student rollouts, compute:

- Difficulty-based reward.
- RL-only alignment reward.
- Combined RL+gated OPSD alignment reward.

Measure:

- Reference-gradient norm.
- Candidate RL-gradient norm.
- Candidate gated OPSD-gradient norm.
- Scaled OPSD-gradient norm.
- Combined candidate-gradient norm.
- Cosine similarities for C and D.
- Near-zero alignment frequency.
- Reward mean, variance, range, and quantiles.
- Within-parent valid-only reward variation.
- Distinct valid tasks with distinct rewards.
- Proposer-gradient norm.
- Runtime and peak memory overhead.

Distinguish genuine variation between different valid tasks from reward variation caused only by invalid proposals.

### 9.2 Examine uniform-reward groups

With four student rollouts and centered binary outcome rewards, uniformly successful or uniformly unsuccessful groups may have zero RL gradient.

For these groups:

g_C(x) = 0

However:

g_D(x) = λ g_SD(x)

may be nonzero.

This is a central research hypothesis, not a numerical error.

Investigate separately:

- All-failure groups.
- Mixed-reward groups.
- All-success groups.

Determine whether D produces differentiated alignment rewards where C cannot.

Do not assume that a nonzero OPSD gradient is necessarily useful. In particular, D may assign high utility to already-mastered tasks or tasks with misleading privileged guidance.

### 9.3 Practical acceptance criteria

C/D qualification should establish that:

1. Reference examples are correctly verified and frozen.
2. The reference gradient is finite and nonzero.
3. Candidate gradients are computed with the intended objectives.
4. Gradient signs and coordinate ordering are consistent.
5. Rewards are finite and correctly assigned.
6. On representative batches, alignment rewards can distinguish at least some distinct valid tasks.
7. Proposer gradients can be nonzero.
8. The estimated training cost is compatible with the available budget.

Do not require every batch to satisfy every learning-signal criterion.

Zero RL gradients on uniform-reward groups are expected.

Do not require the pilot to improve development accuracy.

Do not require convergence or statistical significance.

Do not impose arbitrary minimum cosine similarities or reward magnitudes.

If the initial qualification shows weak valid-only reward diversity, identify whether the issue is:

- A genuine limitation of the reward objective.
- Too few distinct valid proposals.
- An implementation error.
- Near-zero candidate gradients.
- A degenerate reference gradient.

Fix concrete implementation errors. Do not repeatedly redesign the reward to force a positive result.

Record the actual qualification evidence and satisfy the existing controller's `alignment_gate_hash` requirement legitimately.

Do not fabricate a gate hash or remove the controller check merely to launch the experiment.

### 9.4 Strict time limit for qualification

Aim to finish this phase within half a working day of engineering effort.

If the reference integration or controller adaptation becomes a substantial unresolved engineering project, or if the projected compute is incompatible with the deadline, stop expanding the implementation and follow the fallback plan in Section 13.

---

## 10. Phase 3: Execute experiments C and D

### 10.1 Matched experimental configuration

Run both C and D using seed 2026.

Target endpoint:

**512 attempted student training episodes per arm.**

Use the same seed-2026 initialization as the completed Control/A/B study.

Do not initialize C or D from a trained B checkpoint.

Preserve:

- Model and tokenizer revision.
- Initial LoRA weights.
- Shared parameterization.
- Proposer and student optimizer configuration.
- Original proposer-training parent pool.
- Parent scheduling algorithm.
- Proposal group size.
- Student rollout group size.
- Proposal validation and mutation constraints.
- Native tool interface.
- Student and teacher token budgets.
- OPSD and proposer coefficients.
- Development monitoring protocol.
- Evaluation task IDs and inference configuration.

The reference set is additional training guidance used specifically by C/D. Disclose this difference when comparing them with B.

The endogenous task distributions may diverge because different proposer rewards change the learned policy. This is intended.

Do not artificially force every method to generate identical tasks throughout online training.

### 10.2 Run identity and reproducibility

Give the new experiments immutable, distinguishable identities.

Do not rewrite the completed Control/A/B run identities or their data manifests.

Bind new C/D run identities to:

- Exact source commit.
- Frozen training configuration.
- PDB data hash.
- Reference artifact hash.
- Initial checkpoint identity.
- Model and tokenizer revision.
- Seed.
- Worker identity.
- Alignment qualification evidence.
- Prespecified training endpoint.

Check that the original initialization and common scientific settings match the existing seed-2026 study.

Do not demand identical configuration hashes when method-specific reference information or qualification metadata legitimately differs.

### 10.3 Scheduling

Prefer running C and D independently in parallel on separate H200 GPUs if authorized capacity and scheduler conditions permit.

Use the already qualified execution path.

Do not migrate to multi-GPU training merely to accelerate these small-model runs without evidence that it would reduce total time or cost.

Use existing checkpoint/resume support to survive scheduler time limits.

Choose realistic Slurm time limits based on historical runtime and available backfill opportunities.

Before launch, ensure sufficient storage headroom for checkpoints, batch records, and evaluations.

Avoid duplicate jobs and unnecessary repeated model exports.

### 10.4 Intermediate monitoring

Retain the existing development milestones at 128, 256, and 384 attempted episodes, with the final development evaluation at 512.

Use the existing 24-task development monitor.

Development results are diagnostic only.

Do not use final evaluation outcomes for hyperparameter tuning, checkpoint selection, or repeated reward redesign.

If one arm temporarily performs poorly, investigate concrete health metrics before deciding that it has failed.

Low or negative preliminary accuracy gains are not grounds to terminate an otherwise scientifically valid experiment.

---

## 11. Monitoring, observability, and operational efficiency

Use the existing W&B project and checkpoint-backed scalar telemetry.

Reuse already implemented metrics where available. Avoid building an entirely new observability system.

### 11.1 Training progress

Record:

- Attempted student episodes.
- Optimizer steps.
- Effective nonzero RL updates.
- Cumulative generated student tokens.
- Proposer-generated tokens.
- Cumulative allocated GPU-hours.
- Wall-clock time.
- Verified task repairs.
- Mixed-reward group fraction.
- Number of distinct valid generated tasks.
- Proposal acceptance, duplication, and rejection rates.

### 11.2 Gradient and objective health

Record:

- RL-gradient norm.
- Unweighted and weighted OPSD-gradient norms.
- Proposer-gradient norm.
- Reference-gradient norm.
- Joint-gradient norm.
- Actual parameter update norm.
- Optimizer clipping statistics.
- Behavior-policy probability discrepancies and support exits.
- Teacher-student log-probability gap.
- Gate-weight distribution.
- OPSD contribution relative to RL.
- RL/OPSD/proposer gradient interactions.

For C/D specifically, record:

- Alignment cosine distribution.
- Near-zero candidate-gradient frequency.
- Alignment reward diversity.
- Valid-only within-parent reward diversity.
- Alignment reward correlation with task difficulty.
- Fraction of tasks receiving informative alignment rewards.

Where a metric is unavailable or undefined, record it accurately as unavailable or undefined. Do not fabricate zero values.

### 11.3 Agent behavior

Record:

- Generated reasoning and action token counts where supported.
- Episode length and number of tool calls.
- Successful and failed tool calls.
- Accepted and rejected edits.
- Public-test success.
- Hidden-verifier success.
- Turn-limit, token-limit, and context-limit truncation.
- Other relevant termination categories.

### 11.4 Runtime profiling

Measure the incremental cost of:

- Student generation.
- Proposer generation.
- Teacher scoring.
- Candidate-gradient computation.
- Reference-gradient computation.
- Gradient-alignment reductions.
- Proposer backward pass.
- Tool execution and verification.
- Checkpointing.
- Development and endpoint evaluations.

Determine whether reference-gradient computation introduces a material overhead.

Optimize obvious redundant work, but do not start a major optimization project unless the bottleneck threatens experiment completion.

### 11.5 Recovery and storage

Preserve all verified scientific artifacts.

Use the existing reviewed job recovery mechanism.

Do not rerun completed jobs.

Do not overwrite previous checkpoints or results.

Avoid storing unnecessary duplicate rollout adapters and regenerable model exports.

If cleanup is necessary, identify ownership, verify artifact dependencies, and preserve scientific records.

Never delete unique experimental data or another user's files.

Account for failed and resumed allocations in the final compute ledger.

---

## 12. Phase 4: Endpoint evaluation

Evaluate the final C/D checkpoints using exactly the same frozen evaluation tasks and inference settings as the existing seed-2026 Control/A/B arms.

The primary comparison is at 512 attempted training episodes.

Use the existing primary PDB single-bug repair evaluation of 64 tasks.

When capacity permits, also evaluate the existing transfer groups:

- PDB-Multi: 8 tasks.
- Adapted BigCodeBench-derived PDB: 1 task.
- Independently generated BigCodeBench mutants: 9 tasks.

Keep these categories separate.

Do not report their aggregate as an official PDB benchmark score.

Reuse the existing initial-model evaluation when identical initialization, evaluation settings, and seed semantics establish that it is applicable.

Do not repeat identical initial evaluations merely because they belong to a different method directory.

### 12.1 Statistical analysis

For each primary comparison, report:

- Absolute success counts.
- Accuracy.
- Percentage-point difference.
- Relative improvement where meaningful.
- Paired task-level gains and losses.
- Family-clustered paired bootstrap confidence intervals.
- Number of evaluated tasks and parent families.

Important comparisons:

- A − Control
- B − Control
- B − A
- C − B
- D − C
- D − B

For C/D, use the matched seed-2026 results of B and other relevant arms.

Do not compare single-seed C/D scores against a two-seed aggregate B score as the primary matched comparison.

Report the two-seed Control/A/B results separately.

Task-bootstrap intervals measure task/family uncertainty conditional on the trained checkpoints. They do not estimate training-seed uncertainty.

Do not treat repeated evaluations of the same 64 tasks as independent datasets.

Do not describe one-seed improvements as statistically established algorithmic superiority.

The existing test set has already been examined through the completed Control/A/B study. This follow-up should preserve its frozen protocol and avoid outcome-driven tuning. Disclose the sequential nature of the study.

---

## 13. Fallback and longer-training decision

The priority remains C/D. December 2 is a soft presentation milestone; the hard final-report deadline is December 12.

### 13.1 If C/D qualification succeeds

Launch both experiments immediately.

Do not delay the launch to perform optional mechanistic analyses, additional hyperparameter sweeps, or extensive documentation work.

Conduct analyses of existing results concurrently.

### 13.2 If C/D cannot be completed in time

If full 512-episode runs become infeasible because of measured runtime, queue delays, or a persistent concrete blocker:

- Preserve all valid intermediate checkpoints.
- Prefer a common predeclared shorter exposure for C, D, and the relevant B comparator.
- Use 384 or 256 attempted episodes as fallback comparison points, selected according to available compute and completion feasibility, not observed final-test accuracy.
- Evaluate matched intermediate checkpoints if the necessary checkpoints exist and the evaluation budget permits.
- Clearly identify the comparison as a shortened exploratory study.

The current Task 21 controller requires a frozen completed endpoint for final evaluation. If intermediate evaluation is needed, implement a narrow, explicitly identified evaluation path without weakening or rewriting existing completed-study guarantees.

Do not compare D at 256 episodes with B at 512 episodes and describe it as an equal-exposure comparison.

If only one alignment arm can be completed, report what was obtained and explicitly state which research comparisons cannot be answered.

### 13.3 If alignment qualification is blocked

If C/D cannot be qualified within the bounded engineering and compute budget, pivot to extending the existing seed-2026 Control/A/B runs.

Do not spend the rest of the week repeatedly investigating the same alignment blocker.

### 13.4 Longer Control/A/B training

If sufficient authorized compute remains after completing C/D, or if C/D must be deferred, continue all three existing seed-2026 Control/A/B arms.

Preferred additional endpoint:

**512 → 1,024 attempted student episodes.**

A matched 768-episode endpoint is acceptable if the measured remaining capacity cannot support 1,024 episodes.

Choose the endpoint before examining new held-out evaluation results.

Include Control. Continuing only A and B would make it difficult to distinguish improved long-run performance from ordinary additional RL training.

Preserve all scientific settings.

### 13.5 Known continuation implementation issue

The existing Task 21 results documentation suggests that the endpoint is excluded from the scientific hash.

However, the inherited `Task20Control.science` property appears to include both the endpoint and development milestones.

The completed-stage marker also prevents the current runner from automatically extending an already completed run.

Confirm this behavior in:

- `src/selfplay_opsd/task19.py`
- `src/selfplay_opsd/task20.py`
- `src/selfplay_opsd/task19_train.py`
- `src/selfplay_opsd/task21.py`
- Relevant HPC controller code.

Implement a narrowly scoped, provenance-preserving continuation path if required.

Prefer a separately identified continuation linked to the verified parent run, or an equivalently safe versioned mechanism.

Do not change the identity of the original completed 512-episode study.

Restore the full training state:

- Shared LoRA parameters.
- Optimizer state and step counters.
- RNG state.
- Parent schedule position.
- Proposer group counters.
- Episode and token counters.
- Seen-task and curriculum state.
- Protected checkpoint information.
- Other state required by the existing training implementation.

Verify that the optimizer and model state match the original endpoint before continuing.

Do not silently reset the optimizer.

Do not restart from the initial model.

Do not simply delete the completion marker or disable identity verification.

A small verified resume check is sufficient. Do not rebuild the entire checkpoint system.

The original 512-episode results must remain independently reproducible and unchanged.

---

## 14. Phase 5: Mechanistic analyses

These analyses are essential even if C or D does not improve accuracy.

The project should explain what happens during self-play and why the different objectives may behave differently.

Prefer analyses using existing logs, checkpoints, and rollout records.

### 14.1 Learning efficiency

Produce learning curves using:

1. Attempted student episodes.
2. Cumulative generated student tokens.
3. Optimizer steps.
4. Cumulative allocated H200-hours.

Where feasible, include total generated tokens and other meaningful compute measures.

Clearly distinguish training-only compute from end-to-end experiment cost, including evaluation and failed allocations.

Use fixed development-monitor accuracy for learning curves rather than treating success rates on evolving generated tasks as a stationary benchmark.

Show raw evaluation points. Any smoothing should be clearly identified.

For Control/A/B, show both seeds and their mean where appropriate.

For C/D, show seed 2026 individually.

Do not assign fabricated seed variance to C/D.

### 14.2 RL–OPSD–proposer gradient interaction

Investigate the relationships between:

- cos(g_RL, g_SD)
- cos(g_RL, g_proposer)
- cos(g_SD, g_proposer)
- cos(g_C, g_ref)
- cos(g_D, g_ref)

Analyze the relative gradient magnitudes:

- ||g_RL||
- λ ||g_SD||
- α ||g_proposer||
- ||g_joint||

Determine whether gated OPSD tends to complement RL, conflict with RL, or dominate it.

Determine whether proposer updates reinforce or interfere with student updates under shared parameters.

Distinguish observed gradient geometry from causal evidence about eventual performance.

Do not claim that pairwise gradient alignment alone proves beneficial learning.

### 14.3 Central analysis: Does OPSD improve the proposer learning signal?

This should be a major focus of the presentation.

For the same candidate tasks and frozen policy snapshots, compare difficulty, C, and D rewards.

Investigate:

- Near-zero gradient frequency.
- Number of distinct meaningful rewards.
- Within-parent valid-only reward variance.
- C versus D rank correlation.
- Difficulty versus alignment rank correlation.
- Reward differences across all-failure, mixed, and all-success groups.
- Relationship between gradient norms and alignment.
- Examples of tasks whose estimated utility changes substantially when OPSD is included.

Separate tasks into groups according to their four-rollout success count: 0, 1, 2, 3, or 4.

This analysis should determine whether incorporating dense OPSD feedback:

- Recovers alignment signals for tasks lacking RL gradients.
- Changes task rankings within the same difficulty category.
- Improves the diversity of valid proposer rewards.
- Introduces potentially undesirable reward behavior.

If D has more nonzero alignment gradients but does not improve downstream performance, that is still an important research finding.

### 14.4 Optional small counterfactual utility probe

If inexpensive using existing checkpoints and gradients, investigate whether the alignment score predicts actual short-term reference-loss improvement.

For a small set of training-only candidate tasks:

- Restore the same frozen policy before each probe.
- Apply a controlled candidate student update.
- Measure loss changes on a separate training-domain diagnostic reference set, if available.
- Compare observed loss changes with the alignment-based prediction.
- Restore model, optimizer, and RNG state after every probe.

This is optional and should not delay C/D.

Do not use the final evaluation tasks for these probes.

Do not introduce expensive nested training loops.

### 14.5 Curriculum and proposer behavior

Analyze how proposed tasks evolve over training.

Where available, compare:

- Valid proposal rate.
- Number of unique tasks and parents.
- Duplicate task frequency.
- Mutation categories.
- Edit sizes.
- Source complexity.
- Student success-rate distributions.
- Semantic verification failures.
- Invalid or unscored proposals.
- Repeated task families.
- Reward diversity.

Investigate whether C or D encourages more useful, more diverse, or more difficult tasks than B.

Avoid claiming increasing environment complexity unless the data actually demonstrates it.

Select representative qualitative examples showing:

- Original correct program.
- Proposer-generated bug.
- Student attempt and observed failure or repair behavior.
- Training-only privileged teacher information.
- Difficulty-based and alignment-based task assessments, where available.

Keep examples sanitized and suitable for public presentation. Do not expose private evaluation fixtures or hidden test data.

### 14.6 Failure-mode analysis

Task 21 reported that 29–38 of 64 primary evaluation episodes per trained arm encountered truncation.

This is potentially a major bottleneck.

Analyze whether performance differences arise from:

- Better bug localization.
- More accurate code edits.
- More efficient tool selection.
- Fewer rejected edits.
- Shorter or more focused reasoning.
- Fewer turn-limit failures.
- Fewer token-limit or context-limit failures.
- Improved public-to-hidden test generalization.
- Other observable behavioral differences.

Compare initial models and trained models across the existing common evaluation tasks.

Report absolute counts, denominators, and per-category changes.

Do not automatically interpret truncation as proof that increasing the token budget would solve the task.

If time and compute permit, conduct a small separately labeled inference-budget sensitivity analysis on a prespecified subset of tasks.

Use identical increased inference limits across compared methods.

Do not alter the primary evaluation or training budgets retroactively.

### 14.7 Effective distillation strength confound

A and B use the same nominal λ, but gating changes both the direction and magnitude of their effective distillation gradients.

Therefore, the current B-versus-A comparison does not fully isolate selective token weighting from overall distillation strength.

Calculate and report the observed weighted gradient ratios.

If substantial additional compute remains after the primary goals, consider a strength-matched ungated OPSD ablation:

λ_A_matched = λ_B × ||g_gated|| / ||g_ungated||

Calibrate this on representative training-only batches.

Use the existing `calibrated_ungated_scale` mechanism if appropriate.

Do not tune the coefficient based on final-test outcomes.

This is a stretch goal, not a prerequisite for completing the class project.

---

## 15. Required figures and tables

Prepare publication-quality figures and tables suitable for the December 2 presentation.

Prioritize clarity over quantity.

### Main results

Create a table including:

| Method | Seed | Training episodes | Optimizer steps | Primary accuracy | Gain over Control | H200-hours |
|---|---|---|---|---|---|---|

Report the matched five-arm seed-2026 comparison separately from the two-seed Control/A/B summary.

### Required figures

**Figure 1: Research framework**

Show:

- Proposer.
- Executable coding task.
- Student agent.
- Environment and verifier.
- Privileged teacher context.
- Sparse RL and dense OPSD feedback.
- Reference-gradient-based proposer reward.
- Shared parameter update.

**Figure 2: Primary held-out results**

Compare Initial, Control, A, B, C, and D.

Include clearly identified task-bootstrap uncertainty where appropriate.

**Figure 3: Learning efficiency**

Show development accuracy versus:

- Attempted episodes.
- Generated tokens.
- GPU-hours.

**Figure 4: Gradient-alignment mechanism**

Compare C and D using:

- Near-zero candidate-gradient frequency.
- Alignment rewards by solver success-rate category.
- Valid-only proposer reward diversity.
- Task-ranking changes where informative.

**Figure 5: Training dynamics and gradient interaction**

Show:

- RL versus weighted OPSD gradient magnitudes.
- Relevant gradient cosine similarities.
- Effective RL update frequency.
- Relevant gate behavior.

**Figure 6: Failure modes**

Compare truncation, tool errors, unsuccessful repairs, public-pass/hidden-fail cases, and verified successes.

**Optional Figure 7: Curriculum evolution**

Show representative changes in proposed tasks and associated training signals.

Do not fabricate any missing curves, intermediate checkpoints, confidence intervals, or experimental outcomes.

Generate both PDF and PNG versions where practical.

Make figures reproducible from scripts and frozen aggregate records.

---

## 16. Research interpretation and reporting

The final interpretation must be evidence-driven.

Do not assume that D must outperform C, C must outperform B, or B must maintain its current advantage during longer training.

Consider the following possible outcomes.

### If C outperforms B and D outperforms C

This supports the hypothesis that gradient-aligned task generation is useful and that accounting for the actual student objective improves the proposer reward.

However, describe single-seed evidence cautiously.

### If C and D underperform B

Difficulty-based proposer rewards may be more robust under the current limited training exposure, small rollout groups, reference set, and shared-policy dynamics.

Investigate reward variance, gradient noise, reference alignment, and computational overhead.

Do not generalize this result to all gradient-based self-play methods.

### If D produces richer gradients but does not outperform C

Distinguish the ability to provide a nonzero directional signal from the ability to identify genuinely useful training tasks.

This may be one of the most interesting results.

### If longer training strengthens B's advantage over A

Investigate whether gating improves sustained learning or mitigates harmful distillation dynamics.

### If A or Control catches up during longer training

The main effect of OPSD or gating may concern early learning efficiency rather than asymptotic performance.

### If all methods remain statistically indistinguishable

Report the effect estimates, uncertainty, observed learning mechanisms, and limitations.

A carefully executed negative or inconclusive experiment is preferable to a misleading positive claim.

### Required limitations

Explicitly acknowledge:

- Small-scale 4B model.
- Limited optimizer steps and training exposure.
- Two training seeds for Control/A/B.
- One training seed for C/D.
- Finite and relatively small evaluation sets.
- Limited cross-distribution evaluation.
- Reused development monitor.
- Sequential experiment design.
- Gating/effective-strength confounding.
- Differences in actual compute and generated-token exposure.
- Additional reference supervision for C/D.
- Alignment as a first-order proxy rather than direct learning-progress measurement.
- Potential interference between proposer and student objectives under shared parameters.
- High observed truncation rates.
- Any remaining execution or verification limitations.

Do not present this as demonstrated recursive self-improvement or general coding-agent superiority unless the evidence genuinely supports such claims.

---

## 17. Timeline and experiment cutoff

Use the following target schedule, adapting to actual queue conditions.

### November 24–25

- Audit existing evidence and resources.
- Implement the minimal PDB reference-gradient integration.
- Run the bounded alignment qualification.
- Freeze configurations.
- Launch C and D.

### November 25–28

- Execute and monitor C/D training.
- Recover interrupted jobs through the existing workflow.
- Analyze completed Control/A/B results concurrently.
- Prepare framework figures, learning curves, and diagnostic analyses.

### November 28–30

- Complete matched endpoint evaluations.
- Reconcile allocations and experimental records.
- Perform final comparisons.
- Complete the optional longer-training study only if feasible without jeopardizing the primary goals.
- Produce final figures and tables.

### December 1

- Freeze presentation results.
- Complete documentation and presentation-ready summaries.
- Identify unresolved questions and future work.

### December 2

- Five-minute presentation of the best available findings; final results may still be pending.

### December 3–11

- Continue and analyze useful experiments; prepare and review the final report.

### December 12

- Submit the final report (hard deadline).

November 30 is a useful target for assembling preliminary presentation material, not an experimental cutoff. Continue useful experiments and analysis after the December 2 presentation when they can inform the final report due December 12.

Do not launch expensive new experiments late in the schedule unless their completion and evaluation are realistically feasible.

Prioritize a coherent completed study over partially completed optional experiments.

---

## 18. Explicitly out-of-scope work

Unless required to fix a demonstrated correctness problem, do not:

- Change to a larger model.
- Replace PDB with another dataset.
- Restart the entire experiment.
- Introduce additional OPSD algorithms.
- Implement SOAR-style nested optimization.
- Expand to SWE-bench or substantially different agent environments.
- Perform extensive hyperparameter sweeps.
- Change the training or inference limits of the primary study.
- Replace the execution worker.
- Introduce a major distributed-training refactor.
- Redesign the checkpointing system.
- Require extensive new infrastructure qualification.
- Reevaluate every historical checkpoint.
- Repeat already completed baseline evaluations.
- Tune methods based on held-out test outcomes.
- Remove inconvenient negative results.

These are possible future research directions, not current priorities.

---

## 19. Documentation and deliverables

Follow the repository's existing documentation conventions and `AGENTS.md`.

Create or update appropriately named Task 22 documentation, such as:

- `docs/task/22-pdb-gradient-alignment-spec.md`
- `docs/plan/22-pdb-gradient-alignment-plan.md`
- `docs/investigations/task22-alignment-qualification.md`
- `docs/investigations/task22-status.md`
- `docs/investigations/task22-results.md`
- `docs/investigations/task22-mechanistic-analysis.md`
- `docs/task_completion/22-implemented-pdb-gradient-alignment.md`
- `docs/figures/task22/`

Adapt filenames to the repository's established conventions.

Keep the documentation concise, evidence-based, and focused on reproducibility.

Do not spend excessive time producing overlapping planning documents before implementing the experiment.

### Required final deliverables

1. PDB-compatible reference-gradient implementation.
2. Frozen reference artifact and verification evidence.
3. Bounded C/D qualification report.
4. Frozen experiment configurations and launch records.
5. Verified completed checkpoints for C/D, or explicitly documented fallback endpoints.
6. Matched endpoint evaluation results.
7. Reconciled incremental and cumulative GPU allocation ledger.
8. W&B training and analysis summaries.
9. Reproducible analysis scripts.
10. Publication-quality figures and tables.
11. Mechanistic analysis of gradient signals, curriculum behavior, and agent failures.
12. Presentation-ready conclusions and limitations.
13. Updated README describing the current project status accurately.
14. Clear instructions for reproducing or extending the study later.

If a deliverable cannot be completed, document exactly what is missing, why, and how it affects the research conclusions.

Never report an unfinished job as completed.

---

## 20. Final instructions

We have already spent substantial time and compute on this project. Earlier attempts repeatedly became stuck in infrastructure preparation, overly strict readiness requirements, or prolonged investigations without completing the actual experiments.

Do not repeat that pattern.

The current Task 21 results represent a meaningful improvement in experimental feasibility. Build directly upon them.

Your first responsibility is to identify the smallest correct implementation required to run C and D on PDB.

Your second responsibility is to execute those experiments within the available budget and deadline.

Your third responsibility is to produce scientifically meaningful analysis regardless of whether the hypotheses are supported.

Use the following decision hierarchy throughout execution:

**Correctness and reproducibility → complete matched scientific comparisons → useful mechanistic evidence → computational efficiency → optional engineering improvements.**

Do not confuse perfect infrastructure with successful research.

Do not stop merely because the evidence is not yet statistically significant.

Do not stop after proposing what could be done.

Begin by inspecting the current repository, actual experimental artifacts, and available Yale compute. Then proceed directly with the narrowly scoped PDB reference-gradient implementation, qualification, and C/D execution.

Preserve the existing Control/A/B results and use them as the matched baseline.

Present defensible available findings on December 2, including any unfinished work. Aim for a completed, well-analyzed five-arm PDB study, including the seed-2027 C/D extension, for the final report due December 12.

Longer training and additional ablations are secondary objectives.