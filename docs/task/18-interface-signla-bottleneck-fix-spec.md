# Codex Task: Resolve Coding-Agent Interface and Learning-Signal Bottlenecks, Then Execute the Next Controlled Experiments

## 0. Your role and mission

You are the lead research engineer for my coding-agent self-play research project.

**Repository:** https://github.com/tatsu432/self-play-opsd-coding-agent

Your responsibility is to **inspect the current implementation, identify the remaining high-impact scientific and engineering problems, implement appropriate solutions, run bounded controlled experiments, analyze their results, and deliver a defensible research conclusion.**

Do not stop after writing a plan.

We have already spent substantial time and GPU compute debugging this project. I want you to use the existing evidence and implementation rather than repeatedly rebuilding, revalidating, or rerunning work that is already complete.

Prioritize, in this order:

1. Scientific correctness and trustworthy learning signals.
2. Reliable code-editing interfaces and actual repair capability.
3. Generalization to unfamiliar programming problems.
4. Stable, efficient, reproducible training.
5. The original OPSD/RLVR research hypotheses.
6. More speculative ideas, only when supported by the earlier experiments.

The objective is not to make every conceivable improvement to the repository. **Identify and execute the smallest set of experiments that gives us the most scientifically useful information per unit of compute.**

Make routine engineering decisions autonomously. Ask me only when a genuine external authorization, security constraint, or material scope decision prevents progress.

---

## 1. Understand and preserve the current state

Before modifying code, inspect the latest `main` branch, existing branches, documentation, configurations, and implementation.

Start with:

- `AGENTS.md`
- `README.md`
- `docs/methods.md`
- `docs/investigations/task14-research-decision.md`
- `docs/investigations/task15-final-results.md`
- `docs/investigations/task16-generalization-checkpoint-decision.md`
- `docs/investigations/task17-research-decision.md`
- `docs/investigations/task17-repair-interface-results.md`
- `docs/investigations/task17-objective-results.md`
- `docs/task_completion/15-implemented-stabilize-training-complete-ab.md`
- `docs/task_completion/17-implemented-repair-interface.md`

Inspect relevant source files, particularly:

- `environment.py`, `protocol.py`, `mutations.py`
- `sandbox.py`, `worker_runtime.py`, `worker_policy.py`
- `data.py`, `evaluation.py`, `objectives.py`
- `policy.py`, `trainer.py`, `verl_backend.py`
- Task 15–17 analysis scripts, test suites, and HPC launch infrastructure.

The last observed `main` commit is `48b48398871519640e0b808e37daa1aec555e002` (November 16, 2026). Check whether newer commits or active jobs exist.

Read the latest completion records as authoritative where older plans or `AGENTS.md` sections retain historical intermediate statuses.

### 1.1 What has already been accomplished

The original project compares:

| Experiment | Proposer reward | Student objective |
|---|---|---|
| A | Difficulty | GRPO + ungated OPSD |
| B | Difficulty | GRPO + gated OPSD |
| C | RL-gradient alignment | GRPO + gated OPSD |
| D | RL+OPSD gradient alignment | GRPO + gated OPSD |

**Experiments A and B have already completed three seeds each. Do not rerun them from scratch.**

On MBPP+-derived held-out repairs, A achieved 57–64/115 and B achieved 68–71/115, against a base score of 7/115.

On HumanEval+-derived repairs, both remained at approximately 0–3/116, near the base score of 2/116.

Task 16 identified 4,301 rejected patches versus just 13 accepted patches across six trained HumanEval+-derived evaluations.

Task 17 found that mechanically reconstructing rejected edits yielded correct saved states in 242 previously failed episodes. These are retrospective replay results, not new end-to-end agent successes.

A controlled strict-diff versus exact-text comparison on 40 training-only families found:

| Snapshot | Strict diff | Exact-text replacement |
|---|---:|---:|
| Fresh Qwen3-4B base | 5/40 | 19/40 |
| Previously trained B-2026 | 27/40 | 21/40 |

Thus, simpler editing substantially improved the fresh base model but did not establish a gain for the model trained under the old interface.

Task 17 also completed a fixed-curriculum student-objective comparison with three seeds and 80 attempted episodes per arm:

| Student objective | 2026 | 2027 | 2028 |
|---|---:|---:|---:|
| GRPO | 10/16 | 8/16 | 9/16 |
| Gated OPSD | 10/16 | 8/16 | 10/16 |
| Nominal ungated OPSD | 11/16 | 11/16 | 10/16 |
| Calibrated ungated OPSD | 9/16 | 9/16 | 9/16 |

These findings **do not establish robust superiority of gating**.

The proposer also has an information bottleneck: only about 5.1–7.7% of historical attempted proposer groups distinguish difficulty rewards among valid candidates.

Full C/D training remains deferred.

### 1.2 Preserve historical evidence

Do not modify original Task 15–17 configurations, checkpoints, results, dataset manifests, or scientific execution identities.

Do not silently resume a historical experiment with changed model conditioning, reward definitions, or tool interfaces.

Create a clearly named new study/branch for new work. Reuse trustworthy analysis and runtime infrastructure rather than duplicating it.

Keep raw trajectories, private oracles, credentials, and sensitive run artifacts outside Git and public W&B uploads.

The November 16 update records a checksum-verified archival copy of the Qwen3-4B model in Yale project storage. Consult `docs/operations.md` before redownloading the model. The archive is not automatically configured as the Hugging Face cache; use an explicit verified restoration or model-loading path.

---

## 2. Formulate the scientific questions correctly

We should distinguish at least four mechanisms:

**RQ1 — Interface:** How much of the observed repair failure results from the agent's inability to express or apply valid code edits?

**RQ2 — Semantic capability:** Once an intended edit is successfully applied, can the agent produce a correct repair that generalizes to unseen inputs and unfamiliar tasks?

**RQ3 — Training objective:** Under a reliable and fixed interface, does OPSD, particularly gated OPSD, improve learning compared with GRPO or ungated OPSD?

**RQ4 — Proposer learning:** Do adaptive task proposals contain useful learning signals beyond task validity and repeated examples? Can alternative proposer rewards meaningfully change the student update direction?

Do not confound these questions in a single uncontrolled experiment.

In particular:

- Patch acceptance is not the same as verified repair.
- Public-test success is not the same as hidden-test success.
- More training updates do not necessarily imply greater learning.
- In-distribution improvement is not proof of cross-distribution generalization.
- More valid proposals do not necessarily imply more informative training tasks.
- An auxiliary OPSD gradient is not necessarily a beneficial gradient.
- A difference between two independently co-evolving A/B runs does not isolate the effect of gating.

A valid negative result is valuable. Do not tune until one method looks better.

---

## 3. Priority 0: Resolve high-impact numerical correctness questions

Before new RL experiments, investigate the remaining rollout-versus-rescoring log-probability discrepancy.

The previous Task 17 study reported a maximum difference of 5.4899 in one calibrated run, with a small fraction of sampled tokens exceeding 0.2 in that iteration. The causal effect remains unknown.

This is scientifically important because incorrect behavior-policy probabilities can affect importance ratios, clipping, and effective optimization directions.

### Required investigation

Using existing saved trajectories or a small bounded diagnostic:

1. Reproduce sampling and rescoring at an identical frozen model snapshot.
2. Compare exact token IDs, assistant masks, chat-template boundaries, EOS handling, prompt prefixes, and multi-turn histories.
3. Verify temperature, top-p, and other sampling/log-probability conventions.
4. Verify active LoRA adapters, vLLM weight synchronization, optimizer state, numerical precision, and any stale-model effects.
5. Distinguish expected backend numerical differences from mismatches that change PPO/GRPO clipping or gradients.
6. Perform a targeted counterfactual update comparison with the complete original model and optimizer state restored between variants.

Report absolute-difference distributions, affected token fractions, clipping/sign changes, gradient differences, and actual optimizer-update differences.

**Do not replace original behavior-policy denominators with rescored probabilities simply to make diagnostics look better.**

If there is a reproducible defect, fix it, add a regression test, and assign a new scientific identity to affected future experiments.

If the discrepancy is not causally important within the tested regime, document the evidence and move on. Do not allow an unresolved but inconsequential numerical warning to trigger another indefinite debugging cycle.

Use a bounded diagnostic budget. Do not retrain the original A/B study as part of this investigation.

---

## 4. Priority 1: Improve and evaluate the coding-agent interface

The next main scientific task is to determine which interface allows our current Qwen3-4B model to reliably edit Python code.

### 4.1 Understand the current interface

Our current agent operates through a custom plain-JSON tool protocol with:

- `read_file`
- `apply_patch` or an alternative editor
- `run_tests`
- `finish`

The virtual workspace contains:

- `README.md`: specification, read-only
- `src/solution.py`: buggy source, editable
- `tests/test_public.py`: public examples, read-only

The historical `apply_patch` protocol requires a strict unified diff, including exact file headers, hunk coordinates, hunk counts, and source context.

The repository also implements exact-text replacement and a locally tested optional normalized-diff path.

**Do not assume Codex's `apply_patch`, Claude Code's Edit tool, our strict unified diff, and Qwen-native tool calling are equivalent.**

Our current model is `Qwen/Qwen3-4B-Instruct-2507`, a general instruction-tuned model with coding capabilities, not a Qwen3-Coder-specific checkpoint.

The existing harness uses custom plain-JSON responses rather than the model's documented Qwen tool-call serialization.

### 4.2 Candidate interfaces

Evaluate the following in order of implementation cost and scientific usefulness:

**Control: strict unified diff**

Preserve the original implementation for comparison.

**Candidate 1: normalized diff**

Use the existing optional mechanism to recover edits with uniquely matching original context while ignoring incorrect mechanical hunk metadata.

This is particularly informative because it can preserve the old tool prompt and model conditioning.

**Candidate 2: exact-text replacement**

Use the already implemented unique-substring replacement.

The previous Task 17 results establish that its effect depends on the model snapshot. Reuse those results; do not rerun identical experiments.

**Candidate 3: simplified patch or line-range editor**

Consider a Codex-style simplified patch or SWE-agent-style line editor that avoids strict hunk coordinates/counts.

Implement at most one new editing mechanism initially, and only if it offers a clear potential benefit not captured by the first two candidates.

Do not build a complete general-purpose terminal/SWE-agent framework.

### 4.3 Qwen-native tool calling

Investigate whether the model's documented function-call format improves action validity compared with our custom JSON serialization.

Treat tool-call **serialization** and code-editing **semantics** as separate interventions.

For example, compare plain-JSON versus Qwen-native serialization while keeping the same underlying `replace_text` action.

If this requires substantial backend changes, first implement a very small compatibility probe.

Do not introduce constrained decoding without explicitly handling the resulting sampling distribution and on-policy log probabilities.

Do not add SFT at this stage.

### 4.4 Improve editing observations only where useful

Review whether the current responses provide sufficient information for the agent to recover from mistakes.

At present, a successful edit generally returns an `ok` status and source hash, without a readable excerpt of the modified source. The public test runner primarily reports aggregate pass counts.

Consider a compact response containing:

- Whether the edit was accepted.
- The resulting source version/hash.
- A bounded excerpt of the changed lines.
- A concise rejection category when editing fails.
- Useful public-test failure details, without any hidden-test leakage.

Keep observation budgets bounded and comparisons fair.

Do not mix observation improvements into the editing-format comparison unless they form an explicitly separate factorial intervention.

### 4.5 Experimental design

Use frozen model snapshots, matched training-only tasks, identical generation/evaluation budgets, and fixed interface definitions.

At minimum, compare the preserved fresh base snapshot and a checksum-verified trained checkpoint when scientifically useful.

Prefer reusing existing Task 17 results for completed cells and running only the missing comparisons.

Use a task pool that respects parent/family separation. Audit whether suitable unused training-only families actually exist; do not assume there are unlimited unused MBPP+ parents.

**Do not tune the interface using the existing HumanEval+ external results.** That split has already been inspected repeatedly and cannot be described as untouched confirmation.

Primary metric:

- Fully verified repair rate per task/family.

Secondary metrics:

- Valid tool-call rate.
- Edit acceptance.
- Tasks with an applied edit.
- Public-test success.
- Hidden-test success.
- Semantic failures conditional on successfully applied edits.
- Error/rejection taxonomy.
- Agent turns, generated tokens, and GPU-hours.
- Performance stratified by source tokens, specification tokens, and code complexity.

Report paired task/family uncertainty separately from training-seed uncertainty.

Select the next interface based on **verified repair capability, robustness, and cost**, not formatting success alone.

Deliver a clear interface decision: selected implementation, evidence, limitations, and alternatives rejected.

---

## 5. Priority 2: Preserve and audit the sandbox/reward boundary

The agent must never gain arbitrary access to the trainer's filesystem, private oracle, model files, credentials, or trusted test results.

Our current implementation uses a trusted virtual workspace, a restricted editing tool, and isolated Docker/Apptainer execution workers.

Keep this architecture.

### Required invariants

1. Only `src/solution.py` is writable by the agent's tool interface.
2. The specification and public-test file remain read-only.
3. Hidden test cases and correct expected outputs do not enter student prompts, tool observations, or writable workspace files.
4. The self-teacher may receive privileged correct source/inverse-patch information only through the explicitly training-only scoring path.
5. Generated Python runs in the accepted isolated worker, not directly in the trainer or coordinator process.
6. The sandbox has no access to host credentials, home directories, project data, GPU devices, arbitrary network access, or the Docker socket.
7. Resource limits and cleanup work on the actual execution host.
8. Infrastructure failures are distinguished from candidate/program failures.

Remember that candidate programs necessarily receive test inputs during execution; expected outputs are retained in the trusted supervisor. Do not claim cryptographic secrecy of hidden inputs from arbitrary malicious code running inside its own process.

Add or reuse focused tests for disallowed paths, attempts to edit test files, restricted imports, sandbox isolation, and data/oracle leakage.

Use the existing accepted Yale isolation mechanism where its invariants are unchanged. Do not introduce unnecessary rigid requirements such as the previously problematic user-controlled 32-process cgroup limit.

However, **never execute untrusted model-generated Python on the trainer host merely to bypass an isolation blocker.**

A real isolation failure is a hard blocker; redundant paperwork or repeated already-validated checks are not.

---

## 6. Priority 3: Design a partial-reward ablation

Our current student reward is binary:

\[
R_{\mathrm{binary}}(x,y)=\mathbf 1[\text{all public and hidden tests pass}]
\]

This is a clean task-completion objective, but it gives no gradient discrimination among multiple incomplete solutions if all their rewards are zero.

Investigate whether partial rewards provide more informative training while preserving correctness.

### 6.1 Proposed reward variants

**R0 — Binary control**

Keep the existing reward unchanged.

**R1 — Hidden-training-test pass fraction**

\[
R_{\mathrm{partial}}(x,y)=
\frac{1}{|H_x|}
\sum_{h\in H_x}\mathbf 1[\text{test }h\text{ passes}]
\]

Here, \(H_x\) contains hidden *training* test cases only, not held-out final evaluation cases.

**R2 — Partial reward with completion bonus**

\[
R_{\mathrm{hybrid}}=(1-\lambda)R_{\mathrm{partial}}
+\lambda R_{\mathrm{binary}}
\]

Use a small predeclared set of reasonable \(\lambda\) values only if the initial reward analysis shows this is useful.

### 6.2 Important distinctions

Public tests are available for the agent to read and use for debugging. Hidden training tests are better suited to the primary learning signal because the model cannot directly memorize their expected outputs from the prompt.

Do not expose hidden pass fractions to the student as a tool observation unless that is explicitly part of a different feedback experiment.

Keep complete-repair success as the **primary evaluation metric** regardless of training reward.

Do not assume more partial test passes means better semantic generalization. Test suites may contain correlated or redundant cases.

Audit reward hacking and overfitting, including suspicious source modifications, public-only memorization, unexpected runtime behavior, and cases that improve pass fraction while degrading complete-repair performance.

Ensure partially correct initial mutants are accounted for; report raw test-pass fractions and improvement relative to the initial buggy source where useful.

### 6.3 Critical proposer distinction

Our proposer currently uses the binary-solve-rate difficulty reward:

\[
R_{\mathrm{proposer}}=4p(1-p),
\]

where \(p\) is the fraction of fully successful student rollouts.

Do **not** silently redefine \(p\) as the fraction of hidden tests passed.

Student trajectory reward, student full-solve rate, and proposer difficulty are distinct quantities.

Initially keep the proposer reward unchanged and isolate the student reward intervention.

### 6.4 Efficient execution

Before any training, inspect saved trajectories or perform a bounded reward-only probe.

Measure whether partial rewards actually increase:

- Within-task reward variance.
- Nonzero group-relative advantages.
- Informative student updates.
- Correlation with eventual complete-repair success.

Also measure the additional verifier cost. If running hundreds of hidden cases per rollout dominates compute, evaluate a deterministic, prespecified training-test subsample, with its sampling variability reported.

Only conduct a short matched RL training comparison if the reward probe produces a meaningfully different, trustworthy signal.

This is a secondary study. Do not let it delay the interface investigation.

---

## 7. Priority 4: Run the next controlled student-training study

Once a suitable interface is selected and the numerical diagnostics are sufficiently understood, conduct the smallest informative RL comparison.

Use:

- The same frozen 4B base initialization within each seed.
- The same task pool and task order across student-objective arms.
- The same selected interface.
- The same binary reward for the primary comparison.
- The same action, token, and attempted-episode budgets.
- The same independent development evaluation tasks.
- Appropriate pairing across seeds.

Prioritize:

1. GRPO only.
2. GRPO + gated OPSD.
3. GRPO + nominal ungated OPSD.

The calibrated ungated control may be added if necessary to interpret effective auxiliary strength, but the earlier Task 17 results already provide useful evidence.

Start with a short bounded pilot, preferably reusing the Task 17 fixed-curriculum runner. An 80–160-episode regime is a reasonable starting point subject to measured cost and available learning signal.

Use at least two paired seeds if a new training comparison proceeds, and a third only when it materially improves the inference within the compute ceiling.

### Required training diagnostics

Record:

- Verified student repairs.
- Mixed-reward groups.
- Uniform-success and uniform-failure groups.
- Optimizer steps and skipped zero-gradient updates.
- GRPO gradient magnitude.
- OPSD gradient magnitude and teacher/student gap distributions.
- Gating behavior, including positive and negative gaps.
- Invalid action/schema rate.
- Accepted versus rejected edits.
- Public and hidden evaluation outcomes.
- Generated tokens, sampled episodes, and GPU-hours.
- Rollout/rescoring mismatch statistics.
- Checkpoint identity and telemetry completeness.

For every comparison, distinguish matched episodes from matched tokens, optimizer updates, and actual compute.

### Decision criteria

Continue only if the numerical implementation is credible and the pilot produces useful learning evidence.

Scientific underperformance is a result, not an automatic reason to retry with another seed or adjust hyperparameters.

Stop or repair only for a reproducible scientific-integrity or execution defect.

Do not use held-out test performance to choose checkpoints or hyperparameters.

Most importantly, do not repeat the original full online A/B study merely because a new interface exists. The first question is whether the new interface yields a clearer controlled comparison at substantially lower cost.

---

## 8. Priority 5: Revisit the proposer only after the student-side bottlenecks

The original C/D question remains interesting:

Does including the OPSD-gradient component in the candidate task-value signal improve proposer learning?

But the existing proposer has insufficient distinct-reward candidate groups.

Before a full C/D run, perform a training-only proposer-signal study.

### Required diagnostics

Measure:

- Proposal validity.
- Unique versus repeated proposed mutations.
- Distinct bug families and structural coverage.
- Number of valid candidates per proposer group.
- Number of groups with distinct rewards.
- Student solve-rate distributions.
- Candidate-gradient norms.
- Reference-gradient stability.
- RL-only versus RL+OPSD candidate alignment scores.
- Repeat-controlled proposer gradient and actual optimizer-update differences.

Use identical frozen student snapshots, candidate proposals, and optimizer state for counterfactual comparisons.

Do not infer proposer learning from validity improvements alone.

Only consider a bounded C/D pilot if the new candidate distribution produces sufficiently informative, reproducible reward differences and distinct optimizer directions.

Otherwise, document why C/D remains unidentifiable under the current setup.

### Optional alternative proposer rewards

If the current difficulty measure is still uninformative, investigate a small number of principled alternatives:

- Difficulty rewards targeting intermediate solver success rates.
- Small, measured learning-progress proxies.
- Validity/novelty-aware proposer rewards.
- Gradient alignment to a trusted training-only reference set.

Do not run a large proposer-reward search.

Reference-task gradients must be drawn from appropriate training-only data, never from final evaluation labels.

---

## 9. Stretch goal: Adaptive sharing between proposer and student

An additional research idea is to adaptively transfer knowledge between proposer and student rather than keeping the roles always shared or always separated.

The current implementation uses two independent LoRA adapters initialized identically on a shared frozen base model.

Possible comparisons include:

- No transfer: existing separate adapters.
- Periodic student-to-proposer transfer.
- Performance-triggered student-to-proposer transfer.
- Always shared parameters, if supported without extensive reengineering.

Initially prioritize student-to-proposer transfer, because improved solving capabilities may help the proposer generate more challenging, learnable problems.

Do not assume the reverse direction is beneficial.

Consider optimizer-state handling and the fact that interpolation of LoRA factors is not necessarily equivalent to interpolation of effective model-weight updates.

Any adaptive trigger must be predeclared using training/development signals rather than chosen after observing final evaluation outcomes.

**This is a stretch goal, not part of the critical path. Do not implement it before the interface, learning-signal, and proposer-reward problems have been addressed.**

---

## 10. Model choice and possible SFT

Keep `Qwen/Qwen3-4B-Instruct-2507` as the primary model.

Do not assume 4B is too small merely because external transfer is weak. Previous evidence shows that editing-interface choice can substantially change its performance without increasing model size.

Also, do not assume that the optimal tool interface is universal across models.

Investigate compatibility with Qwen's documented tool-calling conventions before concluding that the model lacks the necessary instruction-following ability.

**Do not introduce SFT as a mandatory preparation step.**

First evaluate interfaces with the existing model.

If all reasonable interfaces still show substantial format-following failures, consider a narrowly scoped format/tool-use SFT warm start.

If SFT is used:

- Use separate synthetic or training-only demonstrations.
- Exclude final evaluation tasks and oracle solutions.
- Avoid confounding tool-format learning with supervised semantic repair learning.
- Initialize all RL comparison arms from the same SFT checkpoint.
- Preserve a no-SFT control.
- Record the additional SFT compute and data budget.

Do not independently introduce a larger model, SFT, new reward, and new interface within one comparison.

---

## 11. Execution, compute efficiency, and infrastructure

Use the existing Yale HPC infrastructure and verified execution paths wherever possible.

Before new submission, reconcile live Slurm status and existing logical-run ledgers. Do not duplicate a pending, running, completed, or recoverable job.

Use actual scheduler limits and available CPU/GPU resources rather than assumptions from older runbooks.

Prefer one H200 for bounded experiments unless there is clear evidence that another allocation is necessary.

### Compute budget

For the next diagnostic and pilot phase, use a suggested ceiling of **12 new allocated H200 GPU-hours in total**, including failed attempts and continuation allocations. Treat this as the proposed bounded pilot scope, not as authorization for new paid cloud resources or for violating current Yale allocation rules.

If this ceiling is insufficient, complete the highest-priority evidence first and provide an itemized justification before larger runs.

Do not silently carry unused historical budgets into a new study.

Prioritize compute as follows:

1. Reuse saved data and CPU analysis whenever possible.
2. Small numerical checks only where GPU behavior matters.
3. Bounded, paired interface comparisons.
4. Short matched student-objective experiments.
5. Reward and proposer pilots only if supported by earlier evidence.

Avoid launching many redundant jobs or repeating checks whose dependencies, images, code, and execution conditions have not changed.

### Training stability

Preserve the established transactional checkpoint mechanism, including:

- Both LoRA adapters.
- Both optimizer states.
- RNG states.
- Curriculum state.
- Task/proposal counters.
- Scientific configuration identity.
- Complete iteration boundaries.

Do not resume a partially applied update.

Handle transient SSH, VPN, RPC, or scheduler problems without corrupting state or silently creating replacement runs.

Do not require unnecessary site-level privileges or impractical cgroup settings. The existing verified single-process Yale worker is the starting point.

Do not weaken code-execution isolation to make an experiment launch.

### Observability

Use W&B or the existing equivalent for live monitoring, but keep checkpoints and local events authoritative.

W&B delivery failures must not be interpreted as training failures when trustworthy local checkpoint evidence exists.

Provide enough observability to determine whether:

- The environment is producing useful tasks.
- The student is improving.
- The model is making valid, executable edits.
- Rewards have meaningful variance.
- Gradients and optimizer updates are numerically reasonable.
- The experiment remains within its compute budget.

Keep telemetry concise. Do not build an elaborate new monitoring system when the existing one can be extended.

---

## 12. Engineering standards and efficient workflow

Implement changes incrementally in the existing architecture.

Prefer small, clearly separated configuration and tool-interface extensions over extensive refactoring.

For each material change:

1. Identify the specific hypothesis or reproducible defect.
2. Implement the minimum necessary change.
3. Add focused deterministic tests.
4. Run relevant existing regression tests.
5. Perform a bounded end-to-end or allocated GPU test only when necessary.
6. Freeze the scientific identity before the controlled experiment.
7. Record the actual result and proceed.

Use existing `uv`, Ruff, mypy, pytest, Slurm controller, and W&B integration.

Do not create parallel implementations of working infrastructure.

Avoid repeatedly restarting research planning from Task 1. Read the existing plans and completion records and continue from the current state.

Do not allow speculative edge cases to block progress. Separate:

- **Hard blockers:** incorrect rewards, unsafe execution, invalid model/log-probability alignment, corrupted checkpoints, evaluation leakage.
- **Scientific limitations:** weak learning, uncertain differences, insufficient proposer reward variation, poor generalization.
- **Nonblocking improvements:** optional extra formats, additional charts, redundant readiness checks, speculative optimization.

A scientifically negative result should not trigger an engineering rewrite.

Maintain reasonable code quality, but optimize for a clean research conclusion rather than unnecessary abstraction.

---

## 13. Required deliverables

At completion, provide the following.

### A. Current-state audit

A concise account of:

- What was already complete.
- What remained unresolved.
- Which historical findings were reused.
- Which new problems were actually discovered.
- Which issues were not reproducible or did not warrant further work.

### B. Implemented changes

For each change, identify:

- Files and functions modified.
- Scientific motivation.
- New tests.
- Compatibility implications.
- Whether historical experiment behavior remains untouched.

### C. Executed experiments

Provide a table containing:

- Study ID and method.
- Model and interface.
- Dataset/split and number of independent families.
- Training seed.
- Episode and token exposure.
- Actual optimizer updates.
- Verified repair outcome.
- Invalid/rejected edit rates.
- Allocated GPU-hours.
- Checkpoint/run identifier.
- Completion or failure status.

Distinguish real GPU results from CPU tests, mock experiments, and retrospective replay.

### D. Scientific analysis

Include:

- Paired learning curves.
- Repair success and failure categories.
- Interface acceptance and verified repair differences.
- RLVR/OPSD learning-signal comparisons.
- Reward variance and partial-reward evidence, if tested.
- In-distribution versus cross-distribution comparisons.
- Uncertainty across task families versus across training seeds.
- Main limitations and plausible alternative explanations.

### E. Reproducibility

Provide commands and configurations sufficient to reproduce the important new results using the permitted data and execution environment.

Keep private source/oracle content out of public artifacts.

### F. Research decision

Conclude explicitly:

1. Which interface is currently most suitable for Qwen3-4B under our benchmark?
2. How much of the generalization failure is attributable to edit application versus semantic correctness, to the extent the experiments can identify this?
3. Is the numerical RL implementation trustworthy enough for continued experiments?
4. Does gated OPSD show a credible advantage over GRPO or ungated OPSD?
5. Would partial rewards plausibly improve learning efficiency?
6. Is there sufficient proposer reward discrimination to justify C/D?
7. What should the next experiment be, if any?

Do not present hypotheses as proven conclusions.

---

## 14. Definition of done

The critical path is complete when:

- The latest repository and preserved artifacts have been reconciled.
- The relevant numerical discrepancies have been diagnosed to the extent necessary for a valid training decision.
- A controlled interface comparison has produced a defensible engineering and scientific decision, reusing prior results wherever possible.
- The sandbox and reward boundary remains intact.
- A bounded student-objective comparison has been executed if supported by the earlier evidence and available compute.
- The findings are analyzed and documented with honest uncertainty.
- The repository contains reproducible code, tests, and results documentation.
- Existing completed experiments remain untouched.

The partial-reward experiment, full C/D training, adaptive proposer–student transfer, SFT, and model scaling are conditional or stretch goals. Their absence does not mean the critical path failed.

If infrastructure access genuinely prevents a GPU experiment, complete the applicable implementation, local tests, saved-data analysis, and dry-run preparation; identify the specific missing access rather than inventing results or requiring unrelated checks.

**Most importantly: make actual scientific progress.**

We do not need another cycle of generic suggestions, repeated readiness work, or broad hyperparameter changes.

Use the current evidence, prioritize the strongest unresolved mechanisms, run the smallest reliable experiments, and deliver a clear answer about what our coding agent has learned, why its generalization fails, and which interventions genuinely help.

Begin by auditing the current repository and producing a brief prioritized execution plan, then immediately implement and execute the first justified step. Work through the critical path autonomously, documenting progress and decisions as you go.