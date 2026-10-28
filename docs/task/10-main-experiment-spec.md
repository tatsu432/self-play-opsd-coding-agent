# Codex Task: Finalize Reliable A–D Training Launch, Resolve Numerical Risks, and Enable Live W&B Monitoring

## 1. Mission

You are the lead research engineer for my Yale CPSC 5880 final project, **Self-Play with On-Policy Self-Distillation for Coding Agents**.

Repository: https://github.com/tatsu432/self-play-opsd-coding-agent

Your objective is to make the existing A–D experiments ready for **reliable, efficient, long-running H200 training**, with **live training metrics visible in Weights & Biases (W&B)**.

This is the final pre-launch engineering task. The repository has already passed extensive readiness tests, so **do not begin another broad readiness cycle or architectural rewrite**.

I want you to:

1. Inspect the latest repository and live Yale HPC state.
2. Investigate and resolve, or quantitatively assess, the remaining rollout/trainer numerical discrepancy.
3. Enable and validate **online W&B monitoring**, with live dashboards for A–D.
4. Ensure that long-running Slurm training, checkpointing, continuation, and recovery work reliably.
5. Verify that the configured workload remains reasonably efficient.
6. Make any small, necessary improvements and validate them.
7. Produce exact commands and a clear GO/NO-GO decision for starting the main experiments.

**Do the engineering work, not just propose another plan.**

However, do not submit the full main A–D jobs or incur substantial additional GPU-hours without explicit main-experiment authorization. Read-only checks, local tests, and previously authorized bounded diagnostics can proceed within existing permissions.

The desired outcome is a launchable experiment, not another list of hypothetical requirements.

---

## 2. Research objectives — preserve these exactly

We compare four configurations:

| Method | Proposer reward | Student loss |
|---|---|---|
| A | Difficulty-based | GRPO + ungated OPSD |
| B | Difficulty-based | GRPO + gated OPSD |
| C | RL-only gradient alignment | GRPO + gated OPSD |
| D | RL+OPSD-loss gradient alignment | GRPO + gated OPSD |

The intended comparisons are:

- A vs B: Does gating improve OPSD?
- B vs C: Does RL-gradient-based task proposal improve over difficulty-based rewards?
- C vs D: Does including the gated OPSD component in the candidate-gradient alignment reward improve the proposer?

Preserve the current mathematical objectives, normalization, sampling, evaluation boundaries, model, data, and experimental budgets.

Do not change GRPO to Dr.GRPO, GSPO, or another method. Do not introduce asynchronous RL, change the OPSD coefficient, alter reward functions, or change proposal validity criteria.

Do not modify prompts, generated task distributions, group sizes, episode budgets, learning rates, or optimization steps for performance reasons.

If you identify a genuine numerical correctness issue, you may implement a narrowly scoped correction, but explain and validate the resulting scientific change before adopting it for the main study.

### Current known configuration

At the latest reviewed repository state:

- Active study: `repair-selfplay-2026-efficiency-v2`
- Settings: `configs/task09-diagnostics.yaml`
- Frozen execution commit: `93a8533beeae26e54bb1510caa12991fb2471337`
- Main branch head at review: `925fa8a67867755e96caadb680d3aafeecfd27c0`
- Model: `Qwen/Qwen3-4B-Instruct-2507`
- Hardware: one H200 per experiment
- Training budget: 200 iterations per method
- Seed: 2026
- Four parents × four proposals
- Four solver episodes per valid proposal
- Checkpoint every iteration
- Slurm segments: 23 hours

Check whether the repository has advanced before beginning.

Read:

- `AGENTS.md`
- `docs/yale-launch-runbook.md`
- `docs/task_completion/09-implemented-efficiency-learnability.md`
- `docs/task_completion/08-implemented-task-learnability-probe.md`
- `docs/methods.md`
- `configs/task09-diagnostics.yaml`
- `src/selfplay_opsd/trainer.py`
- `src/selfplay_opsd/verl_backend.py`
- `src/selfplay_opsd/vllm_worker.py`
- `src/selfplay_opsd/environment.py`
- `src/selfplay_opsd/objectives.py`
- `src/selfplay_opsd/observability.py`
- `src/selfplay_opsd/rollout_parity_probe.py`
- `src/selfplay_opsd/scientific_health.py`
- `src/selfplay_opsd/hpc.py`
- `scripts/hpc_job.py`
- `scripts/hpc_remote.py`

Treat the latest verified results, rather than historical sections of documents, as authoritative.

---

## 3. Current readiness: avoid repeating completed work

The latest Task 09 v2 report says:

- All readiness gates passed.
- Four actual main-job previews succeeded.
- Hosted Linux/Docker correctness CI passed.
- The D pilot completed ten real iterations.
- Checkpoint continuation from iteration two was verified.
- A/B/C each completed three real iterations.
- Student and proposer optimizer updates were exercised.
- Worker isolation and backend compatibility passed.
- The study was frozen with a measured resource plan.
- Main A–D training had not yet been submitted.

These are substantial accomplishments.

Reuse successful evidence where dependencies have not changed.

Do not reintroduce the old PID/cgroup isolation blocker. Do not demand another complete GPU acceptance cycle because of changes limited to logging or operational documentation.

If the current frozen study is scientifically valid, preserve it as a launch option.

Do not bypass freeze validation or fake equivalence. If a change genuinely invalidates the frozen execution identity, create the smallest appropriate new immutable study and revalidate only the affected dependencies.

**Readiness should be proportional to actual risk, not hypothetical possibilities.**

---

## 4. Priority 0 — Investigate the rollout/trainer log-probability discrepancy

This is the most important remaining numerical issue.

The current architecture samples trajectories using vLLM but calculates training gradients with the verl/PyTorch policy.

At the frozen student snapshot, the policy used to generate a trajectory and the policy used to evaluate it should closely agree.

However, Task 09 measured discrepancies between rollout token log-probabilities and trainer-rescored token log-probabilities.

### Existing H200 evidence

Initial snapshot:

- 3,621 tokens examined.
- Mean absolute difference: 0.00369.
- p99 absolute difference: 0.1004.
- Maximum difference: 1.138.
- 18 tokens outside the GRPO clipping interval.

Later checkpoint:

- 4,054 tokens examined.
- Mean absolute difference: 0.00330.
- p99 absolute difference: 0.0996.
- Maximum difference: 0.728.
- 21 tokens outside the GRPO clipping interval.

One mixed-reward-group diagnostic found:

- Relative gradient L2 difference: 9.9%.
- Gradient cosine similarity: 0.9985.

These findings are concerning but do not, by themselves, prove that training is scientifically invalid.

### Required investigation

Using existing replay artifacts wherever possible:

1. Determine whether these differences arise from:
   - BF16 numerical differences between vLLM and PyTorch.
   - Different attention or log-softmax implementations.
   - Different token-position handling.
   - Incorrect causal masks or context boundaries.
   - LoRA adapter synchronization or version mismatch.
   - vLLM sampling/log-probability extraction.
   - Prompt/response serialization differences.
   - Other implementation errors.

2. Check numerical behavior across:
   - Different response lengths.
   - Different token positions.
   - Multiple model snapshots.
   - A valid mixed-reward group.
   - Both student and proposer policies, where relevant.

3. Quantify effects on:
   - GRPO importance ratios and clipping.
   - Actual student RL gradients.
   - Teacher–student OPSD gaps and gates.
   - Total student gradients.
   - Candidate alignment scores and proposer rewards for C and D, if relevant.

4. Distinguish rare floating-point outliers from a systematic representation or policy-version error.

### Consider trainer-rescored old log-probabilities

`score_teacher()` already recomputes unprivileged student log-probabilities.

Investigate whether using trainer-rescored old-policy log-probabilities for optimization would improve consistency.

However, do not blindly replace the rollout log-probabilities.

The rollout engine represents the behavior distribution from which actions were actually sampled, while the trainer implementation may define a slightly different numerical distribution. Changing the denominator of an importance ratio can change the estimator.

Compare both approaches carefully and explain their semantics.

If a correction is warranted:

- Implement it narrowly.
- Preserve original sampled token IDs.
- Preserve detached old-policy and teacher values.
- Test the resulting gradients and optimizer updates.
- Confirm whether the same correction is needed for proposer training.
- Maintain correct scientific identity and freeze provenance.

### Decision rule

Do not require exact bitwise agreement between vLLM and PyTorch.

If discrepancies are rare, do not materially affect gradient direction or learning stability, and have no identifiable correctness bug, document their magnitude and proceed.

If there is an actual indexing, adapter, probability, or clipping bug, fix it before main training.

**Do not let the numerical investigation become an unlimited research project.**

Use the existing H200 diagnostics first. Only schedule a small additional GPU test if the existing evidence cannot resolve a consequential uncertainty and the test is authorized.

---

## 5. Priority 0 — ENABLE LIVE WEIGHTS & BIASES MONITORING

**This is a required deliverable. I want to open W&B in my browser and watch A–D training progress and results in near real time, without manually downloading log files.**

The current configuration has:

`logging.mode: offline`

The current `WandbSink` also explicitly passes this mode to `wandb.init()`.

Simply providing `WANDB_API_KEY` is insufficient.

You must implement and validate a genuinely functioning online monitoring workflow.

### 5.1 Choose the simplest reliable integration

Prefer the existing W&B integration in `src/selfplay_opsd/observability.py`.

Investigate the safest way to enable online logging while preserving the frozen scientific study.

Possible approaches include:

- An explicitly validated logging-only runtime override.
- A new operational configuration with online W&B enabled.
- A separate read-only telemetry publisher that streams committed training events to W&B without modifying the frozen trainer, if this avoids unnecessary scientific revalidation.

Choose the simplest maintainable solution that provides actual live metrics.

A logging-only change should not alter model sampling, RNG, optimizer behavior, or scientific outputs.

Do not silently modify the existing immutable frozen execution or bypass study identity checks.

If enabling online logging requires a new freeze, perform dependency-scoped validation and preserve the previous frozen study as a fallback.

### 5.2 Authenticate securely

Use W&B API-key authentication without exposing credentials.

The repository already supports a remote secrets file and a `--secrets-file` argument for HPC submissions.

Inspect and reuse that mechanism where appropriate.

Requirements:

- Never commit the API key.
- Never insert credentials into YAML files.
- Never expose the API key in Slurm job arguments, logs, GitHub, W&B configuration, or generated-code workers.
- Use a user-owned permission-600 secrets file outside the repository if required.
- Ensure that the trainer container receives the key.
- Ensure that resumed Slurm segments retain access to the credential.
- Do not mount secrets into sandboxed generated-code execution.

Use W&B project name:

`self-play-opsd-coding-agent`

Use the actual authorized W&B entity/account. Read existing configuration if available; do not invent an entity.

If W&B authentication is not configured, provide exact secure setup commands and identify the required user action. Complete everything else independently.

### 5.3 Verify actual network connectivity

Check whether the allocated Bouchet compute node can reach W&B over HTTPS.

Do not assume that connectivity from a laptop or login node proves connectivity from the allocated training node.

Use a lightweight, permitted test that does not allocate an H200 unnecessarily.

If direct connectivity works:

- Enable online W&B.
- Verify that a small test run appears in the cloud dashboard.
- Confirm that logged metrics actually update.

If direct connectivity is restricted:

- Investigate a YCRC-supported solution or an authorized read-only telemetry relay.
- Preserve local offline logging.
- Provide a concrete alternative for near-real-time dashboard updates.
- Do not establish unauthorized network tunnels or persistent login-node daemons.

Do not claim that W&B is live merely because `wandb.init()` returned successfully or an offline run directory exists.

A real cloud-side metric must be visible before declaring W&B enabled.

### 5.4 Organize the four experiments

I want four easily comparable W&B runs:

- `A-difficulty-ungated-opsd`
- `B-difficulty-gated-opsd`
- `C-rl-alignment-gated-opsd`
- `D-combined-alignment-gated-opsd`

Group them under the same scientific study.

Use the same metric names and axes across methods.

Prefer one logical W&B run per method across all Slurm segments.

The implementation already uses a logical run ID in online mode. Verify that this behaves correctly when a method stops after a Slurm allocation and resumes several hours later.

Requirements:

- A resumed segment continues the same logical run.
- No accidental duplicate runs.
- No reset of the outer-iteration axis.
- No misleading training progress from duplicated or rolled-back events.
- Preserve the identities of Slurm segments and checkpoints as metadata.
- Do not mix pilot runs with main runs.

If W&B's history model cannot transparently handle a rollback, preserve committed local events as authoritative and implement a safe, documented reconciliation strategy rather than overwriting history incorrectly.

### 5.5 Required dashboard metrics

Organize the dashboard into meaningful sections.

#### Training progress

Display:

- Current outer iteration out of 200.
- Student and proposer optimizer-step counts.
- Elapsed active training hours.
- Allocated GPU-hours.
- Recent average iteration duration.
- Estimated remaining training hours once enough data exists.
- Current logical run and Slurm segment.
- Last committed checkpoint.
- Last successful metric update timestamp.

#### Student learning

Display:

- Verifiable repair success rate.
- Public-test pass rate.
- Hidden full-pass rate.
- Successful episodes / total episodes.
- Reward distribution.
- Mixed-reward-group fraction.
- All-zero and all-success groups.
- RL gradient norm.
- OPSD gradient norm.
- Total gradient norm.
- RL/OPSD gradient norm ratio.
- GRPO loss.
- OPSD loss.
- Total training loss.
- Important numerical diagnostics and clipping statistics.

Loss values must be correctly labelled. Some RL or OPSD surrogate losses may be negative; they are not direct measures of model quality.

#### Proposer learning

Display:

- Attempted proposals.
- Valid proposals.
- Proposal validity rate.
- Invalid reason distribution.
- Duplicate fraction.
- Proposer loss.
- Proposer gradient norm.
- Proposer reward distribution.
- Generated task diversity and mutation statistics, where already available.

#### Gradient alignment (C and D)

Display:

- Alignment cosine statistics.
- Candidate and reference gradient norms.
- Near-zero and exactly-zero candidate gradient fractions.
- Negative-alignment fraction.
- Number of valid alignment scores.
- Neutral alignment reward frequency.
- Any relevant RL-vs-OPSD gradient diagnostics.

For A/B, report these metrics as not applicable rather than zero.

#### OPSD diagnostics

Display:

- Teacher–student log-probability gap statistics.
- Positive/negative gap fractions.
- Gating weight statistics.
- Gate saturation.
- Gap clipping fraction.
- Student/teacher mapping consistency.
- Rollout/trainer log-probability discrepancy.

#### Evaluation

This is critical.

Log development evaluation results against:

1. Training iteration.
2. Allocated GPU-hours.

Show:

- Development repair success.
- Parent-macro development score.
- Best development score and corresponding checkpoint.
- Initial model score.
- Improvement over initial model, if measurable.

Do not confuse development evaluation with held-out final test performance.

Do not evaluate the final test set repeatedly during training.

#### Compute efficiency and infrastructure

Display:

- Proposer generation time and tokens/second.
- Student generation time and tokens/second.
- Rollout and execution time.
- Teacher and student scoring time.
- Gradient alignment time.
- Student/proposer update time.
- Checkpoint time.
- GPU utilization and memory, where measurable.
- CPU utilization and memory.
- Valid tasks per GPU-hour.
- Successful episodes per GPU-hour.
- Infrastructure retry counts.
- W&B delivery errors or backlog.

Use the existing event store and phase timing data instead of repeatedly measuring expensive quantities.

### 5.6 Logging frequency

Log lightweight training metrics after every committed outer iteration.

Log development evaluation metrics on the existing evaluation schedule.

Keep expensive diagnostics at their existing frequency unless a targeted investigation requires otherwise.

Do not add expensive full-gradient calculations or benchmark evaluations just for dashboard visualizations.

### 5.7 Privacy and data boundaries

Only aggregate metrics and approved metadata should be uploaded by default.

The existing W&B code includes sampled proposal and episode tables. Audit those before enabling online uploads.

Do not automatically upload:

- Private test inputs or expected outputs.
- Hidden evaluation oracles.
- Private reference solutions.
- Credentials.
- Raw unrestricted model context.
- Full resumable checkpoints.
- Arbitrary source files.

Keep detailed raw trajectories and full scientific checkpoints in Yale's trusted project storage.

If example trajectories are useful, use a separate explicitly controlled redacted logging option.

### 5.8 W&B reliability

W&B must not become a critical dependency for training.

If the network disconnects:

- Preserve training.
- Preserve all committed local events.
- Queue or retain undelivered telemetry within bounded limits.
- Support later reconciliation.
- Avoid unbounded memory growth or training delays.

A transient W&B failure should not cancel a 23-hour Slurm allocation.

Implement informative warnings, including how far behind the cloud dashboard is relative to the most recent locally committed iteration.

Test authentication failure, network interruption, clean restart, and checkpoint continuation.

**Acceptance: I can open real W&B URLs for the study and observe verified training metrics updating, with correct method grouping and resume behavior.**

---

## 6. Priority 1 — Preserve long-running Slurm reliability

The current controller already provides:

- Verified joint checkpoints.
- Per-iteration checkpointing.
- Clean continuation.
- Bounded transient retries.
- Submission ledgers.
- Distinct scientific run identities.
- A documented manual resume mechanism.

Preserve these capabilities.

### Inspect the current launch configuration

The latest measured resource plan uses:

| Method | Planning GPU-hours | Clean continuation allowance |
|---|---:|---:|
| A | 82.06 | 5 |
| B | 79.94 | 5 |
| C | 84.82 | 5 |
| D | 97.95 | 6 |

Confirm these values against the actual current freeze and controller previews.

Do not assume the estimates are upper bounds.

### Verify operational behavior

Check the following:

1. Stop signals allow an iteration to finish safely.
2. Both optimizers and adapters are saved transactionally.
3. RNG and curriculum state are restored.
4. No partial optimizer step is committed.
5. No duplicate episodes or updates occur after restart.
6. All resumed jobs preserve scientific identity.
7. Bounded retries do not produce duplicate Slurm submissions.
8. Out-of-memory and scientific errors are not retried blindly.
9. W&B resumes or reconciles correctly after a Slurm segment.
10. The next job can be started from the verified checkpoint.

Reuse existing real checkpoint tests. Run new tests only where the changed implementation makes them necessary.

### Handle the foreground watcher limitation

The current `hpc watch --continue-stopped` command depends on a foreground process. A previous pilot watcher exited after an SSH connection interruption.

Determine the best practical solution.

Prefer a small, durable and YCRC-compliant continuation mechanism if readily possible, but do not introduce a complicated scheduler service or a persistent login-node daemon.

A validated manual `status` → `logs` → `resume --continue-stopped` procedure is acceptable.

Make the workflow explicit so that an SSH disconnect does not result in confusion, duplicate training, or lost progress.

### Queue and resource efficiency

Inspect live YCRC constraints:

- Available H200 GPUs and account eligibility.
- Partition and walltime limits.
- Fairshare and priority.
- Maximum concurrent GPUs/jobs.
- Requested CPU/RAM fit.
- Queue wait estimates, when available.
- Whether multiple independent one-H200 method jobs could run concurrently.

If A–D can run concurrently within the authorized account limits, report the potential calendar-time reduction.

Do not automatically use multiple GPUs per method. The current FSDP implementation is not proven to achieve useful data-parallel speedups simply by increasing GPU count.

Do not change the 128-GiB host RAM request without adequate peak-memory evidence.

---

## 7. Priority 1 — Confirm scientific usefulness early

The most concerning issue is not GPU memory or even generation throughput. It is whether the student and proposer receive enough useful learning signal.

### Existing evidence

The most recent ten-iteration D pilot produced:

- 160 attempted proposals.
- 14 valid proposals.
- 56 solver episodes.
- 14 successful episodes.
- Eight actual optimizer steps for each role.
- Five zero RL-gradient diagnostics among eight sampled.

The greedy base evaluation was 0/41 successful repairs.

These results show that the pipeline works mechanically, but not yet that training improves held-out performance.

### Early checkpoints

Prepare an easy-to-run scientific-health analysis at:

- Iteration 5.
- Iteration 10.
- Iteration 20.

Use the existing `scripts/scientific_health.py` as the starting point.

At each checkpoint, examine:

- Proposer validity and diversity.
- Invalid patch/tool-call patterns.
- Student repair success and test usage.
- Mixed-reward-group fraction.
- Zero RL-gradient fraction.
- OPSD contribution and gating statistics.
- Alignment degeneration for C/D.
- Actual GPU-hours consumed.
- Development evaluation when available.
- Changes in rollout/trainer numerical disagreement.

Provide a concise interpretation, not only raw metrics.

Distinguish mechanical learning updates from demonstrated improvement in task performance.

Do not automatically declare scientific failure because the first few iterations show zero success. The dataset is small, and short-run measurements are noisy.

However, if invalid actions dominate, RL signals remain absent, and development scores show no promising movement, report that clearly before continuing expensive training.

### Early stopping

Implement no new arbitrary success-rate gate.

Automatic stopping should remain reserved for real correctness or operational failures, including:

- Nonfinite loss/gradient.
- Invalid checkpoint integrity.
- Unrecoverable worker failure.
- Broken data boundaries.
- Confirmed model/adapter inconsistency.
- Serious numerical corruption.

Scientific underperformance should trigger investigation and a recommendation, not silently change or terminate the experiment.

---

## 8. Priority 2 — Efficiency improvements: only what is justified

The Task 09 optimization branch already implements:

- A validated shortcut for zero C candidate gradients.
- Optional batched proposer generation.
- Configurable vLLM prefix caching.
- Configurable CUDA Graph/eager execution.
- Configurable scoring microbatch size.
- Additional throughput diagnostics.

The frozen main configuration currently disables most new throughput features.

That is deliberate.

### Existing benchmark observations

For 16 proposer requests:

- Eager sequential generation: approximately 54.6 seconds.
- Eager batched generation: approximately 6.6 seconds.
- CUDA Graph sequential generation: approximately 11.5 seconds.
- CUDA Graph batched generation: approximately 1.7 seconds.

These results suggest substantial optimization potential.

However, token log-probabilities and sometimes generated sequences differed. Long-context stability and complete training-iteration throughput were not adequately established.

### Your instructions

Do not automatically enable batching, CUDA Graphs, prefix caching, or larger scoring microbatches.

First determine whether any existing evidence now supports an optimization that is:

- Scientifically acceptable.
- Numerically stable.
- Reliable on long contexts.
- Consistent across A–D.
- Actually faster end-to-end.

If not, leave the conservative configuration unchanged and document the possible future speedup.

Do not spend days trying to achieve perfect bitwise parity just to enable an optional optimization.

Do not rewrite solver generation for asynchronous RL or replace the RL algorithm.

If an easy optimization provides a clear benefit without changing mathematical behavior, implement it with targeted tests and dependency-scoped readiness checks.

Otherwise, defer it until actual main-run performance measurements justify further work.

**The cost of delayed experiments is itself an efficiency concern.**

---

## 9. Exact launch strategy

Once the core requirements are satisfied, prepare the following workflow.

### Stage A — Final preflight

Verify:

- Exact frozen execution source.
- Correct study/config.
- Frozen dataset and model identities.
- Existing passed readiness gates.
- Real main resource plan.
- Clean main directories and submission ledgers.
- Current scheduler/account eligibility.
- Available checkpoint storage.
- Secure W&B credentials.
- Verified online W&B delivery.
- No active or duplicate main jobs.
- Authorization for the intended GPU-hours.

Use the existing controller, not handcrafted Slurm commands that bypass its safeguards.

### Stage B — Main experiment A first

Prepare to submit method A using the currently accepted study after explicit main-run authorization.

I prefer to begin with A rather than submit all four expensive experiments simultaneously.

Record:

- Slurm job ID.
- Logical run ID.
- W&B URL.
- Exact experiment source hash.
- Start checkpoint.
- Allocated resources.
- Configured continuation behavior.

Verify that online metrics begin appearing after committed training iterations.

### Stage C — Review after 5–10 iterations

Use actual main-run data to inspect:

- Scientific-health indicators.
- GPU utilization.
- Phase timing.
- Unexpected retries.
- Model-memory behavior.
- W&B delivery.
- Updated runtime projection.

Do not require positive held-out improvement before the first scheduled development evaluation.

### Stage D — Review at iteration 20

Analyze:

- First post-training development evaluation versus baseline.
- Valid-task and successful-episode yield.
- Actual iteration speed and GPU-hours.
- Mixed-reward groups and gradient signal.
- Numerical diagnostics.
- Expected completion time.

Provide a recommendation to continue A, investigate the training process, or make a scientifically justified change.

### Stage E — Submit B, C and D

After a satisfactory initial review and explicit authorization, submit the remaining methods under the same validated scientific protocol.

Where account permissions and fairshare make it sensible, run independent methods concurrently on separate H200s.

Do not change the training recipe between methods based on early results.

Do not start one method on an optimized configuration while the others use the conservative version, unless a new controlled study explicitly addresses that comparison.

---

## 10. Required tests

Run the relevant existing unit, integration, and hosted CI tests.

Add narrow tests for any new changes.

At minimum verify:

- W&B cloud-side metric delivery.
- Stable run IDs and method grouping.
- Correct iteration axes across resumed segments.
- Proper handling of duplicate/rolled-back events.
- W&B failure isolation.
- No secret exposure.
- Numerical correctness for any changed scoring path.
- Student and proposer optimizer behavior.
- Checkpoint compatibility.
- Main submission preview consistency.

Do not repeat already passed costly worker-isolation, data-preparation, or H200 acceptance checks unless their dependencies changed.

Do not use a new 24 GPU-hour allowance or reset an accounting ledger by changing the study name.

The previously recorded readiness consumption was 7.478055556/24 GPU-hours. Reconcile the live ledger before any new GPU work.

Keep any additional diagnostic allocation small and clearly justified.

---

## 11. Final required deliverables

Produce a final report containing the following.

### A. GO / NO-GO decision

State:

- Is the numerical issue resolved or sufficiently characterized?
- Is training mechanically reliable?
- Is checkpoint recovery dependable?
- Is the frozen scientific design preserved?
- Is W&B genuinely live?
- Is the existing compute plan reasonable?
- Is there any real reason not to begin?

Separate required corrections from optional efficiency improvements.

### B. W&B dashboard

Provide:

- Actual W&B project URL.
- Actual verified run URLs.
- Dashboard organization and available metrics.
- How the user can compare A–D.
- Confirmation of live updates.
- Handling of W&B interruption and Slurm continuation.
- Any missing authentication or network step.

Do not invent URLs or report that live monitoring works without a cloud-side test.

### C. Numerical report

Summarize:

- Root-cause investigation.
- Before/after discrepancy statistics, where a correction was made.
- GRPO gradient impact.
- OPSD gating/gradient impact.
- C/D gradient-alignment implications.
- Whether any scientific change was necessary.

### D. Runtime and efficiency report

Give a table:

| Method | Pilot-yield estimate | Conservative estimate | First-run measured rate | Projected finish |
|---|---|---|---|---|
| A | | | | |
| B | | | | |
| C | | | | |
| D | | | | |

Only fill measured columns when actual runs exist.

Report GPU-hours separately from scheduler queue delay.

### E. Reliability and recovery report

Describe:

- Current Slurm segment duration.
- Checkpoint frequency.
- Retry limits.
- Clean continuation limits.
- How to resume after SSH disconnects.
- How W&B continues across segments.
- What failures require manual intervention.

### F. Exact commands

Provide ready-to-copy commands for:

1. Secure W&B credential setup.
2. Online W&B verification.
3. Synchronizing the selected study.
4. Readiness inspection.
5. Main A submission.
6. Main B/C/D submissions.
7. Status and logs.
8. Live health inspection.
9. Checkpoint continuation.
10. Fetching results.
11. Final evaluation.
12. Offline W&B recovery/synchronization if necessary.

Clearly label commands that were actually executed versus commands prepared for later authorization.

### G. Code changes

Provide:

- Branch and commits.
- Changed files.
- Tests executed and outcomes.
- Any new study identity/freeze.
- Remaining limitations.
- Whether the original frozen baseline remains available.

---

## 12. Final engineering principle

We are past the stage where we should endlessly improve readiness infrastructure.

The main experiments already have real H200 acceptance evidence. The priority now is obtaining scientifically meaningful A–D results before the project deadline.

**Do not delay launch for speculative risks, marginal speedups, perfect bitwise inference parity, or optional architecture improvements.**

But do address genuine numerical correctness issues, checkpoint/recovery defects, and missing live monitoring.

The user must be able to:

- See the study live in W&B.
- See whether the models are learning.
- Understand how quickly GPU-hours are being consumed.
- Recover safely after a Slurm segment or connection interruption.
- Compare A, B, C and D under a consistent scientific protocol.
- Stop or investigate early if training is clearly unproductive.

Work autonomously through code changes and available tests. Make reasonable engineering choices rather than asking unnecessary clarification questions.

If an essential external action is missing—such as W&B credentials or main-training authorization—finish all independent work and provide the precise required step.

**Deliver a validated, observable, reliable main-experiment launch workflow. Do not stop at another proposal or readiness checklist.**