# Task: Optimize and harden the A–D self-play training pipeline for reliable execution on Yale HPC

## 1. Your role and overall objective

Act as a senior research engineer specializing in LLM post-training, reinforcement learning, vLLM, verl, PyTorch distributed training, and Slurm HPC systems.

**Repository:** https://github.com/tatsu432/self-play-opsd-coding-agent

**Primary working branch:** `feat/05-yale-experiment-readiness`

Your task is to **implement, test, benchmark, and document targeted performance and reliability improvements** so that we can confidently run the four main experiments (A–D) on Yale's Bouchet H200 cluster.

Do not merely write a plan or recommendations. Inspect the actual implementation, make justified code changes, run available tests and benchmarks, and produce an actionable launch-ready result.

However, **do not submit the full 200-iteration A–D experiments without my separate authorization**. You may prepare commands, dry runs, benchmark scripts, and validated execution configurations.

The goal is not to introduce the latest RL algorithm for its own sake. It is to minimize wasted GPU-hours, avoid avoidable failures, and obtain scientifically meaningful results with the experimental design we have already established.

Prioritize:

1. Correctness and scientific comparability.
2. Reliable training, checkpointing, and recovery.
3. End-to-end wall-clock and GPU-hour efficiency.
4. Early detection of scientifically ineffective training.
5. Simple, maintainable, extensible code.

Do not introduce a large architectural rewrite if a small change solves the problem.

## 2. Research context and non-negotiable experimental design

The project studies the effects of gated on-policy self-distillation (OPSD) and gradient-aligned task proposal in self-play for coding agents.

The four experimental configurations are:

| Method | Proposer reward | Student objective |
|---|---|---|
| A | Difficulty-based | GRPO + ungated OPSD |
| B | Difficulty-based | GRPO + gated OPSD |
| C | RL-only gradient alignment | GRPO + gated OPSD |
| D | RL+OPSD-loss gradient alignment | GRPO + gated OPSD |

We want to isolate three comparisons:

- **A vs B:** Does gating improve OPSD effectiveness?
- **B vs C:** Does RL-gradient-based task proposal improve over difficulty-based proposal?
- **C vs D:** Does including the gated OPSD component in the candidate gradient improve proposer training?

The current main configuration is `configs/task08.yaml`.

Important settings:

- Base model: `Qwen/Qwen3-4B-Instruct-2507`.
- LoRA student and proposer adapters.
- One H200 GPU.
- 200 iterations per method.
- Seed 2026.
- Four parent tasks per iteration.
- Four attempted proposals per parent.
- Four solver episodes per valid proposal.
- Maximum 512 proposer-generated tokens.
- Maximum 4,096 student-generated tokens per episode.
- Maximum eight student turns.
- OPSD coefficient λ = 0.1.
- Fixed prepared-data and protocol identities.

**Preserve the A–D mathematical objectives and scientific budgets.**

Do not silently change:

- GRPO to Dr.GRPO, GSPO, PPO, DAPO, or another optimizer.
- Advantage normalization or token/episode loss normalization.
- OPSD gating, clipping, coefficient, or teacher conditioning.
- Student/proposer learning rates.
- The reference and candidate gradient definitions.
- Task-verification semantics.
- Proposal validity criteria or failed-proposal rewards.
- Model architecture, initialization, or LoRA configuration.
- Task distribution, data splits, or prompts.
- Sampling temperature, top-p, seeds, token budgets, or group sizes.
- Number of optimization steps.
- Evaluation datasets, frequency, or reported endpoints.

An efficiency optimization must not make the main research comparison scientifically different.

If a promising optimization necessarily changes sampled outputs or the mathematical objective, isolate it as a separately named research variant. Explain the implications and obtain approval before using it for A–D.

## 3. Begin by inspecting the current repository and verified evidence

Before changing anything:

1. Inspect the current branch, latest commits, worktree status, and existing main-run artifacts.
2. Read the current implementation and documentation.
3. Determine which findings below remain valid.
4. Reuse existing successful readiness evidence rather than rerunning it unnecessarily.
5. Work on a new dedicated optimization branch based on the correct current source. Do not overwrite the frozen execution commit or archived evidence.

Read at least:

- `docs/yale-launch-runbook.md`
- `docs/task_completion/08-implemented-task-learnability-probe.md`
- `docs/methods.md`
- `docs/operations.md`
- `configs/task08.yaml`
- `src/selfplay_opsd/trainer.py`
- `src/selfplay_opsd/rollout.py`
- `src/selfplay_opsd/verl_backend.py`
- `src/selfplay_opsd/vllm_worker.py`
- `src/selfplay_opsd/policy.py`
- `src/selfplay_opsd/environment.py`
- `src/selfplay_opsd/objectives.py`
- `src/selfplay_opsd/evaluation.py`
- `src/selfplay_opsd/workload.py`
- `src/selfplay_opsd/observability.py`
- `src/selfplay_opsd/hpc.py`
- `scripts/hpc_job.py`
- `scripts/hpc_remote.py`

### Existing verified readiness

The latest Task 08 record reports:

- All ten readiness gates passed.
- Four actual main-submission previews succeeded.
- H200 numerical backend accepted.
- Isolated execution worker validated.
- Checkpoint recovery and continuation tested.
- Student and proposer optimizer updates exercised.
- A/B/C tested for three real iterations each.
- D tested for ten iterations with a real checkpoint stop/resume.
- No main A–D training job submitted yet.

The frozen execution source was commit:

`983a6cf57ce5caf5c9aa7ddf53f5ad7f313b4e73`

The branch head may have advanced. Inspect the current state rather than assuming that commit is the latest.

**Do not reopen previously resolved YCRC isolation blockers.**

The Slurm/Apptainer single-process worker path has already been validated. In particular, do not require an unsupported nested cgroup PID controller or demand a new process-limit facility merely to proceed.

Preserve the validated isolation guarantees, but do not create additional speculative launch blockers.

### Existing runtime projections

The latest repository estimates 200 iterations plus monitoring on one H200:

| Method | Observed-yield hours | Conservative planning hours |
|---|---:|---:|
| A | 17.16 | 76.56 |
| B | 17.66 | 79.93 |
| C | 18.91 | 86.81 |
| D | 16.51 | 86.14 |
| Total | 70.24 | 329.45 |

These are conditional forecasts, not measured full-run durations.

They differ substantially because more valid proposals induce more solver episodes, scoring, and training work.

We need better projections from representative workload measurements, including queue delay separately from occupied GPU-hours.

### Existing measured bottlenecks

In D's ten-iteration pilot:

| Phase | Measured time |
|---|---:|
| Proposal generation and validation | 651.1 seconds |
| Solver rollout and scoring | 1,039.8 seconds |
| Gradient alignment | 191.6 seconds |
| Student and proposer updates | 192.6 seconds |
| Checkpointing | 63.3 seconds |

The complete allocation was approximately 0.77 H200 GPU-hours, including additional startup, preparation, diagnostics, and other costs.

This suggests that generation, validation, and scoring matter considerably more for wall-clock efficiency than the GRPO optimizer step itself.

Validate these measurements against the raw records and avoid counting overlapping or cumulative timers as independent wall time.

## 4. Priority 0: Instrument and diagnose the real bottlenecks

Before optimizing, confirm what consumes the GPU allocation.

Existing phase instrumentation is useful, but I want sufficient detail to distinguish:

- Proposer prefill and decode.
- Proposer validation.
- Student prefill and decode.
- Waiting on sandbox workers.
- Public-test execution.
- Final verification.
- Old-policy scoring.
- Privileged teacher scoring.
- Reference gradient computation.
- Candidate gradient computation.
- Student backward and optimizer step.
- Proposer backward and optimizer step.
- GPU offload, wake/sleep, adapter synchronization.
- Checkpoint serialization and storage.
- Development evaluation.
- Initialization and uninstrumented overhead.

Add targeted instrumentation only where existing logs do not provide the information.

Collect:

- End-to-end iteration wall time, mean/median/p90.
- Generated tokens per second and generation requests per second.
- Prompt and output token counts.
- Actual number of generation batches and batch sizes.
- GPU compute utilization and memory usage by phase, where measurable.
- CPU memory, utilization, and worker saturation.
- Sandbox wait time, execution time, failure categories, and retry counts.
- Number of valid proposals and solver episodes per GPU-hour.
- Student/proposer optimization steps per GPU-hour.
- Checkpoint size, duration, and storage throughput.
- Allocated versus useful training time.
- Scheduler queue time separately from training runtime.

Avoid expensive full-system profiling continuously. Lightweight timers should be sufficient for ordinary training. More intrusive PyTorch/CUDA profiling can be enabled in a short diagnostic run.

Produce a reproducible profiling script/report that works with existing stored events and Slurm accounting.

Do not invent measurements if the cluster is unavailable.

## 5. Priority 1: Optimize generation and scoring without altering the experiment

### 5.1 Batch independent proposer generations

Inspect `SelfPlayTrainer.proposals()` and the `VerlPolicy` → `vllm_worker` RPC implementation.

Currently the proposer appears to generate 16 candidates through individual sequential calls.

These generations are independent at the frozen proposer snapshot.

Investigate and implement a batched generation interface, for example:

`generate_many(requests) -> list[Generation]`

Each request must retain its individual:

- Prompt token IDs.
- Maximum generation length.
- Sampling seed.
- Decoding parameters.
- Adapter/version identity.
- Canonical proposal slot.
- Original token IDs and sampled-token log-probabilities.

Use the **actual pinned vLLM API**; verify that request-specific sampling parameters and seeds are supported before implementing.

Keep candidate ordering deterministic in the returned records, even if internal execution order differs.

Preserve all 16 attempted proposal slots and their original validity/reward semantics.

Do not change the proposer distribution through rejection sampling, retries, silent patch correction, or filtering.

Benchmark sequential versus batched proposer generation on the same frozen inputs.

### 5.2 Improve solver rollout scheduling

Inspect `episode_wave()`.

The implementation currently serializes model generation behind a lock. The Yale configuration also uses `rollout_concurrency=1`, and the Slurm worker has a single execution slot.

Investigate whether independent solver episodes can be advanced with an event-driven or wave-based scheduler:

1. Maintain a separate state for each episode.
2. Collect prompts from episodes currently ready to generate.
3. Batch ready prompts through vLLM.
4. Dispatch tool requests without blocking unrelated ready generation.
5. Resume episodes when their tool observations are available.
6. Preserve each episode's complete original history and output token sequence.
7. Return episodes in canonical order.

However, **do not redesign the entire agent framework solely for this optimization**.

Start with batching independent proposer requests, then benchmark whether solver-turn batching offers enough improvement to justify the additional complexity.

Do not simply increase `rollout_concurrency`. The isolated execution worker currently accepts one request at a time; uncontrolled concurrent requests could produce worker-busy errors.

If overlapping inference and tool execution is useful, explicitly respect the one-worker execution slot, including backpressure and bounded queues.

Avoid introducing concurrent distributed collectives or unsynchronized PyTorch model calls.

### 5.3 Benchmark vLLM prefix caching and CUDA Graphs

The current vLLM configuration contains:

`enable_prefix_caching=False`

`enforce_eager=True`

Both may sacrifice performance.

Investigate, using the pinned vLLM version:

- Automatic prefix caching.
- CUDA Graph execution.
- Dynamic LoRA loading and adapter-version transitions.
- Sleep/wake behavior when alternating training and rollout.
- GPU memory consumption.
- Compatibility with exact sampled-token log-probabilities.

Benchmark four configurations where supported:

1. Existing settings.
2. Prefix caching enabled.
3. CUDA Graphs enabled.
4. Both enabled.

Do not assume the fastest configuration is safe.

Ensure that caching never reuses incompatible hidden states across student/proposer adapters, policy updates, or resumed model versions.

Test repeated generation after adapter synchronization and after sleep/wake.

If an optimization changes stochastic sampled outputs, distinguish floating-point/scheduling effects from a genuine sampling-policy change. Document that difference and preserve fair A–D comparisons.

Select the fastest **validated** configuration, not merely the one with the highest nominal token throughput.

### 5.4 Optimize student and teacher scoring

Inspect `HFPolicy.score_many()` and `score_teacher()`.

Current scoring uses `microbatch_episodes=1` and repeatedly processes long student/teacher histories.

Benchmark:

- Scoring microbatch sizes 1, 2, and 4.
- Optional length-aware batching to reduce padding.
- Activation checkpointing enabled versus disabled.
- Opportunities to reuse or avoid redundant prefix computation.
- Avoiding unnecessary old-policy re-scoring after numerical parity has already been established.

The H200 has substantial apparent memory headroom during trainer-only phases, but do not extrapolate memory safety from short examples.

Test representative long-context workloads, teacher privilege prefixes, and large valid-task batches.

**Preserve exact scoring semantics.**

In particular:

- Identical target token IDs.
- Identical student/teacher context construction.
- Correct causal positions.
- Identical loss masks.
- Correct episode/task normalization.
- Teacher gradients detached.
- No gradients on prompts, observations, or privileged context.
- No change to the actual OPSD objective.

Do not remove old-policy numerical verification entirely. If its full per-episode cost is significant, investigate a validated sampled/periodic diagnostic mode while preserving adequate numerical checks.

If changing microbatch size alters gradients only through small floating-point ordering effects, quantify the differences.

Measure both forward-scoring speed and full backward/update throughput.

### 5.5 Avoid mathematically unnecessary gradient computations

Method C uses RL-only candidate gradients for gradient alignment.

For a candidate whose four solver rewards are identical, the group-relative advantages are zero.

Under the current A–D objective, that candidate's RL-only gradient is therefore exactly zero.

Inspect whether C unnecessarily executes a full gradient computation for these candidates.

If so, implement a safe shortcut:

- Detect an analytically zero RL-only candidate objective.
- Return the existing neutral alignment reward of 0.5.
- Preserve near-zero/zero-gradient diagnostics.
- Preserve candidate counts, reward ordering, and proposer normalization.
- Avoid a backward pass that cannot produce a learning signal.
- Verify equality against the old implementation in unit tests.

Do not apply this shortcut to D, since its gated OPSD component can generate a nonzero candidate gradient even when RL advantages are zero.

Also investigate whether reference gradients are recomputed more often than required at a frozen student snapshot.

Any reuse must be invalidated after relevant student updates.

### 5.6 Development evaluation efficiency

Development evaluation currently executes many independent tasks sequentially.

Investigate whether the same validated inference batching improvements can accelerate evaluation without changing:

- Evaluation prompts.
- Greedy decoding.
- Task identities.
- Verifier behavior.
- Number of tasks.
- Evaluation schedule.
- Primary or secondary endpoint definitions.

Do not reduce evaluation coverage just to make training faster.

### 5.7 Avoid unnecessary optimization work

Do not prioritize multi-GPU training yet.

The current FSDP2 implementation deliberately repeats the same mathematical microbatches across ranks. Merely adding GPUs is not guaranteed to increase training throughput.

Similarly, do not implement asynchronous RL, Dr.GRPO, or GSPO as part of these A–D experiments.

These approaches may be scientifically interesting, but they change optimization semantics or require substantial architecture work.

Only propose them as future, separately controlled ablations if the measured evidence justifies doing so.

## 6. Priority 1: Make long-running Slurm execution dependable

The main experiments may require several 23-hour Slurm allocations per method.

The current controller already supports verified checkpoints, clean continuation, limited transient retries, and recovery.

Inspect the actual implementation and improve real operational gaps, rather than rebuilding functioning mechanisms.

### 6.1 Checkpointing and restart

Preserve:

- Joint student and proposer adapters.
- Both optimizer states.
- RNG and sampling state.
- Curriculum/proposal history.
- Task and episode counters.
- Scientific, data, and code identities.
- Reference anchors.
- Checkpoint checksums and atomic writes.

The existing implementation checkpoints every iteration. Its measured overhead appears acceptable, so do not reduce checkpoint frequency unless strong evidence supports it.

Verify that an interruption never leads to:

- A partially committed scientific iteration.
- Double-counted training episodes.
- Repeated optimizer steps.
- Duplicate proposal exposures after recovery.
- Lost or inconsistent optimizer state.
- Incorrect cost accounting.

Reuse existing Task 08 recovery evidence where unchanged.

Only rerun targeted checks for components affected by your modifications.

### 6.2 Retries and continuation

Distinguish clearly between:

- A normal clean Slurm segment stop.
- A transient Slurm failure or preemption.
- A sandbox infrastructure failure.
- A model-generated invalid action.
- A scientific/numerical failure.
- GPU OOM.
- Unknown or unreconciled submission state.

Never retry incorrect model-generated code as though it were an infrastructure failure.

Never resubmit an uncertain job without reconciling scheduler and controller state.

Keep bounded retry and continuation policies, and preserve the existing run ledger.

The documented continuation allowances are approximately:

- A: 5
- B: 5
- C: 6
- D: 5

Confirm the actual current values and whether they remain sufficient for the latest workload projections.

### 6.3 Avoid reliance on a foreground laptop watcher

Inspect `hpc watch --continue-stopped` and `hpc resume --continue-stopped`.

Determine whether automatic progression through clean Slurm segments currently depends on a foreground process remaining active.

If so, investigate the simplest reliable supported approach for durable continuation, potentially using Slurm-native dependencies or an idempotent bounded continuation mechanism.

Requirements:

- Respect YCRC scheduling and operational policies.
- Never require a daemon running indefinitely on a login node.
- Never bypass authorization or accounting.
- Never create duplicate submissions.
- Resume only verified checkpoints.
- Honor retry/continuation limits.
- Support inspection and manual intervention after ambiguous or scientific failures.

If durable autonomous continuation is complicated or unsupported, do not let it block the main experiment. Instead provide a robust, clearly documented manual continuation workflow and a reliable method to detect when intervention is needed.

### 6.4 GPU and CPU resource efficiency

Current allocation:

- One H200.
- Seven trainer CPUs.
- One isolated worker CPU.
- Approximately 128 GiB trainer host RAM.
- One GiB independently limited worker memory.

Measured:

- Whole-device sampled peak approximately 71.8 GiB.
- Trainer GPU-process peak approximately 14–15 GiB.
- Actor plus descendant sampled RSS approximately 28.5 GiB.

These are observations, not guaranteed worst-case memory bounds.

Investigate whether the 128-GiB RAM request is unnecessarily high and whether a smaller request could improve Slurm scheduling.

Use actual job-level peak-memory evidence and safety headroom before changing it.

Do not increase the CPU worker count beyond verified per-GPU resource limits.

Investigate scheduling queue time, requested walltime, fairshare and backfill behavior using YCRC's actual tools and documentation.

Do not assume that requesting more GPUs or a longer allocation reduces time to completion.

### 6.5 W&B and observability

The current `configs/task08.yaml` uses offline W&B logging.

I would prefer live monitoring if possible, but training must not depend on successful W&B network access.

Implement or document a reliable approach that supports:

- Offline local recording as the source of truth.
- Optional online W&B when supported.
- Later synchronization without duplicated scientific events.
- Correct handling of multi-segment runs.
- No credential leakage.
- No checkpoint or training failure caused by logging interruptions.

Ensure that the following important metrics are easy to inspect:

- Iteration progress and ETA based on actual recent throughput.
- Valid-task yield and invalid-proposal reasons.
- Student reward, successful episodes, and mixed-reward-group fraction.
- RL and OPSD gradient norms.
- Zero-gradient frequency.
- Gradient-alignment distribution and near-zero fraction.
- Token counts and throughput.
- Phase-level duration.
- GPU memory/utilization.
- Infrastructure retries.
- Evaluation performance over iterations and allocated GPU-hours.

A simple CLI summary from local events is acceptable if online W&B is not feasible.

## 7. Priority 0: Detect scientifically unproductive training early

This is essential.

Read the pilot diagnostics carefully.

Actual Task 08 acceptance measurements include:

| Method | Valid/attempted proposals | Solver episodes | Successful solver episodes |
|---|---|---|---|
| A | 6/48 | 24 | 1 |
| B | 6/48 | 24 | 0 |
| C | 6/48 | 24 | 1 |
| D | 18/160 | 72 | 7 |

Additional concerns:

- Initial greedy development evaluation: 0/41 successful repairs.
- D had 15 entirely unsuccessful groups among 18 valid tasks.
- D recorded 240 patch-tool errors and relatively few valid patches/tests.
- C had five near-zero candidate gradients among six scored candidates.
- D's RL gradient was zero in eight of ten sampled diagnostics.

These measurements do not establish that learning will fail, but they suggest a risk of spending substantial compute on weak or degenerate learning signals.

I do not want to finish 200 iterations per method only to discover that the models never meaningfully improved.

### Implement useful early diagnostics

Create a concise scientific-health report from actual training events, including:

1. Proposal validity and diversity.
2. Most common proposal rejection reasons.
3. Valid solver actions versus malformed/failed actions.
4. Fraction of tasks with all-zero, all-one, and mixed outcome groups.
5. Fraction of RL gradients that are exactly zero.
6. OPSD gradient magnitude and relative contribution.
7. Percentage of useful/defined gradient-alignment scores.
8. Training reward progression.
9. Development evaluation progression.
10. Useful learning signal per allocated GPU-hour.

Distinguish:

- Nonzero optimizer gradients.
- Nonzero RL gradients.
- Positive verifiable rewards.
- Actual held-out evaluation improvement.

These are not equivalent.

For example, a nonzero OPSD gradient does not establish useful repair learning.

### Early-run review

Recommend a lightweight review after the first five to ten iterations and a more substantive review around iteration 20.

Provide criteria for:

- Healthy enough to continue.
- Worth investigating.
- Clear numerical/infrastructure failure requiring a stop.
- Scientifically inconclusive but not necessarily broken.

Do not impose arbitrary success-rate requirements as a new readiness gate.

Do not automatically stop because a small early sample has zero successes. Instead report the evidence and allow a reasoned decision.

If the major problem is the low validity or repair success of the existing protocol, explain it concretely.

**Do not silently repair proposals, modify prompts, refill invalid slots, or change rewards to improve the headline numbers.** Those would change the experiment itself and require a separate scientific decision.

## 8. Benchmarking, correctness tests, and acceptance criteria

Do not promote an optimization merely because a unit test passes or a microbenchmark shows a speedup.

### 8.1 Baseline

First establish the current source and existing runtime measurements as the reference baseline.

Record its exact:

- Commit.
- Model and tokenizer revisions.
- Config.
- Dataset/manifest identity.
- Container and backend identity.
- GPU hardware.
- Training/profiling conditions.

Reuse stored pilot evidence where valid.

### 8.2 Numerical and behavioral equivalence

Create targeted tests comparing the baseline and optimized implementations on controlled inputs.

Check, as applicable:

- Proposal slot identities and ordering.
- Per-request seeds.
- Prompt IDs and sampling parameters.
- Generated output IDs and sampled log-probabilities.
- Teacher/student target mappings.
- Loss values.
- Gradient norms and directions.
- AdamW optimizer updates.
- Student/proposer version synchronization.
- Checkpoint save/load and continuation.
- Verifier results and task validity.
- Correctness across A, B, C, and D.

For deterministic operations, require exact agreement where feasible.

For GPU numerical calculations, define tolerances supported by the actual precision and existing backend tests.

If different batching strategies produce different sampled sequences despite identical sampling distributions, do not pretend they are bitwise equivalent. Explain the source of nondeterminism and verify that the scientific comparison remains fair.

An optimized implementation must not introduce method-dependent computational or sampling advantages.

### 8.3 Realistic performance benchmarks

Benchmark the important workloads on representative H200 inputs:

- Proposer generation.
- Multi-turn solver rollout.
- Teacher/student scoring.
- Student update.
- Reference and candidate gradient alignment.
- Development evaluation.
- One complete training iteration.

Where useful, test both:

- The approximately 11–13% valid-proposal workload observed in the pilot.
- A higher-yield workload with substantially more solver episodes.

Use legitimate fixed valid tasks for stress tests; do not manipulate the actual training proposal distribution.

Measure:

- Baseline and optimized wall time.
- GPU-hours.
- GPU peak memory.
- CPU memory.
- Tokens per second.
- Valid tasks per hour.
- Cost per scientifically useful training signal.
- Median and tail latency.
- Cold-start versus steady-state performance.

Use repeat measurements when affordable and disclose measurement uncertainty.

Prioritize **end-to-end improvements**, not isolated microbenchmark wins that disappear once initialization, validation, scoring, checkpoints, and monitoring are included.

Do not require an arbitrary 2× improvement. If a change provides a small benefit but adds substantial complexity, reject it.

### 8.4 Resource and authorization boundaries

You may execute local/CPU tests and non-submitting checks directly.

For additional H200 profiling, verify that a valid authorization and GPU-hour budget exist. Do not silently use the previously limited readiness allowance or submit expensive GPU workloads without permitted budget.

If fresh GPU authorization is required:

- Complete all independent code changes and CPU tests.
- Prepare a precise bounded benchmark plan and commands.
- Explain the minimum GPU allocation needed.
- Do not block the remaining work.

Never claim that CPU mocks establish H200 performance or production memory safety.

### 8.5 Preserve the existing study freeze correctly

The existing frozen main study must remain intact.

If your optimized code changes the execution identity, identify which readiness evidence becomes stale.

Use dependency-scoped validation: revalidate the relevant numerical, rollout, runtime, and checkpoint behaviors without unnecessarily repeating unrelated worker or dataset acceptance.

Do not falsify hashes, modify evidence, or bypass the freeze validator to make new code appear equivalent.

If an optimization changes the scientific method, create a distinct experiment identity rather than contaminating the frozen A–D study.

All four main methods must use one consistently validated implementation. Do not run A on one scientifically different backend and B–D on another without explicitly designing that comparison.

## 9. Implementation order and scope control

Follow this approximate order:

**Phase 1 — Establish reliable baseline**

- Inspect current readiness.
- Confirm the exact runnable source.
- Audit existing profiling and failure-recovery behavior.
- Identify genuinely missing instrumentation.
- Confirm submission and checkpoint commands.
- Do not repeat successful expensive readiness checks.

**Phase 2 — Low-risk optimizations**

- Improve phase diagnostics if necessary.
- Implement the exact-zero candidate-gradient shortcut for C.
- Benchmark and tune scoring microbatches.
- Benchmark activation checkpointing options.
- Benchmark prefix caching and CUDA Graphs.
- Preserve safe defaults and allow rapid rollback.

**Phase 3 — Generation throughput**

- Implement and test batched proposer generation.
- Evaluate independent solver-turn batching.
- Improve development evaluation batching where worthwhile.
- Keep the implementation limited to performance benefits that can be demonstrated.

**Phase 4 — Operational reliability**

- Audit safe stopping and resuming.
- Address any remaining continuation workflow issues.
- Confirm queue/resource accounting.
- Improve live or offline observability.
- Produce the early scientific-health checks.

**Phase 5 — End-to-end acceptance**

- Run available tests.
- Run authorized GPU benchmarks.
- Compare baseline and optimized performance.
- Validate required checkpoint, worker, and model interactions.
- Reconcile scientific/operational identities.
- Produce final submission commands.

This ordering is flexible if the actual profiling data identifies a different bottleneck.

Implement improvements incrementally, with tests and clear commits.

If a high-complexity idea provides little demonstrated benefit, abandon it rather than delaying the study.

Do not create a second, entirely new RL training framework.

## 10. Code quality requirements

Follow research engineering best practices:

- Keep interfaces explicit and typed.
- Minimize new dependencies.
- Use the existing configuration system.
- Make performance options configurable with safe defaults.
- Keep hardware-specific paths out of scientific code.
- Preserve deterministic seeds and canonical record ordering.
- Ensure bounded queues, memory, and outstanding requests.
- Avoid unbounded retries.
- Keep errors actionable.
- Preserve transactionally safe checkpoints.
- Avoid introducing hidden mutable state that is not checkpointed.
- Maintain compatibility with the existing CPU and H200 backend tests.
- Keep CI green.
- Preserve current sandbox isolation for model-generated code.

Use existing tests and code-quality tools rather than introducing another testing framework.

Add specific tests for new batching, synchronization, and optimization behavior.

Do not loosen safety or scientific correctness requirements to obtain a speedup.

Conversely, do not add rigid requirements unrelated to an observed failure mode.

## 11. Required deliverables

At the end, provide the following.

### A. Executive assessment

Summarize:

- Whether the existing baseline is already safe to launch.
- Which issues were actual correctness/reliability defects.
- Which issues were performance inefficiencies.
- Which concerns were speculative or low-priority.
- What you changed and why.

### B. Implementation

Provide actual committed code changes on a dedicated branch.

List changed files and commits, with a short justification for each substantial change.

Do not stop after writing an implementation proposal.

### C. Performance comparison

Provide a table like:

| Metric | Original | Optimized | Change | Evidence |
|---|---|---|---|---|
| Proposer generation seconds | | | | |
| Solver rollout seconds | | | | |
| Teacher scoring seconds | | | | |
| Student update seconds | | | | |
| Alignment seconds | | | | |
| Full iteration seconds | | | | |
| Peak GPU memory | | | | |
| Valid tasks/hour | | | | |
| Projected GPU-hours A–D | | | | |

Report unmeasured metrics as unmeasured, not estimated facts.

Update the runtime projections from the actual data and state the assumptions.

Distinguish occupied GPU-hours, queue waiting time, and overall calendar makespan.

### D. Reliability assessment

Explain:

- Checkpoint integrity.
- Clean continuation.
- Transient retries.
- Scientific failure handling.
- Worker failure handling.
- Slurm walltime and stop-margin behavior.
- Whether a foreground watcher is needed.
- Remaining manual steps.
- Remaining risks.

### E. Scientific-health assessment

Provide an early-run diagnostic plan specifically for our four methods, including the low proposal validity, sparse positive rewards, zero RL advantages, and gradient-alignment degeneracy observed in the pilots.

### F. Reproducible benchmark and test evidence

Provide:

- Exact commands executed.
- CPU/GPU hardware used.
- Commit and config identities.
- Relevant test results.
- Benchmark methodology and measurements.
- Slurm job IDs, where applicable.
- Any tests that could not be run and precisely why.

### G. Final launch instructions

Produce exact commands for:

1. Synchronizing the accepted source/configuration to Yale.
2. Checking current readiness.
3. Submitting A, B, C, and D.
4. Checking queued and running job status.
5. Checking training metrics and logs.
6. Fetching results.
7. Resuming after a clean segment stop.
8. Inspecting a failed job.
9. Running final evaluations.
10. Syncing W&B if offline logging is used.

Make clear which commands are prepared versus actually executed.

Provide a short **GO / GO WITH CAVEATS / NO-GO** assessment based on real evidence.

Do not redefine successful readiness as requiring every optional optimization to be completed.

## 12. Decision principle

The final project needs meaningful experimental evidence under a limited academic timeline.

The existing implementation has already passed readiness gates and produced real training updates. We should avoid returning to a state where we cannot launch because of hypothetical performance, infrastructure, or safety concerns.

At the same time, the projected 70–330 H200 GPU-hours across A–D is significant. It is worth investigating measured inefficiencies, especially sequential generation, redundant scoring, and unnecessary backward passes.

Use this principle throughout:

**Make the system faster where measurements justify it, make it recoverable where real failure modes require it, and otherwise preserve the already validated experiment.**

If the optimization branch is not demonstrably superior or ready, retain the original frozen baseline as a valid launch option.

Do not silently change the scientific objectives to improve throughput.

Do not require a complete architectural rewrite.

Do not repeatedly rerun unchanged readiness tests.

Proceed autonomously through implementation and available validation. Resolve ordinary engineering choices yourself, document tradeoffs, and report only genuine blockers requiring new authorization, resources, or a scientific design decision.

**The expected outcome is a reliably executable, well-instrumented, computationally efficient A–D experiment—not another readiness plan that prevents the experiments from starting.**