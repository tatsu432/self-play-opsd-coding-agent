# Codex implementation prompt: Gated self-distillation in gradient-guided coding-agent self-play

You are implementing a research codebase for a solo final project in Yale CPSC 5880, Advances in Frontier AI Models. You have no prior conversational context. Treat this document as the complete implementation specification. Implement the repository, meaningful correctness tests, launch profiles, experiment observability, and reproducible analysis. Do not stop after producing a plan or skeleton.

The intended repository is self-play-opsd-coding-agent (https://github.com/tatsu432/self-play-opsd-coding-agent). Inspect the current working directory, git state, existing code, and applicable AGENTS.md before editing. Preserve useful work and unrelated changes. Work in the provided checkout; do not assume the repository is empty or overwrite an existing implementation wholesale. If no checkout is available, create a clean local project and state that fact. Do not publish, push, provision paid machines, or start the four long training runs merely because you implemented their launch commands. Run local checks and short pilots when suitable resources are already available. Finish with exact commands and verified readiness status.

## 1. Research question, scope, and experimental claims

We study whether token-level gating improves privileged on-policy self-distillation in an evolving agentic self-play curriculum, and whether a task proposer should be rewarded using the student's RL gradient alone or its combined RL-plus-distillation gradient.

The solver is a small coding agent that inspects files, edits a buggy Python function, runs public tests, and submits a repair. The learned proposer creates bugs in verified correct seed programs. A fixed verifier checks proposed tasks and student repairs. The privileged self-teacher is a frozen snapshot of the current student with access to the correct implementation and repair. It scores the student's sampled actions; it does not generate replacement student trajectories.

Four primary configurations:

| Config | Student objective | Proposer reward |
| --- | --- | --- |
| A | GRPO + ungated signed-gap self-distillation | Difficulty |
| B | GRPO + gated signed-gap self-distillation | Difficulty |
| C | GRPO + gated signed-gap self-distillation | Candidate RL-gradient alignment with reference CE gradient |
| D | GRPO + gated signed-gap self-distillation | Candidate combined-gradient alignment with the SAME reference CE gradient |

Comparisons: A versus B tests gating in the complete self-play system; B versus C tests RL-gradient proposer feedback; C versus D tests accounting for the student's distillation update when rewarding the proposer. Since curricula evolve separately, A versus B is a system comparison, not an isolated identical-data causal estimate. Additional common-data diagnostics address this limitation.

This is seeded, finite-horizon Python repair self-play. It is not zero-data Absolute Zero, a general repository SWE benchmark, or a demonstration of recursive self-improvement. Four main runs do not establish superiority over GRPO alone: evaluate the base model, and support an optional GRPO-only config without requiring a fifth main run. One training seed is the planned budget. Evaluation confidence intervals cannot measure training-seed variance.

The chosen sampled-token auxiliary objective below is SDAR-inspired, not an exact reproduction of SDAR, full-vocabulary KL, or an unbiased full-trajectory KL gradient. Do not silently replace it with weighted SFT, full-distribution KL, or a different paper's objective. CompassPlay's main method uses verified-solution CE gradients; our C/D candidate gradients deliberately differ. Positive C/D results are hypotheses, not assumed outcomes.

## 2. Engineering priorities and implementation process

Prioritize mathematical correctness, exact data boundaries, reproducibility, measured resource use, and a small working vertical slice. Use typed Python modules, explicit configuration schemas, narrow interfaces, structured events, and actionable errors. Keep research math separate from framework, environment, and logging code. Use a normal installable package with pyproject.toml, a lockfile or equivalent pinned environment, formatting/linting, and pytest. Avoid a new general-purpose orchestration framework, heavyweight services, or duplicating features already supplied by a tested training framework.

Preferred stack: PyTorch + Transformers + PEFT + verl, with vLLM for rollout. Use a thin project extension or subclass at documented extension points; avoid editing installed site-packages or maintaining an unnecessary fork. Verify the pinned verl/vLLM/PEFT combination supports the needed model, LoRA weight synchronization, multi-turn response masking, and backend. Existing on-policy-distillation support does not imply this exact experiment is turnkey.

First inspect current official APIs and a working example from the pinned versions. Record exact package versions, repository commits, model/tokenizer revisions, CUDA/PyTorch versions, and container digest. Do not invent API parameters or pin an untested mix of latest packages. A Hugging Face generation backend and a tiny local model should support correctness tests and CPU smoke runs. They are debug backends, not evidence that large-GPU support works.

Implement stages in order:
1. Data adapter, deterministic task construction, sandbox contract, tool loop, and fixed base-model evaluation.
2. Student/proposer objective functions and tiny-model numerical tests.
3. One complete outer iteration of A/B, then C/D gradient scoring.
4. Checkpoint/resume, hardware profiles, W&B, and a short end-to-end pilot.
5. Full-run and evaluation/analysis commands with documented resource projections.

If GPUs, datasets, network, W&B credentials, or sandbox capabilities are unavailable, complete everything independently possible, use explicit mocks only in tests, and report the exact unverified integration. Never generate pretend experimental results. No pass statement or NotImplementedError may sit on the default production path while claiming it is complete.

## 3. Data and immutable split manifests

Primary source: pinned MBPP+ release from EvalPlus. The expected release has 378 parent problems; inspect actual schema and eligibility. Expected partition targets are 200 training parents, 32 reference parents, 46 development parents, and 100 final in-domain parents. Partition by original problem BEFORE creating mutations. All descendants of a program remain in one partition.

Split development parents deterministically into 34 learning-curve parents and 12 counterfactual-diagnostic parents. This subdivision does not create a new final test set; both are development data. Use a seeded hash assignment, store exact IDs, and never use evaluation performance to choose assignments.

External transfer source: pinned HumanEval+ release, all eligible parent programs. Create one fixed repair task per eligible external parent. Do not train, select hyperparameters, or compute proposer/reference rewards on it.

Before splitting, remove exact/near-duplicate normalized code and materially equivalent problem families that cross intended partitions or the external dataset. Use a transparent deterministic procedure (e.g. normalized AST plus conservative similarity checks), record decisions, and do not claim it detects every semantic duplicate. Preserve complete families together. External overlap removal may reduce counts.

Data preparation must:
- Pin dataset revision/checksum and EvalPlus version; preserve licenses and provenance.
- Validate correct reference programs under the authoritative tests and preconditions.
- Preserve required imports/setup code; reject unsupported non-stdlib dependencies rather than silently installing them during an episode.
- Create README descriptions from the original specification only. Do not include the correct solution, inverse patch, hidden tests, seed mutation operator, or proposer metadata in the student's files or messages.
- Designate the original public example tests as public. Keep additional verifier tests, expected outputs, and canonical repairs in evaluator/trainer-owned storage. Check that each task has useful additional verifier coverage; record public/hidden overlap rather than pretending repeated public tests are new hidden evidence.
- Record exact eligible counts, exclusions, failure reasons, parent IDs, split roles, source hashes, mutation seeds, tool-protocol version, and environment image digest.
- Fail clearly if expected data cannot be built. Do not fabricate tasks or silently redistribute reserved test parents into training to meet a target count.

Frozen evaluation construction: use a non-adaptive seeded mutation operator library, independent of the learned proposers. Attempt up to a configured finite number of mutations per parent (default 20). Aim for 2 valid fixed tasks per learning-curve parent, 2 per diagnostic parent, 1 per reference parent, 2 per final in-domain parent, and 1 per external parent. Select by structural validity and verifier behavior only, never by model success. Freeze exact tasks before main training. Fewer valid tasks are acceptable only with an explicit manifest and coverage report.

Call results "MBPP+-derived repair" and "HumanEval+-derived repair". Our repair scores are not official MBPP+/HumanEval+ generation scores. Base-model pretraining contamination remains possible. Keep an optional standard generation-evaluation adapter separate, clearly named, and outside the required four-run implementation.

Training-only pilot: use parent programs from the training partition. It may choose common environment/hyperparameter settings, then reset all adapters, optimizer states, RNG streams, and proposer histories before the main runs. Never pilot on final-test problems. Freeze a main-study config after the pilot.

## 4. Task representation and safe repair environment

Default workspace:
- README.md: behavior specification.
- src/solution.py: mutated implementation, with any necessary approved setup/imports.
- tests/test_public.py: public examples, read-only.

Prefer a single target implementation file initially. Optional helper modules are an extension, not required for the first experiment. Keep function names/signatures and externally observable API fixed.

Expose exactly four semantic tools via a fixed plain-JSON action protocol, one tool call per assistant message:
- {"action":"read_file","path":"src/solution.py"}
- {"action":"apply_patch","patch":"<unified diff restricted to allowed source files>"}
- {"action":"run_tests"}
- {"action":"finish"}

Paths are normalized and checked by the trusted tool server; no path traversal, symlink escape, arbitrary command execution, patching tests/configs, or network access. Record the protocol schema and system prompt as versioned artifacts. Read README/tests through read_file if needed. Fixed file lists and tool descriptions may appear in the initial prompt; the complete correct source must not.

Use ordinary model sampling without constrained decoding in the main study so behavior-policy probabilities are well defined. Invalid JSON/unknown action returns a bounded structured error and consumes a turn; it must not execute anything. Keep raw generated text and parse status. Unsafe actions are rejected and logged. Do not replace invalid output with model-generated retries outside the episode budget.

Version and freeze actual prompt templates. Default student system template:
  "You are a Python repair agent. Repair the behavior described in README.md while preserving the public API. You may inspect allowed files, edit allowed source files, and run the public tests. Additional tests will check correctness. Each reply must be exactly one JSON object invoking read_file, apply_patch, run_tests, or finish with the documented fields. Do not include prose outside JSON. Public tests are read-only. Use finish when the repair is ready."
The initial user message gives the task specification, allowed file list, tool schemas, and remaining budget, but no mutation hints or correct source. Tool messages contain only bounded public observations. Choose one documented serialization for budget updates and keep it identical in training, teacher replay, references, and evaluation. Budget updates are non-assistant context and have zero loss mask.

Default proposer system template:
  "You create executable Python repair tasks. Given a verified correct implementation and its specification, introduce a small semantic bug while preserving the API. Change at most 3 executable AST nodes and 12 nonblank source lines. Preserve imports and function signatures. Do not modify tests, add I/O or introspection, deliberately crash, or delete the implementation. Reply with exactly one JSON object containing a patch field whose value is a unified diff."
The proposer user message contains only the training parent's specification, correct source, and allowed path/patch format. A deterministic trusted classifier can assign bug-family labels after generation; no proposer-supplied label becomes authoritative. Sampling new tasks does not include ad hoc extra LLM calls to repair malformed output.

Default teacher-only system addition:
  "Training-only privileged information follows: <correct_source>...<inverse_initial_patch>.... Use it to assess the next action token in the provided agent history. The history is the student's actual history, including any incorrect edits and tool failures. The correct implementation describes the desired final behavior; an inverse patch may need adaptation after subsequent edits."
Teacher scoring appends the exact student action targets under teacher forcing; this instruction does not request a separate teacher response. Freeze the actual wording, delimiters, serialization, and injection position before main training.

Default episode limits:
- 8 assistant action turns total.
- 4,096 assistant-generated tokens total; 1,024 generated tokens maximum per turn, also limited by the remaining episode budget.
- 10,240-token maximum student sequence, including observations and all generated tokens.
- 12,288-token maximum teacher sequence, including privileged context. Reserve up to 2,048 tokens for that context at task admission; reject overly large parent tasks before the study instead of dropping privilege in some examples.
- Each tool observation capped at 1,024 tokens with an explicit truncation marker.
- Public test call default 10-second wall limit; final full-verifier call default 30 seconds; calibrate once using correct programs and freeze common limits.
- Per-execution default 1 CPU core, 1 GiB RAM, bounded process count, disk/output size, and no GPU access. Allow explicit per-task time limits derived from correct-program runtime without solver-dependent tuning.

Never silently truncate an already sampled action or its target tokens in training/scoring. End the episode on a budget boundary. A finish action, token/context/turn limit, or terminal tool error ends the episode; evaluate the current allowed source snapshot. Reward is defined solely by the final verifier, not by whether the model wrote a convincing final message. Record termination reason separately. Runtime/resource-limit failures attributable to candidate code count as task failures; infrastructure failures do not automatically become student mistakes.

Security is functional experiment infrastructure because generated code executes. Run code in disposable, resource-limited sandboxes with no egress, host credentials, home mounts, trainer files, AWS metadata, or Docker socket. Mount public tests read-only. Never run generated code directly inside the trainer process. Verify the sandbox can enforce these properties on the actual host. Docker/Apptainer existing on a host is insufficient evidence of isolation. A restricted AST validator helps but is not a replacement for execution isolation.

The authoritative evaluator owns hidden tests and expected results outside the rollout workspace. Copy only allowed final source files into a fresh evaluation sandbox. Reinitialize the evaluator and do not trust state, pytest configuration, cached results, or reported pass counts from the student's public-test environment. Where feasible, keep expected-output comparison in a trusted supervisor and expose only input/output calls to candidate code. Be explicit about any EvalPlus adapter's actual threat model; do not claim cryptographic secrecy of tests from arbitrary Python executing in the same process. Restrict imports/introspection and isolate execution; test the relevant bypass attempts. Any unavoidable security limitation must block a production launch on a shared host, while allowing safe mocked/local tests.

Differentiate statuses: candidate_verification_failure, student_failure, agent_budget_timeout, sandbox_infrastructure_error, missing_dependency, data_oracle_failure. Retry transient infrastructure errors at most a fixed number (default 2), preserving task/episode IDs and generation seeds. Do not mask model failures as infrastructure retries. Persist unresolved errors and stop an affected main run rather than silently dropping evaluation cases or distorting training rewards.

## 5. Proposer task generation and validity

For a training parent, the proposer sees the original specification and verified correct source. It outputs one plain-JSON object with a unified-diff patch introducing a bug. It never controls the authoritative verifier. No proposer explanation is passed to the student.

Initially permit small semantic changes: comparisons/bounds, indexing/slicing, constants/operators, accumulator initialization, and small branch/update changes. Default limits: at most 3 executable AST node changes and at most 12 changed nonblank source lines. Implement a deterministic validator with documented AST-diff semantics; preserve imports, signatures, function names, and approved file structure. Reject wholesale deletion, deliberate unconditional exceptions, added I/O/network/subprocess facilities, introspection, test edits, or unsupported syntax. Keep the restrictions identical in A-D.

Proposal validity requires:
1. Base program passes its authoritative verifier.
2. Patch parses, applies cleanly, preserves the API, and passes structural restrictions.
3. Mutated program fails at least one authoritative semantic test within limits. Timeout/crash-only mutants are invalid initially.
4. Restoring the original source passes the verifier.
5. Mutated code is not a duplicate for the same parent in the run's committed proposal history or earlier committed candidate slot of the current iteration.

Normalize code/AST for deduplication and record keys. Candidate slots have a deterministic order independent of completion timing. The first valid slot owns a duplicate; subsequent identical candidates are invalid. Persist duplicate history across resumes. Reference/dev/test task generation uses separate immutable histories. No data from those partitions enters proposer prompts or duplicate feedback.

Invalid proposals are retained in proposer batches with reward -1 and a reason code. They do not get student rollouts. Do not refill until 16 valid proposals exist: each iteration has a fixed attempted-proposal budget. Difficulty/alignment scores are for valid proposals only. Do not keep only the highest-scoring valid tasks for student training: all valid proposed tasks train the student; proposer reward shapes the future distribution.

## 6. Models, adapters, and frozen snapshots

Default model/tokenizer: Qwen/Qwen3-4B-Instruct-2507, pinned revisions. Dense 4B, non-thinking mode. BF16 weights, no quantization in the primary implementation. Inspect actual module names; do not assume an incompatible target list.

Separate trainable LoRA adapters for student theta and proposer phi, initialized from the same base model:
- rank 32, alpha 64, dropout 0.
- Attention q/k/v/o projections and MLP gate/up/down projections where present.
- Freeze every base parameter, including embedding and LM head unless explicitly included in a later declared variant.
- Independent optimizers and state; proposer updates must not modify student adapter tensors or vice versa.
- Disable base-model dropout for consistent rollout/scoring.

At the beginning of outer iteration k, freeze student snapshot theta_k and proposer snapshot phi_k. Student rollouts and old student logprobs use theta_k. Teacher uses the same theta_k with privileged context. All scoring and candidate gradients are evaluated before the student optimizer step. Proposer sampling/old logprobs use phi_k. Cache detached scoring outputs so no permanently resident third model is required.

Teacher has no optimizer, no separate RL objective, no gradient flow, no EMA, and no independent sampled history. Do not confuse it with a fixed KL reference policy. The reference dataset below supplies a CE-gradient target, not a separate model.

Copying adapters, phase-wise loading, or shared frozen base weights are allowed if isolation is tested. When using an adapter-disabled base as an optional KL model, distinguish its identity explicitly. No teacher/proposer inference may accidentally retain the student's active adapter after a switch. Save weight hashes/version IDs and confirm rollout engine synchronization on every update.

## 7. Exact student objective and tensor contract

Let x be one valid proposed repair task. Sample G=4 independent complete episodes y_i from the frozen student pi_old using the same task and identical initial workspace, with independent keyed generation seeds. R_i is 1 iff the authoritative final evaluator passes all required public-plus-additional verifier tests, and 0 otherwise. No partial-credit, length, format, or public-test bonus in the primary study.

Let mu = mean_i R_i and sigma = sqrt(mean_i (R_i-mu)^2), using population standard deviation. A_i = (R_i-mu)/(sigma+1e-6). If rewards are all identical, set every A_i exactly to zero. Advantages are detached. Do not add cross-task reward whitening, running baselines, or difficulty normalization.

For episode i, M_i,t is a binary mask on positions actually sampled by the assistant, N_i=sum_t M_i,t, and rho_i,t(theta)=exp(log pi_theta(y_i,t|h_i,t)-old_logp_i,t). Define episode averaging E_i[z] = sum_t M_i,t*z_i,t / N_i. Include generated JSON/action tokens and genuinely sampled end-of-turn tokens. Exclude forced role markers/formatting, prompt tokens, tool outputs, padding, and teacher privilege. For an empty generated response, use a documented terminal failure path with N_i=0 and zero loss contribution; never divide by zero or fabricate a target token.

L_RL(x) = -(1/G) sum_i E_i[min(rho_i,t*A_i, clip(rho_i,t,1-epsilon,1+epsilon)*A_i)], epsilon=0.2.

No entropy bonus, critic, separate value loss, or fixed-reference KL penalty in the primary default. Support a fixed-reference KL penalty as an explicitly named optional variant; it must not silently become active through framework defaults. Any addition requires a new study config. Record policy drift diagnostically rather than treating sampled old/current log ratios as a proven nonnegative KL estimate.

The teacher sees h_i,t plus privileged context c+(x): original correct source and the inverse initial mutation patch. Inject this only into a teacher-only system block with an instruction to assess the next action given the agent's ACTUAL current history. It must account for edits/failures the student has already made. Teacher sequence includes the exact student messages and tool observations, never an independently generated successful teacher rollout. Privilege is constant per task, inserted at the beginning and retained on each turn.

Define at the frozen snapshot:
  Delta_i,t = stopgrad(log q_theta_k(y_i,t | h_i,t,c+) - log pi_old(y_i,t | h_i,t)).
  d_i,t = stopgrad(clip(Delta_i,t,-5,5)).
  w_i,t = 1 for A; stopgrad(sigmoid(beta*Delta_i,t)) for B/C/D; beta=1.0.
  a_SD_i,t = stopgrad(w_i,t*d_i,t).

L_SD(x) = -(1/G) sum_i E_i[a_SD_i,t * log pi_theta(y_i,t|h_i,t)].
L_student = mean over valid tasks x of [L_RL(x) + lambda_SD*L_SD(x)], lambda_SD=0.1 initially.

Use the same lambda in all four primary configs. All-valid/all-failed groups have zero RL signal but can retain SD signal; do not filter them out of SD. Normalize by masked token count, not the sum of gate weights or |gap|; renormalizing gates would remove part of the intervention. The denominator is per episode and then per task, so long episodes do not automatically receive higher weight.

This is an explicit local on-policy semi-gradient surrogate. Teacher/gap/gate coefficients are fixed within an iteration. With the single optimizer step below, current parameters equal old parameters at loss evaluation, so it is an on-policy update. Its scalar loss may be negative, and decreasing the numerical loss across iterations is not a reliable health criterion because coefficients and tasks change.

Token-level contract:
- Stored old_logp and teacher_logp are finite FP32 values for exactly the same sampled target token IDs.
- Student logits correspond to [batch,L_student,vocab] conceptually, but gather target logprobs in chunks or a fused kernel without materializing long-lived full-vocabulary tensors. Loss arrays/masks can be [batch,L_student] or an equivalent packed representation.
- Teacher sequence length differs because of privileged prefix. Maintain an explicit mapping from every assistant token to its corresponding teacher scoring position. Causal shifting must be correct.
- Preserve generated token IDs. Do not decode/re-tokenize responses and assume token identity/turn boundaries survived. Validate chat-template prefix consistency and token mapping on multi-turn examples. If using per-turn teacher-forced scoring, use the exact full actual history before that turn and identical action IDs.
- Never compute gradients through teacher, old logprobs, reward, advantages, Delta, or w.
- No gradients through environment/tool observations.

Use training temperature=1.0, top_p=1.0, no top_k/min_p truncation, repetition penalties, or constrained decoding for BOTH student and proposer. This deliberately tightens the earlier provisional sampling default to avoid a mismatch between the behavior policy and the logprobs used by GRPO and Delta. If a different temperature is later enabled, the implemented policy logprobs must include that temperature consistently in sampling, current/old scoring, and teacher-gap definitions; treat truncated sampling as a separate carefully tested variant. Do not compare tempered behavior probabilities with untempered model probabilities as if they were the same policy.

## 8. Proposer objectives

Each outer iteration samples 4 training parents from a common deterministic parent schedule. For each parent, generate K=4 independent bug proposals from phi_k, maximum 512 generated tokens each. This is 16 attempted proposals per iteration. All four proposals for one parent share the exact proposer prompt. Use their reward group to form proposer advantages using the same population-std rule as the student. No grouping of unrelated parent prompts to increase variance.

For valid task x, p_hat(x)=mean_i R_i from its same G=4 student episodes.

A/B reward: r(x)=4*p_hat(x)*(1-p_hat(x)). This is 0, 0.75, or 1 for G=4 and favors mixed outcome groups. Invalid proposal reward=-1.

C/D reward: define reference and candidate gradients in Section 9; for valid task r(x)=(1+cosine(g_candidate(x),g_ref))/2. Preserve negative cosines; do not clamp to positive-only values. Invalid proposal reward=-1. Finite near-zero norm yields neutral valid reward=0.5 with an explicit flag. NaN/Inf is a numerical failure, not a neutral case.

Proposer training: standard clipped GRPO on its sampled patch tokens, one token-mean per proposal followed by mean over four proposals and mean over four parent groups. Clip=0.2, no proposer SD, no critic, no entropy bonus or reference KL by default. One AdamW optimizer step per outer iteration after all rewards are known. Preserve invalid proposal tokens and their old logprobs for this update. A group with identical rewards gets zero proposer advantage and still counts toward exposure/cost statistics.

Do not ask an external LLM to rate task quality, generate feedback, or provide extra solutions. Do not add a task-selection filter based on alignment/difficulty after reward assignment. Structural validity restrictions are common to all configurations.

## 9. Gradient-alignment objective and efficient scoring

Reference set: 32 held-out reference parents, one frozen valid mutant per parent and one verified canonical repair trajectory per mutant. Canonical sequence is read source -> apply correct repair -> run public tests -> finish. Construct it deterministically from the known source/patch and obtain its observations by real tool execution. Verify its final program. These are programmatically constructed action demonstrations, not model-generated chain-of-thought.

Compute reference CE only on the canonical apply_patch assistant action's sampled-equivalent JSON content tokens, with a fixed edit-action mask. Do not train directly on the anchors or compute CE on tool outputs. If protocol serialization requires fixed wrapper tokens, explicitly record whether they are included and keep it fixed across C/D. Context for reference scoring is the normal student-visible history; no privileged block in reference CE.

For task j, L_ref_j = mean negative logprob over its canonical edit-action target tokens. L_ref = mean_j L_ref_j across all available frozen anchors (target 32). g_ref = grad_theta L_ref at theta_k. Recompute once per C/D iteration, reusing it for all candidates in that iteration. Reference data is part of training guidance, so never call it an unbiased validation/test set. Reference prompts/answers are never fed to the proposer.

Candidate gradients, evaluated at theta_k BEFORE optimizer updates:
- C: g_candidate(x)=grad_theta L_RL(x).
- D: g_candidate(x)=grad_theta [L_RL(x)+lambda_SD*L_SD_gated(x)].

Use the same stored on-policy student episodes, advantages, old logprobs, detached gaps/gates, masks, and per-episode/per-task loss normalization used for the real student update. Do not substitute CE on successful solutions, remove negative-advantage episodes, or use an optimizer delta while labeling it the specified gradient. Include all four episodes per task. C's all-equal reward groups have exactly zero candidate RL gradient; D may have a nonzero auxiliary direction. Record this distinction explicitly.

Gradient coordinates: all trainable STUDENT LoRA parameters, deterministic name/order, no base/proposer parameters. Let d be their total scalar count. Every compared vector is in R^d at the same adapter parameterization and snapshot. Accumulate dot products/norm squares in FP32 or higher:
  cosine = dot(g,g_ref)/(norm(g)*norm(g_ref)).
Clamp only tiny floating-point spillover to [-1,1]. Default near-zero threshold is norm <=1e-12, configurable and logged; exact zero and below-threshold cases have separate counts. Do not add epsilon in a way that manufactures meaningful cosine for zero gradients. A zero reference gradient makes all valid candidate rewards neutral for that iteration and raises a health event. A nonfinite gradient aborts/reports the affected iteration.

Efficiency:
- Stream per-episode/microbatch contributions with correct weights. Store one reference gradient and at most one candidate gradient at a time, optionally on CPU. Do not allocate [candidate,base_parameter] Jacobians or per-token gradient tensors.
- No Hessians, second-order differentiation, differentiation through alignment, or student-training inner loops in the main reward path.
- torch.autograd.grad or a backend-safe equivalent may be used. Clear/restore gradient buffers so scoring cannot affect the real update. Scoring never changes parameters, optimizer state, RNG-dependent rollout schedule, or teacher snapshot.
- FSDP/sharded gradients require correct global dot/norm reductions. Fail clearly if an advertised distributed profile cannot implement them; do not silently use local-shard cosine or only a subset of layers.
- Persistent base gradients must not be created. Shared-base adapter switching must preserve the correct student adapter in every scoring operation.
- Gradient collection precision (e.g. BF16 backprop accumulated into FP32) must be recorded. Validate the efficient implementation against a tiny FP32 reference.

g_ref is the SAME CE target for C and D. Change only the candidate objective. No separate RL reference rollouts, no reference SD loss, no validation-gradient objective drift, and no math reference dataset. Raw-gradient cosine is a first-order proxy under the chosen adapter coordinates; AdamW preconditioning and clipping can change the actual step. Evaluate proxy validity through diagnostics instead of claiming an exact guarantee.

## 10. Outer iteration, optimization, and fair budgets

Pseudocode:
1. Restore/confirm theta_k, phi_k, optimizer/RNG states and weight versions.
2. Draw 4 parents from the common keyed parent schedule; sample 4 proposer patches each.
3. Validate/deduplicate in canonical candidate-slot order, with infrastructure errors handled separately.
4. For every valid task, sample 4 independent student episodes from theta_k; evaluate final rewards.
5. Score exact student targets under unprivileged theta_k and privileged teacher theta_k; build detached auxiliary coefficients.
6. C/D only: compute g_ref and per-candidate gradient rewards at theta_k. A/B use difficulty rewards without gradient-alignment overhead.
7. Build the global student loss over all valid tasks. Accumulate microbatches; take EXACTLY ONE student optimizer step if there is at least one eligible trajectory and nonzero update. If the total gradient is zero, record a no-op rather than manufacturing signal.
8. Build the proposer loss from all 16 attempted proposals; accumulate microbatches; take exactly one proposer optimizer step if nonzero.
9. Commit iteration event, proposal history, RNG/counters, checkpoint when due, and synchronize new adapter weights for next rollout.
10. Evaluate development tasks at scheduled completed-iteration boundaries.

No valid proposals: skip student rollout/scoring/update; train proposer on invalidity if its within-group rewards have variance (all -1 produces zero proposer signal). No hidden resampling or minimum-success filtering. Warn if this repeats.

One full mathematical batch means one optimizer step. Do not let microbatch size, worker count, FSDP world size, or framework "minibatch" defaults produce more optimizer steps. For distributed training, handle DDP averaging/FSDP reductions and uneven/empty ranks explicitly, so the result equals the weighted single-process reference. A mixed task count must not cause rank-local losses to be averaged incorrectly. Teacher coefficients remain fixed throughout the step.

With one optimizer step evaluated at theta_k, importance ratios start at 1 and PPO clipping is normally inactive. That is expected, not a bug. Keep the clipped definition for clarity/extensibility; do not introduce extra PPO epochs to make clipping metrics nonzero. Any later multiple-step reuse of data/SD coefficients needs explicit off-policy treatment and a named variant.

Default hyperparameters:
- 200 completed outer iterations per main configuration.
- Seed 2026; use separate stateless/keyed RNG streams for split assignment, parent schedule, proposer sampling, student episodes, eval, and diagnostics.
- Maximum student episodes per iteration=16*4=64; maximum per 200-iteration run=12,800, with actual count depending on validity.
- AdamW student LR=1e-5, proposer LR=5e-6, betas=(0.9,0.999), eps=1e-8, adapter weight_decay=0, constant LR schedule.
- Global gradient norm clipping=1.0, applied after accumulating the true global gradient.
- SD coefficient=0.1; gap clip [-5,5]; sigmoid sharpness=1.0.
- No mixed quantization, model-size switching, or automatic objective changes.

These are provisional pilot defaults, not established optimal choices. Inspect lambda*||g_SD||/||g_RL|| on nonzero-RL pilot batches. If auxiliary influence is negligible or dominant, calibrate the common lambda once from training-only pilot data and freeze it for A-D. Preserve the actual influence difference induced by gating. Record all tested pilot settings and the selection rule; avoid an open-ended sweep.

Default one-shot calibration rule, if enabled: retain lambda=0.1 unless the median gated-batch gradient-norm ratio is <0.01 or >1.0 on at least 8 fixed training-only pilot batches with finite nonzero RL/SD norms. Otherwise rescale lambda once to target a median ratio of 0.1, bounded to [0.001,1.0], record the realized value, and freeze it for all four runs. If sufficient mixed-outcome pilot batches do not exist, report that calibration is unsupported and address environment difficulty before main training; do not infer a ratio from zero RL norms. This rule calibrates scale, not held-out performance.

A-D must start from identical student/proposer initialization, adapter tensors, prompts, parent schedule, split/task manifests, and common hyperparameters. Key random streams by semantic IDs rather than asynchronous completion order. Generated curricula will diverge; do not force identical sampled outputs after policies diverge. A common seed is not a substitute for replicated training.

Primary comparison uses fixed attempted proposals and outer iterations. Report actual valid tasks, student episodes, tokens, optimizer steps, test calls, GPU-hours, and CPU-hours; they can differ naturally. Also compare learning curves at overlapping measured compute budgets. Do not extrapolate beyond observed curves or drop scoring cost to make C/D look efficient. Report repeated parents/mutants and diversity.

## 11. Hardware portability and memory/performance design

Support explicit profiles, not hard-coded device assignments:
- cpu_debug: tiny local model and fixture data; objective/environment tests only.
- single_gpu_80gb: one H100/A100-class GPU, LoRA, microbatch 1, sequential rollout/scoring/optimization; feasibility profiled rather than asserted.
- dual_gpu_80gb: preferred baseline on 2 H100-class GPUs; allow sequential use of both GPUs for training/rollout or an explicit tested role assignment.
- dual_gpu_141gb: 2 H200-class GPUs, same mathematical batches, larger safe microbatches if measured.
- multi_gpu_single_node: 4/8 GPUs with a tested sharding/DP strategy. Use all-adapter-coordinate distributed cosine correctly.

Single-node support is required. Keep interfaces suitable for later multi-node execution; do not claim multi-node readiness without testing it. GPU count, available VRAM, host RAM, CPU workers, sandbox concurrency, rollout concurrency, microbatch/token budget, tensor/data parallelism, parameter/optimizer offload, and checkpoint/output paths are configuration fields.

Preferred initial allocation: 2 GPUs, 16 CPU cores, 128-256 GiB RAM. This is a requested profile, not a guarantee of memory fit. Inspect actual free resources and scheduler allocation. Respect CUDA_VISIBLE_DEVICES; never use unallocated devices or spawn unbounded workers based on the entire host.

Use BF16, activation checkpointing, supported memory-efficient attention, length bucketing/packing only after mask-equivalence validation, bounded asynchronous tool execution, and batched teacher scoring. Cache immutable environments/test setup and dataset artifacts by content hash. Phase-wise load/offload adapters/inference engines to avoid simultaneously resident student+teacher+proposer+reference models. Limit CPU copies of gradients and trajectories according to RAM budget; use memory-mapped/disk-backed storage when appropriate.

Do not call torch.cuda.empty_cache every microbatch, enable wandb.watch on the entire 4B model, or retain full-vocabulary logits. Profile real peaks for rollout, teacher replay, gradient scoring, and optimizer step separately.

OOM policy: record phase/allocation and stop safely. A pre-main-run calibration may lower microbatch/token concurrency, enable a supported offload mode, and retry the SAME batch without changing episode budgets/objectives. During a main run, any runtime memory adaptation must preserve the exact mathematical batch, data, and step count and be logged. If that cannot be guaranteed, stop and resume from a committed checkpoint with a reviewed resource config; do not skip examples or truncate them to survive an OOM.

Provide Slurm launch templates for Yale with configurable account, partition, GPU type/count, CPUs, RAM, wall limit, project scratch, model cache, and container path. Current defaults may include a 24h job limit; query actual account/partition rather than claiming every Yale user has the same limits. Handle pre-timeout SIGUSR1 and SIGTERM by requesting a safe checkpoint, not performing collectives inside a signal handler. Use short resumable jobs. Never run GPU training on a login node. Apptainer profile must state sandbox/network restrictions actually supported; generated-code execution can use an isolated remote worker backend if the shared host cannot safely provide them.

Provide AWS single-node launch/run documentation and container entrypoints, including a one-H100 p5.4xlarge profile and an optional 8-H100 p5.48xlarge profile if implemented. Use user-supplied paths/regions/quotas; never assume quota or credits apply. No credentials embedded in configs/images/logs. Provisioning is not part of this implementation request. Include configurable runtime/GPU-hour limits and a preflight cost estimate from USER-SUPPLIED current instance price; label estimates, do not invent current prices.

Pilot throughput report: at least 10 representative outer iterations on available target hardware, including all phases, plus C/D scoring. Report median/p90 iteration time, startup time, generated/prefill/scored tokens, effective tokens/sec by phase, peak VRAM/host RAM, CPU/test latency, validity, success, and projected 100/200-iteration runtime. At 2,000 tokens/episode, the maximum 12,800 episodes imply ~25.6M student generated tokens/run, excluding other work. Do not turn this rough count into a claimed runtime benchmark. If budget requires 100 iterations, change the common main-study budget before the four main runs, not only the slow configuration.

## 12. W&B observability and local source-of-truth logs

Implement live Weights & Biases tracking with an opt-in online mode and fully functional offline/disabled modes for restricted HPC nodes. Use SDK APIs verified against the pinned version. W&B outages must not lose local scientific records or crash a healthy training step. Offline logs are not live; make that clear and provide sync commands. Never request credentials interactively inside a Slurm job.

Configuration fields: enabled, mode, project (default self-play-opsd-coding-agent), entity optional from environment, study_id/group, run_name, run_id, tags, notes, log/sample frequencies, artifact policies, and upload limits. Use WANDB_API_KEY/environment or existing authenticated SDK state, never a committed key. Persist run identity with checkpoints. Only the coordinator logs aggregated run metrics; distributed workers have structured local logs and hardware metrics aggregated/tagged by rank/device.

One logical main run per config/seed, grouped by immutable study_id. Suggested name: <study_id>-<A|B|C|D>-seed2026. Tags include method, model revision, dataset manifest hash, hardware profile, pilot/main, and resume segment. Log resolved config, git commit/dirty diff hash, dependency lock hash, prompts, tool protocol, model/tokenizer revisions, adapter targets, total/trainable parameters, and resource allocation. Redact secrets, environment tokens, private absolute paths if unnecessary, and generated credentials.

Keep append-only local JSONL events and Parquet analysis tables as authoritative records. Every record carries schema_version, study_id, logical_run_id, segment_id, config, seed, completed_iteration, phase, wall timestamp, and stable task/proposal/episode IDs as appropriate. Stage logs can occur before an iteration commits; distinguish pending and committed data. Persist enough raw metadata to recompute all reported aggregate metrics.

Counters:
- train/outer_iteration (completed iterations), student/proposer optimizer steps, attempted/valid proposals, generated student/proposer tokens, processed/scored tokens, test calls, elapsed active seconds, allocated GPU-hours and CPU-hours.
- Separate cumulative compute counters for training, scheduled dev evaluation, final evaluation, diagnostics, and startup/checkpoint overhead. "End-to-end" includes all relevant phases; do not hide idle reserved GPUs while tools execute. Integrate allocated GPU count over time, including external rollout/teacher GPUs if any.
- Scheduler queue time is recorded separately from active compute; W&B network downtime is not training time removal.

Use custom axes via define_metric or the pinned equivalent. Default training curves use completed iteration. Log evaluation rows with BOTH iteration and cumulative GPU-hours in the SAME event, and expose eval_by_iteration/* and eval_by_gpu_hours/* aliases or dashboard axis choices without conflicting wildcard definitions. Persist a monotonically increasing W&B event index separately from research step counters; avoid logging to old W&B history steps after async phase events or resume.

Required live metric groups (log every iteration unless stated otherwise):

proposer/*
- attempted_count, valid_count, valid_fraction; invalid counts/fractions by reason; infrastructure retry/error counts separately.
- reward mean/std/min/max; within-parent reward std; fraction of groups with zero advantage.
- duplicate_fraction, distinct_parent_count, distinct_mutant_count, bug-family distribution, patch changed-lines/AST nodes, proposal tokens.
- L_RL, gradient norm before/after clip, update norm, nonzero update count, learning rate.

student/*
- repair_success_fraction with numerator/denominator, public_pass_fraction, hidden_full_pass_fraction, public-pass/hidden-fail fraction.
- fractions of tasks with k=0,1,2,3,4 successes out of G=4; mixed_group_fraction; advantage moments.
- L_RL, L_SD, lambda_times_L_SD, L_total (surrogate values, not cross-iteration quality scores).
- episode turns/tokens/test calls (mean/p50/p90), invalid_action_fraction, per-episode and per-action termination/error rates, budget/context/output truncation rates.
- gradient norms/update norms, clip coefficient, learning rate; sampled policy entropy or exact selected-position entropy with the estimator clearly named.
- old/current logprob mismatch before update, absolute mismatch quantiles between rollout and trainer scoring on selected tokens; post-update drift on a fixed subsample. A sampled log ratio can be negative and should not be mislabeled as exact KL.

distill/*
- raw Delta mean/std/p05/p50/p95, positive_gap_fraction, negative_gap_fraction, gap_clip_fraction.
- gate mean/quantiles and low/high saturation fractions; separate positive-gap and negative-gap weights; signed coefficient moments.
- Same gap/gate summaries by assistant turn and by success/failure, sampled at a bounded rate.
- teacher/student mapped-token counts, mapping_error_count, privilege-token count, teacher/student scoring time and tokens.
- On sampled nonzero-RL batches: ||g_RL||, ||g_SD||, lambda*||g_SD||/||g_RL||, cosine(g_RL,g_SD). Computing these extra gradients has cost; measure them every 20 iterations or on the pilot rather than doubling all A/B work.

alignment/* (C/D every iteration; explicit null/unavailable in A/B)
- raw cosine distribution and proposer reward distribution separately.
- reference norm, candidate norm distribution, exact-zero and near-zero fractions, negative-alignment fraction, nonfinite counts.
- zero-RL-group fraction and fraction of those with nonzero SD/combined gradient on sampled diagnostics.
- reference loss, reference checkpoint/version, anchor count, gradient precision, scored candidate count.
- scoring seconds, fraction of iteration time/GPU-hours spent on alignment; no silent exclusion from compute totals.

eval/*
- per-split greedy repair success (task-macro and parent-macro), public/hidden disagreement, generated tokens/turns/test calls, wall time, budget/infra errors, N parents/tasks.
- Frozen evaluation protocol version, checkpoint hash/iteration, exact evaluation seed, dataset manifest hash.
- Last-checkpoint primary and best-development secondary identifiers explicitly distinguished.

perf/* and system/*
- Phase seconds: propose, validate, rollout, reward_eval, old_student_score, teacher_score, reference_gradient, candidate_gradient, student_update, proposer_update, checkpoint, development_eval.
- Phase generated/prefill/scored tokens and throughput with clear denominators; iteration median/p90 over a trailing window.
- GPU allocated/reserved/peak memory and utilization per device; CPU RSS/utilization, sandbox worker active/queued counts, RAM/disk availability, test wall-time distribution.
- Cumulative active wall/GPU/CPU hours, checkpoint bytes/latency, W&B overhead, estimated remaining time from measured recent iterations.

W&B tables/media at bounded frequency (e.g. every 20 iterations):
- Deterministically sampled proposal examples with validity/reason, patch, score, p_hat, IDs, and bug family.
- Success/failure trajectories with public student-visible context and tool outputs, truncated for display with pointer to complete local artifact.
- Gate/gap-by-turn table/plot; C/D cosine distributions; invalidity reasons; common evaluation failure examples.
- No reference answers or final-test answer/hidden-test content in default live tables. Do not place untrusted patch text into executable HTML.

Artifacts:
- Small artifacts: split/task manifests, resolved configs, environment/lock metadata, prompt/protocol versions, pilot report, analysis tables and figures.
- Checkpoint upload policy configurable; default local checkpoints plus manifest/hash in W&B, optionally final/best adapter artifacts. Avoid uploading full frozen base weights or every trajectory/checkpoint repeatedly.
- Bound table rows, histogram samples, output text, and artifact frequency. No full-model wandb.watch. Explicitly record sampled metric coverage.

Health monitoring:
- Hard stop/checkpoint on nonfinite loss/gradients, privilege leakage, token mapping failure, verifier corruption, parameter mutation during scoring, or unsupported sandbox isolation. Do not "recover" by skipping the offending scientific example.
- Warning events for valid_fraction<0.25 for 5 consecutive iterations, no mixed RL groups for 10 iterations, zero reference gradient, alignment near-zero fraction>0.8 for 5 iterations, invalid-action fraction>0.2 over a trailing 5-iteration window, repeated infrastructure failures, >0.05 gap clipping fraction, >0.95 gate saturation fraction, or unexpected pre-update rollout/scorer mismatch.
- These are configurable initial troubleshooting thresholds, not scientific claims or automatic hyperparameter tuning. Log denominators and enough context to interpret them. Warnings do not automatically change the curriculum/objective.
- A heartbeat/state JSON records current phase, latest progress timestamp, and pending work. A supervisor can detect a stalled phase; an in-process logger cannot report if its own process has died. Document the limits of live monitoring.
- W&B alert integration may be configurable with rate limiting/deduplication, disabled by default; local health events and W&B metrics are required. Never set up email/Slack recipients automatically.
- Provide a dashboard recipe (or a tested workspace script if the pinned API supports it) for the four runs: dev success vs iteration/GPU-hours, validity/mixed groups, gaps/gates, gradient norms/cosines, and phase cost/VRAM. Do not claim a dashboard was created in the user's account unless actually done.

W&B resume details: online runs should reuse a persisted run ID with documented resume semantics. Local state is authoritative if W&B has later pending events than a restored checkpoint. Preserve segment IDs/committed flags; never overwrite old history or silently duplicate committed rows. Offline continuation may require separate physical segments and later sync rather than native single-run resume; implement/test the pinned SDK's actual behavior and aggregate logical runs by IDs locally. Do not promise offline resume semantics the SDK does not provide.

## 13. Checkpointing, interruption, and reproducibility

Default checkpoint every 10 completed iterations, on a clean stop request, and before a scheduler deadline when possible. Persist:
- Student/proposer adapter weights and both complete optimizer states; schedulers/scalers if used.
- Base/model/tokenizer revisions, parameter-name order, adapter config, prompt/protocol hashes, loss version, main-study resolved config and compatibility hash.
- Python/NumPy/PyTorch CPU/CUDA RNG states as applicable and stateless seed counters/keys.
- Parent schedule cursor, proposal duplicate history, curriculum/task counters, iteration/optimizer counters, evaluation schedule and checkpoint-selection state.
- W&B/logical/segment run IDs, committed event offsets, resource-time counters.
- Dataset/task/reference manifest hashes and the required canonical trajectories.

Use atomic checkpoint writes: write a temporary directory/files with checksums, then commit/rename or write a completion marker. Do not delete the previous good checkpoint until the new one is verified. Configurable retention: last 2, final, and best-development adapters plus required resume state. Clearly distinguish inference-only adapter artifacts from fully resumable checkpoints.

An outer iteration is transactional. Stage proposal/episode/scoring events; commit after BOTH student and proposer updates and state bookkeeping. On interruption mid-iteration, either restore the preceding committed checkpoint and deterministically regenerate/reload the pending batch, or persist enough stage state to resume correctly; never replay an optimizer update twice or restore only one adapter. Signal handlers request a safe exit; they do not execute unsafe distributed collectives. Committed local metrics must not contain two active versions of the same iteration after resume.

Because periodic checkpoints are less frequent than iteration events, restoring checkpoint k may roll back several later recorded iterations, not just one pending iteration. Persist a lineage/rollback watermark: invalidate or supersede records above k in the abandoned segment before replay, preserve them for audit, and make aggregation select only the active lineage. Apply the same rule to curriculum history, evaluation records, optimizer counters, GPU-hour accounting, and W&B segments. Physical compute already spent on abandoned work still counts in end-to-end consumed resources even when its scientific steps are rolled back.

Resuming on a different GPU count is allowed only when mathematical batches, parameter order, masks, loss normalization, and optimizer state remain equivalent and checkpoint formats can be safely resharded. Document expected floating-point/nondeterministic differences. Do not claim bitwise reproducibility for distributed/vLLM runs unless tested. A changed scientific config creates a new logical run linked to its parent; a pure resource change may continue with an explicit segment/config record.

## 14. Evaluation, checkpoint selection, and statistics

Evaluate the untrained base model first and the initial adapters if they are not exactly no-op. Confirm the shared base baseline is genuinely the same across A-D. Main evaluation uses greedy decoding (temperature 0), the same tools, same action/token/time budgets, fresh workspaces, and no privileged context. Native benchmark generation scores must remain separate from derived repair scores.

Development learning curves: iteration 0 and every 20 completed iterations on the frozen 34-parent monitoring subset (target 68 repair tasks). Counterfactual development tasks are reserved for diagnostics, not used as extra training data. Final locked evaluations: final in-domain repair tasks (target 200 tasks, 100 parents) and external HumanEval+-derived repairs (one per eligible parent).

Primary endpoint: final common-budget checkpoint for each configuration. Secondary endpoint: checkpoint with the highest parent-macro monitoring-development success; tie-break by earlier iteration. Make selection deterministic and freeze the rule. Do not use final test results to choose checkpoints, rerun settings, mutation types, or seeds. The secondary endpoint may be evaluated once under the predeclared rule; label it as selected by development performance. If compute is tight, prioritize the primary endpoint and report the secondary as unrun rather than quietly dropping selection criteria.

Persist per-task outcomes and public/hidden pass status, final patch/source hash, tokens/turns/test calls, timings, errors, and checkpoint identity. No silent exclusion of hard model failures. Retry true infra errors under the fixed policy; if unresolved, record incomplete evaluation and do not present it as a full success-rate estimate with an altered denominator.

Report task-macro and parent-macro success. For paired differences, bootstrap parent programs (e.g. 2,000 resamples with a fixed analysis seed), keeping all mutations of one parent together and pairing configs on identical tasks. Report confidence intervals, numerator/denominator, and effect size. Optionally report paired discordant counts. These intervals quantify evaluation-task uncertainty, not seed variability. Avoid significance or general superiority claims unsupported by one training seed.

Learning curves must be available against outer iterations, actual student episodes/generated tokens, and cumulative allocated GPU-hours. Store raw points; smoothing is visualization-only and optional. Distinguish train-only compute and end-to-end compute. Compare methods only over common observed budget ranges without extrapolation.

## 15. Mechanism diagnostics: explain success or failure

Required counterfactual update experiment at approximately iterations 0,100,200 (or 0,50,100 for a 100-iteration study). Use B checkpoints as the common student/teacher state by default. Generate/save 16 valid diagnostic training candidates using one frozen diagnostic proposer state and training parents; score the SAME pool with C-style and D-style gradients at each common student snapshot. If candidate construction cannot yield 16 valid tasks within a fixed attempt budget, report the realized pool. Do not use test parents.

For each candidate independently:
1. Restore the same student parameters AND optimizer state.
2. Restore relevant RNG state; freeze its G=4 candidate episodes and detached coefficients.
3. Compute C/D alignment scores with the main reference anchors.
4. Apply exactly one real combined gated student optimizer update with the same LR, AdamW state, clipping, and normalization as training.
5. Measure change in CE on separate canonical diagnostic-development repairs, plus greedy diagnostic success when affordable.
6. Restore parameters and optimizer state before the next candidate. Verify restoration by hashes/checksums where practical.

Reference CE decrease is not enough to validate the reward, because it shares the objective used to construct alignment. The separate diagnostic-development measurements are essential. Report Spearman rank correlations, top-quartile update utility, negative-transfer fraction, zero-RL/nonzero-SD behavior, and realized sample size. If either score is constant, report correlation undefined, not zero. Do not claim 16 candidates is a high-power statistical study.

Additional bounded diagnostics:
- A/B gated versus ungated coefficients/gradient effects on the same frozen trajectories and model snapshot, avoiding curriculum confounding.
- Distributions of raw teacher gaps/gates by turn, action type, success/failure, and public/hidden disagreement.
- RL-versus-SD gradient cosine/norm ratios, sampled to control overhead.
- Proposer validity, duplicates, diversity, task difficulty, and curriculum changes over time.
- Representative repairs and failures selected using a documented sampling rule, including regressions rather than only wins.

Keep diagnostics in separate runs/event categories and compute accounts. They must not update main-run checkpoints or feed back into the main curriculum. If budget prevents a diagnostic, record it as missing rather than substituting fabricated values.

## 16. Required outputs and artifact layout

Provide a stable run directory, for example:
  runs/<study_id>/<config>-seed2026/
    resolved_config.yaml
    provenance.json
    state/health.json
    events/*.jsonl
    manifests/*.json
    tables/{proposals,episodes,iterations,evaluations,diagnostics}.parquet
    trajectories/<bounded-sharded-files>
    checkpoints/<iteration>/
    plots/
    summaries/
    wandb/

This is an example layout; adapt to existing repo conventions without losing the information. Logs must include schemas/units and stable joins. Store large arrays/trajectories in bounded shards with references; avoid millions of tiny files on HPC shared storage. Separate trainer-private oracle data from student-visible exports. Upload policies must not accidentally leak oracle/test files into tool workspaces.

Final analysis command must produce, from actual logged data:
1. summary.csv and machine-readable JSON with base/A/B/C/D primary scores, paired differences/CIs, tokens/turns/time, compute, validity, nonfinite/infra/incomplete flags, checkpoint identity, and sample counts.
2. Raw learning-curve tables and publication-usable PNG/PDF or SVG figures: success vs iterations, success vs allocated GPU-hours, curriculum success-group mix/validity, teacher gap/gate behavior, RL/SD gradient interaction, alignment and zero-gradient distributions, phase-time breakdown.
3. Counterfactual score-versus-actual-improvement scatter/rank plots and a correlation/utility table.
4. Per-task evaluation output and qualitative case-study exports with a documented selection rule.
5. A concise results report distinguishing observations from hypotheses, limitations of single-seed/derived-small-program tasks, missing runs, and implementation deviations. It must not claim novelty or a winner solely from noisy point estimates.
6. Reproduction metadata: commands, resolved configs, code/model/data hashes, runtime versions, hardware, environment/container digest, and resume history.

The pipeline should generate partial summaries from incomplete runs with explicit status; no empty fields pretending to be successful results. Analysis must work from local files without requiring W&B availability. W&B is for live visibility and artifact links, not the only copy of research data.

## 17. Suggested package interfaces and configuration

Use interfaces equivalent to:
- DatasetAdapter -> parent programs/provenance/tests.
- TaskBuilder / MutationValidator -> immutable TaskSpec + private OracleSpec.
- RepairEnvironment -> step/reset/finish with structured observations.
- SandboxBackend -> local container/Apptainer/remote executor with capabilities/preflight.
- PolicyBackend -> generate, score_token_targets, sync_adapter, adapter identity.
- PrivilegedTeacherScorer -> detached target logprobs with explicit token mappings.
- StudentObjective / ProposerObjective -> pure tensor functions with masks and normalizers.
- GradientScorer -> reference/candidate vectors and cosine metadata.
- SelfPlayTrainer -> transactional iteration and counters.
- Evaluator / DiagnosticRunner -> frozen evaluations and isolated update probes.
- RunLogger / CheckpointManager -> local-first observability and safe resume.

Prefer a single package (e.g. src/selfplay_opsd/) with configs/, scripts/, tests/, and docs/. Exact names are flexible. Include schemas for ParentProgram, TaskSpec, OracleSpec, ProposalRecord, EpisodeRecord, TokenTargetMap, IterationRecord, EvaluationRecord, and CheckpointMetadata. Private fields should not be serializable into student requests by accident; use explicit allowlisted student-view construction rather than dumping whole task objects.

Separate method config from hardware and logging config. Validate unknown keys, incompatible settings, path permissions, episode-group sizes, token limits, and world-size-dependent batch feasibility before a long run. Scientific config hash excludes benign resource/logging changes but includes model/data/prompt/objective/budgets/seed settings. Document that boundary.

Provide defaults equivalent to this concise schema (expand with full validation; this is not a claim about literal verl API names):

study:
  id: required_user_value
  config: A|B|C|D
  seed: 2026
  iterations: 200
model:
  name: Qwen/Qwen3-4B-Instruct-2507
  revision: resolved_pinned_sha
  dtype: bfloat16
  lora: {rank: 32, alpha: 64, dropout: 0.0}
batch:
  parents_per_iteration: 4
  proposals_per_parent: 4
  student_episodes_per_task: 4
student:
  optimizer: {lr: 1.0e-5, weight_decay: 0.0}
  grpo_clip: 0.2
  sd_lambda: 0.1
  sd_gate_beta: 1.0
  sd_gap_clip: 5.0
  optimizer_steps_per_iteration: 1
proposer:
  optimizer: {lr: 5.0e-6, weight_decay: 0.0}
  max_new_tokens: 512
sampling:
  train_temperature: 1.0
  top_p: 1.0
  evaluation_greedy: true
episode:
  max_turns: 8
  max_generated_tokens: 4096
  max_turn_generated_tokens: 1024
  max_student_sequence_tokens: 10240
  max_teacher_sequence_tokens: 12288
  max_observation_tokens: 1024
evaluation:
  every_iterations: 20
  primary_checkpoint: final_common_budget
checkpoint:
  every_iterations: 10
hardware:
  profile: dual_gpu_80gb
  gpu_count: explicit_or_scheduler_detected
  microbatch_episodes: 1
  cpu_workers: bounded_by_allocation
logging:
  local_events: true
  wandb_mode: offline
  sample_every_iterations: 20

Config A sets gate=none, proposer_reward=difficulty. B sets gate=sigmoid_gap, reward=difficulty. C sets gate=sigmoid_gap, reward=gradient_alignment, candidate_objective=rl_only. D differs from C only in candidate_objective=rl_plus_gated_sd. Reference objective=canonical_repair_ce for both. Store these resolved settings, not just a single opaque method label.

## 18. Required correctness and integration tests

Focus on real failure risks rather than tests that restate implementation. Use tiny fixtures/models for fast math tests; separate optional GPU, sandbox, data-download, and W&B integration markers. Required tests:

Objectives and gradients:
- Hand-computed GRPO advantages including population std, all-zero/all-one rewards, negative advantages, and per-episode normalization.
- Exact masking of system/user/tool/padding/privilege tokens, and correct causal shifts across at least two tool turns.
- Delta=0 gives zero SD gradient; positive gap increases target-token likelihood, negative gap decreases it; sigmoid gating attenuates negative-gap influence.
- Gate=1 reduces gated formula to ungated formula; lambda=0 makes C/D scoring equal on the same trajectories.
- Frozen teacher/old logprobs/advantages/gates have no gradients; teacher/student use identical sampled target IDs under different prefixes.
- Finite-difference or independent autograd checks for signed-gap and alignment gradients with frozen coefficients.
- Correct cosine sign/bounds, near-zero handling, constant-score diagnostics, and no parameter/optimizer mutation during scoring.
- Candidate RL gradient includes failed/negative-advantage episodes and becomes exactly zero for all-equal groups; combined gradient may remain nonzero.
- Streaming/microbatch results match full-batch loss and gradient within stated tolerance; multi-GPU reduction matches a single-process reference when that profile is advertised.

Data/environment/security:
- No parent/descendant overlap between splits, no oracle content in student-view serialization, frozen mutation reproducibility and provenance.
- Correct source passes, valid mutant fails semantically, inverse repair restores; invalid patches/test edits/API changes/dedup rejected deterministically.
- Public-test modifications/pytest configuration tricks cannot alter fresh final evaluation; path traversal/symlink escape/unsafe imports rejected.
- Generated code cannot read trainer credentials/oracle files or access network/metadata in the actual supported sandbox profile.
- Fresh episode isolation, independent group seeds, observation/output/time limits, invalid action behavior, model versus infrastructure failure classification.

State/observability:
- A/B/C/D complete one transactional iteration on fixture tasks; a tiny model test covers real generation/scoring where feasible.
- Student/proposer adapter isolation and rollout weight-version synchronization.
- Checkpoint restores both optimizers/adapters/counters/RNG/history, interruption cannot replay an update twice, and outputs remain equivalent within documented tolerance.
- Local tables/metrics have stable IDs, units, valid denominators, committed flags, and no secret leakage.
- Offline/disabled W&B path works without credentials/network, online smoke test when credentials exist, consistent custom axes, resume segmentation, and metric export.
- Final-test guard prevents evaluation data use in training/rewards/hyperparameter selection; analysis uses paired parent resampling.

Acceptance is not "pytest passed" alone: distinguish tests that ran, tests skipped for missing resources, and integrations still unverified. A CPU mock cannot prove FSDP/vLLM/sandbox/W&B online readiness.

## 19. CLI, docs, and completion requirements

Provide working CLI commands (names can follow repo conventions) for:
- doctor/preflight: inspect devices/allocation/dependency versions, data caches, sandbox isolation, token budgets, W&B mode, estimated disk/RAM, and launch readiness without training.
- prepare-data: build pinned splits, frozen mutation manifests, canonical references and diagnostic trajectories.
- validate-data: oracle checks, split audit, leakage audit, and coverage/exclusion report.
- smoke: tiny/fixture end-to-end A-D with an explicit small budget.
- pilot: target-hardware calibration and phase-level throughput report.
- train --config A|B|C|D --hardware <profile> --study-id ... [--resume ...].
- evaluate --checkpoint ... --split dev|final_id|external --protocol frozen.
- diagnose: counterfactual update probes/common-data gating diagnostics.
- analyze: aggregate actual logs, tables, plots, CIs, and report.
- wandb-sync/export helpers and Slurm/AWS launch/resume wrappers.

Document exact commands with realistic paths/placeholders, required environment variables, clean environment/container setup, W&B online/offline behavior, Yale storage/cache/signal settings, AWS resource/cost controls, expected artifacts, common failure modes, and how to inspect health during a run. Include a short METHODS.md with the actual implemented equations/normalization and a deviations log. Source these docs from the final implementation, not an abandoned architecture.

Relevant starting references (verify APIs against pinned versions; this specification defines the experiment):
- Qwen model: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
- EvalPlus: https://github.com/evalplus/evalplus
- verl: https://github.com/verl-project/verl and https://verl.readthedocs.io/
- W&B: https://docs.wandb.ai/
- Yale hardware/containers: https://docs.ycrc.yale.edu/clusters/bouchet/ and https://docs.ycrc.yale.edu/clusters-at-yale/guides/containers/
- CompassPlay: arXiv:2609.32228 (background; main CE-alignment recipe differs from our C/D).
- SDAR: arXiv:2605.15155 and https://github.com/ZJU-REAL/SDAR (background; do not silently substitute its printed or coded objective for ours).

At completion, give:
1. What is implemented and the important design decisions/deviations.
2. Tests/integrations actually run, with exact results and unresolved blockers.
3. Exact commands for the next safe smoke/pilot and the four full runs.
4. Measured resource/runtime projections if target hardware was available; otherwise label them unmeasured.
5. W&B configuration/dashboard instructions and the local artifact paths.
6. Whether the code is ready for a short pilot, ready for full runs, or still blocked, with concrete reasons.

Continue until the implementation and locally verifiable checks are complete. Do not present scaffolding as a completed experiment, do not fabricate results, and do not change the scientific objective to accommodate an easier framework API.
