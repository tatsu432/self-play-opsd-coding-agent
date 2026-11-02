# Stabilize the existing training system, then complete main experiments A and B

Repository: https://github.com/tatsu432/self-play-opsd-coding-agent

## 1. Mission and authorization

Work in the existing repository and Yale environment. Find and fix the causes of unstable or uninformative training, select a practical configuration using bounded training-only tests, and then RUN, monitor, recover, evaluate, and analyze main experiments A and B. A and B are required deliverables. C and D are stretch goals after A/B completion.

This is an execution task, not a request for another readiness report, unimplemented plan, or list of possible experiments. Reuse the current implementation and valid previous evidence. Do not rebuild the project or repeat A-D from scratch. Fresh paired runs are appropriate when a necessary change makes old checkpoints scientifically incompatible; rebuilding the infrastructure is not.

I authorize the code changes, commits, CPU/GPU jobs, necessary continuations, and evaluation needed to complete this scope on the existing authorized Yale account. Do not repeatedly ask whether to launch or continue. Use actual available allocations and account limits. Do not use another course account or AWS merely because it exists. Ask only for an external action you genuinely cannot perform, such as completing MFA, or a new constraint outside this scope. Do not bypass authentication or platform restrictions.

This instruction supersedes historical repository statements that require new approval at each diagnostic checkpoint, defer A/B for C/D alignment evidence, or stop after an informative 20-iteration demonstration. Preserve real containment, train/test separation, checkpoint integrity, immutable identities, and no-duplicate submission safeguards.

A reliable negative A/B result is acceptable. Manufacturing an advantage for B, repeatedly restarting an unfavorable seed, and substituting a static control for main online A/B are not acceptable.

## 2. Inspect once and reuse what is already known

Start by reading the current AGENTS.md, README, docs/methods.md, docs/operations.md, the authoritative scientific specification, and these investigations:
- docs/investigations/task14-research-decision.md
- docs/investigations/task14-informative-selfplay.md
- docs/investigations/learning-stagnation-diagnosis.md
- Relevant Task 12-14 completion and launch records.

Inspect current HEAD, worktrees, dirty files, actual private results, scheduler state, durable ledgers, and checkpoints. Never assume an old AGENTS.md job-status paragraph is live. Preserve unrelated work. Use an isolated branch/worktree if needed.

The reviewed public baseline was main commit 1c8e1903effa9a63b07bf76de9eed626b59e8094. Verify newer changes rather than resetting to this commit. Historical findings to verify/reuse, not regenerate:
- Corrected old A/B completed 200 iterations. B accepted 162/3,200 proposals and produced only 19 mixed student groups. Nominal iteration counts exaggerated useful RL exposure.
- Task 14 structured online A/B completed 20 iterations, both ending at 12/41 development repairs. They produced nine and twelve mixed groups.
- Matched static controls used only 32 tasks and 160 episodes per arm/seed. GRPO took 8-10 optimizer updates and learned from 0/41 to 11-13/41. Therefore 4B learnability is already demonstrated; saturation and superiority are not.
- Structured AST proposals improved valid/novel yield, but did not establish a general improvement in mixed-group efficiency.
- Function-body replacement increased edit validity while reducing verified repairs. Do not adopt it again based on syntax validity.
- Existing numerical counterfactuals changed PPO denominators AND distillation gaps together. They did not causally isolate those mechanisms.
- Singleton-valid proposer groups largely erase valid reward magnitude under within-parent standardization. This is structural, not something another identical probe will fix.
- C/D gradient separation is not a prerequisite for A/B.

Create a concise status/decision record with verified artifact paths and the unresolved questions. Do not create another large historical narrative or run the same completed study under a new label.

## 3. Preserve the intended A/B comparison

Keep these primary methods:
A: difficulty-reward online proposer + student GRPO and UNGATED signed-gap distillation.
B: difficulty-reward online proposer + student GRPO and SIGMOID-GATED signed-gap distillation.

Keep the current 4B model by default, including its pinned revision and LoRA setup. Do not start a model-size sweep. Larger models are a fallback only after evidence localizes a capability limitation that simpler protocol/data fixes cannot address.

The current auxiliary objective is a deliberate SDAR-inspired extension, not the exact printed SDAR loss. Preserve its signed-gap semantics:
- delta = detached privileged-teacher log probability minus detached frozen unprivileged-student log probability for the SAME sampled target token.
- A's coefficient: clip(delta, -gap_clip, gap_clip).
- B's coefficient: sigmoid(beta * delta) * clip(delta, -gap_clip, gap_clip).
- Auxiliary loss: negative detached coefficient times current student log probability, with the existing episode/task averaging.
- Keep actual student histories, target IDs, loss masks, EOS policy, clipping, and teacher detachment correct.

Do not silently replace this with positive weighted SFT, absolute/positive-only gaps, full-vocabulary KL, a stronger external teacher, success-only distillation, or a new paper's objective. A negative gap must remain negative. Such changes answer different questions and are not needed merely because A/B fails to beat GRPO.

Use the SAME primary lambda in A and B. Default to the existing 0.1 if stable. If training-only evidence supports a smaller value, select ONE common value and freeze it for both. The existing matched-strength ungated control is supporting evidence, not the primary A definition. Explain that the primary contrast includes both selectivity and effective strength; do not claim it isolates selectivity alone.

A/B must have identical common initialization within seed, common non-method hyperparameters, tools, data boundaries, verification, parent-sampling policy, novelty/replay policy, budgets, and evaluation. Their online curricula may diverge as their models learn; that is part of the complete-system comparison. A shared schedule of parents is not a shared schedule of realized tasks.

Necessary common protocol/curriculum corrections are authorized, but version and describe them. Do not mutate historical frozen runs. Resume an old run only if compatibility actually holds. If warm-starting is necessary, use the same training-only warm-start procedure and common checkpoint for each paired A/B seed, disclose it, and never initialize each arm from its own favorable historical checkpoint.

## 4. Diagnose causes using interventions, not another open-ended checklist

Use existing saved rollouts wherever the question does not require fresh sampling. Each new diagnostic must state: hypothesis, evidence missing, intervention, measured outcome, and the concrete configuration decision it will change. If it cannot change a decision, defer it.

Cover the following failure cases with targeted tests; do not interpret "cover" as a Cartesian sweep of all models, losses, tools, and hyperparameters.

A. Probability/backend correctness
- Compare identical token IDs and prefixes at identical adapter versions through rollout, unprivileged scorer, privileged scorer, and gradient-bearing scoring.
- Check tokenizer/chat-template/EOS handling, selected-token log-probabilities, causal positions, precision, attention/LoRA configuration, adapter reloads, eval/train behavior, and snapshot freshness. Only pursue differences implicated by evidence.
- Persist separate fields for actual behavior logps, frozen same-backend unprivileged logps, and privileged teacher logps.
- Diagnose PPO denominator changes while holding SD coefficients fixed, and SD-gap changes while holding PPO denominators fixed. Include an identical-operation repeat to estimate numerical repeat variation.
- Run a no-privilege control: identical teacher/student conditioning should give a zero signed gap within measured scorer repeat tolerance. Quantify the spurious auxiliary update produced by cross-backend discrepancy.
- Prefer computing the privilege contrast from teacher and frozen unprivileged scores on the same scoring backend, while retaining TRUE behavior probabilities for importance ratios. If adopted, explicitly version the corrected gap definition and sampling approximation.
- Do not overwrite behavior probabilities to force ratios to one. If rollout/training policy mismatch needs a correction, implement a principled, documented estimator or use a measured consistent-backend fallback. Do not invent a denominator substitution.
- Select precision/kernel/scoring fixes using effect on loss, updates, and repair behavior; do not demand bitwise equality, an arbitrary maximum logp threshold, or a universal cosine threshold.

B. Loss, gradient, and optimizer correctness
- Check per-task advantages, all-equal rewards, failed/successful episodes, negative/zero/positive gaps, unequal episode lengths, masks, EOS, lambda=0, gate=1, and microbatch accumulation against a small independent calculation.
- Verify one intended optimizer step per logical update, correct trainable adapter, detached teacher, no stale gradients, no double division, no accidental batch-size dependence, finite gradients, correct clipping, adapter export/reload, and exact checkpoint restoration.
- In the current signed-gap objective, delta=0 must give zero SD signal. All-equal episode rewards must give zero current GRPO signal, while SD can remain nonzero.
- Track current gradients separately from Adam momentum. Do not describe zero current RL gradient as zero parameter displacement.
- Only investigate Adam state or BF16 update rounding further if measurements show a material problem.

C. Task supply, repetition, and difficulty
- Separate semantic validity, representation errors, historical duplication, task admission, and student outcomes in one joined funnel.
- Distinguish novelty under an accumulated history from the model's ability to produce valid bugs.
- Check whether the proposer is penalized for repeats without being told which actions remain admissible. Fix that information mismatch rather than repeatedly sampling a policy into an invisible historical blacklist.
- Check mutation-site/operator/parent coverage and extreme task difficulty, not just proposal acceptance.
- A 0/4 or 4/4 outcome is uncertain evidence about difficulty, not proof of population success probability 0 or 1.
- Do not select mixed groups using a rollout group and then treat the same selected rollouts as an unbiased fresh training batch. Use independent screening/past outcomes and fresh on-policy updates.

D. Student protocol and teacher usefulness
- Inspect representative failed trajectories to distinguish rejected edits, ineffective accepted edits, premature finish, budget exhaustion, and semantic failures.
- Reuse the failed body-replacement evidence. Keep unified diff unless a small prospective comparison supports a better option.
- If edit serialization remains dominant, test at most one simpler alternative, such as exact uniquely matched search-and-replace with deterministic diff construction and unchanged semantic verification.
- Judge it by verified repair and useful learning per allocated GPU-hour, not action-parse rate.
- Check that privilege actually improves verified repairs or action discrimination on training-only tasks. A negative mean gap alone does not prove a bad teacher.
- Use a tiny canonical-repair overfit/positive-control test only if the relevant path is unverified or newly broken. Existing GRPO learning means another general "can 4B learn?" investigation is unnecessary.
- Do not introduce protocol SFT unless evidence requires it; if used, keep it identical across arms and report that it changes initialization.

E. Signal, scaling, and statistical interpretation
- Measure mixed groups, advantage-bearing tokens, actual updates, gradient interactions, and fresh task diversity per allocated GPU-hour.
- Check whether varying valid-task counts or a tiny batch makes the effective step noisy. Prefer accumulated microbatches over larger GPU allocations when that gives an affordable stable batch.
- Treat the small 41-task development set as noisy. Do not stop or restart because of a one-task fluctuation, or require B>A during stabilization.
- Do not diagnose a capacity limit solely from a small diagnostic run or a plateau under starvation.

For each alleged cause, distinguish confirmed defect, intervention-supported cause, contributing bottleneck, and unresolved hypothesis. Fixes should improve the relevant end-to-end mechanism. Passing another parser test is not sufficient evidence that learning is fixed.

## 5. Choose a practical recovery route and then stop searching

My preferred direction is to reuse structured proposals, remove history-dependent starvation, and supply a broad verified task stream. Implement the least invasive version that the measured evidence supports.

Useful implementation options, in preferred order:
1. Reuse the structured AST mutator and deterministic rendering. Make admissible options/history visible or constrained. Cache semantic verification and canonical repairs. Sample diverse parents and distinct mutations within a batch.
2. Build a bounded verified training-only mutation catalog/reservoir using the current legal operations. It supports difficulty estimates, controlled exploration, and predeclared replay rather than an ever-growing blanket rejection of previously useful tasks.
3. Use a bounded refill policy plus a disclosed reservoir fallback if online generation still leaves the student idle. Log which tasks are online-generated versus fallback/replay and include all proposal/screening cost.
4. Change the student edit interface only if the targeted repair comparison supports it.

A/B main training must still include an actively trained difficulty-reward proposer and online student updates. A static catalog-only student run is a control, not completion of A/B. If the proposer selects from a finite catalog, explicitly identify this as constrained online task selection, preserve its policy/reward/update semantics, and distinguish it from free-form generative self-play.

Maintain true sampling probabilities:
- Constrained sampling, mixtures, resampling, and without-replacement selection change the behavior distribution. Store and train against the actual distribution.
- Deterministic fallback mutations are not LLM samples. Do not fabricate their token logps or train the proposer on them as if sampled.
- Preserve complete proposal groups and snapshot identities across refill/buffering. Do not assign unavailable quality rewards to unscored candidates or use stale rollout groups.
- Separate validity from novelty in reporting. Replay must be explicit, not silently counted as new data.
- If the main run is overwhelmingly fallback/static tasks, do not present its outcome as evidence for an effective generative proposer. Fix the supply policy or state the actual constrained scope.

A bounded catalog is useful infrastructure, not permission to replace the research objective. Likewise, proposer reward normalization is a known issue, but A/B does not require expensive C/D optimizer-discrimination studies. Change normalization only if needed for the chosen A/B supply route, apply it commonly, and disclose it. Defer alignment-specific redesign.

Use training-only evidence to select:
- Representation/supply route.
- One common lambda and, only if necessary, one common learning rate.
- Task/replay sampling rules and accumulation size.
- Four versus eight student rollouts only if a timed comparison can change the choice. Default to four; a higher mixed fraction alone does not justify doubling cost.

Avoid a broad sweep. Default to at most two substantive stabilization rounds and a planning allowance of about eight incremental H200-equivalent GPU-hours for new exploratory work, excluding reused evidence. This is a prioritization target, not a new approval gate or permission to ignore a real correctness defect. If more diagnosis is necessary, identify the reproduced defect and the exact repair/test; do not repeat generic probes.

After selecting a correct runnable route, freeze it and start A/B. Do not open a third general redesign cycle because results are ambiguous. If no route is scientifically valid, finish the localized repair rather than launching knowingly invalid runs or claiming success.

## 6. Main A/B execution: mandatory, adequately trained, and resumable

Prepare one versioned successor study for the necessary changes. Preserve the original implementations and historical evidence. Reuse the existing CLI/controller, trainer, worker, checkpointing, and W&B integration.

Important existing launcher limitations:
- scripts/task14_online.py currently enforces exactly 20 iterations.
- scripts/task14_static.py stops at 160 episodes and uses a static pool.
Neither is the new main experiment merely by renaming its output directory. Extend or parameterize the supported route, and verify that the actual remote command, ledger, resolved config, and resume path carry the intended exposure target and stop rules.

Default main plan unless measured capacity/runtime or task availability justifies a concrete adjustment BEFORE freezing:
- Same pinned 4B model.
- A and B for paired seeds 2026, 2027, and 2028.
- Target 4,096 fresh student training episodes per arm per seed.
- Four independent episodes per task by default.
- Accumulate toward roughly four valid tasks per student update when practical; implement this without corrupting proposer groups or snapshots.
- Evaluate at baseline and approximately 256, 1,024, 2,048, and 4,096 cumulative training episodes.
- Start the first A/B pair as soon as common stabilization is complete. Continue the first pair and subsequent seeds through the predeclared endpoint, subject to real failures.
- Do not run C/D instead of completing these six A/B arms.

The 4,096 target is a planning default, not a guarantee of sufficient statistical power. Use measured time, memory, task supply, and observed effective exposure to choose the final common target. If adjusting it, publish the calculation and freeze the same rule for both arms before looking for a favorable method difference. Do not shrink it to another 20-iteration/160-episode probe for convenience.

Use episode-based progress rather than nominal proposer iterations. Count started training episodes consistently, including unsuccessful/malformed model attempts; separate infrastructure failures and screening episodes. Finish complete rollout groups. Implement budget-aware batching; if a small final overshoot is unavoidable, predeclare its bound, report exact exposure, and compare common supported exposure checkpoints. Do not stall over perfectly identical counts.

Keep a bounded total proposal-attempt/compute limit and a fallback policy so a valid-episode target cannot cause infinite refilling. These are part of the frozen study. If supply deteriorates, execute the predeclared response and log it; do not silently change curriculum rules differently after inspecting A/B outcomes.

Measure and report actual updates and mixed groups. A sensible design aims for order-hundreds of useful training groups rather than the historical tens, but no fixed mixed-group count or positive development gain is an endless prerequisite. If realized information is lower, explain it and complete/report the frozen experiment.

A/B should share base initialization and common parent schedules within seed, while each has its own live student/proposer and on-policy trajectories. Do not force identical sampled tokens across different policies or replay one arm's behavior as the other's on-policy data.

Use one H200 per run by default if it remains the measured efficient fit. Use up to the entitled concurrency, not assumed independent account capacity. Batch generation/scoring only where verified, overlap isolated tools where useful, avoid GPU work on login nodes, and profile actual phase costs. Add GPUs only for measured benefit.

## 7. Monitoring, failure handling, and continuation

Keep local scientific events/checkpoints authoritative and W&B useful for live observation. Log:
- Cumulative attempted training episodes, sampled tokens, actual student/proposer updates, and allocated GPU-hours.
- Validity funnel, duplicates/replay, proposal/task source, parent/site/operator diversity, task ages.
- All-failure/mixed/all-success groups and useful groups per GPU-hour.
- Repair/patch success, action errors, episode lengths, terminal reasons, public/final verification disagreement.
- RL and lambda-scaled SD norms/interactions at a sampled frequency; signed coefficients, positive/negative gaps, clipping.
- Behavior/scorer mismatch summaries and effect-bearing clipping; actual update norms where cheap.
- Development repair rate, paired task outcomes, checkpoint and resume events, throughput, memory, and phase costs.

Avoid full gradient decompositions every iteration if they dominate runtime. Keep expensive numerical audits on sampled/saved batches. Skip teacher scoring for lambda-zero controls where unnecessary. Use cached deterministic verification without bypassing runtime task isolation.

Reconcile Slurm and durable ledgers before submissions/retries. Resume verified complete checkpoints after supported transient failures; keep full optimizer/RNG/curriculum/sampler/replay-buffer/exposure state. Exercise one dependency-relevant interruption/resume check for changed state, reusing historical infrastructure acceptance. Do not rerun unrelated container, containment, distributed, or D-pilot readiness checks.

Preserve completed logical work after time limits/preemption. Do not repeatedly reset training to base. Diagnose OOM/config/code defects before retrying. If a code/scientific correction invalidates a running study, checkpoint/retain evidence, fix the defect, and create only the necessary successor; do not silently mix protocols.

A weak score, high A/B update cosine, temporarily sparse rewards, or B failing to outperform A is not an infrastructure failure and does not justify favorable-seed retries. Genuine nonfinite optimization, broken sampling/logps, oracle leakage, corrupt state, or a frozen supply-policy failure requires repair.

W&B upload failures should not terminate valid local training. Reconcile cloud history after writers exit. Do not let telemetry maintenance consume the main experiment.

## 8. Evaluation and required final outputs

Before main training, freeze:
- Primary checkpoint selection: final common exposure-budget endpoint.
- Monitoring-development policy and any clearly secondary checkpoint rule.
- Metrics, evaluation splits, paired analysis, seed aggregation, and stopping/budget policy.
- Task supply/fallback policies and exact method definitions.

Use existing development splits for monitoring. Keep final held-out data closed during debugging and configuration selection. I authorize final held-out evaluation of the frozen A/B endpoints once the main runs and selection rules are fixed; do not use final outcomes to choose a new configuration. Do not repeatedly evaluate the test set while tuning.

Report family/parent-clustered paired task uncertainty and training-seed variation separately. Three seeds remain limited. Report exact task counts and integer successes; do not claim precise small improvements from 41 development tasks. An expanded evaluation is optional only if the existing data boundaries support it without contaminating the final set. Distinguish repair-derived benchmarks from official generation scores.

Deliver:
1. Implemented, tested changes and one concise causal account: which failure mechanisms were confirmed, which interventions helped, and what remains uncertain.
2. Frozen resolved A/B configs, code/data/model identities, run/seed/checkpoint manifests, and reproduction commands.
3. Completed main A/B training, or an exact independently unavoidable external blocker with preserved recoverable state. "Controller implemented" or "ready to launch" is not completion.
4. Learning curves versus episodes, generated tokens, actual updates/mixed groups, and allocated compute; supply/format/numerical health curves.
5. Final paired A/B evaluation across seeds, uncertainty, cost, and explicit scope/limitations.
6. Checksum-verified resumable checkpoints and cleaned-up/reconciled job/accounting state. Keep private source/oracles/trajectories and secrets out of public reports.
7. A short status file with the next concrete action, so an interruption cannot restart the debugging cycle.

Update AGENTS.md with a short CURRENT section distinguishing active instructions from historical logs. Preserve historical records, but remove ambiguity about A/B priority, automatic continuation, and obsolete readiness gates. Add focused regression tests for changed scientific behavior and complete repository-required checks; do not build a new generalized experiment framework.

C/D may proceed only after A/B outputs exist and only if their extra signal can be studied meaningfully. Their absence must not delay or relabel A/B completion.

## 9. How to work and when to finish

Do the work continuously through implementation, validation, submission, monitoring, continuation, and analysis. Give concise updates with evidence and next actions, rather than asking to continue at each checkpoint. If a launcher capability is missing, implement/test the narrow capability and use it.

A long-running scheduler job may exceed one conversational turn. Never claim to keep monitoring after you have stopped unless an actual supported watcher/job is running. Persist state and provide a precise resume command when an external interruption prevents continued work. Authentication interruptions require the specific login action, not a new scientific plan.

Do not promise that A/B will win, or call a robust negative result a software failure. The acceptance criterion is: the intended methods received correct, meaningful, adequately accounted training; the A/B comparison was completed; and the result is interpretable.

Start now with current-state reconciliation, then make the minimum evidence-supported repairs and carry A/B to completion.

