# Task: Recover an Informative Self-Play Experiment

Repository: https://github.com/tatsu432/self-play-opsd-coding-agent

## Objective

The completed Task 13 diagnosis found that the corrected A/B/C self-play experiments have very sparse informative learning signals. The primary goal now is to establish an experimentally sound setup in which we can meaningfully compare gated OPSD and gradient-aligned proposer rewards.

**Do not immediately launch another full A–D experiment. Implement and execute the smallest necessary diagnostic and controlled experiments, make evidence-based decisions, and then launch short online runs when justified.**

## 1. Read the existing evidence

Inspect the latest main branch, particularly:

- `docs/investigations/learning-stagnation-diagnosis.md`
- `docs/plan/13-diagnosis-plan.md`
- `docs/task_completion/13-implemented-learning-stagnation-diagnosis.md`
- `docs/methods.md`
- The existing Task 13 structured proposer and static student implementations.

Use the existing measured findings rather than repeating completed probes.

The main problems are low proposal acceptance, history-dependent duplication, mostly uniform student rewards, proposer advantages that rarely distinguish task utility, and student patch-application failures.

Preserve the previous runs and their scientific identities.

## 2. Improve the proposer representation

Promote the proven structured AST-edit proposal format from an inference-only diagnostic to a supported online training option.

Reuse the existing deterministic transformation machinery, mutation validation, novelty checks, semantic verification, and worker isolation.

Do not replace these components unnecessarily.

Before running large training:

- Compare the structured proposal mode with the existing unified-diff mode on a prospectively declared training-only parent sample.
- Measure novel verified mutations, distinct edit locations/operators, and useful proposals per allocated GPU-hour.
- Measure acceptance with both empty novelty history and the actual checkpoint's stored history.
- Record whether structured mutations create tasks with informative student reward variation.

The structured mutation vocabulary is narrower than unrestricted diffs. Report that limitation and check diversity rather than treating the validity gain alone as proof of improvement.

Prefer the existing 4B model unless the new evidence supports scaling.

## 3. Investigate student editing failures

The corrected final development evaluations were dominated by rejected patches.

Perform a bounded, paired inference-only comparison of the existing student editing protocol against one simpler structured edit operation, such as replacing a specified function body.

Maintain executable-code verification, the same legitimate repair objectives, unchanged test secrecy, and comparable tool/turn budgets.

Determine whether the alternative improves valid tool actions and successful repairs without making tasks trivial or providing the answer.

If it does, introduce it as an explicitly new scientific protocol. Do not retroactively change the old evaluation.

## 4. Repair experimental identifiability

Before changing proposer rewards, quantify the current normalization issue.

For a single valid reward among three invalid proposals, standardized group advantages are essentially independent of the valid reward's magnitude.

Implement a reproducible counterfactual analysis using the same proposer snapshots, generated candidates, and student rollouts for:

- Validity-only reward.
- Difficulty reward.
- RL-gradient alignment.
- RL-plus-gated-OPSD alignment.

Compare raw rewards, normalized advantages, actual proposer gradients, and counterfactual optimizer updates.

If richer rewards still fail to change optimization, run a small explicitly labeled ablation using mean-centered but non-standardized advantages. Do not silently change the primary objective.

Measure reference-gradient stability using training-only reference anchors when feasible.

Do not start full C/D training unless the candidate rewards yield meaningful differences after the complete optimizer pipeline.

## 5. Conduct matched-exposure student comparisons

Using the existing static training-task control, compare:

1. GRPO only.
2. GRPO plus gated OPSD.
3. GRPO plus ungated OPSD with a prespecified matched distillation-gradient strength.

Reset student initialization for each arm.

Use the same frozen training-only tasks, declared task schedule, number of attempted episodes, rollout settings, and evaluation checkpoints. Use at least two additional training seeds if feasible.

**Do not match only on actual optimizer step count.** The previous comparison used 424 episodes for GRPO and 160 for gated OPSD at 20 steps.

Use common episode-count checkpoints (for example, 80, 160, and possibly 320), log actual optimizer steps separately, and compare against allocated GPU-hours as well.

Use training-only measurements to select the ungated coefficient before testing. Do not select coefficients based on final held-out outcomes.

## 6. Check numerical correctness without excessive blockers

Investigate recorded rollout-versus-rescoring mismatches, especially whether rare large differences affect policy-ratio clipping, gradients, or optimizer updates.

Retain true behavior-policy log probabilities. Do not silently replace them.

Use focused correctness tests and an actual GPU check where required. Preserve the existing verified worker isolation, checkpoints, and dataset boundaries.

Do not introduce new arbitrary process-count, cgroup, infrastructure, or formal readiness requirements unless they address an actual demonstrated problem.

## 7. Execute bounded online experiments

After the above evidence establishes an informative setup, run short controlled self-play experiments using a new immutable study identity.

Start with bounded 20–50 iteration runs and extend only if the experiments are producing useful, interpretable information. Preserve the original A–D contrast definitions where possible, and clearly label any changes.

Use matched initial model states, parent schedules, experimental budgets, and evaluation rules.

Include the following W&B metrics:

- Proposal validity, unique novelty, rejection reasons, duplicate rate.
- Parent groups with multiple accepted candidates.
- Groups with distinct candidate rewards.
- Advantage difference versus validity-only reward.
- Student mixed-reward fraction.
- GRPO and OPSD gradient norms and interaction.
- Teacher gaps, gates, and numerical mismatch diagnostics.
- Student repair success versus episodes, tokens, iterations and allocated GPU-hours.
- Curriculum diversity and held-out development performance.

Keep the final held-out set closed during intervention selection.

## 8. Research and execution standards

Do not stop after writing a plan or implementing untested interfaces.

Implement the changes, test them, run the bounded Yale experiments using existing authorized resources, inspect results, and document conclusions.

Before new submissions, reconcile existing Slurm jobs and ledgers to avoid duplicate runs. Retain failed-attempt accounting and checkpoint provenance.

Prioritize efficient and interpretable experiments over extensive infrastructure changes. Stop scientifically uninformative runs instead of completing a nominal iteration budget.

Commit code and concise experimental reports to GitHub, preserving private raw artifacts and hidden oracle information outside the repository.

## Expected deliverable

Produce a concise research decision report answering:

1. Did the new proposal format improve genuinely novel, verified task yield?
2. Did it increase student mixed groups and proposer reward discrimination?
3. Does gated OPSD outperform GRPO at matched episode exposure and compute?
4. Does gradient alignment meaningfully change proposer updates?
5. Which A–D runs are now justified, and what evidence supports that decision?

Include reproducible configurations, experimental tables, learning curves, uncertainty estimates, GPU-hour accounting, and a recommendation for the next experiment.

The goal is a defensible research result, including a well-explained negative result if necessary—not merely four completed training jobs.