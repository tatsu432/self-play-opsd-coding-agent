# Task 16: generalization, checkpoints, and the next experiment

**Recovered trajectory analysis identifies patch rejection as a dominant observed obstacle on HumanEval+-derived repairs.** Across the six trained runs, only 0–6 patches per run were accepted on 116 external tasks, while 628–811 patches per run were rejected. On MBPP+-derived repairs, 93–111 patches were accepted and 262–364 rejected per 115-task run. A task can contain several attempts. These are observed tool outcomes, not a causal intervention.

All 1,617 final/base trajectories and all 1,447 committed training/proposal trajectory pairs are now available and analyzed. The original [local-only analysis](task16-initial-local-analysis.md) is retained as historical evidence. The final checkpoint remains the monitoring-development selection in all six arms; no additional GPU evaluation is justified. A/B trajectories evolve independently and equal nominal OPSD coefficients do not imply equal effective distillation strength. No Task 15 full-run matched GRPO-only control exists.

This retrospective analysis uses frozen study `repair-selfplay-2026-task15-ab-v2`, scientific execution `0ac77cde4bc5f70ba2b3eb9e457749e3106a2bc5`. No training, held-out rerun, GPU job or W&B modification was performed.

## Dataset and evaluation audit

All 306 frozen reference/development/final tasks regenerate exactly with their recorded deterministic mutation seeds. The parent/family audit passes; parents were assigned before mutation generation. The reserved training population has 186 parents, of which 127 have eligible structured mutation sites after the frozen exclusion. Admitted reference, monitoring, diagnostic, final ID and external task counts are 14, 41, 20, 115 and 116; their covered parent counts are 14, 23, 10, 60 and 116. Reservations and actually admitted tasks are different quantities.

Final splits share the four-action interface, frozen greedy decoding, eight-turn and common token budgets. Both use `fixed_mutation`; online training uses structured AST proposals. The common arithmetic/comparison/constant vocabulary does not make their task distributions equivalent. The matching historical executed oracle-validation report covers all 306 tasks, with correct originals passing and mutants having semantic failures; this report was reused, not rerun. Current frozen source hashes and task-file checksums are verified locally.

Parent-macro means avoid counting multiple mutations from one parent as independent:

| Feature | MBPP+ final ID | HumanEval+ external |
|---|---:|---:|
| Source lines, including docstrings | 5.00 | 17.63 |
| Source lines after removing docstrings | 5.00 | 7.25 |
| Source tokens (frozen tokenizer) | 48.45 | 181.74 |
| Specification tokens (frozen tokenizer) | 19.97 | 127.31 |
| AST nodes | 48.78 | 66.26 |
| Branch/loop/comprehension nodes | 1.33 | 2.22 |
| Public cases | 3.12 | 2.98 |
| Hidden cases | 104.22 | 764.91 |
| Specification characters | 92.17 | 429.05 |

Much of the raw line-count difference is docstring text: HumanEval+ has a mean 381.7 docstring characters, while these MBPP+ programs have none. Executable structure still differs, but raw lines alone would overstate it. Public-example counts are similar; this contradicts a simple "far fewer examples" explanation, without establishing equal usefulness. Hidden-case counts differ substantially and are not a measure of independent semantic coverage or intrinsic difficulty.

Fresh isolated final verifications of source-hash-identical initial mutants recover initial public/hidden behavior without executing code again. They cover all 116 external tasks: 96 fail public tests, 114 fail hidden tests, and 20 pass public tests while failing hidden tests. Of 108 recoverable ID mutants, 97 fail public and 106 fail hidden tests. The seven remaining ID tasks and reference/development mutant detection profiles require additional isolated evidence. Original sources and cases remain private.

## Final outcomes and failure evidence

The common base scores 7/115 ID and 2/116 external. Trained scores reproduce Task 15 exactly. Table entries count final source changes and public test actions; source retention does not exclude transient edits later undone.

| Model | ID repairs /115 | ID source changed /115 | ID test calls | External repairs /116 | External source changed /116 | External test calls |
|---|---:|---:|---:|---:|---:|---:|
| Base | 7 | 7 | 6 | 2 | 2 | 6 |
| A 2026 | 61 | 68 | 96 | 2 | 3 | 1 |
| B 2026 | 70 | 76 | 91 | 3 | 3 | 3 |
| A 2027 | 57 | 65 | 100 | 1 | 1 | 4 |
| B 2027 | 68 | 79 | 99 | 0 | 0 | 0 |
| A 2028 | 64 | 70 | 103 | 1 | 1 | 1 |
| B 2028 | 71 | 81 | 94 | 0 | 0 | 0 |

Trained ID episodes average 5.83–6.59 turns and 460–815 generated tokens. External episodes average 7.41–8.00 turns and 1,026–1,851 tokens. Thus weak transfer coincides with longer, less testing-oriented episodes, not fewer generated tokens. Turn/token/context limits are distinguished in the private overlapping diagnostics; they must not all be called context exhaustion.

The terminal taxonomy prioritizes success, confirmed infrastructure failure, public-pass/hidden-fail, recorded budget exhaustion, no-edit patch rejection, no-edit parsing failure, other no-edit failure, edited semantic/runtime failure, then unknown. It describes terminal evidence, not a causal root cause. All observed action errors remain separate flags. A failed `finish` is not automatically labeled premature. Public-pass/hidden-fail can occur on the untouched initial mutant and does not imply that a valid patch was applied.

| Model | ID: success / public-pass-hidden-fail / budget / other | External: success / public-pass-hidden-fail / budget / other |
|---|---|---|
| Base | 7 / 11 / 48 / 49 | 2 / 20 / 42 / 52 |
| A 2026 | 61 / 10 / 44 / 0 | 2 / 19 / 95 / 0 |
| B 2026 | 70 / 10 / 29 / 6 | 3 / 20 / 79 / 14 |
| A 2027 | 57 / 9 / 44 / 5 | 1 / 20 / 71 / 24 |
| B 2027 | 68 / 7 / 16 / 24 | 0 / 20 / 32 / 64 |
| A 2028 | 64 / 9 / 42 / 0 | 1 / 20 / 95 / 0 |
| B 2028 | 71 / 8 / 11 / 25 | 0 / 20 / 18 / 78 |

There are no recorded final infrastructure failures and no denominator exclusions. All trained evaluation and training trajectories are now available. Final-turn tool-error subtypes can still be unavailable because the observation is not stored in a subsequent history. Base trajectories show: ID has seven valid patch applications versus 331 rejected applications and 172 parsing failures; external has two versus 408 and 166. These base measurements establish a substantial interface bottleneck for the base; they do not establish its exact trained-model contribution. The updated failure figure uses the recovered action categories; its terminal-category precedence means a budget outcome can coexist with rejected patches.

### Recovered patch outcomes

| Arm | ID accepted / rejected patches | External accepted / rejected patches | External tasks with a rejected patch /116 | External context / malformed-hunk / unknown rejection details |
|---|---:|---:|---:|---:|
| A 2026 | 102 / 267 | 3 / 628 | 96 | 424 / 118 / 86 |
| B 2026 | 93 / 337 | 3 / 745 | 114 | 508 / 149 / 88 |
| A 2027 | 111 / 327 | 6 / 747 | 114 | 350 / 320 / 77 |
| B 2027 | 100 / 279 | 0 / 698 | 115 | 526 / 136 / 36 |
| A 2028 | 111 / 364 | 1 / 811 | 116 | 258 / 438 / 115 |
| B 2028 | 96 / 262 | 0 / 672 | 111 | 603 / 50 / 19 |

Error subtypes are inferred from saved tool observations using recorded error text; category labels are descriptive and mutually exclusive. The final turn's error observation is not always retained, so unknown is not zero. Syntax and path categories are not a complete audit of all possible parsing failures. Many episodes have both budget exhaustion and rejected patches: terminal precedence does not measure the total frequency of patch failures. Public-pass/hidden-fail can reflect the original unedited mutant.

Within-seed A/B ID overlaps (both / only A / only B / neither) are 54/7/16/38, 43/14/25/33, and 54/10/17/34. A retains all seven base ID successes in every seed. B retains seven, six and six, adding 63, 62 and 65 previously unsolved tasks. On external tasks, A retains one, zero and zero of the base's two successes and gains one new task per seed; B retains one, zero and zero and gains two, zero and zero. This is task-level regression evidence on very few successes, not proof of broad catastrophic forgetting.

Paired ID parent-macro B-minus-A effects and family-bootstrap 95% intervals (2,000 replicates, deterministic seed 2026) are:

| Seed | Difference (percentage points) | Family-bootstrap interval |
|---|---:|---:|
| 2026 | 8.33 | [-0.81, 17.86] |
| 2027 | 8.33 | [-5.30, 22.31] |
| 2028 | 5.83 | [-4.31, 16.67] |

The separate three-seed mean is +7.50 percentage points, with a t-based 95% interval [+3.91, +11.09]. Three seeds provide weak support for the interval's population assumptions. Family/parent uncertainty and seed uncertainty answer different questions. External results remain near the base floor: the paired seed mean B-minus-A is −0.29 percentage points, with a separate seed-t 95% interval [−2.76, +2.19]. Point estimates cannot identify an intrinsic benchmark difficulty or a causal gating effect.

Descriptive mutation/operator, public-case, line-count and structural-complexity strata are exported privately. Successes remain too sparse externally for a credible unregularized multivariable explanation. Post-outcome action variables are analyzed as behavioral diagnostics, not causal pre-task predictors.

## Development selection and overtraining

Exactly five evaluations per arm are observed: baseline and the four actual milestone crossings, including 260/1,028/2,052/4,100 episode overshoots where present. The monitoring-only selector maximizes parent-macro accuracy across 23 parents and selects the earliest measured checkpoint on ties. It selects the final checkpoint in every arm:

| Arm | Iteration | Episodes | Development task successes /41 | Parent-macro accuracy | H200-hours through evaluation | Bootstrap selection of final |
|---|---:|---:|---:|---:|---:|---:|
| A 2026 | 241 | 4096 | 15 | 43.48% | 25.943 | 48.25% |
| B 2026 | 245 | 4096 | 19 | 52.17% | 24.990 | 96.40% |
| A 2027 | 239 | 4096 | 12 | 36.96% | 21.606 | 43.85% |
| B 2027 | 242 | 4096 | 16 | 43.48% | 24.929 | 88.70% |
| A 2028 | 244 | 4096 | 17 | 47.83% | 24.652 | 99.05% |
| B 2028 | 236 | 4100 | 14 | 39.13% | 23.555 | 76.95% |

The selected-minus-final effect is exactly zero on these observations because they identify the same checkpoint. B's development-selected endpoint exceeds A in seeds 2026 and 2027, but A is higher in 2028. There is no observed terminal development decline in any arm. Earlier dips occur (for example B-2028 falls from 8/23 parent-equivalent successes at 1,028 episodes to 6.5/23 at 2,048 before ending at 9/23); they do not establish persistent overtraining.

Joint family resampling reruns the selector across all checkpoints. Final selection is much less stable for A-2026/A-2027 than for several other arms; the 41 tasks are not 41 independent parents. Out-of-bootstrap selected-minus-final mean differences range from −0.35 to −7.50 percentage points. These are selection-stability diagnostics with small, variable omitted-family sets, not independent holdout estimates or bias-free confidence intervals. Searching a few milestones cannot establish a globally optimal checkpoint. No earlier-checkpoint GPU evaluation is warranted by the specified rule. Current checkpoint files remain on Yale and their historical checksum verification is documented in Task 15; fresh checkpoint checksum verification was not repeated.

## Sample and compute efficiency

Training allocation costs reconcile to 145.723889 H200-hours, including interrupted segments and replayed work. Queue time is excluded; initialization and evaluation within allocations are included. A/B costs by seed are 25.951/24.998, 21.614/24.938 and 24.660/23.563 hours. B is cheaper at the final exposure in two seeds and more expensive in one. Final generated training tokens A/B are 2,112,165/2,066,597; 1,692,242/1,973,718; and 1,941,426/1,957,017.

The following is a checkpoint-availability comparison: each arm may choose its best *measured development checkpoint* within the same budget. It is not an evaluation of each evolving model at that exact wall-clock time, and no interpolation or extrapolation is used.

| Seed | Largest common measured hour budget | A available development parent accuracy | B available development parent accuracy | B − A |
|---|---:|---:|---:|---:|
| 2026 | 24.990 | 41.30% | 52.17% | +10.87 pp |
| 2027 | 21.606 | 36.96% | 39.13% | +2.17 pp |
| 2028 | 23.555 | 34.78% | 39.13% | +4.35 pp |

B has the better observed availability frontier at these terminal common hour budgets. These are descriptive point differences after development selection; they do not carry an independent significance claim, and checkpoint-selection noise can affect them. This does **not** establish uniform compute dominance: the earliest observed 35% development threshold favors A in 2026 (2.373 versus 24.990 hours), B in 2027 (6.942 versus 21.606), and B slightly in 2028 (23.555 versus 24.652). Thresholds are exploratory and crossings may occur between the sparse observations. At the largest common token budget, available B-minus-A development differences are +10.87, +2.17 and −13.04 pp. There is no consistent token-budget advantage. Student-step budgets are descriptive because steps differ in episode/token content.

At an exact 4,096-episode cutoff, B-2028's final 4,100-episode point is unavailable. The primary study's explicitly allowed overshoot is retained; it would be misleading to claim a new exact-equality measurement or interpret the preceding checkpoint as its 4,096-episode policy. Final ID episode-matched evidence remains favorable to B in all three seeds, while development and compute definitions need separate qualifications.

Committed scoring timings attribute approximately 1,695–1,884 seconds per arm to teacher scoring, 2,076–2,304 to unprivileged scoring, and 2,155–2,400 to proposal generation. These omit discarded work, initialization and some unrecorded phases; they are not allocation costs or a counterfactual overhead estimate against GRPO. Missing development-phase timings remain null.

## Training signal and proposer behavior

| Arm | Verified student repairs / episodes | Mixed groups / groups | Mixed groups per allocated hour | Valid proposals / attempts | Distinct-valid-reward proposer groups / attempted groups | Historical repeated valid samples |
|---|---:|---:|---:|---:|---:|---:|
| A 2026 | 2200/4096 | 169/1024 | 6.51 | 3001/3460 | 49/865 | 2486 |
| B 2026 | 2278/4096 | 224/1024 | 8.96 | 3060/3552 | 65/888 | 2540 |
| A 2027 | 2086/4096 | 155/1024 | 7.17 | 2879/3384 | 60/846 | 2342 |
| B 2027 | 2254/4096 | 189/1024 | 7.58 | 2987/3520 | 52/880 | 2413 |
| A 2028 | 2029/4096 | 126/1024 | 5.11 | 2921/3380 | 43/845 | 2403 |
| B 2028 | 2243/4100 | 172/1025 | 7.30 | 2810/3264 | 63/816 | 2269 |

All six arms have zero fallback groups. Cumulative curriculum mutation counters end at only 220–254, despite 2,810–3,060 valid proposals and 1,024–1,025 trained task groups per arm. Historical repetition and within-update repetition overlap; their counts must not be added as disjoint failures. A zero rejected-duplicate count does not imply novelty, because Task 15 intentionally admits historical repeats.

Proposal validity rises from 72.8–81.2% in the first measured window to 87.4–89.2% in the last. Student repair success rises from 27.7–32.0% initially to 52.5–63.3% in the last window. Mixed-group fractions fluctuate and mostly decline from early values; B supplies more mixed groups per allocated hour than A in each seed. Training success and mixed-group yield are different quantities.

B has fewer invalid training actions than A in every seed: invalid fractions A/B are 49.30/45.41%, 46.30/43.61%, and 52.17/44.09%. Total action counts are recovered from the recorded turn-distribution mean × count; action-type and patch breakdowns are reconstructed from the recovered trajectories. OPSD can update on uniform groups, so their zero current GRPO advantage does not imply zero total gradient or zero optimizer displacement. The recovered candidate groups and matched student rewards reconstruct nonzero proposer-group fractions of 11.79–15.56%; the per-candidate counter alone could not establish these fractions.

These observations show increasing validity and repair proficiency on the realized narrow curriculum. They do not identify a causal contribution from proposer learning: there is no matched frozen-proposer comparison, and repetitions dominate valid samples. Only 5.1–7.7% of attempted proposer groups distinguish difficulty rewards among valid candidates. Increased structured validity has not removed the reward-identifiability concern underlying deferred C/D.

## Ranked explanations and research decision

1. **Patch protocol failures are a directly observed obstacle.** The six trained external runs contain 4,301 rejected patches and 13 accepted applications, with 96–116 of 116 tasks per run having at least one rejected patch. Context mismatches and malformed hunks dominate identifiable saved error messages. Budget exhaustion overlaps with these errors; it is not evidence of a separate exclusive cause. The observational comparison does not establish whether repairing the interface alone would yield successful semantic repairs.
2. **Representation and task structure plausibly contribute.** External specifications and docstrings are longer, and the patch-context error burden is high. Public-case counts are similar, and 96/116 external initial mutants fail public tests. Which representation feature causes the problem requires a controlled training-only comparison.
3. **A weak baseline and narrow repeated curriculum limit what can be inferred about transfer.** The base solved 2/116 external repairs; per-task regressions are measurable but too few to establish broad forgetting. The evolving proposer/student comparison does not identify the proposer's causal contribution.
4. **Public-test insufficiency explains a subset.** Twenty external initial mutants pass public tests while failing hidden tests. This is not evidence that the majority of accepted patches overfit public tests.
5. **Terminal overtraining and infrastructure failures lack support in these records.** The selected checkpoint is the final checkpoint in all six arms and final denominators are complete. Sparse evaluation cannot rule out unmeasured peaks.

The next experiment should be a small training-only, matched-snapshot repair-interface study informed by the recovered error subtypes. Preserve the current task oracles and compare accepted edits and verified repairs under identical tasks and rollout seeds; investigate patch context and hunk formatting before changing the curriculum. Keep both final splits out of design and tuning. Defer full C/D training: Task 14's RL-versus-combined update cosine (~0.992) was close to the identical-reward repeat (~0.995), and Task 16's difficulty-reward discrimination remains sparse. A future C/D pilot would require matched frozen snapshots/tasks, repeated updates, restored optimizer state, and disjoint reference-gradient estimates.

## Reproduction, coverage, and limitations

All six run ledgers and selected trajectory shards were fetched through the authenticated Yale controller without model checkpoints. The read-only controller inspection succeeded; no job was submitted. The analysis uses the existing execution and frozen data, with the recovered copies selected by `--run-root`:

```bash
uv run python scripts/analyze_task16.py \
  --manifest /private/tmp/selfplay-task15-v2/runs/task15/manifest.json \
  --data data/protocol-v3 \
  --base /private/tmp/selfplay-task15-v2/runs/task15/base \
  --run-root runs/hpc/yale-hpc-cpsc4880 \
  --output runs/task16/recovered-analysis-verified \
  --tokenizer "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"
```

The private output directory contains per-task records, input hashes, taxonomy, training health and bootstrap results. It must remain ignored. Final evaluation trajectory coverage is 1,617/1,617; all 1,447 active training and proposer iteration shards were available. Six checkpoint files remain on Yale; their historical checksum validation was not repeated in this analysis. Initial public/hidden outcomes remain unknown for seven ID mutants and unobserved reference/development tasks unless separately replayed through the accepted isolated worker. The final test comparisons remain exploratory.

## Figures

Markers on development curves are actual evaluations; connecting lines are visual guides. All figure data are privacy-safe aggregates.

![Development accuracy by episode exposure](../figures/task16/development-episodes.png)

![Development accuracy by allocated compute](../figures/task16/development-compute.png)

![Monitoring-selected and final checkpoints](../figures/task16/checkpoint-selection.png)

![Recovered terminal failure categories](../figures/task16/failure-categories.png)

![Final ID paired effects and family uncertainty](../figures/task16/final-id-paired.png)

![Proposal and student training-signal diagnostics](../figures/task16/training-signal.png)

The [completion record](../task_completion/16-implemented-generalization-checkpoint-analysis.md) records validation, provenance and remaining limitations.
