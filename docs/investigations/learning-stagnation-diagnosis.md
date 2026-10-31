# Why corrected repair self-play stagnates

Task 13 diagnosis, 2026-10-31. The strongest supported explanation is a combination of a brittle edit representation, repeated mutation generation against accumulating novelty history, and very sparse objective-discriminating feedback. Model capacity and OPSD-caused forgetting remain unresolved. Additional unchanged iterations are a poor first diagnostic.

## Evidence and operational reconciliation

The CPU analysis joins active committed events to the exact proposal and episode shards they reference. It covers **8,000 proposals, 439 accepted tasks (unique parent/mutation pairs within each run), and 1,756 student episodes**: corrected A/B through 200 and C through 100. All 500 proposal shards and 500 episode shards are present. C's missing 41–100 trajectories were fetched through the existing `YaleSSH.fetch` transport, targeting the trajectories subtree without checkpoints. Input SHA-256 manifests accompany the results. That CPU analysis did not execute generated Python or load a model; the later GPU probes described below run generated code only in the accepted isolated worker.

Live Slurm inspection found no active jobs. The existing A jobs 31364215/31372772, B jobs 31372774/31381376, and C jobs 31429026/31455833/31469208 completed with exit 0. Their allocated H200-hours are **8.105278, 8.101944, and 6.656667**, respectively. The corrected remote study has A/B/C main directories and training ledgers; it has no D main directory or training ledger. Corrected D evidence is a ten-iteration readiness pilot, not a main result.

C CPU job **31528917 completed with exit 0 in 29 seconds**. Its reconciliation code verifies the latest checkpoint before logging; the fetched telemetry record is cloud-verified through 100 with canonical revision `07e12f64924f324c994551ebb4a8f98037da9561849c9bf2796557cc9971a5e6`. No replacement verification job is needed. All three run manifests have file SHA `4be6da81532938b3e03bf5e7e839c200335e2b3728b5bcbfbc7c7161c39288cf`; controller specifications bind canonical manifest `fd47b9117ffa627531a24b938999d9e063f1e6eee47e4b0b91b1bbf46c4ed714` and corrected frozen execution `c307c9800d36a879a295dbaa2741cf5c18d6fa1b`. Method-specific scientific hashes agree between resolved configuration and event records.

Private reproducible outputs are under `runs/task13/diagnosis/{A,B,C}/`. Each contains `diagnosis.json`, per-proposal and per-group records, parent summaries, conceptual and actual validation funnels, exposure CSV, health CSV, and the reused scientific-health report. Remote accounting, ledger inventory, and inspection evidence are under `runs/task13/`.

## Q1. Why is proposal acceptance around 5%?

### First recorded rejection, mutually exclusive

| Outcome | A, n=3,200 | B, n=3,200 | C, n=1,600 |
|---|---:|---:|---:|
| Accepted | 161 (5.03%) | 162 (5.06%) | 116 (7.25%) |
| JSON rejection | 15 | 51 | 19 |
| Diff application rejection | 1,632 (51.00%) | 1,632 (51.00%) | 854 (53.38%) |
| Python syntax rejection | 6 | 5 | 2 |
| AST edit-size restriction | 79 | 120 | 66 |
| Duplicate | 1,204 (37.63%) | 1,140 (35.63%) | 492 (30.75%) |
| No verified semantic failure | 103 | 90 | 51 |

There were no recorded additional schema rejections after JSON parsing, safety/import/API rejections, or token-admission rejections in these runs. This describes first failures: earlier formatting rejection can conceal downstream constraints.

JSON itself is usually correct: 99.53%, 98.41%, and 98.81% parse. Precise diff generation is the immediate representation bottleneck. The most frequent diff failures are hunk counts (625/644/352), context mismatch (544/576/199), and invalid hunk lines (452/399/295), for A/B/C respectively. Header failures are only 9/10/5.

### Actual ordered validation funnel

The production parser interleaves context matching and hunk-count checks. Deduplication occurs before semantic execution, not after it. The actual funnel below avoids imposing a false order on its subchecks. Entries show **passed / reached (conditional pass rate)**; rejected counts are the differences.

| Gate | A | B | C |
|---|---:|---:|---:|
| JSON | 3,185/3,200 (99.53%) | 3,149/3,200 (98.41%) | 1,581/1,600 (98.81%) |
| Proposal schema | 3,185/3,185 (100%) | 3,149/3,149 (100%) | 1,581/1,581 (100%) |
| Exact diff application | 1,553/3,185 (48.76%) | 1,517/3,149 (48.17%) | 727/1,581 (45.98%) |
| Python and structural eligibility | 1,468/1,553 (94.53%) | 1,392/1,517 (91.76%) | 659/727 (90.65%) |
| Novel mutation | 264/1,468 (17.98%) | 252/1,392 (18.10%) | 167/659 (25.34%) |
| Verified semantic failure | 161/264 (60.98%) | 162/252 (64.29%) | 116/167 (69.46%) |
| Token admission | 161/161 (100%) | 162/162 (100%) | 116/116 (100%) |

The 12-stage conceptual CSV separately reports pass, fail, unknown, fractions of attempts, and conditional rates among assessed proposals. It does not treat unexecuted stages as failed. Generation returned in every saved proposal; normal EOS versus length termination is unavailable. Semantic validity of duplicates is inherited only when their normalized AST matches an observed accepted mutation. This is evidence of previously valid code generation, not an additional accepted task.

Independent lexical checks identify overlapping failures without fixing or executing proposals. For example, 417/438/175 proposals have both hunk-count and context problems. These counts overlap first-rejection categories and must not be added to the funnel.

### Repetition, novelty history, and time

| Window | A accepted / attempts | A duplicates | B accepted / attempts | B duplicates | C accepted / attempts | C duplicates |
|---|---:|---:|---:|---:|---:|---:|
| 1–20 | 34/320 | 53 | 30/320 | 45 | 29/320 | 49 |
| 21–40 | 28/320 | 105 | 29/320 | 96 | 25/320 | 96 |
| 41–80 | 53/640 | 242 | 48/640 | 222 | 45/640 | 231 |
| 81–100 | 15/320 | 119 | 15/320 | 116 | 17/320 | 116 |
| 101–140 | 14/640 | 264 | 19/640 | 252 | unavailable | unavailable |
| 141–200 | 17/960 | 421 | 21/960 | 409 | unavailable | unavailable |

A's duplicates comprise 200 within the same iteration and 1,004 against prior accepted history. B's are 198 and 942; C's are 156 and 336. None has unresolved duplicate ownership. All recorded sampling seeds are distinct within each parent group. This excludes repeated recorded seeds as an explanation, not all possible RNG or inference defects.

Repeated exact text occurs in 660/800 A groups, 675/800 B groups, and 327/400 C groups. Identical-text pair counts are 2,155/4,800, 2,041/4,800, and 967/2,400. Among pairs where both outputs have an eligible normalized AST, 1,650/2,101, 1,585/1,954, and 701/907 match. Repetition therefore occurs both at the token-output and normalized-program levels; it is not solely equivalent patches written differently. Parent-level distinct mutation sites/operators and text/token diversity are recorded in `parents.json`. The existing `bug_family` field is too coarse: its comparison label indicates a comparison anywhere in the resulting program, not necessarily the edited operator.

For the **49 parents observed in both 1–20 and 141–200**, paired parent-macro changes are:

| Metric, late minus early | A | B |
|---|---:|---:|
| Accepted fraction | −11.39 percentage points [−16.58, −6.63] | −8.63 [−13.74, −3.83] |
| Duplicate fraction | +24.36 [14.80, 35.46] | +24.02 [14.62, 34.23] |
| Syntax/structural eligibility | +12.71 [3.19, 22.96] | +15.14 [5.18, 25.60] |

Intervals are 2,000 seeded paired-parent bootstrap resamples, conditional on this observed parent set; they do not measure training-seed uncertainty. Matching 1–20 to 81–100 uses 30 recurring parents and likewise shows increased duplication in A/B/C. Parent identity alone does not hold model snapshot, historical novelty, or mutation difficulty fixed.

Thus declining yield is better supported as **concentration against accumulating novelty history** than as worsening general format compliance. Format failures remain a major absolute bottleneck. Actual mutation-space exhaustion, causal policy collapse, and a loss of code-generation ability are not established. Fixed probes with reset novelty history distinguish those possibilities. Training proposal lengths remain about 95–104 tokens; mean sampled-token surprisal declines in A from 0.022 to 0.012 and B from 0.022 to 0.014. This is a concentration-compatible statistic, not an entropy estimate or causal proof.

## Q2. Which validity rules should change?

Keep correct-source restoration, genuine semantic failure, verifier correctness, path confinement, worker isolation, supported execution limits, and train/held-out separation. Preserve original-study restrictions for any continuation or comparison claiming its identity.

| Requirement | Role and diagnosis |
|---|---|
| One allowed file, fixed API/imports, no arbitrary module execution | Correctness/safety and task comparability; keep execution protection. Import/API preservation is also a scientific task-family choice, not a universal necessity. |
| Genuine semantic mismatch; crash/timeout alone is invalid | Defines useful verified repair tasks; keep. |
| Exact context, headers, hunk counts, no fuzz | Deterministic current protocol. Do not relax silently, but generating these fields need not be the model's job. |
| At most 3 AST edit roots / 12 nonblank changed lines | Convenient bounded mutation family and frozen comparability; not an isolation boundary. It rejects 2.47–4.13% of all attempts here, much less than diff application. |
| No `pass`, no `raise`, unchanged comments/docstrings/top-level declaration count | Conservative representation/task choices, some broader than their motivating risks. Inspect separately before a new study; none is a measured leading first-rejection cause here. |
| AST novelty against all accepted history | Useful diversity rule, but acceptance becomes history-dependent and increasingly hard. Keep its distinction from code validity in reporting. |

**First intervention:** structured edits with stable AST field paths, an explicit finite transformation vocabulary, deterministic patch construction, and the unchanged validator and isolated semantic verifier. This removes diff arithmetic/context serialization but restricts expressivity and needs measured site/operator diversity. Full-source or fragment replacement followed by a deterministic diff is a less restrictive alternative, with greater output cost and more opportunities to exceed mutation constraints. Neither guarantees semantic bugs or novelty. Start with the 4B model to isolate representation from scaling.

## Q3. Is 4B capacity the bottleneck?

The 4B model already generates many syntax-eligible mutations, including repeated previously verified bugs; novel syntax-eligible candidates pass semantic verification 61–69% of the time. This argues against treating inability to generate any valid code as the central diagnosis. Paired larger-model probes below show that changing model size alone does not solve the exact-format problem in these settings.

The first paired base-model GPU probe completed after the original CPU diagnosis. On the same 40 training parents and four seeds, the diff representation produced 43/160 verified outputs and 20 unique parent/mutation pairs; structured edits produced 95/160 and 33. The paired-parent bootstrap difference in verified fraction was +32.5 percentage points (95% interval +15 to +50); the difference in unique mutations per parent was +0.325 (+0.025 to +0.625). The representations differ in expressivity, so this is a useful-yield comparison rather than a controlled equivalence of mutation spaces. The B-at-200 checkpoint produced 93/160 verified structured outputs and 33 unique pairs, of which 28 were absent from its recorded training history. On this changed structured prompt, its paired verified-fraction difference from the base proposer was −1.25 percentage points (95% interval −5.625 to +2.5); this does not establish how B training affected its original diff behavior.

An attempted extra-sample batch used a faulty launcher that failed to pass its slot offset. Its completed structured arm produced exactly the same 160 parent/slot/seed/text records as the original and is not independent evidence; the other two affected jobs were cancelled. The attempts consumed 765 allocated H200-seconds in total. The command wiring has since been corrected and covered by a regression test; replacements use new immutable run identities.

The corrected extra samples use slots 4–7 on the prespecified first 16 parents. Combining both batches gives 18/128 verified diff outputs and six unique parent/mutation pairs, versus 91/128 verified structured outputs and 17 unique pairs. The paired difference is +57.0 percentage points in verified fraction (parent-bootstrap 95% interval +31.3 to +80.5) and +0.688 unique mutations per parent (+0.125 to +1.188). Within these same parents, moving from four to eight samples changed diff uniqueness from six to six and structured uniqueness from 15 to 17, despite many additional valid outputs. This small, selected subset supports diminishing returns from repeated sampling; it does not establish a population-wide group-size effect.

The A-at-20 checkpoint has also completed the same-parent, same-seed diff probe: 64/160 verified proposals and 25 unique parent/mutation pairs, versus 43 and 20 for the base proposer. Its paired verified-fraction difference was +13.125 percentage points (95% parent-bootstrap interval +5.625 to +21.875); 22 of its 64 verified outputs matched that checkpoint's earlier accepted history. These descriptive fixed-probe observations support separating output capability from history-dependent admission. They do not establish a multi-seed training effect.

A at iteration 200 produced 79/160 verified diff outputs and 33 unique pairs, but 76 of those verified outputs and 30 of the 33 unique pairs matched its saved history. Only three unique verified pairs were novel relative to that history. Compared with the base, its verified fraction increased by 22.5 percentage points (95% paired-parent bootstrap interval +11.25 to +34.375). This does not prove the full mutation space is exhausted or establish an effect across training seeds.

C at iteration 100 produced 71/160 verified diff outputs and 29 unique parent/mutation pairs on the same probe. Its paired verified-fraction difference versus the base was +17.5 percentage points (95% parent-bootstrap interval +6.875 to +28.75). However, 54 of the 71 verified outputs matched C's checkpoint history; only 8 of the 29 unique verified pairs were absent from that history. This directly demonstrates that validity under a reset history and novelty under the live history can differ substantially. It does not prove the mutation space is exhausted.

The matching B-at-200 diff probe produced 68/160 verified outputs and 25 unique pairs, of which only four were absent from its training history. Under the structured prompt at the same checkpoint, 28 unique pairs were absent from that history. The paired increase was 0.60 novel unique mutations per parent (95% parent bootstrap interval 0.375–0.85). Novel verified output fraction increased by 43.125 percentage points (95% interval 28.125–58.75). These are controlled by checkpoint, parents and seeds, but the structured vocabulary represents a narrower mutation family and the results are from one training seed.

On the matched 40-parent probe, Qwen2.5-Coder-7B produced 4/160 verified diff outputs (four unique pairs) and 1/160 structured outputs (one pair), while non-thinking Qwen3-8B produced 6/160 diff outputs (five pairs) and 73/160 structured outputs (37 pairs). Relative to 4B structured, 8B structured's verified fraction was 13.75 percentage points lower (paired-parent bootstrap interval −26.875 to −1.25), while its unique-pair count was four higher; the paired difference of 0.10 unique pairs per parent has interval −0.20 to +0.425. This does not establish a uniqueness advantage. The 7B diff arm's 115 leading JSON parse failures all began with Markdown fences; 135 structured outputs had a single complete fence. Removing those fences offline leaves only eight diff outputs and five structured outputs passing the static mutation validator; these 7B recoveries have not been semantically verified. The 8B diff arm had 114 outputs beginning with an extra exact `diff --git` header; removing it lets 41 pass the static validator. CPU replay job `31537818` checked all 114 using the accepted isolated worker and matching cached tokenizer: 37 passed semantic verification and admission, adding to the six originally valid outputs for **43/160 and 30 unique pairs**. This is a post-hoc normalization of the same samples and preserves the original strict-format result. It provides direct evidence that one exact header rule can be relaxed without weakening the subsequent checks. The result is sensitive to model-specific serialization and prompt format, and does not establish a broad model-capability ranking. The [Coder model card](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) and [Qwen non-thinking guidance](https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html) describe the models; the trainer image and lock are unchanged. Equal temperatures and seeds do not imply equivalent token distributions across models.

After the post-hoc header normalization, 8B diff and the original 4B diff both produced 43 verified outputs; 8B had 30 unique parent/mutation pairs versus 20. The paired uniqueness difference was 0.25 per parent (95% parent-bootstrap interval −0.025 to +0.525), which does not establish a reliably positive difference. The normalization was selected after inspecting 8B's outputs, so a prospective comparison with both formats specified in advance remains necessary for a model-selection claim.

As a descriptive throughput check, the 4B and 8B structured jobs took 242 and 196 allocated H200-seconds respectively, corresponding to about 491 and 680 unique pairs per allocated GPU-hour on this small fixed workload. Those rates include startup and differ in backend and caching; the 8B job used the cache warmed by its earlier diff arm. They are not portable throughput benchmarks or sufficient evidence to select a larger model.

## Q4. What is the proposer optimizing?

Mostly binary acceptance. For rewards `[-1,-1,-1,r]`, let `d=r+1>0`. Centering gives `[-d/4,-d/4,-d/4,3d/4]`, with population standard deviation `sqrt(3)*d/4`. Ignoring epsilon, standardized advantages are `[-1/sqrt(3),-1/sqrt(3),-1/sqrt(3),sqrt(3)]`, independent of r. Production adds `1e-6` to sigma, creating tiny scale differences; this is why exact and approximate comparisons must be separated.

| Group/update evidence | A | B | C |
|---|---:|---:|---:|
| Parent groups | 800 | 800 | 400 |
| 0 / 1 / 2 / 3 / 4 accepted | 671 / 100 / 26 / 3 / 0 | 668 / 106 / 22 / 4 / 0 | 308 / 70 / 20 / 2 / 0 |
| Nonzero-advantage groups | 129 | 132 | 92 |
| Accepted candidates have differing rewards | **4** | **5** | **7** |
| Exactly identical to validity-only advantages | 784 | 781 | 308 |
| L2 difference ≤ 1e-5 | **796/800** | **795/800** | **393/400** |
| Mean L2 difference | 0.004018 | 0.005288 | 0.000535 |
| Maximum L2 difference | 1.018278 | 1.018278 | 0.078611 |
| Optimizer iterations containing a difference > 1e-5 | 4 | 5 | 7 |
| Total proposer optimizer steps | 98 | 102 | 70 |

The classification is unchanged at tolerance 1e-4. At 1e-6 more epsilon-scale differences remain. Zero-vector cosines are undefined, not zero. Per-group reward variances, valid-only variances, vectors, distances, and defined cosines are in `groups.json`.

Only **1.75% of C's groups** change their normalized advantages relative to validity-only acceptance beyond the stated numerical tolerance, even though 18 candidate gradients are nonzero. This tolerance measures numerical similarity, not a proven threshold for an optimizer effect. A higher-quality reference gradient cannot make an exactly zero RL-only candidate gradient nonzero. D might supply gradients on uniform groups, but singleton-valid groups would still largely erase score magnitude. Score richness and actual proposer discrimination must both be measured.

The matched C/D counterfactual has now completed at verified C checkpoint 99. It sampled 64 proposals in 16 original parent groups; two candidates were accepted, both in the same group. Their student outcomes were 0/4 and 4/4, so both RL-only candidate gradients were exactly zero and both C rewards were neutral 0.5. Combined RL+gated-OPSD candidate norms were 0.06957 and 0.004334, yielding D rewards 0.44816 and 0.31870. The group's normalized advantages differed by L2 0.13213. The complete-batch proposer-gradient cosine was 0.95646, with norms 0.01107 and 0.003231; on the discriminating group the counterfactual Adam-update cosine was 0.98893, with norms 0.007932 and 0.007751. Both adapters and optimizer state were restored and verified. Thus D supplied information where C had none, and the observed grouping allowed some of it through, but the measured optimizer directions were similar. This is **one** discriminating parent group, not a population estimate or evidence that D improves repair performance. Details and original inputs remain private in run `913f441b4f52675eb2479c16`.

The optimizer audit found no demonstrated defect. `proposer_examples` normalizes within contiguous same-parent groups and weights attempted sequences equally after per-sequence token averaging. The PPO loss uses actual stored behavior log probabilities, `exp(current-old)`, sign-sensitive clipping at 0.2, and detached advantages. Microbatch contributions sum with the declared weights. Distributed norms are reduced over unique coordinate shards; gradients clip at norm 1 and optimizer steps skip zero total norm. Adapter versions are invalidated after updates. Recorded proposer gradient norms are finite; clipped steps are 0/2/2 in A/B/C. Prior Adam state can influence a nonzero step.

This static/recorded audit does not re-prove GPU numerical equivalence. Existing actual-D counterfactual Adam-update cosine **0.58968** remains a material sensitivity, not a confirmed bug. Fresh proposer rescored probabilities, clipping fractions, entropy, and per-step parameter displacement are not stored sufficiently for a complete retrospective reconstruction. Do not substitute rescored probabilities to force ratios toward one.

## Q5. Why are student RL updates uninformative? Should G increase?

| Student evidence | A | B | C |
|---|---:|---:|---:|
| All-failure / mixed / all-success tasks | 85 / 16 / 60 | 98 / 19 / 45 | 60 / 18 / 38 |
| Mixed fraction | 9.94% | 11.73% | 15.52% |
| Student optimizer steps | 98 | 102 | 70 |
| Steps with zero current group-relative RL signal | 82 | 84 | 54 |
| Generated student tokens | 146,579 | 159,412 | 164,645 |
| Accepted tasks per allocated GPU-hour | 19.86 | 20.00 | 17.43 |
| Mixed groups per allocated GPU-hour | 1.97 | 2.35 | 2.70 |

C's 98 neutral candidate scores have exactly zero candidate RL gradient; the 18 non-neutral scores correspond to mixed groups. Reference gradients are present. Zero current RL signal does not imply zero OPSD gradient or zero Adam displacement. Conversely, a mixed reward group guarantees nonzero advantages, not a theorem that parameter gradients cannot cancel.

For independent rollouts conditional on success probability p, `P(mixed)=1-p^G-(1-p)^G`. At p=0.5, G=4→8 changes 87.5%→99.22%; at p=0.05, 18.55%→33.66%. Doubling sampling is not automatically twice as informative.

Integrating each observed four-rollout task under Beta(0.5,0.5) gives posterior-predictive mixed fractions for **new** G=4/G=8 groups of A 31.4%/45.2%, B 32.3%/46.1%, C 33.6%/47.6%. A uniform Beta(1,1) prior instead gives 46.2%/64.0%, 46.8%/64.6%, and 47.7%/65.6%. G=16 is also tabulated. These priors do not reproduce the observed extremely bimodal four-rollout population well; their wide sensitivity is a warning against treating either projection as measured efficacy. They explicitly avoid setting p=0 or 1 after 0/4 or 4/4.

The subsequent static-pool screen provides an observed paired comparison: 12/32 training tasks (37.5%) had mixed rewards across eight rollouts. Averaging all 70 four-rollout subsets of those same outcomes gives a 27.95% mixed fraction; the eight-minus-four difference is 9.55 percentage points (task-bootstrap interval 4.20–15.89). The first four rollouts alone were mixed on 10/32 tasks. This is one fixed training-only pool and initial student, not an independent sample from the evolving A–C curricula. Dividing these fractions by their group sizes gives about 0.070 mixed groups per rollout at four versus 0.047 at eight; fixed overhead and learning impact still require actual timing and training outcomes. The full matched training control is ongoing.

The static GRPO arm's initial development evaluation completed 0/41. Its first batch used two tasks and eight episodes, with seven successes and one mixed group, and performed one student optimizer step and zero proposer steps. This confirms the intended static update path is active; one batch does not establish learning.

At the five-student-step review, GRPO had processed six batches, 12 task exposures and 48 episodes, including 29 successes and eight mixed groups; the proposer had taken no steps. Five nonzero recorded student gradient norms were finite. There were six rollout/rescoring mismatch warnings (maximum absolute sampled log-probability difference 0.991) and two rolling invalid-action warnings; 113 of 308 recorded actions were invalid. These warnings limit interpretation and their causal effects are unverified. The arm continued to its scheduled development checkpoint.

At ten student steps, GRPO had committed 21 batches, 42 task exposures, 168 episodes, 74 successes and 13 mixed groups, with zero proposer steps. All recorded gradient norms were finite. There were 20 rollout/rescoring mismatch warnings and 15 rolling invalid-action warnings; the maximum sampled log-probability difference was 2.535. Its complete ten-step development evaluation improved from 0/41 to **10/41** (parent-macro 0.3043). This single-seed static control supports student learnability independent of learned-proposer yield; it does not identify the cause of the original plateau. The warnings are material measurement limitations, not proof of their causal effect on the student. Training and the matched gated arm are ongoing.

At 15 steps the arm had committed 31 batches, 248 episodes and 19 mixed groups. All recorded gradient norms were finite and proposer steps remained zero; the maximum sampled rollout/rescoring difference rose to 3.203. This is a health observation, not a new development evaluation.

GRPO subsequently reached the planned 20 student steps after 53 batches: 106 task exposures, 424 episodes, 189 successes and 24 mixed groups, with 192,190 generated student tokens and zero proposer steps. All recorded gradient norms were finite; 48 rollout/rescoring warnings and 38 rolling invalid-action warnings were recorded, with maximum scoring difference 3.203. Its completed final development evaluation was **12/41** (parent-macro 0.3478), following 0/41 initially and 10/41 at ten steps. This is a single-seed descriptive result, with the numerical caveat above. The matched gated arm has started.

Under these illustrative projections, mixed groups per rollout decrease at G=8. Whole-allocation efficiency might differ because of fixed startup/proposal costs; it must be timed. Improve the task-difficulty distribution first. Keep proposer and student group sizes separate: under an illustrative independent q=0.05 candidate acceptance model, proposer G=4→8 gives at least one accepted 18.55%→33.66%, at least two 1.40%→5.72%. Observed within-parent repetition violates independence. G=8 may provide extra reward discrimination, but does not address history-dependent duplication by itself.

## Q6. Why does development stagnate?

**Confirmed:** useful RL exposure is tiny. A accumulates only 16 mixed groups in 200 iterations and no new mixed group after iteration 140; B ends at 19. Most late updates can therefore be OPSD-driven even when the optimizer steps. Early development improves from 0/41 to 14/41, 13/41, and 12/41; final/current results are 11/41, 12/41, and 11/41. Additional accepted tasks and tokens did not produce a sustained subsequent gain. This is a descriptive plateau, not proof of incapacity.

Patch protocol remains a student bottleneck. Recorded training patch successes/errors are 347/1,099 in A, 271/1,265 in B, and 231/1,385 in C. Test calls are 339/249/220. Unsuccessful finish episodes are 339/378/173; this is an observable proxy, not proof that each finish was premature. At A/B final development, 27/25 of the 30/29 failures are classified as patch rejected. The tool histories count each action's immediate observation once; cumulative histories are not counted repeatedly.

The original wholesale newline regression did not recur. Across corrected training patch actions, literal backslash-n occurs in 9/1,446 A patches, 2/1,536 B patches, and 0/1,616 C patches; this is not a claim of no escaping anywhere. Earlier examined development checkpoints had no literal-backslash-n rejected patches. Do not conflate isolated corrected occurrences with the original A collapse to zero development success.

Token-weighted OPSD measurements further limit interpretation:

| Statistic on recorded student tokens | A | B | C |
|---|---:|---:|---:|
| Mean teacher-minus-behavior gap | −0.1353 | −0.1453 | −0.1431 |
| Mean gate | 1.0000 | 0.4927 | 0.4927 |
| Mean absolute clipped, gated coefficient before lambda | 0.05816 | 0.005020 | 0.004614 |
| Gap clipping fraction | 0.782% | 0.852% | 0.870% |

The coefficient magnitude difference is much larger than a simple factor of two on these separately generated trajectories. It does not measure parameter-update strength: token vectors can cancel, optimizers have state, and A/B saw different tasks. Action-category and success/failure strata are recorded, as are sampled RL/SD gradient interactions. In most sampled uniform batches RL norm is zero while lambda-scaled OPSD norm is nonzero. Sample coverage is sparse and cannot establish the exact decomposition of every optimizer displacement.

Negative gaps often downweight sampled tokens; teacher scoring of unsuccessful episodes is not equivalent to reinforcing failure. The teacher's privilege, gate selectivity, and signed token coefficients must be interpreted at token level. Lower loss or larger gaps alone do not establish better repairs. A later matched-strength ungated control is necessary to attribute a benefit to selectivity rather than changed effective distillation strength.

There is no observed initial development reward drop followed by recovery: the first monitoring checkpoint improves. Early dense fixed-probe measurements are absent, so a short transient between checkpoints is unidentifiable. Persistent reward sparsity and rising duplicate history are observed; a generic “GRPO gets worse before learning format” explanation is unsupported. Policy displacement/KL and fixed-probe initial-versus-trained capability remain unmeasured.

The generated learning-efficiency PNG/PDF compares outcomes against iteration, unique accepted tasks, mixed groups, student tokens, and scheduler-integrated GPU-hours. Event-reported hours remain separate because they undercount allocation overhead relative to Slurm. No final-ID or external held-out outcome was opened for intervention selection.

## Q7. Are A–D scientifically identifiable?

They describe complete co-evolving systems under a common frozen protocol. They do not isolate all intended mechanisms:

- A/B change both gating selectivity and effective distillation coefficients, and use independently evolving curricula. One training seed and a one-task final difference do not establish a gating advantage.
- B/C barely expose the sophisticated reward to normalized proposer advantages. Most groups train acceptance or have zero advantages. A negative result is weak evidence against alignment in a regime where alignment rarely changes optimization.
- C/D needs a matched-snapshot counterfactual before expensive D training. Nonzero combined gradients are not sufficient if their scores disappear under grouping or merely encode teacher-format preference.
- Behavior/rescoring sensitivity remains unresolved. It is a robustness limitation, not authorization to replace the estimator silently.

Use matched static tasks to isolate the student objective, fixed parents with reset novelty history to isolate proposer policy behavior, and reward/advantage/update counterfactuals to test alignment identifiability. New representations, objectives, models, or budgets require new study identities.

## Q8. Shortest credible recovery path

| Observation and confidence | Action | Information gained / stop condition |
|---|---|---|
| Exact diff failure dominates; high confidence | Use the completed structured probe to design the next study | Verify useful yield and edit diversity prospectively with the existing semantic checks. |
| Recurring-parent duplication rises; high confidence about association | Use the completed base/early/late checkpoint probes | Separate policy concentration from novelty accounting. Do not claim exhaustive mutation-space depletion. |
| Only 0.5–1.75% of proposer groups discriminate real reward | Keep full D deferred | One matched C/D group had different rewards but similar optimizer directions; do not generalize it. |
| Very few mixed student groups; high confidence | Finish the active static GRPO versus gated OPSD control | Test learnability independent of proposer yield; interpret failure using actual informative exposure and protocol behavior. |
| 4B versus larger-model evidence is format-sensitive | Use the completed 7B/8B probes and header replay | Prospectively compare normalized formats before selecting a larger model for training. |
| Gating-strength confound confirmed by objective, effect unresolved | Later matched-strength control | Separate strength from selectivity on shared tasks; avoid broad lambda search. |

The [follow-up plan](../plan/13-diagnosis-plan.md) fixes datasets, sampling, resource caps, dependencies, and interpretation. The paired 4B, checkpoint, larger-model and C/D probes are complete; static training is active. Reference-domain ablations are lower priority; the author's unpublished math-reference observation remains anecdotal evidence.

## Reproduction, observability, and limits

```bash
uv run --frozen python scripts/diagnose_learning.py \
  --run A=runs/hpc/yale-hpc-cpsc4880/747bd2e097258445de647e47 \
  --run B=runs/hpc/yale-hpc-cpsc4880/f3271727f5fd25c07466e50d \
  --run C=runs/hpc/yale-hpc-cpsc4880/35bfbd1bc074a8963e916f77 \
  --accounting runs/task13/accounting.json --output runs/task13/diagnosis
```

The new offline dashboard and per-iteration `health.csv` add the missing counts of groups with two accepted candidates, reward-discriminating groups, and advantage distance from validity-only. Existing group mix, alignment norms/neutrality, losses, finite gradients, gap/gate summaries, and sampled gradient interactions are reused. These three proposer aggregates are the highest-value additions for a future study's W&B logger. This task does not rewrite or append new metrics into historical W&B runs; their canonical reconciliation remains intact. Raw sources, tests, trajectories, and oracle data stay private.

The diagnosis establishes measured bottlenecks and estimator limitations. It does not establish which intervention improves held-out generalization, whether GRPO alone learns this environment, whether 7B/8B is more cost effective, or whether D changes meaningful update directions. Those require the small controlled follow-ups. No infrastructure rebuild or full GPU readiness rerun is justified for this CPU observer.
