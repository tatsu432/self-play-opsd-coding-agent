# Experiment A regression investigation

The locally fetched active scientific events contain 200 committed iterations. All 200 proposal shards, 200 episode shards, and 451 development trajectories have now been retrieved from Yale. The final checkpoint payload remains on Yale; checksum verification succeeded in CPU job 31301421. Unmeasured causal effects and historical gradient decompositions remain explicit limitations.

## Confirmed from the active event lineage

Development successes at iterations 0/20/40/60/80/100/120/140/160/180/200 are 0/12/12/10/9/0/0/0/0/0/0 out of 41. Of the 12 tasks solved at iteration 20, two first failed at iteration 40, one at 80, and nine at 100. None recovered in a subsequent recorded evaluation. This is regression from the peak, not demonstrated degradation below the initial 0/41 baseline.

All 41 evaluation episodes ended with `finish` at iteration 80 and later. Total generated tokens fell from 36,121 at iteration 20 to 13,849 at iteration 100. This is evidence of an action-distribution shift; the event aggregates alone cannot establish whether finishing was premature or why the repair failed.

There were 173 valid proposals, 130 all-failure groups, 38 all-success groups and five mixed groups. The mixed groups appeared in iterations 2, 4, 42, 53 and 57. Of 111 student steps, 106 have only uniform groups and hence exactly zero current GRPO gradient under the implemented group-relative advantage. Residual Adam momentum may reflect earlier RL gradients. Six gradient diagnostic batches were recorded; sampled gradients cannot establish the gradient norm of every unsampled batch. OPSD dominance is confirmed at the current-gradient level for these 106 steps; its causal role in regression is unverified.

The largest invalid categories were 1,103 duplicates, 730 hunk-count mismatches, 627 invalid hunk lines/extra files and 317 patch-context mismatches. Error labels alone cannot distinguish accumulated novelty exhaustion from proposer policy degradation.

## Numerical evidence and next steps

The Task 10 audit found material proposer sensitivity under behavior-versus-rescored probability counterfactuals, including actual-D Adam-update cosine 0.58968, without proving a mapping or estimator bug. The user accepted this risk for A only. The original behavior denominator remains the correct recorded experimental estimator; replacing it for numerical convenience would change the experiment. C/D need their own risk review, especially with mostly uniform groups.

The primary hypothesis is that weak environment-reward signal combined with ungated distillation contributed to the collapse, but a protocol or semantic failure and shared curriculum effects remain plausible. Existing data do not rank these causes reliably. Complete task-level trajectory analysis and assess the implementation before making a causal or B-readiness claim. Do not run held-out tests to select a remedy.

Reproduce the local aggregate analysis with `uv run python scripts/scientific_health.py runs/hpc/yale-hpc-cpsc4880/a87f3772b284d35a375cf69e --regression --output runs/task11-regression`.

## Initial trajectory finding

**Confirmed:** At iteration 20 the 41 evaluation episodes included 17 successful patch calls, 171 failed patch calls and 16 test calls. At iteration 100 they included 75 failed patch calls, no successful patch calls and no test calls. At iteration 200 they included 82 failed patch calls, no successful patch calls and no test calls. One observed tool error reports incorrect patch headers. This strongly supports a breakdown in patch application, but its precise failure categories, causal mechanism and any contribution from semantic repair quality require further analysis.

The next-turn tool observations identify all 75 failed patch calls at iteration 100 and all 82 at iteration 200 as `Only src/solution.py may be patched; two exact diff headers required`. At iteration 20 the recorded errors were 75 context mismatches, 61 hunk-count mismatches, 28 invalid hunk lines, and seven without a subsequent stored observation. This establishes a change in observed patch-format failure; it does not establish why the model adopted that format or whether changing the parser would preserve the scientific protocol.

Yale CPU job 31301421 verified the `000200` manifest checksums for `inference_adapters.pt`, `metadata.json` and `state.pt`. Its metadata identifies iteration 200, the expected logical run, scientific hash `cc618347a25d950424f5c7465718d630b66ad1b54c329417865e1a88e0f63fc3`, data manifest `d9140367e20587051b328dda20a7d93916ee5cadc7e72c2564523cf6ab6beade`, and frozen scientific code `e3260b72cdfe3fa41a2f9ef509343f3c3143bf624f0d8faa047871de6b118e8f`. Earlier CPU job 31301419 failed because Yale's Python lacks `hashlib.file_digest`; it did not modify the checkpoint.

W&B aggregate recovery ran in Yale CPU job 31301954 after the original jobs completed. The existing single-writer reconciler verified checkpoint checksums and read back canonical revision `b835bbbf15272b024b0fe3ac2857aa0a6f8c37cf2623119230a8222a36be64a6` at iteration 200 from the original logical run. Cloud recovery is verified; original scientific events remain authoritative.

Reconstructing duplicate ownership in proposal-slot order finds 919 duplicates of earlier-iteration accepted mutations and 184 duplicates within the same iteration. The 173 accepted mutations cover 102 parents; the stored mutation family labels are 124 `operator_or_constant` and 49 `comparison`. None of the saved proposer outputs reached the configured 512-token generation cap. These findings show substantial recurrence, but do not distinguish policy concentration from exhaustion of admissible variants.

Across 164,777 stored scored training tokens, teacher-minus-behavior gaps averaged -0.1569; 21.74% were negative and 21.07% positive, with the rest exactly zero. This aggregate is token-weighted; a teacher preference is not a verified-repair reward. Outcome/action/turn stratification and any matched-gradient replay remain open.

**Confirmed behavioral mechanism:** After decoding the model's tool-call JSON, all 75 failed patch strings at iteration 100 contain literal backslash-`n` sequences; eight also contain at least one actual newline. At iteration 200 all 82 failed patch strings contain literal backslash-`n` sequences and none contains an actual newline. By contrast, all 171 failed patch strings at iteration 20 contain actual newlines and none contains literal backslash-`n`. The model has double-escaped patch line breaks, so the existing parser sees no valid two-line unified-diff header. This explains the immediate patch-application failure without establishing why training induced the behavior. Auto-unescaping would change the protocol and requires a separate scientific decision; it should not be silently applied to B–D.

In the stored training shards, token-weighted mean gap was -0.1704 over 147,090 tokens in failed episodes and -0.0450 over 17,687 in successful episodes; the corresponding fractions with absolute gap above 5 were 0.99% and 0.23%. Failed patch actions accounted for 122,764 tokens (mean gap -0.1686), versus 14,255 tokens in valid patch actions (mean -0.0961). First-turn tokens averaged -0.0028 versus -0.1666 for later turns. These descriptive comparisons share trajectories and are not causal tests; signed token coefficients do not directly indicate whether a trajectory should be reinforced or punished as a whole.

The onset also appears in sampled training episodes: no patch actions contained literal backslash-`n` in iterations 1–40, two of 424 did in 41–80, 80 of 124 did in 81–100, and all 379 patch actions in 101–200 did. The first observed training occurrence was iteration 60. Thus the change is present on training trajectories before the zero-score evaluation at iteration 100; this timing does not prove an OPSD causal mechanism.

For the 12 tasks solved at iteration 20, those episodes had 12 valid patches and 12 test calls. At iteration 100, the same tasks had 24 rejected patches, no accepted patches, no test calls, and all 12 invoked finish. Two examples, task `11582d141f8e` (parent Mbpp/450) and task `406e354ec669` (parent Mbpp/135), went from read/valid patch/public test/finish with public and hidden success at 20 to two reads/two rejected patches/finish with both final test sets failing at 100.

As a non-executing format sensitivity diagnostic, replacing literal backslash-`n` with newline and stripping leading newlines before passing the resulting patch to the existing parser made 23/75 rejected patch calls apply at iteration 100; 27 still failed context, 22 hunk counts, one hunk range and two invalid lines. At iteration 200, 24/82 applied and 58 still failed validation. This transformation may alter intended code literals and was not applied to training or evaluation; successful patch application is not proof of a correct repair. The result indicates additional patch-quality problems beyond escaping.

## Longitudinal training evidence

| Iterations | Proposals valid/attempted | Successes/episodes | Groups: failure/mixed/success | Student steps | Steps with zero current RL signal |
|---|---:|---:|---:|---:|---:|
| 1–20 | 30/320 | 40/120 | 19/2/9 | 15 | 13 |
| 21–40 | 27/320 | 44/108 | 16/0/11 | 17 | 17 |
| 41–80 | 48/640 | 58/192 | 32/3/13 | 29 | 26 |
| 81–100 | 18/320 | 20/72 | 13/0/5 | 12 | 12 |
| 101–140 | 18/640 | 0/72 | 18/0/0 | 14 | 14 |
| 141–200 | 32/960 | 0/128 | 32/0/0 | 24 | 24 |

Training successes after iteration 100 are therefore also zero, independently of development outcomes. The six sampled gradient batches (iterations 40, 60, 120, 140, 160, 200) all had exactly zero measured RL norm; their lambda-scaled OPSD norms were approximately 0.1363, 0.0706, 0.0018, 0.2520, 0.000149 and 0.000030. These are selected diagnostic batches, not all updates. Exact RL-gradient cancellation in mixed groups cannot be excluded without matching replay; we only infer guaranteed zero current RL for uniform groups.

The 3,200 proposer rewards were 3,027 at -1, 168 at 0, four at 0.75 and one at 1.0. Thus only 5/173 valid tasks and 5/3,200 proposals had positive difficulty reward. Across the run 572 proposer examples had nonzero group-relative advantage; overwhelmingly the reward distribution separates invalid proposals from valid uniform-outcome tasks. There were 184 attempted parents, 102 with an accepted mutation, and at most 44 proposals per parent. Mean output length was 96.9 tokens for invalid and 79.9 for valid proposals; none reached 512 tokens. These facts support repeated duplicates and strict-format failures as major observed causes, but do not establish exhaustion or distinguish it from proposer policy concentration.

## Objective and implementation review

The code constructs detached teacher-minus-behavior gaps, clips them to [-5, 5], and uses a unit gate in A or sigmoid gate in B–D. It multiplies the signed coefficient by 0.1 before minimizing negative weighted current log probability. Four binary rewards are population-normalized within each task; equal outcomes explicitly produce zero advantages. Loss weights are `1 / (episode action tokens × episodes per task × valid tasks)` and optimizer steps accumulate all examples once. Both ordinary and FSDP paths skip an exactly zero gradient and otherwise clip before their separate AdamW optimizer steps. The student records sampled targets and behavior probabilities, and teacher forcing uses the same targets and actual interaction history with the privileged prefix. There is no fixed-base KL in A–D. These observations agree with the frozen method description; this review found no proven implementation defect that explains the escaping change. Teacher privilege includes the correct source and inverse patch, but the recorded change alone does not establish leakage.

The absence of fixed-reference KL permits drift but does not quantify it; neither an exact fixed-base divergence nor matched counterfactual gradient direction is available from these records. Retained joint checkpoints are 0, 20, 199 and 200; the pre-update snapshots for most iterations were pruned. No later snapshot is substituted for them. A targeted replay would need a matching saved snapshot and fresh explicit GPU authorization. The prior audit's measured proposer sensitivity remains unresolved and is not evidence that the estimator is wrong. C/D need special caution: C yields neutral alignment for all-equal groups, while D can respond to the distillation gradient, and existing D counterfactual evidence is limited to the earlier saved snapshots.

## Evidence ranking and decisions

1. **Confirmed:** successful repair actions collapsed into malformed double-escaped patch actions; this prevents edits and is the immediate cause of the observed task-level failure at iterations 100 and 200.
2. **Confirmed:** the frozen curriculum produced predominantly uniform-reward groups and at least 106 student updates with no current RL gradient. The OPSD gradient remains active in sampled zero-RL batches.
3. **Plausible, unverified:** repeated ungated OPSD updates, including on failed actions, contributed to the systematic formatting drift. The temporal association and dominance are not a causal intervention.
4. **Plausible, unverified:** proposal repetition, sparse difficulty rewards and fixed-base policy drift contributed to broader instability. No numerical estimator or parser correctness defect has been demonstrated.

A's completed result is usable as a negative development result with these limitations. Recommend a conditional staged B under the original protocol after review; B is the cleanest next system-level comparison for gating. A GRPO-only control is useful if B also regresses or causality remains ambiguous; it needs a separate study and authorization. New prompt/parser tolerance, changed proposal representation, a KL penalty or altered rewards change the scientific treatment and require a new study and affected reruns. Do not treat the double-unescape counterfactual as a correction to the frozen A–D protocol.

W&B logging warnings contain 72 `CommError` and 14 `OSError` entries; the locally fetched warning events do not identify a more specific network, authentication or service cause. The successful CPU-node online reconciliation confirms recovery, not the original outage cause.

## Provenance and classification limits

The run used frozen execution `54f087fdc5cf5170800d825bf1ee92b89a43c8ed`, run ID `a87f3772b284d35a375cf69e` and logical W&B ID `9c125e660c942541076d5d50`. The two fetched event segment SHA-256 values are `aa30982b11e78aae727cdca5706e046be14d54e54287f1f42b83315967982479` and `bc5972669d317cdbadc4d0b1e7280148d7b87c85affac488144ebba3689c528e`; the manifest file SHA-256 is `934412b2181ead4377d383130a19ad0d58eaab6bb9f74c5d8005416d09412fd2`. Archived trajectories and reports are kept under the local ignored `runs/` tree and the Yale durable run directory. The generator reads active lineage, excludes incomplete evaluation sets, and reports missing shards and checkpoint-payload availability.

Its coarse failure labels use this precedence: public pass/hidden fail, double-escaped rejected patch, other rejected patch, invalid action, accepted but unsuccessful edit, budget termination, then unknown/other. These labels identify observed episode features, not necessarily the first causal mistake; an episode can exhibit multiple symptoms. The episode action counts, finish status, test outcomes and final-source hashes remain available per task and checkpoint in the local JSON output. No raw source or trajectories are included in this report or W&B reconciliation.

Inspection of all 2,777 stored proposals in the four largest rejection categories confirms each parses as a JSON object and begins with both expected diff headers. All 730 hunk-count failures, 317 context failures and 1,103 duplicates have actual line breaks and no literal backslash-`n`; one of 627 invalid-hunk/extra-file failures contains a literal backslash-`n`. None of those outputs reaches the 512-token limit. The primary observed proposer issues in these categories are hunk structure/context and repeated accepted mutations, not JSON decoding or missing headers. The labels follow actual validator results, not subjective inference about semantic novelty.

The 106 uniform-only steps are an exact lower bound on zero-current-RL steps. The five mixed-group iterations were 2, 4, 42, 53 and 57; their summed gradient could still cancel or be zero, and exact decomposition was not saved for them. Hence the total is between 106 and 111, not a measured 106 exactly. Student AdamW momentum may preserve contributions from earlier mixed updates even when the current RL gradient is zero.
