# Task 13: measured diagnosis and efficient recovery experiments

## Completed CPU investigation

The [diagnosis](../investigations/learning-stagnation-diagnosis.md) answers Q1–Q8 using all corrected A/B/C committed records. `scripts/diagnose_learning.py` reconstructs both logical eligibility and actual validation order, proposal reward counterfactuals, repetition, recurring-parent comparisons, student behavior/OPSD strata, uncertainty-sensitive group-size projections, and exposure/cost curves. Outputs are in the ignored `runs/task13/` workspace. The required completion record lists checks and evidence.

The evidence changes the priority: exact diff application rejects 51–53% of proposals; accumulated duplicates reject another 31–38%. Real proposer rewards materially change normalized advantages in only 4/800 A, 5/800 B, and 7/400 C groups. First improve useful candidate generation, then test student learnability; do not start full D or extend C unchanged as the next diagnostic.

Continue the GPU experiments below through submission, monitoring, analysis, and subsequent decisions. The user authorizes efficient GPU spending without another budget confirmation. Twelve H200-hours is an initial envelope, not an obligation to spend it or an absolute ceiling. Reconcile jobs and actual allocations before every submission, retain all failed-attempt costs, and never duplicate an uncertain submission.

**Probe implementation update:** the `task13-probe` operation launches the paired base-model arms through the existing Yale worker and GPU allocation checks. Both completed successfully: diff 43/160 verified proposals and 20 unique mutations; structured 95/160 and 33 unique. Paired-parent bootstrap differences were +32.5 percentage points in verified fraction (95% interval +15 to +50) and +0.325 unique mutations per parent (+0.025 to +0.625). Continue with the retained checkpoint comparisons and additional independent samples.

## Experiment 1: paired inference probe — run first

**Hypotheses:** deterministic edit construction removes a large representation burden; trained proposer repetition is partly policy concentration and partly novelty history; larger models may improve useful yield, but representation could make scaling unnecessary.

**Frozen parents and checkpoints:** select 40 parents from `protocol-v3` training only by sorting `digest([2026, parent_id])`, before inspecting new probe outcomes. Use the first 16 as the eight-proposal subset. Record IDs, sources' hashes, selection algorithm, and model snapshots in an immutable private probe manifest. Do not use development/final/reference parents as generation inputs. Use base 4B plus corrected A checkpoint 20 and 200, B 200, and C 100. Remote inventory confirms these retained checkpoints; A/B checkpoint 100 no longer exists and must not be requested. Verify checkpoints on an allocated node using the existing project verifier before loading.

**Stage 1a, 4B controls, initial cap 2 H200-hours:** use existing diff prompts for base/A20/A200/B200/C100, four proposals per parent: 800 outputs. Also run base and B200 with structured edits, four per parent: 320 outputs. Add four independent proposal slots for base/current-diff, base/structured, and B200/current-diff on the preselected 16-parent subset: 192 additional outputs. Total planned maximum is **1,312 outputs** for this stage. Run model variants sequentially to avoid duplicated residency. If it cannot finish within the cap, save the completed canonical prefix and identify missing paired cells; extend only if the unfinished comparison can change the decision.

**Representation:** structured JSON identifies an AST field path and a transformation from a frozen vocabulary: integer constant ±1 and compatible arithmetic/comparison operator replacements. Enumerate legal sites without executing source, apply one specified transformation to a copied AST, deterministically render and diff, then run the unchanged validator and accepted isolated semantic verifier. Preserve imports, API, semantic mismatch requirements, and worker limits. Reject nonexistent paths/operators. Report the narrower action space and its coverage; this is not a full mutation-space enumerator. No grammar-constrained decoding in the first comparison, so representation is the intended changed variable. Defer full-source/fragment generation unless the finite vocabulary proves too restrictive.

**Generation controls:** temperature 1, top-p 1, unrestricted top-k, keyed seeds `digest([2026, parent_id, slot])`, the current 512-token proposal ceiling, and the same parent information and semantic checks. Record actual prompt and completion tokens, EOS/length termination, sampling configuration, revision, template, and mode. Keep no-feedback independent sampling across slots; do not retry invalid samples. The token ceiling is a common budget, not equal expressive capacity across tokenizers.

**Novelty control:** score every output with empty probe novelty history and separately against the stored accepted-history snapshot for its checkpoint. Report within-group, cross-probe, and prior-training duplication separately. Never alter historical acceptance or curriculum state. The checkpoint comparison is paired by parent; it is not a paired training-seed experiment.

**Stage 1b, scaling, additional cap 2 H200-hours:** if 1a leaves representation versus model capability unresolved, add Qwen2.5-Coder-7B-Instruct and Qwen3-8B, using both formats, four outputs per parent: 640 outputs. Resolve and record immutable revisions before launch, assess inference support in the pinned image, and explicitly set Qwen3-8B non-thinking in its chat template. Do not modify the frozen trainer image/lock or attempt a 9B trainer port. If a model needs a different inference image, give that arm a separate pinned environment identity and report the confound.

**Metrics and uncertainty:** format/applicability, structural eligibility, semantic validity before novelty, unique verified tasks after novelty, AST site/operator diversity, identical-token and identical-AST repetition, and allocated GPU-hours per unique useful task. Use 2,000 paired-parent bootstrap resamples for differences. Compare equal-output cells and compute-matched prefixes. Include model loading/startup in allocation cost; keep validation-only and generation timing separately. Posterior probabilities for additional student rollout groups require actual repair sampling, not proposer outputs.

**Decision:** prefer 4B structured generation if it improves unique verified yield per GPU-hour with an uncertainty range that supports the added implementation cost. Expand paired sampling if uncertainty still spans choices worth pursuing. Scale only if a larger model provides a practical additional gain after controlling representation. If fixed-parent syntax remains stable but history-scored novelty falls, focus on diversity/curriculum rather than model size. If neither intervention improves useful yield, stop adding model arms and move to the static student diagnostic.

## Experiment 2: static student learnability

**Hypothesis:** the student can improve correct repair behavior when proposer rejection and novelty exhaustion are removed. Compare Static-GRPO with Static-Gated on identical tasks. A learned-proposer GRPO comparison comes later, only if this test establishes a useful control.

**Execution update:** `task13-static` now runs both arms sequentially in one bounded six-hour H200 allocation. It constructs and verifies a shared fixed pool, screens each task with eight rollouts, restores the same initial adapters, optimizers and RNG before each arm, and uses the existing student trainer with empty-token proposer placeholders. Its study identity binds the pool and runner hashes; proposer metrics on those placeholders are not meaningful. The scheduler stop marker is separate from corrected main-run state. Check actual completion and exposure for each arm before interpreting the comparison.

**Pool:** construct 32 verified training-split repair tasks, at most one per parent, using the existing deterministic mutation machinery and/or unique admitted training tasks. Preserve original sources, inverse repairs, public cases, private semantic verifier, and admission limits. Use a training-only eight-rollout base screening pass per task under the same repair protocol. Freeze the task manifest and choose a training schedule with up to half the tasks exhibiting mixed outcomes; if too few are mixed, report the actual count and do not screen indefinitely for a favorable pool. Limit pool construction to four deterministic mutation attempts per candidate parent. Do not use dev success to select training tasks.

**Matched arms:** reset to the same base student adapter and RNG initialization; disable proposer sampling and updates. Each optimizer batch uses two fixed tasks with four rollouts each, cycling a frozen schedule. Share all rollout budgets, optimizer settings, sequence weighting, and evaluation rules; set lambda zero for GRPO and 0.1 with the existing gate for Static-Gated. Token-level behavior probabilities retain their original meaning. Task groups remain separate for GRPO normalization.

**Budget and reviews:** initially reserve up to three H200-hours per arm, including screening/evaluation attributed explicitly rather than hidden. Stop after 20 actual optimizer steps, with reviews at steps 5/10/20; extend to 40 only for unresolved learning dynamics. For GRPO, also cap attempted batches at 80 so uniformly rewarded tasks cannot cause an unbounded wait for optimizer steps. Report raw batches, task exposures, optimizer steps, mixed groups, tokens, and allocated hours; arms need not reach identical numbers before a cap. Compare common exposure/step prefixes rather than pretending all budgets can be exactly matched. A plausible initial workload is 160 repair episodes per completed arm plus screening; timings must be measured in the first five batches because current self-play yield is not an all-valid throughput benchmark.

Evaluate the same frozen training probe and existing 41-task monitoring development set at initialization and steps 10/20, optionally 40. Evaluation is greedy and separate from training. No final-ID/external evaluation is used for selection. Track verified repairs, successful patch application, JSON/tool validity, test use, current RL gradients, signed OPSD coefficients, and actual parameter displacement on the bounded pilot. If both arms fail, run a separately labeled tiny canonical-edit CE optimization control on four training tasks for at most ten steps, restoring initialization afterward; success tests the optimization path, not generalization.

**Interpretation:** Static-GRPO improvement with useful signal supports a curriculum/proposer bottleneck. Gated success with GRPO failure supports an OPSD/feedback contribution but does not isolate reward sparsity from optimization automatically. Failure of both requires protocol, capability, verifier, and optimizer inspection; a short negative pilot is not a definitive impossibility result. If both learn, isolate gating strength and proposer rewards next on a shared task schedule. Use a single predeclared matched-strength ungated control based on training-only coefficient/norm measurements, not a broad lambda sweep.

## Experiment 3: C/D matched-gradient counterfactual

**Hypotheses:** combined RL+gated OPSD supplies information on uniform student groups; the current proposer grouping may still erase it.

Use retained corrected **C checkpoint 99** as a single shared snapshot, preserving both adapters/optimizers and RNG. Its immediate iteration-100 proposal set contains no admitted tasks, so existing iteration-100 records alone cannot provide the required candidate comparison. Generate fresh independent same-parent proposal groups at this snapshot using the frozen original format and sampling controls. Draw from the experiment-1 training-parent manifest, with at most 16 groups / 64 proposal attempts and at most 16 verified unique candidates. Do not manufacture two-valid groups by mixing parents or borrowing candidates from other policy snapshots.

For each accepted candidate sample four student rollouts once. Reuse these exact sequences for RL-only and combined candidate gradients, using production behavior probabilities, lambda/gate, coordinate order, and the same frozen coding reference gradient. Measure norms, zero/near-zero states, cosine, rewards, normalized proposer advantages, rankings, and proposer gradient cosine where defined. Recompute sampled token scores at the fixed snapshot without silently overwriting behavior log probabilities. Include invalid slots at −1 and preserve their original groups. Single-valid and multiple-valid group results must be separate.

Cap at **two H200-hours** initially. Use up to four groups with advantage differences above 1e-5 for actual counterfactual Adam displacement, restoring parameters, optimizer, gradients, and RNG before each branch and after the diagnostic. Hash state before/after; never write into main checkpoints or commit training iterations. If no multi-valid/discriminating group appears, that is the result of the bounded on-policy probe. A separate explicitly synthetic all-valid grouping may explain mathematical possibilities, but cannot substitute for observed groups.

Proceed toward a bounded D training pilot only if richer scores survive group normalization and produce meaningful update differences, with protocol/gradient integrity intact. If scores differ but advantages and update directions remain effectively the same, prioritize a new grouping/objective study. If combined gradients mostly capture teacher format preference without useful behavior evidence, keep D deferred. Reference-domain experiments remain lower priority.

## Execution interfaces, resource requests, and integrity

The CPU analysis interface is implemented and executable now:

```bash
uv run --frozen python scripts/diagnose_learning.py \
  --run A=runs/hpc/yale-hpc-cpsc4880/747bd2e097258445de647e47 \
  --run B=runs/hpc/yale-hpc-cpsc4880/f3271727f5fd25c07466e50d \
  --run C=runs/hpc/yale-hpc-cpsc4880/35bfbd1bc074a8963e916f77 \
  --accounting runs/task13/accounting.json --output runs/task13/diagnosis
```

The following interfaces include proposed routes that still require implementation. Implement and test each scoped route and continue to submission; do not substitute the existing unrelated `diagnose` operation:

```bash
# Implemented: stateless inference and validation, diff or structured arm.
uv run selfplay hpc submit --operation task13-probe --settings configs/task13-probe.yaml --profile single_gpu_80gb --gpu-type h200 --account cpsc4880 --dry-run
# Implemented: matched GRPO and gated arms on one fixed training-task pool.
uv run selfplay hpc submit --operation task13-static --method B --settings configs/task12-corrected.yaml --profile single_gpu_80gb --gpu-type h200 --account cpsc4880 --dry-run
# Implemented: read-only C checkpoint counterfactual.
uv run selfplay hpc submit --operation task13-counterfactual --method C --checkpoint-iteration 99 --settings configs/task12-corrected.yaml --profile single_gpu_80gb --gpu-type h200 --account cpsc4880 --dry-run
```

The focused runners and controller/allocated-job routes are implemented. The fixed-task runner uses the existing GRPO objective while bypassing proposer generation and updates; no global A–D readiness gate is weakened. New operations bind immutable inputs and explicit caps. On a supported preview, submission removes `--dry-run`; no extra user budget permission is required. Larger-model inference support and the additional eight-sample proposer subset still require implementation.

Use one entitled H200 in `gpu_h200`, account `cpsc4880`, seven trainer CPUs/128 GiB plus the accepted one-CPU/1-GiB no-GRES worker step: eight CPUs/129 GiB total. H200 is chosen for the already accepted backend and measured approximately 72-GiB whole-device workload/headroom, avoiding new hardware uncertainty. Stage-1a/1b allocations are each at most two hours, static arms three hours each initially, counterfactual two hours. Reserve full requested time and count actual scheduler use, including failures and startup. Derive safe stop margins from measured initial phases and preserve partial diagnostic outputs; training stops only at joint checkpoint boundaries.

New configuration studies are `repair-diagnosis-2026-probe-v1`, `repair-diagnosis-2026-static-v1`, and `repair-diagnosis-2026-counterfactual-v1`. Pin models, data, prompts, code, images, and sampling in each manifest. Reuse existing isolation and image evidence by dependency; do not run model, image hashing, or generated code on login nodes. For a scientific code/objective change, require scoped objective/adapter/checkpoint validation and a new identity. Broader 8–9B training, full A–D reruns, and final evaluation are not automatic consequences of a positive probe.

Highest-value observability additions for new studies are the three per-iteration proposer aggregates already exported offline: groups with 2+ accepted candidates, reward-discriminating groups, and advantage distance from validity-only rewards. Existing student mix, alignment, gaps/gates, and sampled gradients remain sufficient until a specific uncertainty calls for more instrumentation. Never upload raw oracle source, hidden cases, or trajectories.
