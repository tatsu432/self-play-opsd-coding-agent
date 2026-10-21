# Implemented methods and boundaries

The experiment specification in `docs/task/01-experiment-implementation-spec.md`
defines the study. This document describes the code, including its debug and
currently unverified production integrations. Repair scores are **MBPP+-derived
repair** and **HumanEval+-derived repair**, not official generation scores.

## Student update

For each valid task, sample four complete independent episodes at the frozen
student snapshot. Final fresh verification gives binary rewards. Within a task,
advantages use population standard deviation and epsilon `1e-6`; equal-reward
groups are explicitly zero. There is no cross-task whitening.

For sampled assistant target tokens, `rho = exp(current_logp - old_logp)` and
`RL = -mean_episode(mean_token(min(rho*A, clip(rho, .8, 1.2)*A)))`.
Episode token counts include all assistant action turns and sampled EOS. Forced
markers, prompts, observations, padding and privilege never receive loss.
An empty episode contributes zero, retaining its place in the group denominator.

The teacher is the same frozen student snapshot, scoring the **same target IDs**
with the original correct source and inverse initial patch in a teacher-only
system prefix. No teacher actions are generated. Per-turn scoring uses the full
actual student history, including wrong edits and failed tools.

`delta = stopgrad(teacher_logp - old_logp)`;
`d = clip(delta, -5, 5)`;
`w = 1` for A and `sigmoid(delta)` for B/C/D;
`SD = -mean_episode(mean_token(stopgrad(w*d) * current_logp))`.
The student loss averages `RL + lambda*SD` equally over valid tasks, initially
`lambda=.1`. Gate weights never change the denominator. Loss values can be
negative and are not cross-iteration quality estimates.

The implementation accumulates bounded turn microbatches with weight
`1 / (episode_assistant_tokens * episodes_per_task * valid_tasks)`.
This equals the specified episode/task normalization, including unequal lengths.
Exactly one accumulated AdamW step follows all scoring. An exactly zero gradient
skips the optimizer, including its momentum state. There are no extra PPO epochs,
entropy bonuses, critics, automatic KL terms or task-selection filters.

## Proposer and alignment

Four parents from a keyed common schedule receive four attempted proposals each.
Invalid proposals remain with reward -1 and no student episodes. No refill occurs.
Duplicates are owned by the first valid canonical proposal slot. All valid tasks
train the student. Proposer GRPO groups are per parent; its independent optimizer
receives one update over all attempted proposals.

A/B reward is `4*p*(1-p)`. C/D use `(1+cosine)/2` with the **same** reference CE
gradient. Reference CE averages each frozen canonical edit JSON action, including
JSON wrappers and excluding forced EOS. References are training guidance, not
validation. C's candidate gradient is RL only; D's is RL plus gated SD. Failed and
negative-advantage episodes remain in the gradient. Equal-reward groups can have
zero C gradient and nonzero D gradient.

Coordinates are all student LoRA parameters in sorted name order. Dot products
and norm squares use FP64 after FP32 gradient collection. Distributed scoring
reduces unique coordinate shards globally. Norms at or below `1e-12` yield neutral
reward .5 and explicit zero/near-zero flags; nonfinite values abort.

The optional `GRPO` method sets the SD coefficient to zero. An explicitly named
`GRPO_KL` variant additionally supports a positive `objective.fixed_base_kl`,
scoring the frozen adapter-disabled base at identical sampled target positions.
Its penalty uses the sampled k3 estimator `exp(ref_logp-current_logp) -
(ref_logp-current_logp) - 1`; this is not labeled exact full-vocabulary KL.
Neither optional method is a required fifth main run.

## Execution, data and recovery

Sampling uses temperature 1, top-p 1 and disabled top-k/min-p truncation and
repetition penalties. Evaluation is greedy. Original target IDs are retained;
explicit causal student/teacher position mappings are persisted. Teacher, rewards,
old probabilities, gates and advantages are detached.

The virtual workspace exposes exactly README, solution and public-test files.
Only the solution can change. Public `run_tests` executes the public case inputs
through the same isolated function-call supervisor used by final verification;
it does not run model-controlled pytest configuration. Read-only public test
source documents the public cases and comparator semantics. Every final
verification creates a new worker and copies only allowed source through bounded
JSON. Expected outputs stay in the trusted supervisor; candidate code sees its
current inputs. This is OS isolation, not a cryptographic test-secrecy guarantee.

Pinned datasets and original MBPP examples have SHA-256 checks. Correct source is
formatted with `ast.unparse` before mutations. Families use alpha-normalized AST
equality and conservative AST-token 5-gram Jaccard similarity >=.9, then connected
components. This does not detect every semantic duplicate. Exclusions, overlap,
reserved split counts, reduced coverage and fixed mutation attempts are recorded.

Full checkpoints contain both adapters/optimizers, RNG streams, curriculum history,
counters, selection state and compatibility hashes. An iteration commits only
after both updates. Intermediate proposal/episode caches are reusable only at the
same snapshot/config; slot order is restored explicitly after JSON serialization.
Rollback supersedes later scientific events while retaining physical compute.
Signals request a safe stop; interrupted partial updates are never checkpointed.

## Production implementation and validation limits

Production uses a thin verl `BasePPOActor` extension, pinned verl FSDP2 utilities,
and a separate vLLM rollout process. A shared frozen base has isolated student and
proposer adapters. Rollout and training alternate GPU residency; the teacher does
not add a third model. All FSDP ranks evaluate the same mathematical microbatches,
so averaged gradients equal that batch's objective; only coordinates are sharded.
This conservative baseline duplicates data compute rather than optimizing data
parallel throughput. GPU count changes do not change the mathematical batch.

Package compatibility is locked and upstream APIs/source were inspected. Actual
FSDP2/vLLM synchronization, memory fit and resharding require target-hardware
acceptance. `backend-check` saves a single-process gradient reference, checks two
post-update weight reloads, and verifies atomic state restoration. Multi-GPU
profiles require comparison against that reference before pilot readiness.
Yale's trainer container does not imply safe generated-code isolation: use a
verified local Apptainer worker or the authenticated remote Docker worker.

Primary evaluation is the final common-budget checkpoint; the secondary is the
best parent-macro monitoring development score, ties broken by earlier iteration.
Analysis uses 2,000 seeded paired parent bootstrap resamples. Counterfactual probes
restore both model and optimizer before each candidate and measure a separate
diagnostic-development CE set. Undefined constant-score correlations remain null.
One training seed cannot establish training-seed variance or general superiority.

## Deliberate operational choices

Generation is serialized on the coordinator; bounded episode threads overlap
independent tool execution. Results return in canonical proposal/episode order.
This preserves seeded generation and keeps distributed collectives out of tool
threads. Per-episode elapsed sums can overlap; allocated GPU cost uses outer wall
time and the declared reserved GPU count.

The pilot retains the configured common lambda and reports gradient influence;
automatic rescaling is disabled. A changed lambda requires an explicit configuration
change and a new matching pilot/freeze. Training-only canonical proxies provide
pilot alignment references, leaving reserved reference/development tasks untouched.

Optional W&B checkpoint artifacts contain inference adapters only. Full resumable
state and private oracles remain local. Upload delivery is best effort and does
not make a local checkpoint contingent on remote availability.
