# 01 — Experiment implementation plan

Source: `docs/task/01-experiment-implementation-spec.md`. Approved in conversation
on 2026-10-21. Implement every hardware/launch profile; first production validation
target is Yale on two 80 GB GPUs. Available validation initially is local ARM macOS.

## Architecture

One typed, installable `selfplay_opsd` package. Pure objective functions and immutable
public/private records have no framework/environment dependencies. Application
workflows depend on narrow policy, sandbox, dataset, persistence and logging ports.
Strict configuration composes method, hardware, execution and logging profiles.
Student-visible exports are explicitly allowlisted. Local logs are authoritative.

## Ordered implementation

1. Establish uv-managed pinned CPU/CUDA environments, Docker trainer/worker images,
   CI, strict configuration, provenance, and upstream compatibility evidence.
2. Build data adapters, family/split manifests, frozen mutation/reference tasks,
   mutation validators, four-tool repair environment, isolated execution backends,
   and base evaluation. Preserve EvalPlus special oracle semantics.
3. Implement exact population-std GRPO and detached signed-gap objectives; preserve
   original generated IDs and explicit student/teacher causal-position mappings.
   Complete A/B iterations, then C/D all-LoRA-coordinate CE-gradient alignment.
4. Add atomic checkpoints, intermediate artifacts, rollback lineage, independent
   optimizers/RNG histories, local-first W&B, resource accounting, health monitoring,
   every requested hardware profile, and Yale/AWS launch/resume commands.
5. Add frozen evaluations, isolated counterfactual updates, parent-paired bootstrap,
   plots/reports, CLI, exact operating commands and completion evidence.

## Scientific invariants

A/B differ only in gating. C/D share the same canonical edit-action CE reference;
only candidate RL versus RL+gated-SD gradients differ. Four parent groups, four
attempted proposals each, four student episodes per valid task. Invalid proposals
remain in proposer updates. All valid tasks train the student. One optimizer step
per mathematical batch. No teacher gradients, replacement trajectories, constrained
sampling, extra PPO epochs, public-test reward, silent filtering or hidden retries.

Default Qwen3-4B-Instruct-2507, BF16, independent rank-32 adapters, seed 2026,
200 iterations, checkpoint every 10, dev evaluation every 20, offline W&B. Pilot
settings must be training-only, recorded, frozen, and reset before the main runs.

## Recovery and verification

Stage scientific records until both updates commit. Restore both optimizers,
adapters, RNGs, duplicate histories, counters and manifest identities atomically.
Rollback supersedes abandoned scientific events but retains consumed compute.
Reuse intermediate artifacts only when snapshot/input hashes match.

Test independent math/finite-difference results, multi-turn token mapping, isolation,
data leakage, deterministic mutations/splits, all A–D iterations, microbatch and
distributed equivalence, fault-injected resume, W&B offline/outage behavior, and
paired analysis. CPU fixtures cannot establish GPU or actual sandbox readiness.

## Acceptance and deliverables

Local correctness → actual sandbox checks → target GPU integration → ten-iteration
pilot including C/D → frozen full-run readiness. Implement all requested cases;
report unavailable validation separately. Never fabricate results or runtime
projections. Do not provision, publish or launch the four main runs automatically.
Create `docs/task_completion/01-implemented-experiment.md` with exact evidence,
deviations, remaining blockers, commands and readiness status after the work.

Existing `.gitignore` changes and unrelated files are preserved. Documentation is
currently ignored by git; files are delivered locally without staging user changes.
