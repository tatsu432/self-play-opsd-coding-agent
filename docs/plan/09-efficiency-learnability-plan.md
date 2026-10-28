# Task 09 — Efficiency and scientific-health implementation

Source: [Task 09 specification](../task/09-efficiency-learnability-spec.md). The
scientific authority is [Task 01](../task/01-experiment-implementation-spec.md).

## Baseline and boundaries

Start from `edbc89a` on a dedicated branch. Keep the Task 08 frozen execution
`983a6cf`, study `repair-selfplay-2026-protocol-v2`, protocol and data intact.
The new code uses `repair-selfplay-2026-efficiency-v1`; its mathematical settings
match Task 08. The user authorized Task 09 GPU readiness work within the shared
allowance; main A–D submission remains separately unauthorized. Reuse unchanged
worker and data proofs.

## Implementation sequence

1. Read Task 08 raw events and scheduler accounting. Report phase wall time,
   overlap, useful yield, learning signal and conditional cost. Keep queue delay
   separate. Investigate the large old-policy log-probability outliers on H200.
2. Skip C candidate backward when all four rewards are equal, retaining exact
   neutral alignment and diagnostics. Keep D unchanged. Add a bounded proposer
   `generate_many` RPC with per-request seed, prompt, budget and sampled log-probs;
   default to sequential generation pending H200 validation.
3. Expose hardware-only switches for scoring microbatch, activation checkpointing,
   prefix caching and CUDA Graphs. Compare each on fixed low-yield and high-yield
   workloads. Accept only numerical and end-to-end wins; do not change sampling,
   rewards, objectives, budgets or evaluation endpoints.
4. Preserve transactional checkpoints, two transient retries, and bounded clean
   continuation. Document manual `status`/`fetch`/`resume` after a clean stop;
   no speculative login-node daemon. Keep offline events authoritative and
   optional W&B delivery best effort. Add a local scientific-health report.
5. Run local checks and bounded H200 benchmarks within the shared allowance.
   On one H200, cap the benchmark stage at four allocated GPU-hours; measure
   cold/steady generation, scoring, memory and a full iteration. Recompute
   projections and stop margins. Revalidate changed backend/runtime/checkpoint
   dependencies, freeze the new study, and obtain four controller previews
   before any new-study launch claim.

## Acceptance

CPU tests cover slot order, seeds, sampled log-probs, zero-gradient behavior,
objective parity and recovery. H200 checks must cover adapter synchronization,
sleep/wake, long histories, LoRA versions, numerical and optimizer parity,
memory fit, low/high-yield throughput and a full iteration. Report unavailable
measurements as unmeasured. Review scientific health after iterations 5–10 and
about 20; zero early successes alone never trigger an automatic stop.
