# Task 10 — reliable online main launch

Implement the accepted Task 10 plan without submitting main A–D jobs.

1. Audit rollout/trainer probability semantics and existing replay evidence;
   quantify RL, OPSD and C/D implications using saved tokens and snapshots.
   Keep the behavior-policy denominator absent a demonstrated correctness bug.
   At most one additional one-hour H200 replay, after ledger reconciliation.
2. Harden the existing W&B sink: aggregate-only allowlist, checkpoint-backed
   publication, stable logical runs, bounded delivery/retry, local replay and
   canonical lineage tables for rollback. Add verification/reconciliation CLIs.
3. Verify credentials without disclosure, CPU-compute HTTPS and cloud readback,
   restart continuity, and a dashboard covering the Task 10 metric inventory.
   Credentials, raw contexts, private oracles and checkpoints stay off W&B.
4. Create an online operational configuration/study, reuse unchanged evidence
   through exact dependency checks with provenance, bind the measured plan,
   freeze on CPU and preview all four methods. Preserve Task 09's freeze.
5. Validate focused tests, scientific-output/RNG invariance and hosted CI.
   Document actual evidence, limits and executable setup/launch/recovery commands.

Keep one H200/method, 200 iterations, seed 2026, per-iteration checkpoints,
23-hour segments, stop leads 4616/4540/4717/5211 seconds, clean continuations
5/5/5/6, two transient retries, unchanged objectives/data/prompts/sampling and
conservative throughput settings. Review A first at iterations 5, 10 and 20
after explicit main authorization; B/C/D require subsequent authorization.
Scientific underperformance prompts review, never an arbitrary automatic gate.

Acceptance requires cloud-visible metrics and resume, trustworthy canonical
lineage, failure isolation, no secret/raw-text upload, a valid freeze and four
controller previews. Missing authentication must be reported as a real blocker;
local/offline tests do not establish live W&B or target-hardware readiness.
