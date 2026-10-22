# 03 — Establish readiness for the A–D study

Implement the approved experiment-readiness plan. Use Yale `cpsc4880` only;
the old AWS aliases are unavailable. Establish actual runtime isolation, image,
data-validation, backend, base-evaluation, pilot and study-freeze evidence before
declaring A–D eligible for submission. Stop before main training.

The readiness budget is 24 allocated GPU-hours across all attempts. Prefer the
two-H100 profile; consider H200 only after a measured H100 memory-fit failure
and within the remaining budget. Preserve 200 main iterations, seed 2026 and
common scientific settings. Never weaken worker isolation or silently reduce
the scientific workload. Report an external blocker if Yale cannot enforce the
required isolation or the readiness budget is insufficient.

Complete artifact retrieval, remote freezing, idempotent preparation, clean-stop
continuation, immutable resume identities and trustworthy readiness reporting.
Use Docker/uv for images and offline W&B. Test failure paths and run the bounded
live readiness sequence, preserving exact evidence and resource accounting.
