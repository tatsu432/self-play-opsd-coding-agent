# 05 — Establish safe, efficient readiness for A–D

Approved in conversation on 2026-10-22. Source:
`docs/task/05-yale-experiment-readiness-spec.md`.

1. Send the authorized YCRC request with jobs 30217591/30218129, exact worker
   invocation, required isolation and a question about CPU image-build support.
   Record message identity. Follow up after three business days if unanswered;
   rerun worker acceptance only after a documented setup change.
2. Prove source-to-SIF builds in one 2-CPU/4-GiB/15-minute allocation, then add
   `hpc images build --kind trainer --build-mode yale`. Convert the Dockerfile
   with pinned uv-managed tooling; validate instruction semantics, context paths,
   immutable inputs and hashes. Build on 8 CPUs/32 GiB/up to four hours with
   150 GiB scratch headroom. Publish SIF/manifests atomically after package/CLI
   verification. Retain `--build-mode docker`; do not create a local large archive.
3. Finish code and CPU/hosted CI before collecting GPU evidence. Run independent
   one-H100 and two-H100 backend probes. Once worker isolation passes, validate
   target data, evaluate base, run the ten-step D pilot with stop/resume at 2,
   and run one full iteration each of A/B/C with fresh initialization. Smoke
   outputs are separate and never initialize main runs.
4. Budget initial allocation ceilings: reference 1, target 2, base 4, D pilot 12,
   A/B/C combined 4 GPU-hours, leaving 1 for retries. Actual accounting releases
   unused reservations. Preserve the cumulative 24-hour cap and pilot 12-hour cap.
   Use H200 only after managed H100 CUDA OOM and fresh target proof. Never shrink
   workload or retry scientific failures to obtain passing evidence.
5. Require per-method smoke evidence at freeze/readiness and four live-validated
   main previews. Preserve seed 2026, 200 iterations, common settings, fresh
   initialization, offline W&B and allocation-time isolation. Fetch evidence and
   write exact launch/recovery commands, measured projections and completion.

Verification: conversion semantics, context/provenance tampering, interrupted
publication/reuse, duplicate builds, method-specific updates/checkpoints, stale
evidence, strict sandbox gates and shared budget; then the hosted CPU workflow.

External stops: unsupported build capability, unsupported worker isolation or
insufficient readiness budget. Record exact evidence and required next action;
never declare A–D ready while a gate is missing. No additional host provisioning,
main training, local cache deletion or storage purchase is authorized.
