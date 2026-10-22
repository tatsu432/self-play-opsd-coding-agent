# 05 — Safe, efficient Yale A–D readiness

Implement the user-approved Task 05 plan. Finish with a frozen study and four
validated submission previews; do not submit main training. Resume the cumulative
Task 03 allowance of 24 allocated GPU-hours on cpsc4880, H100 first and H200 only
after a recorded H100 memory failure. Task 04's CPU-only restriction applied to
that repair and does not prohibit this newly authorized readiness work.

Contact YCRC using the approved support draft. Resolve worker execution only
through a supported site setup retaining every isolation property. Stop equivalent
failing probes until a documented setup change. An existing approved Yale managed
service is a contingency, not authorization to provision infrastructure.

Implement a Dockerfile-derived Apptainer trainer build on Yale CPU compute nodes,
with a bounded small-image feasibility check first. Keep Dockerfile/uv.lock as the
authoritative inputs, verify conversion semantics and artifact provenance, and
avoid large local archives. Preserve the Docker archive route.

Add full-workload one-iteration A/B/C acceptance runs with fresh initialization,
separate artifacts, verified checkpoints and objective-specific update evidence.
Retain the ten-iteration D pilot and iteration-2 continuation proof. Require all
evidence before freeze/readiness. Preserve immutable identities, budget accounting,
duplicate prevention, isolated generated-code execution and resumable main runs.

Run CPU regressions and hosted correctness CI before GPU acceptance. Produce a
launch runbook and completion record distinguishing actual acceptance from
external blockers. Do not buy storage, delete shared state, weaken safety, or
silently change scientific settings or the GPU allowance.
