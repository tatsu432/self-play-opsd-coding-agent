# 06 — Make A–D ready for main submission

Approved in conversation on 2026-10-22; source:
`docs/task/06-close-yale-launch-blockers-spec.md`.

1. Add trusted allocation diagnostics and five-minute one-/two-H100 checks.
   Join NVIDIA UUIDs/minors and verify device access enforced by Slurm against
   the requested count and live allocation; reject ambiguity/extra access.
   Select verified UUIDs and check actual container CUDA identity/occupancy.
   Publish raw allowlisted diagnostics before resolution, including failures.
2. Collect CPU-only worker diagnostics: runtime/systemd versions, owned bus,
   delegated controllers, cgroup ancestry and namespace/mount configuration.
   Update YCRC ticket 79403; use supported local delegation, or an existing
   approved Yale worker service. Verify actual limits and descendant cleanup.
   Repeat acceptance only after documented setup/implementation changes.
3. Extend preparation with pending/running/passed/stale/failed gate status,
   failure categories, job/evidence identity and concrete next actions. Reconcile
   active/uncertain submissions and accounting on restart. Retry supported
   transient failures only; repair application failures before another attempt.
4. Finish implementation and local/hosted CI before expensive acceptance.
   Run allocation checks, single/dual backend acceptance, worker/runtime data
   validation, base evaluation, ten-step D pilot with iteration-2 continuation,
   A/B/C fresh one-iteration acceptance, freeze and readiness. Reuse compatible
   verified images; never initialize main from acceptance checkpoints.
5. Finish only when live doctor passes all eight gates and all four main previews
   pass. Publish measured runtime/cost and exact monitoring/recovery commands,
   then the completion record. No main submission is part of preparation.

Validation covers divergent indices/minors, remapped masks, missing/extra device
access, malformed inventory, occupied devices and early diagnostic publication;
worker enforcement/cleanup/delegation; stale identity, duplicate prevention,
controller restart, uncertain submissions, accounting and method/resume paths.

Defaults: Yale/cpsc4880 only; cumulative 24 GPU-hours (0.03 already consumed at
initial inspection); allocation checks reserve at most 0.25 combined. H100 first;
H200 only after proven managed CUDA OOM. No new infrastructure. If neither local
delegation nor an existing approved Yale service is available, retain an explicit
incomplete external dependency and continue support resolution without rerunning
equivalent failed probes or declaring completion.
