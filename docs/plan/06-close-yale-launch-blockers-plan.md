# 06 — Make A–D ready for main submission

Historical Task 06: conflicting isolation mechanism, repeat-check and H200 rules
are superseded by Task 07 (`docs/task/07-fix-ycrc-issue-spec.md`).

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

## 2026-10-27 continuation after YCRC response

YCRC ticket 79403 now confirms that users cannot configure cgroups on Bouchet
and that Slurm does not currently provide a process-count limit. YCRC recommends
Slurm job arrays or job steps for per-task CPU/memory limits and Apptainer for
filesystem/network isolation. This does **not** satisfy the existing per-worker
32-process requirement by itself. Do not run candidate code under this partial
contract, silently remove the limit, or treat a per-user `RLIMIT_NPROC` as a
per-worker cap.

1. Record the response in the runbook and operational checkpoint. Keep
   generated-code execution blocked pending full enforcement.
2. Evaluate this no-cgroup path, contingent on YCRC confirming standard
   Apptainer subordinate UID/GID mappings are available and allowed: start each
   worker in a distinct `--fakeroot` mapped UID, launch it as a one-CPU/1-GiB
   Slurm job step with `--gres=none`, set a hard and soft `RLIMIT_NPROC=32`,
   drop all capabilities and set `no_new_privs`, and use seccomp to deny
   namespace/UID changes that could raise or evade the limit. The UID allocator
   must prevent simultaneous workers from sharing a mapped UID. Validate
   fork/vfork/clone/clone3 and thread limits, attempts to raise limits or change
   IDs/create namespaces, network/filesystem/GPU isolation, and
   timeout/cancellation cleanup. Reject per-user limits shared across workers
   or host processes. Require adversarial local validation and a trusted
   allocation preflight on Bouchet before generated-code use. If any prerequisite
   fails, reject this route and keep the worker gate blocked.
3. If a compliant mechanism cannot be proven, ask YCRC concisely whether any
   existing approved service or site-supported mechanism provides the missing
   per-worker process bound. Keep the current isolation contract until YCRC or
   the user supplies an approved complete solution.
4. In parallel, reauthenticate through normal Duo MFA, reconcile the existing
   H100 diagnostic job and restart its watcher only after confirming job state.
   Continue trusted backend checks within the cumulative 24 GPU-hour cap. Do not
   use H200 without the explicit exception or a genuine H100 CUDA OOM after
   ownership and initial-free-memory checks.
5. Once GPU acceptance and a fully compliant worker path both pass, proceed
   through runtime/data validation, base evaluation, D stop/resume pilot, A/B/C
   acceptance, freeze and four main submission previews. Main training remains
   a separate authorization gate.

YCRC source response: ticket 79403, email thread `1a1640b41863817b`, message
`1a17e27e17b5f49f`, dated 2026-10-27. Slurm documents job-step resource
requests at <https://slurm.schedmd.com/srun.html>; Apptainer documents that
per-container `--pids-limit` relies on cgroup delegation at
<https://apptainer.org/user-docs/master/cgroups.html>. These sources support
the investigation; they do not establish that the process-limit gate passes.
The concise mapping/policy question is in the same thread, message
`1a17e39dc4e11cbf`; it has not yet received a response.
