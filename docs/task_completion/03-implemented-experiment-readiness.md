# 03 — Experiment readiness implementation and live result

Source: `docs/task/03-experiment-readiness-spec.md`; approved plan:
`docs/plan/03-experiment-readiness-plan.md`. Observed 2026-10-22.

**The readiness workflow is implemented. The actual A–D study is blocked at
Yale worker isolation and is not eligible for main submission. GPU consumption
under task 03 is zero of the authorized 24 GPU-hours.**

## Implemented workflow

- `hpc images build --kind worker|trainer|all` builds Linux AMD64 Docker
  archives, checks capacity and archive identity, transfers them resumably, and
  submits CPU-only SIF conversion. Published manifests pin archive, Docker image,
  SIF, source commit and dependency-lock hashes.
- `hpc prepare-study --gpu-hours-limit 24 --watch` advances through worker
  isolation, target data validation, one-H100 reference, two-H100 comparison,
  shared base evaluation, ten-step D pilot with a stop/resume at iteration 2,
  CPU study freeze, and four main dry runs. It stops on failed or stale evidence;
  main training is never submitted by preparation. A recorded H100 CUDA memory
  failure permits H200 fallback with its own pilot namespace and the same budget.
- CPU and GPU requests are checked against live per-node Slurm metadata.
  Reservations are locked before submission and reconciled to allocated GPU
  count and elapsed time; unknown accounting retains its reservation. The pilot
  has a separate 12 GPU-hour allocation ceiling. Setup infrastructure retries,
  transient training retries and explicit clean continuations are bounded.
- The host-side worker uses only standard-library Python and starts isolated
  Apptainer workers. It uses an existing user-owned D-Bus socket when available,
  generates an allocation-specific token, publishes complete reports atomically,
  and shuts down with the allocation. It does not create user services or alter
  site privileges. Each actual allocation rechecks isolation.
- Target execution validation writes outside the read-only frozen dataset.
  Shared scientific settings and runtime evidence are checked consistently by
  pilot, freeze, readiness, main launch and checkpoint resume. Freeze runs on
  Yale beside its source artifacts and rejects conflicting evidence.
- Resume/watch use the original persisted specification and immutable checkout.
  Joint checkpoint hashes and scientific/runtime identities are verified.
  Attempts have separate stop files; scientific completion requires the target
  iteration, not just Slurm COMPLETED. Main continuation bounds and deadline
  margins derive conservatively from the measured D pilot.
- `hpc fetch` follows actual operation output paths. `hpc fetch-study` preserves
  study/pilot/readiness layouts. Downloads retain local ownership and avoid
  applying Yale's group/setgid modes on macOS. Main previews require current
  freeze/evidence hashes before any allocation.

## Live Yale results

All jobs below used account `cpsc4880`, partition `day`, **zero GPUs**.

| Job | Operation | Result | Allocation elapsed | CPUs |
|---|---|---|---:|---:|
| 30217338 | First image conversion | Checkout gate rejected an operational symlink; fixed and regression-tested | 0 s | 8 |
| 30217522 | Worker Docker archive → SIF | COMPLETED, exit 0 | 60 s | 8 |
| 30217591 | Initial worker isolation | FAILED before worker startup: D-Bus session address absent | 12 s | 2 |
| 30218129 | Worker isolation with normal bus discovery | FAILED: compute-node user bus socket absent | 14 s | 2 |

Slurm allocation consumption is approximately **0.148 CPU-hours and 0 GPU-hours**,
excluding local builds/tests. There were no GPU backend jobs, base evaluations,
pilots, study freezes or A–D main runs. No additional Yale identity or AWS host
was used. Prepared data manifests and scientific hyperparameters were preserved.

Worker SIF SHA-256:
`76a43eef3424a6e72d516a73c967d0610e12c13288eba6cbd762d8635e860ec3`.
Docker AMD64 image ID:
`sha256:4fa7bbdb29217e128ab2e775296fbcda279aab6d5257750390a5d37c366a4c60`.
The image manifest is in the managed workspace at `images/worker.json`.

The last live worker probe ran committed code
`b5560f73ecc7bb7fb9510966e3e1d833693c386f` on
`c1104u05n03.mghpcc.ycrc.yale.edu`, kernel
`5.14.0-570.140.1.el9_6.x86_64`, UID 35094. Its node report confirms cgroups v2
and a user runtime directory, but **no user D-Bus socket**. Apptainer refused
container creation while applying resource limits:

```text
cannot use cgroups - DBUS_SESSION_BUS_ADDRESS is not set
```

The candidate process never started, so the per-worker isolation checks were
not established. They were not bypassed or weakened. Normal bus discovery did
not find a socket to use; it deliberately did not fabricate an address.
[Apptainer documents](https://apptainer.org/docs/user/latest/cgroups.html)
that non-root resource limits require cgroups v2 and delegated systemd control.

Artifacts were fetched successfully to
`runs/hpc/yale-hpc-cpsc4880/studies/repair-selfplay-2026/readiness/`, including
`worker.json` and `node-environment-30218129.json`. Immutable Slurm ledgers and
logs remain in the managed project workspace. Last-stage failures remain
auditable even when the current worker report is replaced by a later attempt.

The local filesystem had approximately **22 GiB free**, below the controller's
60 GiB trainer-build preflight minimum. The trainer image was not built after
the worker gate failed. Actual trainer compatibility and GPU memory fit remain
unverified.

## Verification

The full suite passed with 84 tests and 2 capability skips before the final
atomic-publication and download-permission fixes. The final 20 focused tests
passed, covering publication under concurrent readers, actual Docker service isolation,
budget races/accounting, bounded setup retries, original-specification resume,
freeze compatibility, CPU shapes, immutable checkout links and Slurm signaling.
Ruff, formatting, mypy and Bash syntax checks passed. The final download fix
was also verified by fetching the live Yale reports successfully.
No CPU/debug probe is presented as a production GPU readiness result.

## Required next action

Obtain a **YCRC-supported Yale execution setup** that provides per-worker
CPU/memory/PID limits under Slurm accounting, together with private networking
and filesystem isolation. Then rerun the CPU worker gate using that supported
setup. A site change requires fresh worker/runtime evidence; it must not be
replaced with a report-file edit or disabled probe.

Suggested support request (draft only; not sent):

> For course account cpsc4880_ts2236, CPU Slurm job 30218129 on Bouchet could
> not start an unprivileged Apptainer worker with per-container CPU, memory and
> PID limits. The node uses cgroups v2, but `/run/user/35094/bus` is absent;
> Apptainer reports that DBUS_SESSION_BUS_ADDRESS is unset. Is there a supported
> way to provide delegated controls inside the Slurm allocation, or an approved
> Yale execution service? We need non-root, network-isolated, read-only workers
> with 1 CPU, 1 GiB RAM, 32 PIDs and 64 MiB temporary storage, while preserving
> Slurm resource accounting and cleanup.

Also provide at least the trainer build's required free disk capacity. Once
both environment gates are resolved, rerun image/runtime preparation and let
the existing controller advance through the remaining measured gates within
the unchanged 24 GPU-hour cap. Keep main A–D launch as a separate action.
