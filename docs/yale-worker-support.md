# Yale worker delegation: support draft and fallback design

Task 05: the user authorized contact. The request was sent on 2026-10-22 from
the connected Yale account to research.computing@yale.edu, subject
“Bouchet cpsc4880: supported per-worker Apptainer isolation and CPU image builds”.
Gmail message/thread ID: `1a1640b41863817b`. It includes the exact worker
invocation and asks about Dockerfile-derived CPU builds as well as isolation.
YCRC's automated acknowledgement created ticket
[79403](https://help.ycrc.yale.edu/helpdesk/tickets/79403), with a normal response
target of 24 business hours. It also warns that Bouchet is urgently low on shared
storage; inspect capacity before a large build. This acknowledgement supplies no
technical approval or worker configuration.
Follow up on the same thread after three business days (2026-10-27) if unanswered.
YCRC has not approved or provisioned a managed service.

## Task 06 diagnostic update

CPU-only job **30293989** completed on `c1104u05n04.mghpcc.ycrc.yale.edu`.
Apptainer is `1.5.3-1.el9`, Slurm `25.11.8`, systemd `252-51.el9_6.6`.
The user bus is absent and the user manager reports “No medium found”.
All inspected cgroup controls in the job ancestry are non-writable by this user.
The `pids` controller is available at `slurmstepd.scope` but is not enabled in
its `cgroup.subtree_control`; job descendants expose only `cpuset cpu memory`.
Thus supplying a bus address alone cannot establish the required PID limit.

The authorized technical update was sent in the same ticket/thread, Gmail
message `1a164e8afe1c5056`. It asks for supported delegation retaining Slurm
accounting or the operator/contact for an existing approved Yale service. No
equivalent worker acceptance probe was repeated and no services were changed.
Full report: `runs/readiness/repair-selfplay-2026/worker-diagnostics-30293989.json`
on Yale; fetched under the corresponding local study readiness directory.
These diagnostics do not establish isolation or approval.

## Support request draft

Subject: Bouchet — supported per-worker resource isolation inside Slurm

For course user `cpsc4880_ts2236` (UID 35094), Apptainer worker probes in CPU
Slurm jobs **30217591** and **30218129**, account `cpsc4880`, partition `day`,
failed before candidate startup with:

```text
cannot use cgroups - DBUS_SESSION_BUS_ADDRESS is not set
```

The second probe ran on `c1104u05n03.mghpcc.ycrc.yale.edu`, kernel
`5.14.0-570.140.1.el9_6.x86_64`. The node report observes cgroups v2 and an
existing `/run/user/35094` directory, but no `/run/user/35094/bus` socket or
`DBUS_SESSION_BUS_ADDRESS`. We have not established controller delegation or
the site's `apptainer.conf` settings. Normal discovery of an existing owned
socket did not resolve the issue. We have made no privilege or service changes.

Can YCRC provide a supported way to enforce unprivileged per-container limits
inside the allocated Slurm job, or a Yale managed execution service with
equivalent isolation and resource accounting?

Each worker must run as non-root with 1 CPU, 1 GiB RAM, 32 PIDs, and a bounded
64 MiB temporary filesystem. It needs no network egress, a read-only root,
no home/project/private-oracle mounts, credentials, GPU, host socket, or
capabilities. Output is bounded at 4 MiB + 4096 bytes; public execution and
verification deadlines are 10 and 30 seconds. All descendant processes must
be cleaned up on timeout, cancellation, and allocation termination.

The goal is running generated Python for a controlled coding experiment.
Allocation-level accounting alone is insufficient when several candidates
share an allocation. We need independently enforced per-worker limits.

Existing diagnostics are retained under
`runs/hpc/yale-hpc-cpsc4880/studies/repair-selfplay-2026/readiness/`, including
`node-environment-30218129.json`; original ledgers and logs remain on Yale.

## Proposed fallback: YCRC managed Yale worker service

If nested unprivileged limits are unavailable, YCRC would operate an execution
service on Yale CPU infrastructure with site managed container controls. A
trusted broker would create a fresh restricted container for each request and
retain runtime privileges outside the trainer and candidate. The trainer would
use the existing authenticated worker protocol over HTTPS or approved loopback
SSH forwarding from its compute allocation. Tokens stay outside Git and logs.

The broker must enforce the same image digest, limits, isolation probes,
bounded concurrency, deadlines and descendant cleanup as the current worker.
Candidate containers receive only the request; expected outputs and private
oracles stay in the trusted verifier. Service resources require YCRC approved
accounting, admission limits, cancellation and owner-scoped request cleanup.
The trainer receives no container-management socket or service privileges.

Before use: obtain YCRC's supported design and authorization, implement its
connection/lifecycle integration, run all actual isolation probes and deployed
data verification, and generate fresh runtime evidence. Site approval alone
does not establish isolation. This document does not enable a remote-service
bypass in the current allocation worker gate.

Apptainer documents that non-root resource limits require cgroups v2 and
delegated controllers; a bus address alone does not establish delegation.
[Apptainer resource-limit requirements](https://apptainer.org/docs/user/1.4/cgroups.html).
