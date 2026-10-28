# Task 08 evidence dependencies

| Historical evidence | Decision | Reason |
|---|---|---|
| Worker 31033392, all 19 checks | Reuse through `hpc reuse-worker` | Worker entry files, seccomp policy, limits and SIF are unchanged; retain original report path/hash |
| Trainer/worker SIFs | Reuse | Pinned lock, image recipes and installation inputs unchanged; application code is mounted read-only |
| Runtime 31045342, 306 tasks | Historical; revised runtime/admission required | Protocol and canonical anchors change; original task/oracle correctness remains provenance |
| H200 backend 31040984 | Retain; rerun current backend contract | Protocol is an actual input to the current backend probe/identity; do not relabel old numerical output |
| Base 31046386, 0/41 | Historical sampled result | Greedy forwarding was broken; revised protocol needs a corrected baseline |
| D 31049338/31050754 and A/B/C 31054795/31055413/31056122 | Preserve as failed attempts | Zero yield does not demonstrate learning, and protocol-dependent evidence is stale |

Fresh containment/ownership/occupancy preflight still runs in each generated-code
allocation. Reuse does not exempt cleanup checks or grant another allowance.
A failed protocol probe blocks baseline/pilot/method allocations; independent
CPU data-revision and regression work may finish. No main job is authorized.
