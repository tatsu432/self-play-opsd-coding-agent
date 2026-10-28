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

## Final compatible evidence — 2026-10-28

Reused worker proof retains its original SHA/provenance; both SIFs were reused
without rebuilds. New runtime 31071377 and numerical backend 31071410 actually
validated the revised protocol/data dependencies. Correct greedy base 31072777,
D 31075857/31077825 and A/B/C 31080094/31080096/31080097 passed. Freeze 31082462
binds their report hashes plus the conditional resource plan, and all four actual
main controller previews pass. No main job was submitted.

Scientific hash `d8526f680fbbe3aff938d8688307817bb25ff496cd3d1cc02de3daecb5a5c6e1`
and allocation verifier `89730a635e803a168ac7285b1261396bda723aaaccdf664bf8a344dce86d6fc9`
remain unchanged across controller/reporting/documentation followups. Numerical
compatibility hash is `c819f3b09a746a9c311f8e490cd265746cb1c883ae6a8d3b5cabbcb4abdcfe42`.
Final controller metadata correction changes an unused preview field to agree
with the existing frozen 200-iteration training settings; actual training physics
and the frozen execution source are unchanged. Focused regression and four fresh
controller previews verify this correction without another GPU allocation.

See the completion record for exact identities, all report paths, sample limits,
accounting and per-method workload-dependent forecasts. Prior numerical tensors
and checkpoint bytes are preserved in the documented historical archive.
