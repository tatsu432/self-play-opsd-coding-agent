# Task 08 — Implementation and real acceptance in progress

Status: **launch readiness false**, execution continuing on 2026-10-28. This is
an interim record, not task completion. Main A–D jobs remain unauthorized.

Code commits: `0138dbc`, `a291098`, `e1836a0`. Hosted CI
[40627889719](https://github.com/tatsu432/self-play-opsd-coding-agent/actions/runs/40627889719)
passed 238 tests, one skipped, three deselected, including Linux/Docker. Earlier
runs 40626463095/40626756410 each exposed the same sampler/subprocess mock race;
the mock was isolated and a separate sampler attribution regression added.

Implemented greedy forwarding, explicit pilot diagnostics, shared authorization
accounting/migration, complete protocol hashing, immutable data revision, bounded
A/B/C windows, separate optimizer mechanics, a fixed protocol probe, and device/
process memory sampling. The strict parser, scientific objectives, model,
sampling, mathematical budgets and frozen task boundaries are unchanged.

| Evidence | Outcome |
|---|---|
| Worker 31033392 | Reused; all 19 checks, original report SHA-256 `b0b4781ee150260269690f074f2f509b24998c243800ce36383acf34988ebe8c` |
| Probe 31069728 / `7f6319f6b8b125ac11eaa7af` | Passed, 32 attempts, five valid mutations, 20 episodes/9,252 tokens, no updates/refills |
| Revision 31069732 / `d4630697f6f230e3021769b7` | Passed all 306 admissions/verifications and original-file equality; new anchors |
| Runtime 31071377 / `1efeedad6e8ff2df7ca56863` | Passed deployed 306-task validation and isolation |
| Backend 31071410 / `1b79a8543eecd835654471e5` | Passed A–D numerical objectives, adapter reloads and checkpoint acceptance |
| Base/D/A–C/freeze/previews | Pending current compatible execution |

Probe actions: 55 valid reads, one valid patch, one valid public-test call, nine
valid finishes, 53 patch tool errors and 38 invalid actions. Invalid proposal
reasons: 11 hunk-count, seven hunk-line, one context mismatch, eight duplicates.
These limitations are retained; five useful tasks resolve the old empty-workload
condition without requiring positive repair improvement. This probe proves no
optimizer or representative main-throughput readiness by itself.

Final settings: `configs/task08.yaml`; study `repair-selfplay-2026-protocol-v2`;
protocol `repair-json-v2`, prompt hash
`07968cec687ca52d8ce02c4a6d52eaca529f830855690cab1caff71c096f4322`.
Model/tokenizer revision remains `cdbee75f17c01a7cc42f958dc650907174af0554`.
New data: `data/protocol-v2`, manifest file SHA-256
`934412b2181ead4377d383130a19ad0d58eaab6bb9f74c5d8005416d09412fd2`, canonical
hash `d9140367e20587051b328dda20a7d93916ee5cadc7e72c2564523cf6ab6beade`.
Lineage binds original file manifest `44fe6caffc0548bebb01bf39208f8b2f5de10ab89ee8487dca20dd436c0154f5`
and prompt `ec1f46db1df8afe308934c54222a37cb6c9d73d072995699e19ea9419ccfd9c5`.
Parents, tasks, oracles, splits and coverage remain byte-identical. Admissions:
14 reference, 41 monitoring, 20 diagnostic, 115 final-ID and 116 external tasks.

Selected GPU request remains one H200, seven trainer CPUs/128 GiB and an independent
worker CPU/1 GiB. The probe consumed 540 allocated GPU-seconds (0.15 hours).
Cumulative reconciled readiness consumption is 1.241111111/24 GPU-hours, with no
outstanding reservation at this snapshot. The 12-hour pilot sublimit persists.
No defensible main learning-throughput projection or freeze exists yet.

Reports and source checksums are under `runs/task08/`; fetched raw artifacts are
under `runs/hpc/yale-hpc-cpsc4880/`. The old full D fetch reported directory metadata
errors, but all three joint checkpoints (0/2/10) verified by checksum; raw D and
baseline trajectories were fetched independently. Historical A/B/C raw proposals
also confirm the shared format failure. All historical failed runs remain intact.


Before replacing shared backend outputs, the complete original numerical report,
1.85-GB gradient reference and 1.06-GB joint checkpoint were copied on Yale to
`runs/readiness/repair-selfplay-2026/backend-v2-archive-31040984`. Transfer metadata
is recorded in `runs/task08/backend-archive.json`. The redundant local full-backend
transfer was stopped after this durable archive completed. Default `hpc fetch`
now excludes tensor payloads; use `--full` when those artifacts are required.

Selected A/B/C method checks now have independent submission slots; a D suite
still conflicts with every selected method, including legacy/uncertain slots.
This removes the remaining same-report-parent collision without weakening
submission/budget locking. Controller-only changes leave scientific code hash
`d8526f680fbbe3aff938d8688307817bb25ff496cd3d1cc02de3daecb5a5c6e1`
and protocol/data/runtime dependencies unchanged. Focused controller tests pass.

Pilot reporting now uses the trainer's actual cumulative step counters rather than
summing cumulative per-iteration counters. Its simple time extrapolation is
explicitly labelled conditional on observed workload and excludes unmeasured
monitoring/continuation overhead; the final cost analysis must account for those
costs and higher-yield scenarios separately. No optimizer or objective changed.
