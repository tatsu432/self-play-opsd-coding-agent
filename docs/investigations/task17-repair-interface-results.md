# Task 17: repair-interface implementation and retrospective evidence

## Verified semantic replay and admission recovery

CPU job `32794598`, run `9d9776357a4e23ae82cd34dc`, completed all 1,991
reconstructed actions using 449 unique isolated task/program verifications.
Among trained external attempts, **755/1,107** pass the full verifier, spanning
**249/696** recorded episodes and **96/116** parent problems. Of those episodes,
242 originally failed. This establishes correct edits at recorded source states;
it does not measure final accuracy under counterfactual future agent actions.
Trained ID has 500/837 passing reconstructed actions across 193 episodes.
Base has 13/14 passing ID and 25/33 passing external reconstructed actions.
All 1,617 trajectories are now present. No GPU time has been used yet.

Preparation `32794599` / `6be54fe0c6ab8d8a6c492675` failed before inference:
eight attempts admitted 35/40 interface and 14/16 development parents. The
failed artifacts and before-ledger snapshot were archived on Yale, with the
ledger's artifact location updated and archive history retained. The corrected
preparation retains first successful samples, uses 32 bounded attempts while
caching identical rejected mutants, and fills deficits only from unused
training families. This is oracle-only admission, without student-outcome
selection. Preparation now also performs current runtime validation, avoiding
a separate orchestration stage for each data-selection revision. Earlier blocked
or unreplayed status notes below are historical and will be consolidated after
the matched interface experiment.


## Active execution update — access restored

Normal Cisco VPN and Duo SSH authentication succeeded. Live inspection found no
active account jobs before Task 17. CPU runtime validation `32792142` / run
`15b407b13adfc94780658de9` completed exit 0 in 323 seconds from `a7046a4`;
its report was fetched before further dependency changes. All 231 base shards
were recovered, and `runs/task17/recovery-full-v1/` now covers all 1,617 episodes
and 11,516 actions, with 1,991 statically admitted reconstructions: 1,944 trained
plus 47 base. The trained-distribution counts below are unchanged. Semantic
replay and model experiments remain unexecuted; no GPU time has been allocated.
Hosted CI [42843475673](https://github.com/tatsu432/self-play-opsd-coding-agent/actions/runs/42843475673)
passed 436 tests with one skip and four deselections, including Docker worker
checks. Earlier access-blocked/partial-coverage notes below are historical.


## Current decision

**Patch encoding is a substantial observed obstacle; its causal contribution to
repair failure remains unmeasured.** The new exact-replacement interface and
bounded experiment routes are implemented. No frozen-model interface experiment,
semantic recovery replay, adaptation, calibration or objective training has run.
The next action is isolated CPU replay and the matched frozen-model comparison,
not full self-play or C/D training.

On 2026-11-10, `selfplay hpc inspect` failed with SSH exit 255 because
`bouchet.ycrc.yale.edu` could not be resolved. Local Docker also has no running
daemon (`~/.docker/run/docker.sock` absent). Generated programs were not executed
on the host. Newly allocated GPU-hours: **0**; no Task 17 job was submitted. Live
Yale job/account state remains unverified and must be reconciled before launch.

## Completed analysis

The read-only audit consumes checksummed Task 16 trajectories and replays only
trusted parsing/application/source validation, never candidate execution. It
reproduces all **1,386 trained** final evaluation trajectories: six arms ×
(115 MBPP+-derived + 116 HumanEval+-derived repair tasks), totaling 9,708 actions.
The 231 common-base trajectory files referenced by Task 16 are absent at their
recorded local paths now; their earlier aggregates remain historical evidence.
They are explicitly missing, not zero outcomes in this analysis.

| Trained evaluation distribution | Tasks/episodes | Accepted patches | Rejected patches | Metadata-recoverable actions | Recoverable actions passing static source/API checks |
|---|---:|---:|---:|---:|---:|
| MBPP+-derived repairs | 690 | 613 | 1,836 | 864/1,836 (47.1%) | 837/1,836 (45.6%) |
| HumanEval+-derived repairs | 696 | 13 | 4,301 | 1,132/4,301 (26.3%) | 1,107/4,301 (25.7%) |

Recoverable means the complete old hunk text identifies one unique exact span of
the actual current source; only hunk offsets/counts are ignored. Added/removed
content is unchanged. Malformed file headers, ambiguous locations, overlapping
or reordered hunks and unsupported body syntax are rejected. Pure insertions
without anchoring old context are not reconstructed. Multiple rejected attempts
can recover to the same program: these are action counts, not independent tasks.
The reconstructed edit is tested independently; subsequent recorded model actions
are not interpreted as a valid counterfactual continuation.

| Static replay rejection category | MBPP+ | HumanEval+ |
|---|---:|---:|
| Context mismatch | 661 | 2,913 |
| Hunk counts | 997 | 1,141 |
| Hunk offsets | 7 | 0 |
| Malformed/unsupported file headers | 115 | 116 |
| Forbidden paths / extra file content | 56 | 131 |
| Total | 1,836 | 4,301 |

The taxonomy uses reproducible static replay, including last-turn errors whose
observations were historically absent. It therefore refines Task 16's categories
rather than treating its unknown observations as measured error text. The
header/path parser can report a header error before reaching another defect;
these categories describe the first deterministic rejection. No historical
syntax/API/import rejection occurred in these strict-parser tool-error records;
this does not imply malformed model output could always reach source admission.
There were also 50 ID and 215 external JSON/schema-invalid actions.

At least one statically admitted reconstructed edit occurs in **292/690 ID** and
**349/696 external** episodes. Rejections recur at the same unchanged source and
category: 1,153 ID and 3,226 external attempts are beyond the first such attempt.
These are broad repeated-error categories, not necessarily identical patches.

The audit exports token lengths, public-case counts, mutation descriptors, AST
features, preceding-turn position, checkpoint, method and family with each action.
The private tables support stratified follow-up without re-reading private source
into reports. The inference comparison analysis separately records verified
repair, applied edits, test use, public-pass/hidden-fail, repeated actions, tokens,
turns and termination, refusing incomplete/mismatched pairs.

![Static rejection taxonomy](../figures/task17/rejection-taxonomy.png)

[PDF figure](../figures/task17/rejection-taxonomy.pdf).

## What this supports

- The reproduced 4,301 external rejections and only 13 acceptances confirm the
  interface obstacle observed in Task 16.
- Unique metadata-only recovery is possible for at least 1,107 rejected external
  actions under static historical source/API admission.
- **No recovered program has been semantically verified in Task 17.** These counts
  do not establish repair success, estimate a causal effect, or predict zero-shot
  exact-replacement performance.
- H1–H4 await the frozen-model experiment. H5 (mainly semantic versus syntactic
  failure) is unresolved; failure at both stages can coexist.
- Existing Task 15 B-over-A outcomes do not isolate gating. The proposed frozen
  curriculum and gradient calibration are unexecuted; Task 17 supplies no new
  student-objective superiority or efficiency evidence.
- Existing sparse proposer reward discrimination and near-repeat C/D update
  differences still do not justify full alignment training.

## Reproduction and private artifacts

```bash
uv run python scripts/analyze_task17.py --allow-missing --recovered runs/task16/recovered-analysis-verified --data data/protocol-v3 --output runs/task17/recovery-v4 --figures docs/figures/task17
```

Use a fresh output directory when rerunning; existing outputs are preserved.
`--allow-missing` is explicit because the 231 base shards are absent. Full-coverage
mode fails on missing files. Consumed existing trajectories are checked against
the prior manifest; all new inputs and missing paths are recorded. Run identity,
outcome and final-source reconstruction inconsistencies fail analysis.

Private results: `runs/task17/recovery-v4/{actions.csv,episodes.csv,summary.json,
replay-candidates.json,input-manifest.json}`. Earlier local diagnostic outputs v1
(zero matched episodes) , v2 (before feature enrichment) and v3 (before dependency-manifest enrichment) are not authoritative;
v4 is the reported output, with complete analysis dependency hashes. The deterministic reservations are saved separately
as `runs/task17/partitions.json`; they are not isolated-verified tasks.

See the [execution plan](../plan/17-repair-interface-plan.md) for the exact next
operations and the [completion record](../task_completion/17-implemented-repair-interface.md)
for validation and remaining work. Final held-out splits were used only for
retrospective diagnosis, not interface prompt tuning or hyperparameter selection.
