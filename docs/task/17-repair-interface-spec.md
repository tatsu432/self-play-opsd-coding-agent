# Task 17 implementation specification

The supplied [research specification](17-generalization-spec.md) governs this work.
Historical Task 15 execution and artifacts remain immutable.

## Interfaces and identity

Preserve strict unified diff as the control. `episode.edit_protocol: replace_text`
advertises exactly one editing action: `action`, `path`, `old_text`, `new_text`.
Only `src/solution.py` is writable. Require a nonempty exact old span, unique even
when occurrences overlap; reject missing/ambiguous spans. Preserve whitespace,
imports and function APIs; validate syntax/source policy before updating state.
Text additionally requires unchanged import declarations; the historical strict
diff control retains its original import allowlist behavior. Report this
admission difference and observed rejections. Empty replacement text permits deletion. Inputs and resulting source are bounded
to 65,536 UTF-8 bytes. Rejections leave source unchanged.

The optional `normalized_diff` protocol retains the strict prompt and action
schema. It first uses the unchanged strict parser. After a parser failure only,
it may locate complete old hunk text at a unique exact span, ignoring hunk
coordinates/counts without changing added or removed content. Ambiguous or
inadmissible candidates retain the original rejection and unchanged source.
Recovered edits additionally preserve import declarations. Successful actions
are recorded as `strict` or `metadata_recovered`, without changing the student
tool observation. The oracle never selects an edit. This implementation is
locally tested but has not been selected or run as a model experiment; any future
`task17-normalized` operation is separate from the primary four-cell comparison.
Its labels A/C mean base/B2026. A new executed runtime check is needed for this
new code before another allocation.

New trajectories persist tool observations, including the last turn. Historical
records without observations still load. New episode identities include episode
configuration, preventing interface/budget collisions. Prompt/scientific hashes
change; old checkpoints retain their historical metadata. B2026 final weights
are imported with fresh optimizers after original checksum/model/data/science
validation; this is not a training resume.

## Required comparison

Study `repair-selfplay-2026-task17-interface-v1`, existing 4B model/backend,
greedy decoding and unchanged eight-turn/token budgets. Four operational labels:

| Label | Snapshot | Interface |
|---|---|---|
| A | Base | Strict diff |
| B | Base | Exact replacement |
| C | Task 15 B2026 final, iteration 245 | Strict diff |
| D | Same B2026 snapshot | Exact replacement |

Reserve one training-only parent per family: interface 40, curriculum 40,
calibration 8, independent development 16. Exclude `Mbpp/590`. Selection uses
source length/structure and deterministic hashes, not model outcomes. Prepare
mutations through the existing isolated verifier, at most 32 fixed attempts
per parent, caching identical rejected mutants and refilling only from unused
training families; report admission deficits instead of silently reducing denominators.

Interface evaluation includes 40 originals and 16 presentation variants: eight
source-comment extensions using the existing specification and eight repeated
specifications. Executable ASTs and verification oracles are preserved. Parent
families cluster the variants. Primary effects use originals; variants are
exploratory. Existing HumanEval parents remain external, so this experiment does
not provide untouched HumanEval confirmation.

## Conditional objective comparison

Use the existing static student trainer with Task 15 `same_scorer_v2`, immutable
PPO behavior denominators, four rollouts/task and a common frozen curriculum.
Operational labels A/B/C/D map to U/B/U-calibrated/GRPO. Distinct study IDs and
output directories prevent the two ungated arms from colliding. Initial
coefficients are 0.1 for U/B; calibration estimates U-calibrated's coefficient on
eight separate training-only parents. Hold it fixed.

No control file is fabricated. After actual Phase 1 evidence, freeze
`configs/task17-objective-control.json` with `phase1_informative`, `interface`,
`episodes` (80 or 128), `pool_sha256`, and checksummed evidence entries
`{path,sha256}`. The calibrated arm additionally requires
`calibration_coefficient` and `calibration_sha256`. Allocate the first 32 episodes
as a bounded pilot via a stop after iteration 4. Independent development and
existing monitoring evaluations occur at baseline, half exposure and endpoint.
Seeds are 2026–2028; training stops at exposure, not an optimizer-step target.

## Budgets and decisions

Use separate locked authorization ledgers: Phase 1 at most 10 allocated
H200-hours; Phase 2 at most 24 additional hours, including failures and backend
checks. Reconcile live scheduler/ledgers before submission. Reservations include
full requested wall time until actual accounting is verified. No full C/D
training. No execution of generated programs outside accepted isolation.

Select exact replacement only if actual verified repair evidence supports it.
Acceptance-only gains do not establish improved capability. A limited matched
adaptation remains conditional on observed schema unfamiliarity and remaining
Phase 1 budget; no adaptation has been run or selected. If results are uncertain,
retain strict diff. Phase 2 remains conditional on informative Phase 1 evidence.

Finalize calibration before starting the training pilots: a preliminary control
contains actual Phase 1 evidence, then the measured coefficient and calibration
checksum enter the final shared control. Training continuations require its exact
SHA-256. Calibrated coefficients must be finite positive numbers. Initial base
adapters are verified explicitly. Scientific fingerprints include the reused
static trainer and calibration helper as well as the Task 17 runner/recovery code.
The clean-stop continuation route retains completed-inference duplicate guards.
