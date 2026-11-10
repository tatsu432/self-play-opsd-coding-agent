# Task 17 execution and recovery plan

## Current state

Implementation, all 1,617 historical trajectory audits, isolated semantic replay,
verified pool preparation and four sequential primary GPU allocations are done.
The complete strict-control artifacts are local. Text-cell completion and aggregate
counts were observed remotely, but final text artifacts/status and accounting
still require retrieval. Yale VPN disconnected and SSH refresh did not complete;
the stale login was stopped. No Phase 2 calibration/pilot/training or normalized
model experiment has run. See the [results](../investigations/task17-repair-interface-results.md),
[decision](../investigations/task17-research-decision.md) and
[execution record](../task_completion/17-implemented-repair-interface.md).

## Resume retrieval and Phase 1 analysis

Restore the normal Yale VPN connection, then authenticate and collect:

```bash
uv run selfplay hpc login
uv run python scripts/collect_task17_interfaces.py
```

The collector uses the existing run IDs only, checks final Slurm states, fetches
private artifacts and reads actual allocation time, then writes the cell manifest
and runs paired analysis. It never submits a job. Failed SSH cannot truncate
previously valid status files. Analysis refuses an existing output directory;
use `--output runs/task17/interface-analysis-v2` for a deliberate later rerun.
The local prepared pool is already fetched at
`runs/hpc/yale-hpc-cpsc4880/86e2df49bb04a9321f7d755d/pool.json`.

Do not repeat preparation, replay or the four primary cells. Their live ledger
identities and immutable execution commits are in the completion record. A/C
pending wall limits were reduced from two hours to one with recorded adjustments;
original conservative reservations remain until accounting reconciliation.
The four observed actual allocations total 1.15 H200-hours.

Review the original 40-family paired effects and family-bootstrap uncertainty
separately from the 16 correlated presentation variants. Report acceptance,
verified repairs, testing, public-pass/hidden-fail, errors/repeats, turns/tokens,
termination, runtime/cost, and source/specification/structure strata. Inspect
actor/scheduler identity checks and final completion markers. Publish actual
outcome, acceptance and efficiency PNG/PDF figures only after successful
analysis, and visually inspect them. All raw source/oracles stay private.

Select the simplest effective interface from verified repairs, not acceptance
alone. The optional same-prompt metadata normalization is implemented and tested
but unselected/unexecuted; consider at most one separately identified diagnostic
if the primary result leaves a concrete conditioning-versus-application question.
Do not change the primary strict control or cycle through minor prompt variants.

## Conditional Phase 2

After the Phase 1 decision, freeze the evidence-backed objective control in
`configs/task17-objective-control.json` as described in the
[implementation specification](../task/17-repair-interface-spec.md). No placeholder
control exists. Upload checksummed evidence to the managed run workspace. Use
common frozen curriculum, four rollouts/task and an 80- or 128-episode endpoint,
chosen before training from measured cost. Existing final splits are exploratory,
not untouched confirmation.

Current code changed after the original interface jobs. Synchronize a clean
committed checkout, reuse the accepted worker and run one current CPU runtime
check for the selected interface. Use explicit `hpc reuse-runtime SOURCE_STUDY
--settings TARGET_SETTINGS` only when code/model/data/images/limits match; its
wrapper retains the executed raw proof and original provenance. This avoids
repeating the same 306-task verifier audit for each objective configuration.

Run `task17-calibrate` on the eight reserved families, then record the measured
finite positive coefficient and calibration checksum in the final shared control
before training. Generate and commit resolved per-arm settings using
`configure_operation`; do not change the control after pilots start. Student
labels A/B/C/D mean ungated/gated/calibrated-ungated/GRPO, not proposer C/D.
Scientific fingerprints include all reused trainer/calibration helpers.

Prioritize matched GRPO/gated seed-2026 pilots with
`--operation task17-student --stop-after-iteration 4` (32 attempted episodes).
Inspect finite gradients, actual mixed groups, edit behavior, checkpoint identity,
denominator warnings and measured cost. Verify the complete joint checkpoint
before `hpc resume RUN_ID --continue-stopped`. Extend only if an informative
matched comparison fits the separate 24-hour ceiling, including failed allocation
and continuation costs. Preserve G/GRPO replication before expanding optional
arms; keep exposure matched. True PPO behavior denominators, teacher stop-gradient,
masking and `same_scorer_v2` remain unchanged. Initial calibration does not match
later optimizer trajectories.

## Finish and integrate

Complete scientific decisions, objective learning/efficiency figures if run,
checkpoint/W&B reconciliation and actual cost accounting. Keep full proposer C/D
deferred unless the earlier signal problems are meaningfully addressed; no full
C/D training is authorized here. Update concise README and completion evidence,
run appropriate final checks, then merge the finished branch into local main and
push. Until the retrieval/analysis and conditional research decisions are handled,
preserve the unfinished feature branch on GitHub without claiming completion.
