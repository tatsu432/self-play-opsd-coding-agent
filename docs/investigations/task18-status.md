# Task 18 execution and reproduction record

Task 18's bounded phase is complete (2026-11-16). The selected interface is
Qwen-native serialization plus exact-text replacement. All existing allocations
are terminal; **never resubmit them**. New RL remains disabled after material
numerical update sensitivity. See the [results](task18-results.md),
[research decision](task18-research-decision.md) and
[completion record](../task_completion/18-implemented-interface-signal-bottlenecks.md).

## Identity, data and safety

The branch began from main `48b48398871519640e0b808e37daa1aec555e002`.
Study `repair-selfplay-2026-task18-v1` has one locked **12 new allocated
H200-hour** allowance across operations, seeds, failures and continuations.
Historical budgets are not imported. Actual charge is **4,589 seconds /
1.274722 hours**, including the 351-second compatibility failure; every
reservation is reconciled. No job remained active for the project user at final
Slurm inspection; other account users' jobs are unrelated.

The frozen pool SHA-256 remains
`602eaf506f8ac2cf17924596dae7c0ff3891ef5034a546d4d3751b9803412c83`.
It has 104 independent families / 120 rows; only one originally reserved
eligible parent remains unused after replacements. New interface cells use
40 previously inspected training families, not an untouched confirmation set.
No new HumanEval evaluation or tuning occurred. Historical Task 15–17 configs,
checkpoints, results and execution identities remain unchanged.

All ten scratch model files matched the preserved project archive's SHA-256
manifest; no download/restoration was needed. See [operations](../operations.md).
The accepted isolated worker, restricted virtual files, API/import validation,
read-only specification/public tests and private-oracle boundary remain intact.
Generated Python executes only in that worker. Hidden expected outputs do not
enter student prompts/observations. New raw trajectories, oracles, tensors and
credentials remain ignored under `runs/`, not in Git or public W&B artifacts.

The legacy `PROMPT_HASH` field is retained. Native conditioning is additionally
bound by serialization, control checksum, Task 18 code hash and the pinned
tokenizer; a matching legacy prompt hash alone does not imply equal prompts.
The reusable CPU proof certifies data/verifier isolation, not generation or loss
mathematics; real native and numerical GPU probes provide separate evidence.

## Immutable executions

All GPU rows below completed exit 0 except the explicit never-started
cancellation. Full native cells generate 32 new episodes and reuse eight
checksum-bound probe records. Do not add probe and combined outcome counts as
independent evidence.

| Operation | Controller run | Job | Execution commit | H200 seconds |
|---|---|---:|---|---:|
| Complete-state numerical replay | `7635b59fe85fddd9e88b4c39` | 33575206 | `61b5457` | 473 |
| Normalized base | `bd76e6a6fbb83abf263b0d83` | 33575526 | `61b5457` | 925 |
| Normalized trained | `d1e5601f947494053b477b4a` | 33575845 | `d490a49` | 599 |
| Native v1 base compatibility failure | `069afeb246a83ece920c06cb` | 33576276 | `d490a49` | 351 |
| Native v1 trained, cancelled pending | `4fbbb1272d0b3fd48a2394a4` | 33576446 | `d490a49` | 0 |
| Cached scoring / fresh rollout | `dd5cd03f30f77dccaa411e81` | 33580028 | `0d55297` | 267 |
| Native v2 base probe | `bcab3d7c1784f7df90862f6b` | 33587596 | `b8144b0` | 208 |
| Native v2 trained probe | `c540fda607c5768db29fc04d` | 33587602 | `b8144b0` | 186 |
| Native base expansion | `02c9b1ee439d1828fa9636ab` | 33589271 | `9047078` | 1,009 |
| Native trained expansion | `4e84c405144439f65a0e38ba` | 33589312 | `9047078` | 571 |

The first native parser wrongly rejected assistant commentary around one tool
call. Its 2/8 base result is a harness diagnostic, not compatible-native evidence.
The trained v1 job was cancelled while PENDING, with zero elapsed allocation and
an operator-action ledger record. Fix `fd9a3e4` preserves all generated IDs and
content while executing exactly one delimited, validated call. The corrected
probes both scored 7/8 versus plain JSON's 4/8, with lower invalid-call rates,
passing the frozen expansion gate. Full results are 27/40 and 31/40.

Pending resource changes and failed adjustment attempts are retained in each
ledger. Numerical wall was shortened two→one hour with original two-hour
reservation retained; cache 30→20 minutes; corrected probes 30→15 minutes.
Full native walls were shortened 30→20 minutes. Variable base latencies led to
restoring the pending trained wall to the original 30 minutes; Slurm rejected
an increase for the already-running base, which finished within 20 minutes.
Original reservations were never reduced or reset.

CPU runtime jobs `33574512` / `96daa9183384957cd9ea58b5` (commit `dd068f0`)
and `33586413` / `5523522c27d08d3e1e5d7b9e` (commit `fd9a3e4`) both completed
exit 0 and verified all 306 tasks, allocating zero GPUs. The second followed
an explicit verifier-dependency change. Original executed proof SHA-256
`cf3d60139e8c7fe703b4e95a4035491f0b6b40b9a156f6b96999c9c8fde150e5`
is preserved in the private compatibility archive. Later runner/analysis-only
changes reused the matching proof after checking the explicit dependency files
and every non-code runtime identity; no repeated dataset audit was required.

## Evidence and analysis

- `runs/task18/interface-manifest-final.json` and `interface-analysis-final/`
  bind eight common-40 cells to controller completion, code/prompt/data/model/
  snapshot identities, budgets, source checksums and family coverage. Copied
  probe records are compared field by field; new/reused counts must be 32/8.
  The 32 new-family expansion is reported separately from the screen.
- `runs/task18/interface-analysis-native-v2-probes/` records both gate decisions.
  The control freezes their evidence and probe-source hashes. The final control
  selects native serialization but retains `training_ready: false`.
- `runs/task18/saved-audit-v1/` reconciles 120 historical iteration commits.
  Recovered calibrated-2026 arrays reproduce the 5.48991494 maximum;
  `recovered-token-audit-v1/` checks all 495 turn prefixes/lengths. Eleven
  recovered trajectory files match remote SHA-256 values in
  `runs/task18/recovery-checksums.json`.
- Numerical artifacts are under the controller run IDs above in
  `runs/hpc/yale-hpc-cpsc4880/`. Both original complete checkpoints and all
  three checkpoint-file checksums were verified on GPU. State restoration
  includes both adapters, both optimizers and RNG. The worst batch's generating
  checkpoint 3 was already retired; retained calibrated-2028 checkpoint 9 and
  calibrated-2026 checkpoint 5 do not establish its causal effect.
- `runs/task18/saved-reward-bounds-v2/` contains logical hidden-fraction bounds,
  not statistical confidence intervals or new program executions.
- `runs/task18/accounting-final.txt`, `budget-final.json` and artifact-checksum
  records retain scheduler charges and byte verification. No new online W&B
  cloud-history completeness claim is made; local artifacts are authoritative.

## Reproduction without duplicate jobs

Fetch or inspect an existing run by its ID; do not submit it again:

```bash
uv run selfplay hpc status 02c9b1ee439d1828fa9636ab
uv run selfplay hpc fetch 02c9b1ee439d1828fa9636ab
```

The recorded commits and `configs/task18.yaml` / their exact archived
`configs/task18-control.json` files define each execution. The final checked-in
control records the decision; it must not be substituted for an older job's
immutable control. New replication requires a separately identified study and
budget decision, not a silent continuation of these completed cells.

Read-only analysis commands (choose a fresh output directory):

```bash
uv run python scripts/analyze_task18.py --operation interfaces \
  --manifest runs/task18/interface-manifest-final.json \
  --pool runs/hpc/yale-hpc-cpsc4880/86e2df49bb04a9321f7d755d/pool.json \
  --output runs/task18/interface-analysis-reproduction
uv run python scripts/analyze_task18_tokens.py \
  --source runs/task18/recovered-calibrated-2026 \
  --output runs/task18/token-analysis-reproduction
uv run python scripts/analyze_task18_rewards.py \
  --pool runs/hpc/yale-hpc-cpsc4880/86e2df49bb04a9321f7d755d/pool.json \
  --source runs/task18/recovered-calibrated-2026 \
  --output runs/task18/reward-analysis-reproduction
uv run python scripts/plot_task18_interfaces.py \
  --summary runs/task18/interface-analysis-final/summary.json \
  --output docs/figures/task18
```

The manifest references private local fetch/status paths, including
`reused_probe_dir` and `reused_probe_status_path` for native full cells; update
local path prefixes if relocating artifacts, preserving hashes and IDs.
Historical controls are reused, never silently rerun or relabeled as new data.
All figures have standalone PNG/PDF exports and were visually inspected.

Local validation: **452 passed, 20 skipped, 18 deselected**; Ruff lint/format
and mypy (55 source files) passed. These are distinct from the real Yale CPU
and H200 results above. GPU/download/sandbox/online-W&B markers were excluded
locally; accepted worker execution and full-model probes ran on Yale.
