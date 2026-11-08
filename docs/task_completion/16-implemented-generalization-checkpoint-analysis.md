# Task 16: generalization and checkpoint analysis

## Execution status

Implemented the deterministic analysis CLI, reusable analysis functions, selective
trajectory retrieval, focused tests, ten private CSV tables, requested JSON
artifacts, six PNG/PDF figures and the [research report](../investigations/task16-generalization-checkpoint-decision.md).
Independent local analyses are complete. **Task 16's trained action-subtype/root-cause
investigation remains incomplete** because the trained trajectories are on Yale
and the current machine cannot resolve `bouchet.ycrc.yale.edu`. Fresh live Slurm,
account and checkpoint reconciliation is likewise unavailable. Missing evidence
is explicit, not converted to zero counts or invented failure mechanisms.

All six monitoring-development parent-macro selectors choose the final checkpoint;
therefore the conditional earlier-checkpoint pilot is unnecessary. Newly allocated
GPU-hours: **0**. No training, checkpoint evaluation, proposer diagnostic or C/D
job was submitted. Historical Task 15 artifacts and W&B runs were not modified.

## Measured findings

- All 1,617 final/base outcomes, 30 development evaluations and 1,447 committed
  training iterations reconcile with frozen task/parent identities and complete
  denominators. All 306 frozen mutations regenerate. The 186 reserved training
  parents yield 127 eligible structured parents after the frozen exclusion.
- Trained HumanEval+ final source changes occur on 0–3/116 tasks per arm, versus
  65–81/115 on MBPP+. Public-test calls are 0–4 versus 91–103. Exact trained edit
  rejection counts remain unavailable. Base trajectories are complete and show
  408 rejected versus two accepted external patches.
- No observed terminal development decline: all six selections are final.
  Bootstrap selection stability remains weak in some arms; this does not identify
  an unmeasured optimal checkpoint.
- Training costs reconcile to 145.723889 allocated H200-hours. At each seed's
  largest common hour budget, B's best *available measured* development checkpoint
  is ahead by 10.87, 2.17 and 4.35 percentage points. Earlier thresholds, token
  budgets and final development endpoints do not consistently favor B. This is
  not a new simultaneous equal-time model evaluation.
- Structured proposal validity rises while repeated samples remain frequent;
  only 5.1–7.7% of attempted proposer groups discriminate difficulty among valid
  candidates. This does not establish a useful causal proposer-learning effect.
- Recommend an evidence-driven training-only repair-interface investigation in a
  separate future study; write up A/B now and defer full C/D.

## Provenance and artifacts

Frozen study: `repair-selfplay-2026-task15-ab-v2`; scientific execution
`0ac77cde4bc5f70ba2b3eb9e457749e3106a2bc5`. Starting analysis checkout:
`0f20ce6410971ed3f7e1701a8ba645c0fd89ef91`. Dataset manifest identity:
`fd47b9117ffa627531a24b938999d9e063f1e6eee47e4b0b91b1bbf46c4ed714`.
Model/tokenizer revision: `cdbee75f17c01a7cc42f958dc650907174af0554`.

Private outputs are under `/private/tmp/selfplay-task16/runs/task16/analysis/`.
They remain ignored. `input-manifest.json` hashes 309 consumed files and lists
4,286 missing dependencies (trained final/training shards and local checkpoint
markers). Generated fetch lists contain referenced final-evaluation, student and
proposer JSON shards only. All 231 base trajectories are available; none of the
1,386 trained final trajectories or committed training/proposal shards is local.

| Artifact | SHA-256 |
|---|---|
| Task 16 canonical consumed-input mapping identity | `d41e9bd12af7647830c73e53757df7bccca90c6235b83636eccd2d741dbfcd74` |
| Task 16 `input-manifest.json` | `17c856717ecee133e4ed3697b3bea1ed575a158766e75f44e8873c174cb6d8ab` |
| Task 16 `summary.json` | `20c4e8ccbb23f3b113c100cbe8906f63f5f9fa342c4454b5b064bf37c2067dd1` |
| Unchanged Task 15 allocation manifest | `d6c038d223659d1da1458508395b35608af10c1c5a474eba1c4639b199864263` |
| Unchanged Task 15 analysis summary | `6c1fe6e719a1a2781a40dec520d9cb4c511b2a29091d28ea047e9fc8e818f3aa` |

The mapping identity is the project's canonical JSON digest of the path/hash
mapping; it differs from the byte-level hash of the input-manifest file. Numerical
resampling uses 2,000 replicates and seed 2026. Task/family uncertainty and training
seed variation are exported separately. Unchanged-source frozen evaluations
recover initial public/hidden outcomes for all 116 external and 108 ID mutants;
remaining initial detection profiles need isolated verification.

## Validation

- Full local CPU suite: **354 passed, 34 skipped, 4 deselected**. The final added
  source-change attribution case and Task 16 contracts subsequently passed:
  **22 focused tests**. Synthetic tests validate implementation, not scientific
  claims or Yale containment.
- Ruff lint and formatting pass across source, tests and scripts; mypy passes
  all 48 source files. `git diff --check` passes.
- Two real-data runs produce **byte-identical output across 22 JSON/CSV files**.
  Figure PDF metadata is intentionally outside numerical determinism checks.
- All six aggregate PNG figures were visually inspected. Their PDF counterparts
  are exported. Connecting development lines guide the eye; markers are actual
  evaluations. The failure figure labels missing trained subtype evidence.
- No actual Yale backend/isolation validation is newly claimed. The evaluator's
  retained historical oracle evidence is reused. The network failure is a DNS
  dependency, not a scheduler or scientific failure.
- Hosted CI status is recorded below after the implementation push.

## Reproduction and remaining work

See the [report's reproduction commands](../investigations/task16-generalization-checkpoint-decision.md#reproduction-coverage-and-limitations)
and `uv run python scripts/analyze_task16.py --help`. `--run-root` supports
selectively fetched copies without rewriting the historical manifest.

Restore normal Yale DNS/network/VPN access; use `uv run selfplay hpc login` if
normal authentication then requires it. Reconcile `hpc inspect` and existing
ledgers. Fetch ordinary records and each of the six generated relative trajectory
lists with `hpc fetch RUN_ID --trajectory-list PATH.json`, without `--full`.
Rerun analysis with `--run-root runs/hpc/yale-hpc-cpsc4880`. This fills exact
trained parsing/patch/action diagnostics and permits stronger root-cause ranking.
Inspect the retained checksum manifests on Yale rather than copying model weights.
A separate isolated CPU replay is needed only for initial-mutant public/hidden
profiles not recoverable from the saved hash-identical final verifications.

The implementation exposes missing data rather than treating these remaining
subtasks as complete. No additional checkpoint evaluation is indicated by the
measured selector.
