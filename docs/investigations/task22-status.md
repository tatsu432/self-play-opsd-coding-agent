# Task 22 execution and reproduction

## Primary complete; transfer in progress · November 25, 2026

The existing primary jobs completed successfully and were verified and fetched:
C `34673660` / `c3bcab0e4d4b3846c90299b2` scored 33/64, and D `34674426` /
`190d42771d72c3fe86acc77b` scored 25/64. The paired D-minus-C difference is
−12.5 percentage points, with a task-family bootstrap interval [−25, 0],
conditional on these single training runs. Do not duplicate either evaluation.

C transfer `34690978` / `c00321c48eb744c98a2e8fb4` completed successfully:
cross-source 1/1, mutants 5/9 and harder transfer 1/8; its primary result was
reused. D transfer `34693437` / `cb99c34dfae8e5bd65b7bf87` is pending. Its
pending wall was reduced from four hours to 90 minutes based on the matched
C transfer's 34:09 allocation, retaining the original four-hour reservation;
the change is recorded in `runs/task22/pending-monitor-wall-adjustments.jsonl`.
Inspect the existing job and ledger; never submit a duplicate. Verify and fetch
D, reconcile all charges, generate final analysis and figures, complete the
research and README reports, run checks, then merge and push the finished work.

## Matched C/D running · November 24, 2026

Both 512-episode logical training runs completed successfully and their
controller receipts and checkpoint identities were verified. C ended at
`000019` (run `b1eee3f0d23d27e43f66f75a`) and D at `000021` (run
`9474b28f2fceda65627c1cdf`). Their reused 512-episode development monitors
scored C 11/24 and D 14/24; these are not held-out results. Primary held-out
evaluations were submitted exactly once: C job `34673660` / run
`c3bcab0e4d4b3846c90299b2` is running, and D job `34674426` / run
`190d42771d72c3fe86acc77b` is pending. Inspect live Slurm and the shared
ledger before acting; never duplicate. Verify, fetch and analyze these
existing evaluations, then run the prescribed transfer evaluation as budget
permits. The completion and final conclusions remain pending.

On November 25, both 512-episode checkpoints were committed. The second
segments stopped cleanly while evaluating the reused endpoint development
set: C job `34617090` used 42,959 H200-seconds and retained 15 of 24 task
records; D job `34617349` used 42,889 seconds and retained 7 of 24. Neither
`complete-main.json` existed at the stop. Guarded continuations C `34671026`
and D `34671002` were submitted once from checkpoints `000019` and `000021`,
respectively. Both started and reported running health under their new job
IDs. Their pending wall limits
were reduced from 12 to 2 hours based on the remaining monitor tasks; the
original budget reservations remain and the intent/results are recorded in
`runs/task22/pending-monitor-wall-adjustments.jsonl`. Inspect Slurm and the
ledger before acting and never duplicate. After completion verify each marker
and checkpoint before held-out evaluations.

The initial twelve-hour segments ended in Slurm `TIMEOUT`, with their
last complete checkpoints preserved: C `000008` at 224 episodes and D
`000009` at 220. C used 43,225 H200-seconds and D 43,214. Guarded
continuations were submitted once from those checkpoints: C job `34617090`
and D job `34617349`. Both started and reported running health under their
new allocation IDs. Their logical run IDs
and frozen scientific identities are unchanged. Inspect their scheduler and
ledger state before acting; never duplicate. The qualification and both
initial segments together used 87,872 H200-seconds (24.4089 hours), with
both new twelve-hour reservations additional under the 95-hour limit.
Both resumed jobs subsequently committed 256 attempted episodes: C checkpoint
`000009` and D checkpoint `000011`. Their 256-episode development monitors
both scored 10/24 (at 128, C scored 9/24 and D 8/24). This reused development
set is not held-out confirmation. Continue toward the primary
512-episode endpoint; 256 is a prespecified fallback, not the main result.
Both then committed 288 episodes (C `000010`, D `000012`) and continued.
Their next checkpoints reached C 320 episodes (`000011`) and D 316
(`000013`); both continuation jobs remained running.
Both subsequently committed 384 episodes, C at `000014` and D at `000016`.
Their reused development monitors scored C 11/24 and D 10/24. Both jobs
continued toward 512 episodes. The 384 checkpoint is a prespecified fallback.
Both then reached 408 episodes (C `000015`, D `000017`) and continued.
Both subsequently committed 440 episodes (C `000017`, D `000018`), with
their continuation jobs still active.
Later checkpoints reached C 480 (`000018`) and D 468 (`000019`); both jobs
continued toward the 512-episode endpoint.
Both have since committed 512 episodes: C `000019`, D `000021`. Their final
development monitors and completion markers were not yet complete at this
observation; do not treat either job as complete or duplicate it.

Qualification job `34554246` / run `1a8172783b2dbd834a3332af`
completed and passed on archived B snapshots at 0 and 128 episodes. The
first batch had finite reference norm 5.1292 and nonzero proposer gradients
with valid reward differences for C and D. The controller-verified receipt
is `runs/task22/qualification-verified.json`. The main configuration
was frozen from the successful gate in commit `41167aa`.

C job `34557078` / run `b1eee3f0d23d27e43f66f75a` and D job
`34557117` / run `9474b28f2fceda65627c1cdf` were each submitted once.
Both are running with twelve-hour reservations and the common 95-H200-hour
Task 22 ledger. At 128 attempted episodes, the reused 24-task development
monitor scored 9/24 for C and 8/24 for D. All recorded joint gradients were
finite through that checkpoint. These are development measurements. Inspect Slurm and the ledger before any recovery or
submission. Do not duplicate either run. Monitor checkpoints, reconcile
actual allocations, evaluate matching endpoints, and complete the analysis.

Both arms subsequently committed 160 attempted episodes (C checkpoint
`000006`, D checkpoint `000007`) while their original jobs were still running.
The latest batches had finite optimizer steps and 22/32 and 21/32 verified
training rollouts respectively; these are not independent evaluation scores.
The D batch recorded zero near-zero alignment signals. The budget ledger
committed 24.398 H200-hours, including both 12-hour reservations and the
completed qualification; actual running-job charges are not yet reconciled.
Later, C committed 192 episodes at checkpoint `000007` and D committed 188
at `000008`, with both original jobs still running. Their most recent
batches had finite optimizer steps; C recorded 20/32 and D 19/28 verified
training rollouts. These rollout counts are training telemetry, not evaluation.

## Earlier qualification launch · November 24, 2026

Yale login succeeded. Qualification job `34554246` / controller run
`1a8172783b2dbd834a3332af` was submitted once from commit
`2eb8481b68a54dd8691a7b52cb1302887438f36c`, with a three-hour H200
reservation under the 95-hour Task 22 ledger. It is running on
`a1124u19n01`. The initial adapter identity matches Task 21, and the
reference cache is prepared. Inspect the live scheduler and ledger before
any action. Never duplicate this run. Group project storage was 98% used,
with approximately 91 GiB free at launch. Continue monitoring.

## Local readiness · November 24, 2026

Task 22 adds PDB reference alignment to the existing shared LoRA trainer. Its
private candidate manifest starts with the 11 verified PDB feasibility
examples, with zero parent and family overlap with the proposer curriculum,
development monitor, solver probe, primary evaluation, and transfer splits.
The manifest includes training-only reserves for deterministic eligibility
failures. The original PDB bundle and Task 21 checkpoints are unchanged.

Local hashes and policy signatures verified B batches `000001`, `000007`,
`000011`, and `000016` at the corresponding pre-update snapshots at 0,
128, 256, and 384 attempted episodes. The controller and trainer check the
remote checkpoint contents and saved behavior and teacher scores before use.
The successful qualification must produce the final verified reference actions
and a controller-verified gate. At this earlier local checkpoint, Task 22 had no allocated GPU time or new
endpoint results. The subsequent qualification is recorded above. The previously reconciled Task 21 total is
217.155 H200-hours. See the [qualification record](task22-alignment-qualification.md)
and [baseline analyses](task22-mechanistic-analysis.md).

Baseline plots are reproducible now. They use only the completed Task 21
evidence. The alignment figure requires an executed Task 22 qualification.

## Qualification

Run from a clean Task 22 checkout with the original Task 21 dataset available.
The freeze script stages copies of the original manifest and bundle and a
content-addressed private reference manifest. It never trains a policy.

```bash
uv run --frozen python scripts/freeze_task22.py
uv run --frozen selfplay hpc login
uv run --frozen selfplay hpc doctor --settings configs/task22-qualification.yaml --gpu-type h200
uv run --frozen selfplay hpc task21-budget --settings configs/task22-qualification.yaml
uv run --frozen selfplay hpc sync --settings configs/task22-qualification.yaml
uv run --frozen selfplay hpc reuse-worker repair-selfplay-2026-task21-pdb-prompt-v2 \
  --settings configs/task22-qualification.yaml
```

Check for existing Task 22 reservations and logical runs, confirm the
`cpsc4880` H200 account and projected capacity, and check both the project
filesystem and user quota. The controller requires 20 GiB free on shared
storage. Reserve additional space for the concurrent checkpoints and
evaluation records. Preserve verified records before reclaiming project space.

Preview the exact qualification specification, then submit it once. Set the
same incremental limit for every Task 22 operation, no higher than 95 hours.
Qualify for three hours first. A recoverable stop may use up to two additional
hours, subject to the shared five-hour qualification limit.

```bash
uv run --frozen selfplay hpc submit --operation task21-align-qualify \
  --settings configs/task22-qualification.yaml --method control --profile single_gpu_80gb \
  --gpu-type h200 --account cpsc4880 --stage-wall-minutes 180 --task20-budget-hours 95 --dry-run
```

Record the submitted logical run ID and Slurm job ID before monitoring.
Use `selfplay hpc status RUN_ID` and `selfplay hpc watch RUN_ID --continue-stopped` when continuation fits the forecast. If
the recorded attempt stopped cleanly before the gate, use
`selfplay hpc resume RUN_ID --continue-stopped` under the same ledger.
Verify completed allocations with `selfplay hpc task21-verify RUN_ID` and
fetch with `selfplay hpc fetch RUN_ID`. The default fetch excludes large
checkpoint and model exports. Preserve the raw qualification marker and
controller receipt as ignored files.

## Matched training and evaluation

Freeze the main control with the fetched `complete-align-qualify.json` and
the controller verification JSON. Stage the new committed configuration
without changing the qualified code. The controller checks that the
qualifying source commit is an ancestor and the scientific source hash
matches the submitted checkout.

```bash
uv run --frozen python scripts/freeze_task22.py --gate GATE_JSON --verified VERIFIED_JSON
uv run --frozen selfplay hpc sync --settings configs/task22-main.yaml
uv run --frozen selfplay hpc submit --operation task21-main --settings configs/task22-main.yaml \
  --method C --profile single_gpu_80gb --gpu-type h200 --account cpsc4880 \
  --stage-wall-minutes 720 --task20-budget-hours 95 --dry-run
```

Repeat the preview and one submission for D. Reconcile Slurm state and
the budget before every continuation. The wall limit is a reservation
and can be reduced for backfill when measured time supports it. Evaluate
the verified endpoint with `task21-evaluate-primary --checkpoint-iteration N`.
The full `task21-evaluate` stage reuses checkpoint-backed primary task records
and adds the three transfer splits. Both use the same run settings, arm,
hardware flags, and budget as training, with an evaluation wall limit based
on measured Task 21 runtime.

For a frozen 384 or 256 fallback, use `--endpoint 384` or `--endpoint 256`
on the same evaluation command with a matching numbered checkpoint. Stop
active training before evaluating an intermediate checkpoint. The protected
B comparator iterations are 15 and 10, respectively. The controller keeps
the original completed-endpoint guard for 512.

If the prespecified matched continuation is feasible, use
`python scripts/freeze_task22.py --extend-to 1024` or `--extend-to 768`.
Commit and stage the three generated arm controls. Run `task21-main` and
endpoint evaluation for each with the new namespace, the common Task 22
ledger, and dependency-checked worker evidence. Each run restores the
verified original endpoint state before any optimizer step and saves its
parent marker and checkpoint identities.

## Analysis and release

Baseline regeneration:

```bash
uv run --frozen python scripts/analyze_task22.py --output runs/task22/baseline-analysis
```

For the final analysis, pass `--budget` with the reconciled Task 22 ledger,
`--qualification` with the qualification receipt, and `--manifest` with
a private JSON mapping arm labels such as `C-2026` to
`{"training_run_ids":["RUN_ID"],"evaluations":["VERIFIED_RECEIPT.json"]}`.
Fetch each referenced evaluation directory. Set `--exposure` for a
shortened comparison. Run `scripts/sync_task22_summary.py` on the aggregate
summary before using its explicit `--publish` option. Update the results,
README, completion record and release branch after accounting, evidence,
and deliverables reconcile.
