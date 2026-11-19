# Task 19 implementation and execution status

The implementation adds a shared Qwen3.5 LoRA, one AdamW state, Dr. GRPO,
native thinking and exact-text tools, executable semantic mutations, and
training-reference gradient alignment. Task 19 uses a separate dependency
lock and SIF identity. Its initial control has empty calibration and
qualification hashes. The controller requires real evidence before launching
the five-arm study.

Local validation passed 462 tests, with 20 skipped and 18 deselected. Ruff
lint and format, mypy, and the Task 19 lock check passed. CPU tests cover the
native protocol, objective, joint step and checkpoint restoration. An isolated environment with Transformers 5.13.0,
PEFT 0.19.1 and PyTorch 2.10.0 exercised the real Qwen3.5 model classes with
tiny CPU weights and the pinned tokenizer. All 14 frozen reference edits
round-tripped to the original correct source. Five figure templates rendered
to PNG and PDF with synthetic test values. GPU image construction, numerical
qualification, measured costs, evaluation and method comparisons are pending.

Yale authentication succeeded. CPU image check job `33706240` completed exit 0.
Trainer image build job `33706414` completed exit 0. H200 qualification job
`33707814`, controller run `170dd97ea69313f0badb1268`, was submitted once from
commit `122604ebbe432da8225aab3eb4315fbbbeeeff12`. It failed before worker startup because the allocation dispatcher rejected the
Task 19 operation. Slurm charged 2 H200-seconds. The dispatcher is corrected
and its local tests passed.
Replacement qualification job `33720283`, controller run
`01a3a1ca9cbf435ad60584db`, was submitted once from
`2436cedded42bbdd4b224d6272f057cf234906e3` after a passing preview. It is
pending on priority with a four-H200-hour reservation. Inspect this existing
job and its ledger before any submission. The replacement failed after 4,362 H200-seconds because its nine student
groups had eight all-success groups and one all-failure group. Its numerical
check therefore had no informative RL gradient, and did not establish a
cached-scoring mismatch. The revision `task19-v2` reuses the checksum-bound
frozen batch and extends it with up to 20 additional training tasks before
retrying the numerical check. No Task 19 benchmark or training result is
available.

## Yale execution

Run these from the pushed Task 19 branch. After `uv run selfplay hpc login`
and Duo approval, use `uv run selfplay hpc inspect` and the matching resource
query to confirm the current account, jobs and H200 allowance.

```bash
uv run selfplay hpc resources --settings configs/task19-yale.yaml \
  --profile single_gpu_80gb --gpu-type h200 --account cpsc4880
uv run selfplay hpc sync --settings configs/task19-yale.yaml
uv run selfplay hpc reuse-worker repair-selfplay-2026-task18-v1 \
  --settings configs/task19-yale.yaml
uv run selfplay hpc images build --settings configs/task19-yale.yaml \
  --kind task19-trainer --build-mode yale --check-only
```

After the CPU build check finishes, submit the same command with
`--check-only` removed and monitor the existing build to a pinned SIF. If
the exact immutable commit already has a passing build check, reuse it.

Use the following options for the allocated scientific stages.

```bash
uv run selfplay hpc submit --settings configs/task19-yale.yaml \
  --profile single_gpu_80gb --gpu-type h200 --account cpsc4880 \
  --operation task19-qualify --method control --dry-run
```

After a successful preview, submit once with `--dry-run` removed. Track the
returned controller run ID through `hpc status`, `hpc watch`, and `hpc fetch`.
Qualification records actual model compatibility and calibration artifacts.
A recorded cache parity failure exposes the bounded `task19-qualify-uncached`
operation. Use the same worker and model image for either route.

Run `task19-base --method control`, then `task19-pilot` for each of
`control A B C D`. Pass the existing private W&B secrets file with
`--secrets-file` for online aggregate metrics. Keep raw trajectories,
reference repairs, prepared data, and full checkpoints private. For a clean
allocation stop, continue the same controller run with
`uv run selfplay hpc resume RUN_ID --continue-stopped` after inspecting it.
The same completed pilot is the checkpoint source for its main segment.

Once all five pilot markers and their Slurm charges are verified, run
`uv run selfplay hpc task19-freeze --cap VERIFIED_TOTAL_H200_HOURS --export runs/task19/main-control.json`. It writes
one common endpoint from actual pilot costs, the evaluation reserve and a 20%
contingency. Then submit `task19-train` once per arm. These segments continue
the same scientific runs and include the frozen final ID and external
evaluations. Review `hpc task19-budget` after every stage.

## Analysis

Fetch the base and five arms with `hpc fetch RUN_ID --full` and reconcile
`hpc task19-budget` into the private analysis directory. The analysis manifest
maps `base`, `control`, `A`, `B`, `C` and `D` to objects with a local
`path` and a `controller_runs` array. Include each arm's pilot and main run IDs.
Use the controller-published `main-control.json` for the common endpoint.

```bash
uv run python scripts/analyze_task19.py \
  --manifest runs/task19/analysis-manifest.json \
  --control runs/task19/main-control.json \
  --budget runs/task19/budget-final.json \
  --output runs/task19/analysis
```

The final report should record results conditional on seed 2026, paired family
intervals, compute and failure taxonomy. The previously inspected external
benchmark is exploratory. Historical Task 15–18 evidence remains in its own
results records.

Task 19 v2 extended qualification job `33730057` / controller run
`4297c7d19ed430ce68361596` started from immutable commit
`2d58885daf00f6eb747f8e75bd67f536f7fb4d9a`. The original nine groups are
checksum-bound, and the job can add at most 20 training tasks. Its four-hour
reservation and the earlier 4,364 H200-seconds share the 80-hour ledger.
Inspect this job before taking action and never duplicate it.

The v2 qualification job `33730057` failed after 4,631 H200-seconds because
no proposal group contained distinct valid difficulty rewards. Its 29 verified
task groups comprised 23 all-success groups, five all-failure groups, and one
mixed group. Some proposer groups contained both valid and invalid proposals.
The specification requires continuing controlled pilots when proposer
reward discrimination is weak. Task 19 v3 therefore reuses the exact saved
batch, records the lack of valid reward variation, and qualifies numerical
behavior using the mixed group. Loss calibration uses up to four student and
four informative proposer groups, with invalid-proposal penalties retained.
The v1 and v2 jobs used 8,995 H200-seconds in total.

The v3 qualification job `33739684`, controller run
`556bb9e4c2b309660f082527`, was submitted once from immutable commit
`5b1eb075c98411b499d15ba91a31dd917de6fe7e` after a passing preview. It is
pending on priority with a four-hour reservation. Its saved batch source is
v2 `frozen-training-batch.json` SHA-256
`4cd5991768348988e351bbebf43ed59c58bc14c822a5a20dfdb2c2ea30a3194d`.
Inspect this run and the shared ledger before further actions.

The v3 numerical job `33739684` failed its prespecified 5% relative threshold
in 447 H200-seconds. Gradient difference was 8.003%, cosine 0.999642; restored
AdamW displacement difference was 5.967%, cosine 0.998284. Same-operation
repeat variation and clipping-decision changes were zero, and sampled cached
replay matched the saved behavior probabilities. These measurements compare
cached generation with full scoring, while keeping actual behavior denominators.
The Task 19 ledger totals 9,442 H200-seconds (2.622778 hours).

Task 19 v4 begins the planned single-backend uncached qualification with newly
sampled behavior trajectories. It keeps the previously selected token limits.
Its execution cost remains to be measured. The v1–v3 batches, controls,
accounting and failed job identities remain intact.

The uncached v4 qualification job `33740964`, controller run
`464255f66a34a218d763e9e0`, was submitted once from commit
`a0b78624a286c1eead3ef86389fa5a98f4fd5d7f` after a passing preview. It has a
four-hour reservation in the shared ledger. Inspect this existing job before
any retry or submission.

The v4 worker was still running after 2 hours 54 minutes without a complete
wave checkpoint. Slurm reported active GPU use; progress within the wave is
unknown. Its immutable code cannot save the individual episodes already in
progress. An isolated, unmerged recovery draft is at commit `989a028`, branch
`wip/task19-qualification-recovery` in `/private/tmp/selfplay-task19-recovery`.
It saves fully scored episodes with their scientific identity and adds a
guarded same-run continuation and a separate v5 configuration; it cannot
recover unsaved v4 episodes. In that
checkout, 464 tests passed, 20 skipped and 18 deselected, and focused recovery
tests, Ruff and mypy passed. Review its controller, budget and execution
identity handling before using it in any new study namespace. Never launch a
replacement while v4 remains active or its terminal accounting is uncertain.

Job `33740964` subsequently timed out after 14,409 H200-seconds without a
complete wave or numerical result. Its original immutable execution and
artifacts remain preserved; do not repeat it. The shared Task 19 ledger now
records 23,851 H200-seconds (6.625278 hours) of actual allocation. The
reviewed recovery changes and v5 configuration have been incorporated into
the Task 19 branch. V5 saves scored episodes and identity-bound numerical
results, and supports guarded continuation of the same controller run. The
older v4 execution did not save individual episodes and cannot be resumed
from the v5 checkpoint format. V5 qualification was submitted once from
commit `c613dc7d8eb4559e815f89354053c37e68b3b570` as Slurm job `33765671`,
controller run `c8d4d093e75060f335f28078`, with a four-hour reservation after
a passing preview and worker proof. Inspect that run and the shared ledger;
never submit a duplicate. While it runs uninterrupted its progress is valid,
but its continuation code sorts saved task keys and may change calibration
order. Do not resume it blindly. A tested ordering fix is isolated at
`d7ded10` on `wip/task19-qualification-recovery`; review an identity-bound
migration or another safe resolution against its saved evidence before any
continuation. A guarded v5 reordering utility and tests are in the task branch;
it requires a verified terminal controller job, archives the original bytes,
and handles only a partial first wave or a completed wave with no pending
tasks. It rejects ambiguous later partial waves. It has not been run on the
live checkpoint. Numerical and calibration phases may still exceed one allocation.

During its first uninterrupted allocation, v5 saved twelve fully scored
episodes by 3 hours 2 minutes, forming groups with 4/4, 4/4 and 3/4 verified
successes. The mixed and uniform groups are training-only diagnostic inputs,
not benchmark results. The fourth task and numerical qualification remain
unfinished; inspect the current progress and scheduler before any action.

V5 job `33765671` timed out after 14,411 H200-seconds with fourteen scored
episodes saved and the first wave unfinished. The shared ledger now totals
38,262 seconds (10.628333 H200-hours). After confirming terminal state and
matching identities, the guarded utility archived the original progress bytes
with SHA-256 `4cc96e212397390587fc8f32680ca5203531515a637dcf9d0fec61ca68ce2581`
and restored the proposal-derived task order; the reordered file SHA-256 is
`9fac56112503b4a062863117a61ebc1fdbffb4b22cd180265f7f59ff957aef8e`.
Parsed checkpoint data and all fourteen episodes were verified unchanged.
The same controller run `c8d4d093e75060f335f28078` was continued once as job
`33783098` with a new four-hour reservation. Inspect that job and the ledger;
do not submit a duplicate. If another continuation is necessary, repeat the
terminal-state and ordering audit before acting.

Continuation `33783098` started and preserved the fourteen earlier episodes
unchanged, then completed all sixteen training-only episodes. The four group
success counts are 4/4, 4/4, 4/4 and 3/4. The frozen batch SHA-256 is
`c3d885680074b9566d279c440729e048200997808cbb07e6afd369fbae3611d7`.
The uncached numerical diagnostic passed: across 64 compared sampled tokens,
replay error, support exits, clipping changes and same-operation repeat
variation were zero. Relative gradient and restored AdamW update differences
were 1.355887% and 1.938686%, with cosines 0.999908 and 0.999812. Calibration
is running and has not yet produced evidence. Inspect this existing allocation
rather than submitting another run.

Calibration has no intermediate checkpoint in the live immutable execution.
An isolated draft at `cc47244` on `wip/task19-qualification-recovery` saves
complete calibration group measurements and random state; a focused CPU test
checks equality with uninterrupted computation. It has not been applied to
the live job and requires further review, evidence migration and qualification
before any potential use. Inspect the current allocation first.
The related draft `7492261` checks the v5 batch and numerical checksums,
scientific settings, model, worker, dataset, adapter and unchanged rollout,
policy and objective files before reusing the batch in a future v6 namespace.
These drafts remain isolated and have not been submitted.
The optional v6 configuration is prepared at `fbcdff0`; the broader draft
checks passed 468 tests with 20 skipped and 18 deselected, plus Ruff and mypy.
Reusing the frozen batch still requires the draft's runtime identity checks
and a fresh numerical qualification under its changed execution code.
Draft `f1d9ad9` adds calibration checkpoints between student turns, preserving
all four rewards, fixed normalization, accumulated gradients and random state.
A CPU test matched the original group gradients exactly; the draft's broader
checks passed 469 tests, with 20 skipped and 18 deselected, plus Ruff and mypy.
GPU numerical qualification remains required before use.
The frozen v5 batch contains 16 episodes, 107 turns and 28,943 sampled tokens.
Training checkpoints are currently committed between updates, which can contain
up to 16 episodes; the measured uncached rollout and calibration times make a
four-hour allocation a material interruption risk. Assess the safe allocation
or finer-grained recovery before submitting pilots, and retain the shared ledger.

V5 continuation job `33783098` timed out after 14,407 H200-seconds without
writing calibration. Its 16-episode frozen batch and passing numerical evidence
remain preserved. Total reconciled Task 19 charge is 52,669 H200-seconds
(14.630278 hours). The v6 recovery now binds that batch and its numerical
evidence to the original identities, repeats qualification under the new code,
and checkpoints calibration between student turns. The integrated local suite
passed 469 tests (20 skipped, 18 deselected), Ruff and mypy. After live
reconciliation and a passing preview, v6 was submitted once as job
`33796439`, controller run `408501bbc06f8e6327990c7e`, from immutable commit
`6a41e5f3b18793c9b5cfd04db47712d29e2f4ff1`. Inspect this run and its ledger;
never duplicate it. Monitor the private calibration progress summary and use
actual throughput to assess the remaining budget for all five arms.
The v6 batch identity checks passed and its fresh uncached numerical diagnostic
passed: replay maximum and clip changes were zero; relative gradient and
restored-update differences were 1.2853% and 1.8991%, respectively. Calibration
has started with its first checkpoint; no scale or training outcome is yet known.
Because a training update can exceed the four-hour checkpoint interval,
pilot and training submissions now permit an explicit wall time up to twelve
hours; the default remains four hours. The shared ledger reserves the entire
requested duration, and no longer allocation has yet been submitted.

The first two v6 calibration turns (154 sampled tokens) took about seven
minutes; the full frozen batch has 28,943 sampled tokens, so uncached execution
poses a serious feasibility risk. To investigate precision as the spec requires,
an isolated FP32 cached candidate was prepared on `wip/task19-fp32`, with
fresh behavior probabilities and the same numerical acceptance thresholds.
Its local checks passed 471 tests (20 skipped, 18 deselected), plus Ruff and
mypy; this does not establish GPU correctness. After a passing preview, job
`33797457` / run `af5597e8e9526b06b403b9da` was submitted once from commit
`3722a234c092cd40c49b5f67e8613de5712fc27c`. It is a different precision
diagnostic, not a duplicate of v6. Both jobs share the 80-hour ledger. Inspect
their live states and never duplicate either; choose a training path only after
identity-bound numerical results and measured cost justify it.
V7 was pending due to H200 capacity at the first check. V6 continued and saved
three student turns; a simple token-rate extrapolation suggests calibration
alone could take over 20 hours, but this is only an early estimate and does not
account for changing context lengths or processing costs.
After four v6 student turns were saved, a `USR1` safe-stop request was sent to
job `33796439` to preserve the remaining budget while the FP32 comparison ran.
It exited cleanly after saving a fifth turn, with health `stopped`, in 4,499
H200-seconds. Preserve the checkpoint and ledger; do not rerun it.
V7 started and saved its first 16 training-only episodes in approximately
19 minutes of allocation. FP32 cached execution passed the unchanged numerical
limits: 64 comparisons, zero replay maximum and clip changes, relative gradient
difference 0.001548% and restored-update difference 0.000829%, with cosines
above 0.9999999999. Calibration is underway; no scale or training outcome is
yet available. The source is now on the task branch with the same scientific
files as the immutable job; preserve the separate v6/v7 run identities.
V7 qualification job `33797457` completed exit 0 in 1,521 H200-seconds.
Its training-only sample contained 16 episodes, 31,317 generated tokens and
101 turns; summed generation time was 896.4 seconds. This is an initial
throughput observation, not a pilot cost or benchmark accuracy estimate.
Calibration selected OPSD scale `0.09882062828499757` and proposer scale
`0.062588360570449`; calibration SHA-256 is
`84ddf061e553ed86f2cf686e0aa8b1cd4da3b67f1ea0bd29fae545ddd281b1b7`.
The two proposal groups contained two distinct valid tasks each, but valid
tasks within each group had equal measured solve rates. The nonzero proposer
calibration gradient therefore includes valid-versus-invalid discrimination;
it does not establish meaningful discrimination among valid tasks. Preserve
this limitation and assess proposer rewards in the bounded C/D pilots, as the
specification requires.
The FP32 base evaluation was submitted once after a passing preview as job
`33799453` / controller run `662c76b708905dfc47d55b2f` from immutable commit
`4376159e6ee92d734ce4ea11833c9514bf415e48`, with a six-hour reservation.
Inspect its scheduler and ledger, never duplicate it, and use checkpointed
evaluation records on continuation if needed. The five matched pilots follow.
On its allocated node the base trainer waited in an outbound HTTPS connection
before model initialization; a bounded connectivity check timed out. Future
Task 19 allocations are configured to load the already-qualified pinned model
and tokenizer from the shared cache offline, failing explicitly if unavailable.
The existing base job retains its original immutable code and must be inspected;
this change does not alter or duplicate that active allocation. An explicit
empty-base recovery guard permits a replacement only after the prior job is
terminal, saved no files, and its scientific settings, dependencies and source
files match. The old run and its allocation remain in the shared ledger.
After 18 minutes blocked in HTTPS with no saved files, base job `33799453` was
cancelled and reconciled at 1,121 H200-seconds. The guarded offline replacement
was submitted once as job `33801734` / run `eec3440e2c5b56338c30b555`, from
`783c3c37d2f855b46c513541a1eb21b4b1b0968c`; its ledger records the superseded
run `662c76b708905dfc47d55b2f`. Inspect this existing job, never duplicate it,
and include both base allocations in final accounting. At submission the shared
ledger held 16.613889 charged plus the new six-hour reservation.
The replacement started on the previously affected node, loaded the pinned
model from cache, and reached health `running`; task evaluation is in progress.
It completed the 41-task monitoring split with 31 verified repairs and the
115-task ID split with 88 verified repairs. The 116 external baseline tasks
are in progress. These results use the new model and interface and are not a
controlled comparison with the historical base.
After Yale reauthentication, job `33801734` was verified COMPLETED with exit 0
in 17,082 H200-seconds. The external baseline is 93/116 on previously inspected
families. Monitoring, ID and external summaries share the verified initial
snapshot. Total reconciled Task 19 allocation is 21.358889 H200-hours.
The 41 verified initial monitoring records were staged for all five arms using
the guarded reuse procedure. Never duplicate this completed baseline.
After passing previews, the five 32-episode pilots were submitted once from
`e2fbe083a3c0aff7a20c8d8a7a2d6c5547910190`, each with a four-hour reservation:

| Arm | Slurm job | Controller run |
|---|---:|---|
| Control | 33828139 | `826f6ae808dc186b477bc945` |
| A | 33828371 | `2ae1a78b54cd011fc21623ef` |
| B | 33828377 | `9016ab34ca892ee6f6ae633d` |
| C | 33828380 | `f5c5a6c6f900e5c11b9a9f34` |
| D | 33828391 | `fd4b69da407f990319580e8d` |

Inspect each existing job and its ledger before any action; never duplicate.
W&B runs use offline transport for later synchronization from the authenticated
local workspace. The total committed reservation after submission is at most
41.358889 H200-hours, subject to reconciliation of actual charges.
At the last confirmed observation, control and A were running with matching
initial snapshots and the reused 31/41 monitoring result; B, C and D were
pending. The SSH master then disconnected and the Duo refresh timed out. These
Slurm jobs are independent of the login: after reauthentication, inspect all
five controller runs and the shared ledger before any action, and never assume
their current state or submit duplicates.
After a successful Duo refresh, all five pilots were observed RUNNING and all
five initial snapshots matched the base. Control had committed 32 attempted
episodes and A 16; B/C/D had not committed their first updates. Control's two
updates were finite and nonzero, but neither had distinct valid proposer rewards;
the second had a zero student RL gradient. A's first update was finite and
nonzero and likewise lacked distinct valid proposer rewards. These are training
diagnostics, not independent evaluation outcomes. Inspect the existing jobs
and ledger before continuing; W&B is captured offline pending local sync.
B, C and D subsequently committed finite, nonzero first updates at 16 attempted
episodes. B had no distinct valid proposer-reward groups. C had one of two such
groups and four near-zero candidate alignments; D had two of two and no
near-zero candidate alignments. All used the matched first rollout batch. These
are early training-signal observations, not independent endpoint performance.
All five subsequently committed 32 attempted episodes with finite, nonzero
shared updates. In their second updates, the common difficulty-based arms had
no distinct valid proposer-reward groups; C had none in three groups and six
near-zero alignments, while D had one distinct group of three and no near-zero
alignments. Endpoint monitoring evaluations are in progress. These counts are
training diagnostics and do not establish downstream improvement.
After the offline, recovery and reuse safeguards, the full local suite passed 476
tests (20 skipped, 18 deselected), Ruff lint/format and mypy (61 source files).
After the complete baseline is reconciled, `uv run selfplay hpc
task19-stage-baseline --settings configs/task19-yale.yaml` can copy the
checksum-verified initial monitoring task records to all five arms before any
pilot submission. All arms use the same initial adapter, task seeds and data;
each pilot still verifies every task identity on read. The helper refuses
incomplete baselines, mismatched evidence, altered copies and existing arm
slots. This reuses the single initial evaluation, not later milestone results.
The baseline node and Yale login could not reach `api.wandb.ai`, while the
local workspace could. Future pilot and training jobs therefore record W&B
offline files for later local synchronization. A per-allocation
`state/telemetry-transport-<job>.json` records the true offline transport;
the existing scientific runner's `telemetry.json` has a hardcoded `online`
mode label and must not be treated as proof of cloud delivery. Fetch the W&B
files with full artifacts, synchronize from the authenticated local workspace,
and verify cloud history before claiming delivery. Offline SDK capture was
exercised locally, and the local W&B API authenticated successfully; cloud
synchronization of Task 19 has not yet occurred.

## 96-episode continuations (2026-11-18)

All five 32-episode pilots completed exit 0. Their actual allocations were
4,696/4,804/4,905/4,764/4,804 H200-seconds for control/A/B/C/D. The shared
ledger reconciled 28.018056 H200-hours across all Task 19 attempts. Before
inspecting endpoint treatment outcomes, the controller verified the pilot
checkpoints and froze a common 96-episode endpoint, with 23.725 H200-hours
reserved in the projection for five final evaluations and a 20% contingency.
The private control is `runs/task19/main-control.json`. The base external
families were previously inspected and are not untouched confirmation.

The five main segments were submitted once from immutable commit
`7150a8b31cc94c876998c6f5362186596c59af29`, each with a four-hour reservation:

| Arm | Slurm job | Controller run |
|---|---:|---|
| Control | 33838200 | `562da119c0ee91f60ff3c932` |
| A | 33838204 | `ad1f59aa48704da5e8887dcd` |
| B | 33838208 | `51f3b27715d01a0b843ec36f` |
| C | 33838211 | `ae25de421c2c804155ac8bec` |
| D | 33838215 | `4ce3511806b176d79c16a3a3` |

All were pending at the first scheduler check. Inspect the existing jobs and
ledger before any action; never submit duplicates. They resume the verified
pilot checkpoints and include final ID and external evaluation. If stopped,
reconcile checkpoint and accounting before using the supported continuation.
W&B transport is offline; retrieve and synchronize saved segments locally,
then verify cloud history before claiming delivery. Do not merge as complete
until the five endpoints, final accounting, analysis, and documentation are
complete.

Control and A subsequently started, followed by B and C; D remained pending.
The five pilot artifact sets were fetched locally. The W&B CLI reported a
successful control-pilot offline sync, but API verification found no run. A
subsequent online initialization created the matching control run as an empty
placeholder; retrying the CLI append still left zero cloud history rows. The
pilot `.wandb` file contains four local history records. Treat cloud delivery
as unverified, retain the original files and local sync records, and reconcile
the placeholder before any replay to avoid duplicate history. Main segments
continue recording offline independently.
All five main jobs subsequently started. For control-pilot telemetry only, a
guarded local replay checked that the cloud placeholder had no history, read
four numeric history records from the preserved `.wandb` file, and logged them
to the matching run with original timestamps and runtimes retained as explicit
metrics. The W&B API subsequently returned four rows at steps 0–3. The replay
script and SDK logs are private under `runs/task19/`; do not replay those rows
again. Remaining pilot and continuation telemetry still requires reconciliation
and independent verification. W&B history can become visible after a delay.
The same guarded replay was performed once for each A–D pilot, preserving the
original timestamps and runtimes as metrics and the offline source files.
Cloud API verification returned four rows each for B, C and D; A's upload
reported success but returned zero rows at the first check and needs later
verification. Do not repeat these uploads without checking the cloud history;
visibility can lag delivery. Continuation telemetry has not yet been replayed.
A later API check confirmed A's four pilot rows as well. Thus all five pilot
histories are independently visible, four rows per arm. These are the replayed
aggregate records; original offline files remain the source of record.
All five main jobs subsequently committed iteration 3 at 48 attempted
episodes and began their 41-task monitoring evaluations. They were still
RUNNING at the last check; these milestones are development monitoring, not
final ID or external outcomes. Inspect the existing jobs and ledger before
continuing, particularly if their four-hour segments reach the safe stop.
For local disk pressure, the redundant local `state.pt` binaries for pilot
checkpoints `000000` and `000002` were removed after verifying the protected
Yale copies against their SHA-256 manifests. Their metadata and all local
`000001` binaries remain: the latter had already been pruned remotely and may
be unique. Re-fetch protected binaries before any analysis that requires full
local checkpoints. A reproducible local Hugging Face Xet chunk cache was also
cleared; the local model hub snapshots and remote run data were not removed.
After the Mac reboot and a fresh Yale login, all five original main jobs were
verified RUNNING. Their 48-episode monitoring sets were complete; control had
reached 72 episodes and B 64, while A, C and D remained at 48. The shared
ledger showed 48.018056 H200-hours committed, including all five original
four-hour reservations. Continue inspecting these exact jobs; do not duplicate.
All five arms later committed the 72-episode milestone. Control completed
its 72-episode monitoring set and committed the frozen 96-episode training
checkpoint; its endpoint monitoring was in progress. A–D remained in their
72-episode monitoring evaluations. All five original main jobs were RUNNING
at this observation; final ID/external evaluation is still outstanding.
All five main arms subsequently committed the common 96-episode training
checkpoint. Control completed endpoint monitoring and began the 115-task final
ID split; A–D were beginning endpoint monitoring. All original jobs remained
RUNNING. These are not completed endpoint evaluations; preserve their IDs and
resume only after terminal scheduler and ledger reconciliation if needed.

After reconnecting, the original five main allocations were terminal with clean
`stopped` health at 96 episodes. All endpoint monitoring sets were complete;
final ID task records for control/A/B/C/D numbered 86/9/37/28/24. Their actual
charges brought the shared ledger to 47.598611 H200-hours. After confirming
no original jobs remained active, the supported continuation was submitted
once for each existing logical run from checkpoint `000007`:

| Arm | Continuation Slurm job | Existing controller run |
|---|---:|---|
| Control | 33864243 | `562da119c0ee91f60ff3c932` |
| A | 33864248 | `ad1f59aa48704da5e8887dcd` |
| B | 33864251 | `51f3b27715d01a0b843ec36f` |
| C | 33864328 | `ae25de421c2c804155ac8bec` |
| D | 33864330 | `4ce3511806b176d79c16a3a3` |

All were initially pending. Including their original four-hour reservations,
committed budget was 67.598611 of 80 H200-hours. Inspect these jobs and their
ledgers before any action; never duplicate. The final ID and external splits
remain incomplete. Reconcile actual charges after terminal state, verify
checkpoints and completion markers, fetch required records without deleting
unique local files, synchronize continuation telemetry, analyze and document.
All five evaluation continuations subsequently started. The terminal main
segment's eight numeric W&B history rows per arm were recovered from its
preserved offline files with guarded append after the four verified pilot rows;
original segment timestamps, runtimes and steps were retained as explicit
metrics. API checks confirmed 12 history rows for each arm. The recovery
scripts, source hashes and logs remain private under `runs/task19/`. Do not
replay these segments again. The active continuation's telemetry is separate
and still requires reconciliation after it stops.

After a fresh login on November 19, continuation jobs `33864243`, `33864251`,
`33864328`, and `33864330` completed exit 0 with full final ID and external
splits and completion markers for control, B, C, and D. Never duplicate.
A continuation `33864248` stopped cleanly after 102/116 external tasks; its
96-episode checkpoint and completed ID split remain intact. The reconciled
charge for all terminal allocations was 64.568889 H200-hours. A's existing
controller run `ad1f59aa48704da5e8887dcd` was resumed once from checkpoint
`000007` as Slurm job `33902234`. Inspect this existing job and the shared
ledger; never duplicate. Completion requires its remaining 14 external tasks,
checkpoint verification, final accounting and full analysis/documentation.
