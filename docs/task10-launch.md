# Task 10 — online launch and recovery

**Current recommendation: NO-GO for an unqualified main launch pending the
measured proposer numerical-sensitivity decision.** Main A–D training still
requires explicit scientific-risk acceptance or an approved correction, and
separate GPU-hour authorization. Use this workflow with
`configs/task10-online.yaml`, study `repair-selfplay-2026-main-online-v1`.
Frozen execution is `54f087fdc5cf5170800d825bf1ee92b89a43c8ed` (CPU freeze
31206522). The original Task 09 v2 freeze remains available as a separate offline fallback.
See the [implementation record](task_completion/10-implemented-main-experiment.md)
for executed jobs, validation, costs and current GO/NO-GO status.

## Credentials and verified delivery

Run the 0.22.3 login tool locally in a terminal. The [0.22.3 release](https://github.com/wandb/wandb/releases/tag/v0.22.3) supports
new-format keys longer than 40 characters; the pinned trainer SDK 0.22.2
rejects them in the interactive credential-save path. The isolated login tool
does not modify the trainer lock/image. Do not paste the key into chat:

```bash
uv tool run --from wandb==0.22.3 wandb login
uv run python scripts/setup_wandb.py --from-login "$HOME/.config/selfplay/wandb.env"
```

The setup helper uses the authenticated default entity. Username is `tatsu432`;
`tatsu432` and `tatsu432s` are user-supplied candidate entities, not verified facts.
If needed, specify the authorized entity with `--entity`. The helper refuses to
overwrite an existing file. It creates the credential file with mode 600 from
creation. Do not place it inside this checkout. Securely provision the same file
on the compute-visible Yale home filesystem, preserving mode 600 and ownership:

```bash
ssh yale-hpc-cpsc4880 'umask 077; mkdir -p "$HOME/.config/selfplay"'
scp "$HOME/.config/selfplay/wandb.env" yale-hpc-cpsc4880:/home/cpsc4880_ts2236/.config/selfplay/wandb.env
ssh yale-hpc-cpsc4880 'chmod 600 "$HOME/.config/selfplay/wandb.env"'
```

No command includes the key. Only the private file path goes into scheduler
arguments/ledgers. The shell loads it without tracing; the trusted trainer gets
`WANDB_API_KEY` and `WANDB_ENTITY` through Apptainer's explicit environment.
Generated-code payloads receive neither credentials nor the credential file.
The submission's persistent secrets-file path is retained by continuation.

From a clean committed checkout, verify HTTPS, actual cloud history, and a second
independent process resuming the same test run on a **CPU-only compute node**:

```bash
uv run selfplay hpc sync --settings configs/task10-online.yaml
uv run selfplay hpc submit --settings configs/task10-online.yaml --operation wandb-verify --method D --profile single_gpu_80gb --gpu-type h200 --account cpsc4880 --secrets-file /home/cpsc4880_ts2236/.config/selfplay/wandb.env
uv run selfplay hpc status VERIFY_RUN_ID
uv run selfplay hpc logs VERIFY_RUN_ID
uv run selfplay hpc fetch VERIFY_RUN_ID
```

This operation requests zero GPUs, two CPUs, 8 GiB, and at most 15 minutes. Its
report must say `ready=true`, `cloud_verified=true`, `sequence=2`, and contain
both cloud sequences plus the actual project/run URL. Test runs belong to
`telemetry-verification`; they are synthetic transport checks, not main results.
Online main previews/submissions reject missing cloud/resume proof or a missing
persistent secrets-file path. Re-check credentials if rotated. Never declare
cloud delivery from a local directory, an SDK enqueue, or `wandb.init()` alone.

If direct egress fails, preserve local events. A foreground local reconciler
can upload fetched checkpoint-backed aggregates after stopping the original
writer; this is manual recovery, not a validated continuous relay. Do not claim
near-real-time fallback until an actual supported relay has been tested. Do not
create SSH tunnels or login-node daemons.

## Freeze, inspect and preview

The online study reuses the original measured evidence only when scientific,
model/data/prompt/image, allocation-verifier, hardware and checkpoint-frequency
dependencies match exactly. Source reports/checkpoints stay intact. Reused method
proofs record their original study and SHA-256 and verify checkpoints in place.

The following preparation commands allocate CPU only for freeze:

```bash
uv run selfplay hpc reuse-study repair-selfplay-2026-efficiency-v2 --settings configs/task10-online.yaml
uv run selfplay hpc submit --settings configs/task10-online.yaml --operation freeze-study --method D --profile single_gpu_80gb --gpu-type h200 --account cpsc4880
uv run selfplay hpc doctor --settings configs/task10-online.yaml --gpu-type h200
uv run selfplay hpc inspect
```

Evidence reuse is idempotent before freeze and refuses a frozen target. Do not
repeat it after successful freeze. A conflicting freeze needs a distinct study;
never overwrite the accepted freeze or edit its evidence in place.

After cloud verification, run four real controller previews (no GPUs allocated):

```bash
for method in A B C D; do
  uv run selfplay hpc submit --settings configs/task10-online.yaml --operation train --method "$method" --profile single_gpu_80gb --gpu-type h200 --account cpsc4880 --secrets-file /home/cpsc4880_ts2236/.config/selfplay/wandb.env --dry-run
done
```

Require 200 iterations, fresh initialization, the frozen execution commit, one
H200, seven trainer CPUs/128 GiB plus one worker CPU/1 GiB, 23-hour segments,
stop leads A/B/C/D 4616/4540/4717/5211 s, and 5/5/5/6 clean continuations.
Check space, no active/duplicate main jobs, and explicit main GPU-hour authority.
The shared readiness allowance is never the main-training budget.

## Later authorized main launches

**Prepared commands; do not execute without main-experiment authorization.**
Start A first:

```bash
uv run selfplay hpc submit --settings configs/task10-online.yaml --operation train --method A --profile single_gpu_80gb --gpu-type h200 --account cpsc4880 --secrets-file /home/cpsc4880_ts2236/.config/selfplay/wandb.env
```

After the initial review and separate authorization for remaining methods:

```bash
for method in B C D; do
  uv run selfplay hpc submit --settings configs/task10-online.yaml --operation train --method "$method" --profile single_gpu_80gb --gpu-type h200 --account cpsc4880 --secrets-file /home/cpsc4880_ts2236/.config/selfplay/wandb.env
done
```

Record Slurm ID, controller/logical IDs, actual W&B URL, execution commit,
initial checkpoint, resource request and continuation allowance. Independent
methods can use separate H200 allocations when authorized and schedulable.
Do not change the recipe between methods based on early outcomes.

## Dashboard and interpretation

One logical cloud run per method, grouped by scientific study, uses these names:

- `A-difficulty-ungated-opsd`
- `B-difficulty-gated-opsd`
- `C-rl-alignment-gated-opsd`
- `D-combined-alignment-gated-opsd`

Use `dashboard/*` custom charts and latest `canonical/training` and
`canonical/development` aggregate tables as authoritative views. Scalar history
is append-only audit data and can contain superseded or repeated deliveries;
do not use it as canonical history after rollback. New segments replace the
canonical tables from active local lineage and clear stale scalar summaries.
An empty rollback view explicitly clears the corresponding chart. No cloud
history rewriting or exactly-once SDK-delivery guarantee is claimed.

| Section | Available event-derived metrics / interpretation |
|---|---|
| Progress | Iteration/200, both step counters, active hours, logical/segment IDs, last checkpoint/commit timestamp, recent iteration seconds; ETA starts after five observations and is conditional on observed yield |
| Student | Repair/public/hidden full-pass fractions, successes/episodes, Bernoulli reward mean/std, all-zero/all-success/mixed groups, invalid actions/test use, RL/OPSD/total signed surrogate losses, total gradient norm |
| Gradient influence | Sampled RL and lambda-scaled OPSD norms, ratio and cosine, at the existing diagnostic frequency; missing diagnostics stay missing |
| Proposer | Attempted/valid/cumulative proposals, validity/duplicates, reward distribution, fixed safe invalid-reason taxonomy, hashed detailed reason categories, mutation sizes, loss/norm/retries |
| C/D alignment | Cosine and candidate-norm distributions, reference norm, score count, exact/near-zero/negative/neutral fractions; A/B are explicitly not applicable |
| OPSD/numerics | Gap/gate distributions, positive/negative gap fractions when local shards exist, saturation/clipping, mapped-token/error counts, rollout/trainer discrepancy mean/max/tail fractions |
| Development | Only completed dev-monitor evaluations; parent-macro and episode success against iteration and actual sampled allocation hours; initial/best/checkpoint/improvement summaries |
| Efficiency | Existing proposer/student generation throughput, proposal/validation/rollout/reward/scoring/alignment/update/checkpoint/evaluation phase times, host/descendant RAM, CPU sample and GPU memory samples |
| Infrastructure | SDK enqueue timestamp/errors/watermark and local recovery warnings; cloud visibility is separately verified, never inferred from enqueue |

`sd_loss` is the lambda-scaled signed-gap surrogate; losses can be negative.
Loss reduction is not a repair-quality metric. Separate RL/OPSD gradient norms
are sampled, not interpolated between diagnostics. No new full-gradient work is
performed for charts. GPU utilization and optimizer-time importance-ratio
clipping are not currently present in main iteration events; use the numerical
report for the replay clipping analysis, not invented dashboard zeros. GPU and
host peaks remain sampled lower bounds.

`compute/allocated_gpu_hours` comes from Slurm `ElapsedRaw` and allocated GPU
counts across all attempts, including startup and abandoned work. Typed and
untyped GPU TRES are not double-counted. The host samples once per minute;
rows include sample age and Slurm job ID. Accounting lag/missing samples leave
the physical-hour axis unset. `compute/trainer_gpu_hours_estimate` is a separate
trainer-elapsed estimate. Paid-rate denominators use actual samples when present.
No zero-hour measurements or phase sums substitute for missing accounting.

Only aggregate numerics and approved identity metadata are uploaded. Raw
trajectories, source contexts, hidden cases/oracles, reference solutions and
resumable/adapter checkpoints remain local. Existing detailed example tables
remain local even if an older config enables checkpoint uploads.

## Health, recovery and evaluation

Monitor using the controller's returned RUN_ID:

```bash
uv run selfplay hpc status RUN_ID
uv run selfplay hpc logs RUN_ID
uv run selfplay hpc fetch RUN_ID
uv run python scripts/scientific_health.py FETCHED_METHOD_DIRECTORY --through-iteration 5 --allocated-gpu-hours RECONCILED_HOURS
uv run python scripts/scientific_health.py FETCHED_METHOD_DIRECTORY --through-iteration 10 --allocated-gpu-hours RECONCILED_HOURS
uv run python scripts/scientific_health.py FETCHED_METHOD_DIRECTORY --through-iteration 20 --allocated-gpu-hours RECONCILED_HOURS
```

Use the actual fetched method directory printed by fetch. Default fetch omits
large raw shards/checkpoints; `--full` is needed for trajectory-level pattern
analysis or local checkpoint-backed telemetry reconciliation. Reports explicitly
state which shards are missing. Supply matching-window reconciled GPU-hours
when analyzing a historical cutoff. Review updates, valid/mixed groups, zero RL
signal, C/D degeneration, numerical disagreement, timing and development evidence.
No post-training dev improvement is required before the first evaluation at 20.
Underperformance triggers review and recommendation, not an automatic stop.

The foreground watcher is optional and disconnect-sensitive. After SSH loss,
reauthenticate normally, inspect the existing run, and then continue a verified
clean stop without duplicate submission:

```bash
uv run selfplay hpc login
uv run selfplay hpc status RUN_ID
uv run selfplay hpc logs RUN_ID
uv run selfplay hpc resume RUN_ID --continue-stopped
```

Joint checksummed checkpoints restore both adapters/optimizers, RNG and
curriculum. Partial optimizer states are never resumed. Clean continuations use
the frozen per-method allowance; transient retries remain capped at two for
confirmed PREEMPTED/NODE_FAIL/BOOT_FAIL. OOM, scientific, authentication, quota,
worker-boundary and ambiguous submission failures require manual investigation.
W&B outages do not cancel training; local events are the replay source.

After stopping the original W&B writer and fetching a full trusted run:

```bash
uv run selfplay wandb-reconcile --run-dir FETCHED_METHOD_DIRECTORY
# Produce a safe aggregate-only offline recovery segment if disconnected:
uv run selfplay wandb-reconcile --run-dir FETCHED_METHOD_DIRECTORY --offline
uv run wandb sync --entity tatsu432s --project self-play-opsd-coding-agent --id LOGICAL_WANDB_RUN_ID --append --skip-console SAFE_AGGREGATE_OFFLINE_SEGMENT
```

Prefer online `wandb-reconcile` after connectivity returns. An offline recovery
archive has its own local SDK ID: specify the original logical W&B ID with
`--id --append` when explicitly synchronizing it, or a companion cloud run will
be created. Scalar appended history remains audit data; rerun online reconciliation
to verify the final canonical cloud revision.

Do not bulk-sync historical pre-Task-10 offline runs: they may contain raw
example tables. The reconciler verifies the latest checkpoint, takes a writer
lock, rebuilds canonical aggregate tables, and verifies the cloud revision in
online mode. It refuses concurrent writers in the same filesystem. Do not run
it against a copied directory while a remote writer for the same cloud ID is
active; the filesystem lock cannot span copies. Reconciliation never loads the
model/checkpoint pickle or changes scientific events. No daemon is installed.

Final evaluation is prepared for later separately authorized GPU work. Use the
common-budget checkpoint; do not repeatedly evaluate held-out final splits:

```bash
for method in A B C D; do
  for split in final_id external; do
    uv run selfplay hpc submit --settings configs/task10-online.yaml --operation evaluate --method "$method" --profile single_gpu_80gb --gpu-type h200 --account cpsc4880 --checkpoint-iteration 200 --split "$split" --endpoint primary --secrets-file /home/cpsc4880_ts2236/.config/selfplay/wandb.env
  done
done
```

These commands preserve the frozen main execution. Development monitoring and
final test performance must remain separately labelled.
