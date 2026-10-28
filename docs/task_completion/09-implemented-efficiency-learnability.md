# Task 09 — implementation and H200 readiness

Status: **GO WITH CAVEATS** for the new v2 study: all readiness gates and four
actual submission previews passed. Main training remains subject to the
separate authorization in Task 09 §1; no main A–D job was submitted. Task 08
remains a separately frozen launch-ready baseline. Historical status and
estimates below are retained for provenance and superseded by this section.

Study `repair-selfplay-2026-efficiency-v2`, settings
`configs/task09-diagnostics.yaml`, frozen execution `93a8533beeae26e54bb1510caa12991fb2471337`:
runtime 31151547, greedy base 31156334 (0/41), D pilot 31158975/31161158
(verified resume at two, ten iterations, 14/160 valid, 56 episodes, 14
successes and eight steps per optimizer), A/B/C suite 31169497 (each three
iterations, six valid tasks, 24 episodes and three steps per optimizer), and
CPU freeze 31197007 passed. Worker and backend proofs were reused with their
dependency-scoped provenance. Hosted Linux/Docker correctness CI 40748915396
passed for the frozen source.

The resource plan binds the reports and measured Slurm accounting: suite 2996 s,
pilot 2695 s, base 1058 s; file SHA-256
`6e20c52ab4ee5ebcb6fb646461cae7a6b17702afbc033a1b484abca8790d0fe7`.
Observed-yield projections for A/B/C/D are 17.69/17.33/18.00/15.80 GPU-hours;
conservative planning projections are 82.06/79.94/84.82/97.95 hours, or
344.78 total, including monitoring and excluding final evaluations. These are
conditional scenarios based on short samples, not guarantees; longer responses
or different yield may exceed them. Shared suite overhead is apportioned once.
Four previews used 23-hour segments, 200 iterations, fresh initialization and
stop leads 4616/4540/4717/5211 s with 5/5/5/6 continuations. Preview IDs are
`f244f8eec337077c09b7d40d`, `8af244041ef9c1fee8c67d05`,
`e223ab1b20daeb7de548f11f`, `108c801d89e63a856490fdcf`; all reported
`submitted=false`. The final shared readiness ledger was 7.478055556/24
GPU-hours consumed with no outstanding reservation. Reconcile live state
before operations.

Before launch, the B and D main directories contained only diagnostic
`state/vllm.log` files. They were moved intact to
`runs/readiness/repair-selfplay-2026-efficiency-v2/pre-main-diagnostic-logs/`
outside the main namespace; B and D log SHA-256 values are
`32d925aba41feb68af6cf4e358e7e7a71e4d91f7e07981672a38a56cd9d872cc`
and `36f3d523c7879d7f9f5691eb0f14dbdeab272540216929bf349fc500818a815a`.
No main training artifacts or active jobs were present.

Main risks remain: greedy base repairs were 0/41; five of eight sampled D RL
gradients were zero. The sampled rollout/scoring discrepancy affected one
mixed-reward diagnostic gradient by 9.9% relative L2 (cosine 0.9985); its
learning effect is unknown. Review scientific health at iterations 5–10 and
20. Faster batching, graph and cache variants are not promoted because their
sampled behavior differed; longer-context memory and end-to-end optimization
benchmarks remain unmeasured. The controller's foreground watcher is not
durable; use the documented verified manual continuation if disconnected.
See the [launch runbook](../yale-launch-runbook.md) for exact commands.

### Latest continuation and numerical diagnostic

The v2 runtime validation 31151547 and base evaluation 31156334 passed. D
pilot 31158975 reached its verified iteration-two checkpoint; continuation
31161158 completed all ten iterations with verified resume at iteration two,
14 valid proposals, 56 episodes and eight actual steps for each optimizer.
Mixed-reward gradient diagnostic
31162121 completed successfully. Its diagnostic-only counterfactual compared
the actual rollout-log-probability RL gradient against a ratio-one rescoring
for one mixed-reward group `[0, 1, 0, 0]`: relative gradient L2 difference was
0.09909 and cosine 0.99853. This single group does not establish the effect on
learning, and no objective or optimizer step was changed. The pilot watcher
exited on a lost SSH connection and was restarted after reconnection; it
reported the existing run scientifically complete without another submission.

On 2026-10-28 normal Yale Duo reauthentication restored access after a
connection interruption. The last observed shared budget
was approximately 5.83/24 GPU-hours consumed, with reservations for the pilot
continuation and diagnostic; it is not a reconciled final total. A/B/C v2
acceptance, measured plan publication, freeze, previews and hosted CI remain
outstanding. Main training remains unauthorized.

Latest readiness: allocation 31091793 and backend 31091980 passed; runtime
31092884 and greedy base 31133057 passed. D pilot 31138473/31141205 completed
ten iterations with a verified resume from two, 15 valid proposals and 60
episodes. A/B/C suite 31144727 passed all method gates. CPU freeze 31149887
and four controller previews succeeded, but the immutable freeze has no
resource plan: the previews fall back to a 484-second stop lead and one clean
continuation, without a high-yield or monitoring envelope. **Do not launch this
Task 09 freeze.** Preserve it for audit and bind measured planning evidence in
a correctly frozen study. Optional optimization benchmarks remain outstanding.
The shared ledger reported 5.052777778/24 GPU-hours committed, with no active
GPU reservation at this checkpoint. Reconcile live state before further work.

The D pilot recorded maximum pre-update rollout/scoring log-probability
differences up to 2.712; the existing backend probe checks only short synthetic
sequences. The maxima alone do not establish prevalence or cause. Follow-on
instrumentation records token-weighted mean absolute difference and fractions
above 0.05 and 0.2, without changing losses. It changes the scientific code
identity and uses new study `repair-selfplay-2026-efficiency-v2` in
`configs/task09-diagnostics.yaml`. It requires target-hardware validation and
does not make the v1 freeze launch ready.

Trusted H200 replay 31153613 reproduced the initial-snapshot discrepancy: mean
absolute difference 0.00369, p99 0.1004, maximum 1.138, and 18/3621 tokens
outside the GRPO clip interval. A checksummed iteration-nine checkpoint and
iteration-ten trajectories gave mean 0.00330, p99 0.0996, maximum 0.728 and
21/4054 outside the interval. These are observed counts, not an assessment of
the impact on learning. The replay verified the checkpoint's policy version,
manifest and prompt and did not execute generated candidates.

On 16 fixed initial proposer requests, warm sequential and batch generation
took 52.80 and 6.29 seconds, with identical output IDs, but the maximum difference
between sampled log-probabilities was 1.229. Scoring microbatches 2 and 4 had
maximum differences of 1.037 versus size 1. These options are **not accepted for
main training**. Follow-on jobs 31154488, 31154500, 31154505 and 31154513
completed baseline, prefix cache, CUDA Graphs and combined comparisons. Warm
sequential/batched times were 54.59/6.55, 55.79/5.80, 11.51/1.66 and 11.48/1.55
seconds respectively. Batch versus sequential IDs matched for 15/16, 14/16,
14/16 and 16/16 requests; matching-ID token log-probability differences exceeded
0.2 on 3, 7, 4 and 11 tokens. Whole-device sampled peaks were 73647 MiB for
eager and 74669 MiB for graph configurations. These short samples do not prove
long-context memory fit or end-to-end speed. None of these settings is promoted.

The C shortcut was compared with actual backward computation in a tiny CPU
model and matched exactly. H200 replay job 31155817 also matched exactly on
the checksummed iteration-nine checkpoint and equal-reward iteration-ten group
(candidate norm zero, reference norm 6.2220, reward 0.5). Updated-study
allocation check 31155886 passed.

The new measured conditional planning estimates are A 75.84, B 72.90, C 75.08,
and D 88.65 GPU-hours; observed-yield estimates are 16.31, 15.94, 16.39, and
15.20 hours. These are scenarios, not guarantees, and exclude final evaluations.
The corresponding stop leads are 4349/4243/4322/4812 seconds and clean
continuation allowances 5/5/5/6. `scripts/analyze_task09_workload.py` reproduces
them from fetched events and reconciled accounting. The A/B/C suite used one
2683-second allocation: event-to-fixture intervals are attributed to each
method and remaining shared time is split equally, without charging the same
allocation three times. This is an estimate, with limited samples and lower-bound
memory observations. It is not attached to the premature immutable freeze.

## Historical blocker (resolved for v2)

Task 09 cannot be declared ready for the main experiment until its changed
execution path passes H200 numerical, memory, performance and recovery checks,
followed by the relevant readiness gates, a new freeze and four previews. The
user authorized GPU readiness work within the cumulative 24-hour limit. The
resource plan was absent at freeze; preserve that immutable record and bind
measured planning evidence to a correctly frozen study. The requested H200
optimization comparisons also remain incomplete.

The local Docker daemon was unavailable, so the new sandbox-backed trainer
tests and hosted Linux/Docker CI remain unverified. This is an additional
validation gap. Source commit `3019394ac60619a8970cf8f43fa84b1ed10b1fb9`
and the frozen dataset were synchronized to Yale; this does not establish
compatibility. The unchanged Task 08 frozen study has passed readiness and four
previews, but submitting its main runs requires separate authorization under
Task 09 §1 and the project operating instructions.

## Changes and evidence

- Added the exact-zero C candidate-gradient shortcut. It retains the neutral
  score and distributed alignment diagnostics. D still computes its combined
  gradient. The existing reference gradient is already computed once per
  iteration, so no reuse change was made.
- Added bounded, ordered proposer batching through `GenerationRequest`,
  `generate_many`, and the vLLM RPC. The pinned [vLLM 0.11.0 API](https://docs.vllm.ai/en/v0.11.0/api/vllm/entrypoints/llm.html)
  supports per-request sampling parameters, seeds and an ordered result list.
  The default batch size is one until actual H200 parity and speed are proven.
- Added hardware settings for proposer batch size, prefix caching and eager
  execution. Existing microbatch and activation-checkpointing settings remain
  available. Added proposer/solver request and token-throughput counters.
- Added `scripts/scientific_health.py`, a read-only summary over committed
  events and available raw shards. Pass an actual reconciled allocation with
  `--allocated-gpu-hours` for rates; it does not infer paid time from phase sums.
- Added `configs/task09.yaml` with a new study ID. Its scientific settings,
  excluding study ID, equal Task 08's. Existing freeze and evidence remain intact.

CPU verification: `uv run pytest -q` passed **217 tests**, 36 skipped (including
Docker sandbox coverage); `uv run ruff check src tests scripts`,
`uv run ruff format --check src tests scripts`, and
`uv run mypy src/selfplay_opsd` passed. Batch-RPC order/seed/log-prob tests pass
locally. The full trainer batching and C shortcut tests require the sandbox
and were skipped locally; hosted CI and accepted Yale tests remain pending.

## Existing scientific-health baseline

The local report reproduced Task 08 committed evidence:

| Method | Iterations | Valid/attempted | Successful episodes | Outcome groups (zero/mixed/all-one) | Near-zero alignments |
|---|---:|---:|---:|---:|---:|
| A | 3 | 6/48 | 1 | 5/1/0 | — |
| B | 3 | 6/48 | 0 | 6/0/0 | — |
| C | 3 | 6/48 | 1 | 5/1/0 | 5/6 |
| D | 10 | 18/160 | 7 | 15/2/1 | 0/18 |

D had 8 zero RL gradients in 10 sampled diagnostics and 240 patch-tool errors.
The report distinguishes these from actual development improvement, which is
not established by these short windows. Review after iterations 5–10 and around
20: continue when finite updates, valid work and improving or inconclusive
development evidence coexist; investigate persistent invalid actions, zero RL
signals and degenerate alignment; stop on numerical, isolation or checkpoint
failure. No early success threshold is a readiness gate.

## Performance and operations

| Metric | Frozen Task 08 baseline | Task 09 optimized |
|---|---:|---:|
| D proposal generation and validation | 651.1 s / 10 iterations | Unmeasured |
| D rollout and scoring | 1,039.8 s / 10 iterations | Unmeasured |
| D alignment | 191.6 s / 10 iterations | Unmeasured |
| D student and proposer updates | 192.6 s / 10 iterations | Unmeasured |
| D checkpoints | 63.3 s / 10 iterations | Unmeasured |
| Peak whole-device memory | 71.81 GiB sampled | Unmeasured |
| A–D conditional observed-yield projection | 70.24 GPU-hours | Unmeasured |
| A–D conservative planning projection | 329.45 GPU-hours | Unmeasured |

These phase values are from the stored Task 08 analysis and must not be summed
as if every timer were disjoint. Queue delay and new-study calendar makespan are
unmeasured. Task 08 checkpoint/continuation evidence remains valid for its
unchanged source. New code needs dependency-scoped numerical, rollout and
checkpoint validation. Current `hpc watch --continue-stopped` depends on a
foreground process; manual `hpc resume RUN_ID --continue-stopped` is the bounded
fallback after checking `hpc status`, `hpc logs` and `hpc fetch`. Ambiguous jobs
are inspected, never resubmitted blindly.

## Next authorized stages

1. Obtain hosted Linux/Docker CI for the new branch and fix any failures.
2. Reconcile the existing backend job and shared budget before profiling.
3. Benchmark default and candidate settings on fixed representative workloads,
   verify sampled-token/log-probability and optimizer parity, memory and recovery;
   accept only demonstrated end-to-end gains.
4. Rebuild/reuse images by changed dependencies, run affected readiness gates,
   recompute resource/continuation plans, freeze the new study and preview A–D.
   Keep main submissions pending separate authorization.

Read-only checks: `uv run selfplay hpc doctor --settings configs/task09.yaml
--gpu-type h200` and `uv run selfplay hpc inspect`. Use normal `hpc login` if
MFA is required. The existing Task 08 launch commands remain in the
[launch runbook](../yale-launch-runbook.md); they still pin frozen execution.
