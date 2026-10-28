# Task 09 — CPU implementation and pending H200 acceptance

Status: implementation staged on `feat/09-efficiency-learnability`; **the new
study is not launch ready**. Task 08 remains launch ready on its frozen source.
No Task 09 GPU job or main A–D job was submitted.

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
2. Obtain separate authorization for at most four allocated H200 GPU-hours of
   Task 09 benchmarking, then reconcile live Slurm and shared budget ledgers.
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
