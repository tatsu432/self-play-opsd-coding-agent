# Task 17 implementation and execution plan

## Implemented locally

Exact replacement, observation persistence, interface-aware episode identity,
static mechanical recovery, deterministic family partitions, isolated pool and
replay runners, paired frozen-model runner, conditional calibrated/static student
runner, and paired interface analysis. See the [implementation specification](../task/17-repair-interface-spec.md).

The [results report](../investigations/task17-repair-interface-results.md) records
actual CPU evidence and external blockers. Implementation and CPU checks do not
establish Yale worker acceptance, model performance or training improvement.

## Resume Phase 1 when Yale DNS/access returns

Use the normal Yale connection; do not bypass authentication. First inspect live
Slurm, account resources and existing ledgers. No Task 17 job has been submitted.
Commands below assume a clean committed checkout and verified existing images:

```bash
uv run selfplay hpc inspect
uv run selfplay hpc sync --settings configs/task17.yaml
uv run selfplay hpc reuse-worker repair-selfplay-2026-task14-online-v1 --settings configs/task17.yaml
uv run selfplay hpc submit --operation runtime-check --method B --profile single_gpu_80gb --settings configs/task17.yaml
uv run selfplay hpc submit --operation task17-prepare --method B --profile single_gpu_80gb --settings configs/task17.yaml
```

Inspect each existing run ID until it completes; never resubmit an uncertain
operation. The prepared private pool is
`/runs/repair-selfplay-2026-task17-interface-v1/prepared/pool.json` inside the
container. It contains isolated verification evidence; local partitions alone
are not a verified pool.

Submit strict control cells sequentially, inspecting allocation cost and the
first eight completed tasks before continuing the design:

```bash
uv run selfplay hpc submit --operation task17-interface --method A --seed 2026 --profile single_gpu_80gb --gpu-type h200 --settings configs/task17.yaml
uv run selfplay hpc submit --operation task17-interface --method C --seed 2026 --profile single_gpu_80gb --gpu-type h200 --settings configs/task17.yaml
```

Then run the dependency-matching CPU runtime check for
`configs/task17-text.yaml` and submit interface cells B and D with that settings
file. Both settings use the same data/study and budgets. Do not repeat unchanged
worker containment or historical checkpoint searches. A target backend check is
needed only for a demonstrated compatibility gap; it counts against Phase 1.
The runner checks context admission before any rollout.

For mechanical semantic replay, upload only the private checksummed
`runs/task17/recovery-v4/replay-candidates.json` to the verified workspace's
`runs/repair-selfplay-2026-task17-interface-v1/replay-input/` directory using the
existing SSH transport, then use `--operation task17-replay` with the Task 17
settings. This CPU operation verifies each reconstructed edit independently at
its historical source state; it does not continue counterfactual trajectories.

Fetch each operation with the existing `hpc fetch RUN_ID` route. Create a private
cell manifest with `cells: [{arm,run_dir,allocated_gpu_hours}]`, including actual
failed/partial allocation costs, then analyze:

```bash
uv run python scripts/analyze_task17_interfaces.py --manifest runs/task17/interface-manifest.json --pool runs/task17/prepared/pool.json --output runs/task17/interface-analysis
```

Do not manufacture success/acceptance figures before these results exist.
Report original-task paired parent-macro/family-bootstrap effects separately
from presentation variants. Review repair success, edit acceptance, test usage,
repeated errors, turns/tokens, length strata, termination and actual allocation.

## Conditional Phase 2

After the Phase 1 decision, freeze the evidence-backed objective control described
in the specification. Run `task17-calibrate` once. Transfer its measured coefficient
and checksum into that control before selecting U-calibrated. Generate per-arm
settings by applying `configure_operation` to the common settings; commit those
resolved settings and control before synchronization. This binds controller and
trainer to the same effective objective, coefficient and endpoint.

Use `--operation task17-student` with A=U, B=gated, C=calibrated-U, D=GRPO and the
corresponding resolved settings. Pilot seed 2026 with
`--stop-after-iteration 4`; inspect its complete joint checkpoint and numerical
health before a clean continuation. `hpc resume RUN_ID --continue-stopped` uses
verified checkpoint/runtime/science identity. Reconcile the separate 24-hour
ledger before each continuation/seed. Keep common exposure; reduce endpoint to
80 before freezing if measured costs make 128 infeasible. Preserve G/B replication
before expanding optional arms. No Phase 2 control, calibration or training has
been fabricated while Phase 1 is unexecuted.

## Completion

Finish empirical decisions and required outcome/efficiency figures after access
returns. Reconcile W&B from authoritative local events if delivery fails. Update
README and completion record with actual results/costs, then merge finished work
into local main and push. Full C/D remains outside this authorization.
