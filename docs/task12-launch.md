# Corrected study launch

The corrected study `repair-selfplay-2026-syntax-fix-v1` is frozen and passed all readiness gates and four actual dry-run previews. The user has authorized submissions needed to complete the corrected study, including main GPU allocations and necessary continuations. Inspect live scheduler and run state before submitting. See [Task 12 completion](task_completion/12-implemented-corrected-study-readiness.md) for identity, evidence, costs and scientific caveats. The original A and partial B remain historical and must not be resumed into this study.

Use `configs/task12-corrected.yaml`, immutable `data/protocol-v3`, account `cpsc4880`, profile `single_gpu_80gb`, one H200, seven trainer CPUs/128 GiB plus one worker CPU/1 GiB, seed 2026, and 200 iterations. Main execution is pinned by the controller to frozen commit `c307c9800d36a879a295dbaa2741cf5c18d6fa1b`. W&B credentials are in the private mode-600 file on Yale, outside Git. The shared readiness ledger had used 9.838611111/24 allocated GPU-hours, with no reservation at closeout; main allocations are accounted separately.

Before any later authorized submission, establish the normal Duo-backed SSH control connection and inspect current state and budget:

```bash
uv run selfplay hpc login
uv run selfplay hpc inspect
uv run selfplay hpc doctor --settings configs/task12-corrected.yaml --gpu-type h200
```

Check for existing submissions before taking action; never duplicate an uncertain submission. Actual no-allocation previews can be repeated using:

```bash
for method in A B C D; do
  uv run selfplay hpc submit --settings configs/task12-corrected.yaml --operation train --method "$method" --profile single_gpu_80gb --gpu-type h200 --account cpsc4880 --secrets-file /home/cpsc4880_ts2236/.config/selfplay/wandb.env --dry-run
done
```

Begin with the following bounded first A allocation through iteration 20. Review scientific health and telemetry at iterations 5, 10 and 20 and verify the joint checkpoint before deciding on continuation.

```bash
uv run selfplay hpc submit --settings configs/task12-corrected.yaml --operation train --method A --profile single_gpu_80gb --gpu-type h200 --account cpsc4880 --secrets-file /home/cpsc4880_ts2236/.config/selfplay/wandb.env --stop-after-iteration 20
```

The frozen measured plan uses 23-hour segments and stop leads of 4730/4753/4662/5755 seconds for A/B/C/D. Its total 346.58 GPU-hour planning scenario includes training and development monitoring only; final evaluations and diagnostics are extra, and real costs may differ. Review the initial A results before deciding on continuation and subsequent methods. Fresh corrected A and B are required for a controlled comparison of their gating behavior.
