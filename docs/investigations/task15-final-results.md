# Task 15: stabilized online A/B results

Study `repair-selfplay-2026-task15-ab-v2` compares ungated A with sigmoid-gated B at the same SD coefficient (0.1), using structured online proposals and unified-diff repair. The frozen training execution is `0ac77cde4bc5f70ba2b3eb9e457749e3106a2bc5`; all six paired arms completed their prespecified exposure endpoint. This is a three-seed comparison of evolving proposer/student systems, not a controlled test of an isolated gate on identical trajectories.

## Held-out results

| Seed | A final ID | B final ID | A external | B external |
| --- | ---: | ---: | ---: | ---: |
| 2026 | 61/115 | 70/115 | 2/116 | 3/116 |
| 2027 | 57/115 | 68/115 | 1/116 | 0/116 |
| 2028 | 64/115 | 71/115 | 1/116 | 0/116 |

The common base model scored 7/115 and 2/116, respectively. B exceeds A by 7–11 repairs on the final ID set for each seed; external repair is near zero for both. The 2028 B endpoint is 4,100 attempted episodes because the last group of four was completed; every other endpoint is 4,096.

On the 60-parent final ID set, paired parent-macro B-minus-A differences are +0.0833, +0.0833 and +0.0583 for seeds 2026–2028. The corresponding family-clustered bootstrap 95% intervals are [-0.0081, 0.1786], [-0.0530, 0.2231] and [-0.0431, 0.1667]. Across the three paired seeds, the mean is +0.0750 with a t-based interval [0.0391, 0.1109]; this interval relies on only three training seeds and its assumptions are not established. These family/parent and seed uncertainties answer different questions and are not interchangeable. On the 116-parent external set, the paired seed differences are +0.0086, -0.0086 and -0.0086; the seed mean is -0.0029 with interval [-0.0276, 0.0219]. All six first-ID and external completion records have their full manifest denominators; no held-out checkpoint was selected by performance.

## Training and operational health

Each 4,096-episode arm used 1,024 four-rollout groups; B-2028 used 1,025. Mixed groups A/B by seed were 169/224, 155/189 and 126/172. All six recorded zero cumulative fallback groups. Committed records attribute all 1,024/1,025 training task groups per arm to online proposals; the historical-replay counter also records repeated proposal samples, so online origin is not a claim of historical novelty. Cumulative verified proposals A/B were 3,001/3,060 of 3,460/3,552 attempts (2026), 2,879/2,987 of 3,384/3,520 (2027), and 2,921/2,810 of 3,380/3,264 (2028). Student optimizer steps A/B were 241/245, 239/242 and 244/236; proposer steps were 88/105, 98/102 and 93/108. These are observed group outcomes; all-equal groups do not prove population task difficulty. Current zero RL signal is not the same as zero Adam displacement.

The last A-2026 external evaluation required six allocated CPUs rather than eight after Yale changed its H200 limit. The first `sbatch` was definitively rejected with no job; the accepted retry kept the same logical identity and checkpoint, recorded the resource change in its ledger, reserved one separate CPU for the generated-code worker, and completed 116/116. Its allocated-node isolation report passed all checks. The final checksum-verified joint checkpoints remain on Yale; private event and analysis files are not committed.

V2 training used 524,606 allocated H200-seconds (145.724 hours) across 28 allocations, including interrupted segments. Fourteen held-out jobs used 52,704 H200-seconds (14.640 hours). Abandoned V1 training attempts used 72,658 H200-seconds (20.183 hours); they were not included in the A/B results. Task 15's separately identified GPU backend, stabilization, and targeted numerical jobs used 8,281 H200-seconds (2.300 hours), including the V2 backend check. These categories total 182.847 H200-hours; reused Task 14 evidence, CPU jobs, and jobs outside the Task 15 ledgers are not charged to that subtotal. Scheduler-rejected submissions incurred no allocation.

The V1 parent failure was traced to one training parent whose prepared original failed its isolated verification; the fixed successor excludes that parent from both arms after an isolated audit of all 128 structurally eligible training parents. The held-out splits were not changed. The source of the parent discrepancy itself remains unresolved. Earlier targeted saved-rollout audits showed that substituting rescored probabilities for the actual behavior denominator can materially alter a base-snapshot diagnostic update. The executed implementation preserves behavior probabilities for PPO and computes the signed privilege gap using the same scorer. Those diagnostics are targeted samples and do not establish population-level effects.

## Reproduction and limitations

Private outputs reside under `/private/tmp/selfplay-task15-v2/runs/task15/analysis/`; raw outcomes and task sources are not public. The deterministic analysis command is:

```bash
uv run python scripts/analyze_task15.py --manifest runs/task15/manifest.json --data data/protocol-v3 --output runs/task15/analysis
```

The privacy-safe `scripts/analyze_task15_health.py` additionally exports committed-event supply, mixed-group, action-error and probability-mismatch curves; its output stays private. The private `summary.json` SHA-256 is `6c1fe6e719a1a2781a40dec520d9cb4c511b2a29091d28ea047e9fc8e818f3aa`; the training-allocation manifest SHA-256 is `d6c038d223659d1da1458508395b35608af10c1c5a474eba1c4639b199864263`. These paths are ignored, machine-local research artifacts. The analysis validates paired common settings, matched initial states, final task sets, exposure completion and full denominators. Its bootstrap is over evaluation families; training variation is separately summarized across three seeds. The intervention changes both trajectories and supply realized over training, so the held-out gap should not be interpreted as a causal estimate for the gate alone. The external transfer result is limited by a floor near zero. The base/scorer mismatch mechanism remains unresolved beyond the targeted replays.

Training event records and joint checkpoints are authoritative. Five runs reported no SDK delivery errors; B-2028 reported 90. After all writers exited, CPU job 32037439 checksum-verified the latest joint checkpoints, rebuilt aggregate-only canonical telemetry, and independently verified the cloud revision for all six logical runs. Verified iterations A/B by seed were 241/245, 239/242 and 244/236. B-2028's 90 prior delivery errors are retained as a historical limitation; the final canonical checkpoint-backed cloud state was verified. The reconciler used no GPU.
