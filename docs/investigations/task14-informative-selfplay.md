# Task 14: informative self-play investigation

This report is provisional. Task 14 is not complete; see the live job identities in `AGENTS.md` and the execution plan in `docs/plan/14-fix-plan.md`.

## Prospective base-model proposer comparison

The two arms use 40 training parents selected before generation by `digest([2026, "task14", parent_id])`, excluding the 40 Task 13 probe parents; each uses four matched seeds. The verified worker and original semantic checks are unchanged. Both jobs completed exit 0.

| Representation | Verified outputs | Unique parent/mutation pairs | Allocated H200-seconds | Unique pairs/H200-hour |
|---|---:|---:|---:|---:|
| Unified diff | 58/160 | 22 | 589 | 134.5 |
| Structured AST edit | 90/160 | 34 | 257 | 476.3 |

The parent-paired difference in unique mutations was +0.30 per parent (2,000-resample bootstrap 95% interval −0.025 to +0.60). The verified-output fraction difference was +20 percentage points (interval +1.875 to +37.5). Structured accepted outputs covered 28 parent/location pairs and seven operator types. They use a narrower finite edit vocabulary, so acceptance and throughput do not establish equivalent coverage. Rates include all allocated startup time. Student mixed-reward measurements are pending.

Reproduce from private fetched artifacts and actual scheduler seconds:

```bash
uv run python scripts/analyze_task14_probes.py \
  --diff runs/hpc/yale-hpc-cpsc4880/276931a7445252e963f37b1b \
  --structured runs/hpc/yale-hpc-cpsc4880/36db89f113049ef37c32b79d \
  --data data/protocol-v3 --output runs/task14/base-proposal-comparison.json \
  --diff-gpu-hours 0.163611111111 --structured-gpu-hours 0.071388888889
```

## B200 proposer snapshot and stored novelty history

Both corrected B200 arms loaded and checksum-verified the retained checkpoint on the allocated node and completed exit 0 on the same 40 prospective parents and four seeds. Against an empty history, diff produced 70/160 verified outputs and 24 distinct parent/mutation pairs; structured produced 91/160 and 36. Against actual checkpoint history, 22 diff pairs and eight structured pairs were already present, leaving **2 versus 28** unique novel verified mutations. The paired difference in unique novel mutations was +0.65 per parent (2,000-resample bootstrap 95% interval +0.40 to +0.925).

The diff and structured allocations used 763 and 236 H200-seconds respectively, equivalent to 9.4 and 427.1 unique history-novel mutations per allocated GPU-hour. Structured outputs covered 29 parent/location pairs and seven operator types. These inference-only figures do not establish student reward variation or benefit from online training. The first B200 attempt failed due to a transcribed identity hash, and another allocated attempt was cancelled; together they used 88 H200-seconds and are retained for accounting. Two further cancelled attempts never started. Corrected replacements were submitted after the identity was checked and the first corrected arm verified model loading and generated records.

```bash
uv run python scripts/analyze_task14_probes.py \
  --diff runs/hpc/yale-hpc-cpsc4880/d113371007fb12d958f92ccf \
  --structured runs/hpc/yale-hpc-cpsc4880/5c34adbea7983c01775ef5a8 \
  --data data/protocol-v3 --output runs/task14/b200-proposal-comparison.json \
  --diff-gpu-hours 0.211944444444 --structured-gpu-hours 0.065555555556
```

## Base-student editing comparison

On the same 16 training-only static tasks and four matched seeds, the base student repaired 7/64 episodes with unified diffs and 2/64 with function-body replacement. The paired success difference (body minus diff) was −7.8125 percentage points (2,000-resample parent-bootstrap 95% interval −18.75 to +1.5625). Function-body replacement made 81/140 edit attempts valid, versus 7/160 for diffs, but greater syntactic success did not translate into more verified repairs in this sample. The two arms generated 27,282 and 37,335 student tokens respectively. The B200 paired editing probe remains in progress, so protocol selection is provisional; diffs remain the default. Training-only diff calibration was started without using any held-out outcome.

Numerical audit, matched-exposure controls, reward counterfactual, B200 student comparison, and online eligibility remain unfinished. Do not draw conclusions about OPSD or gradient alignment from these proposer results.
