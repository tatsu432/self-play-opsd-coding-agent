# Task 14: informative self-play investigation

This report is provisional. Task 14 is not complete; see the live job identities in `AGENTS.md` and the execution plan in `docs/plan/14-fix-plan.md`.

## Prospective base-model proposer comparison

The two arms use 40 training parents selected before generation by `digest([2026, "task14", parent_id])`, excluding the 40 Task 13 probe parents; each uses four matched seeds. The verified worker and original semantic checks are unchanged. Both jobs completed exit 0.

| Representation | Verified outputs | Unique parent/mutation pairs | Allocated H200-seconds | Unique pairs/H200-hour |
|---|---:|---:|---:|---:|
| Unified diff | 58/160 | 22 | 589 | 134.5 |
| Structured AST edit | 90/160 | 34 | 257 | 476.3 |

The parent-paired difference in unique mutations was +0.30 per parent (2,000-resample bootstrap 95% interval −0.025 to +0.60). The verified-output fraction difference was +20 percentage points (interval +1.875 to +37.5). Structured accepted outputs covered 28 parent/location pairs and seven operator types. They use a narrower finite edit vocabulary, so acceptance and throughput do not establish equivalent coverage. Rates include all allocated startup time. The B200 checkpoint-history comparison and student mixed-reward measurements are pending.

Reproduce from private fetched artifacts and actual scheduler seconds:

```bash
uv run python scripts/analyze_task14_probes.py \
  --diff runs/hpc/yale-hpc-cpsc4880/276931a7445252e963f37b1b \
  --structured runs/hpc/yale-hpc-cpsc4880/36db89f113049ef37c32b79d \
  --data data/protocol-v3 --output runs/task14/base-proposal-comparison.json \
  --diff-gpu-hours 0.163611111111 --structured-gpu-hours 0.071388888889
```

Student editing comparisons, numerical audit, matched-exposure controls, reward counterfactual, checkpoint-history comparison, and online eligibility remain unfinished. Do not draw conclusions about OPSD or gradient alignment from these proposer results.
