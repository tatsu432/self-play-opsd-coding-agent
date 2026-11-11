# Self-play OPSD coding-agent experiment

We study whether coding agents can improve at repairing Python programs through
self-play, and whether that improvement transfers to unfamiliar problems.

A proposer introduces verified bugs and a student repairs them using tools and
verifiable tests. We compare GRPO with ungated or gated on-policy
self-distillation (A/B); C/D add gradient-alignment rewards for the proposer.
See [methods](docs/methods.md) for the full design.

## Progress and results

All six three-seed A/B runs are complete. On held-out MBPP+-derived repairs,
A solved 57–64/115 tasks and B 68–71/115; the base solved 7/115. On
HumanEval+-derived repairs, A/B solved 0–3/116, near the base's 2/116.
Across the six trained external evaluations, 4,301 patch attempts were rejected
and 13 accepted. These results concern repair tasks and do not isolate gating
from changes in the evolving proposer and student. C/D training is deferred;
Task 17 implements exact-text replacement and a matched frozen-model study.
Isolated metadata-only replay found correct reconstructed states in 242 previously
failed trained HumanEval+ episodes. These are saved-state diagnostics, not new
agent accuracy. On 40 training-only families, base strict/text solved 5/40 and 19/40;
trained B2026 solved 27/40 and 21/40. The comparison used 1.15 H200-hours.
Exact text was selected for a controlled objective study. At 80 episodes, GRPO
and gated OPSD each solved 10/16 and 8/16 independent development tasks in the
first two seeds, respectively; further replication is underway. See the [objective results](docs/investigations/task17-objective-results.md). See the [Task 17 evidence](docs/investigations/task17-repair-interface-results.md)
and [current research decision](docs/investigations/task17-research-decision.md).

See the [A/B results](docs/investigations/task15-final-results.md) and
[transfer, checkpoint and compute analysis](docs/investigations/task16-generalization-checkpoint-decision.md)
for uncertainty, limitations and the research decision.

## Getting started

```bash
uv sync --frozen
uv run selfplay doctor --settings configs/local.yaml --isolation
uv run selfplay smoke --output runs/local-smoke
```

See the [operations guide](docs/operations.md) for setup, Yale execution,
checkpoints, evaluation and reproducibility.
