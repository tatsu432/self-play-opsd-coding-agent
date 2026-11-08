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
the recommended next step is a focused repair-interface investigation.

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
