# Self-play OPSD coding-agent experiment

A controlled Python repair experiment comparing four self-play methods that pair
GRPO with on-policy self-distillation and verifiable rewards. Methods C and D
also use gradient alignment to train the proposer. The benchmark uses repair
variants derived from MBPP+ and HumanEval+.

The implementation provides reproducible `uv` environments and Docker workers,
GPU launch profiles, W&B monitoring, isolated code execution, resumable training,
evaluation and analysis. Production GPU acceptance is documented separately and
must pass before the main runs.

## Quick start

```bash
uv sync --frozen
uv run selfplay doctor --settings configs/local.yaml --isolation
uv run selfplay smoke --output runs/local-smoke
```

## Documentation

- [Operations guide](docs/operations.md): data preparation, containers, worker
  setup, hardware checks, pilot, launch, recovery, evaluation and W&B.
- [Implemented methods](docs/methods.md): objectives, normalization, data boundaries
  and validation limits.
