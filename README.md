# Self-play OPSD coding-agent experiment

We study whether coding agents can improve at repairing Python programs through
self-play, and whether that improvement transfers to unfamiliar problems.

A proposer introduces verified bugs and a student repairs them using tools and
verifiable tests. We compare GRPO with ungated or gated on-policy
self-distillation (A/B); C/D add gradient-alignment rewards for the proposer.
See [methods](docs/methods.md) for the full design.

## Progress and results

The six three-seed A/B runs are complete. On held-out MBPP+-derived repairs,
A solved 57–64/115 and B 68–71/115, versus base 7/115. HumanEval+-derived
repairs remained 0–3/116, near base 2/116. These evolving proposer/student runs
do not isolate gating. See [A/B results](docs/investigations/task15-final-results.md)
and [transfer analysis](docs/investigations/task16-generalization-checkpoint-decision.md).
Retrospective replay recovered correct saved states in 242 failed external
episodes; that is not new end-to-end accuracy.

Task 17's matched 80-episode objective study found no robust gated advantage:
independent development endpoints were GRPO 10/8/9, gated 10/8/10, nominal
ungated 11/11/10 and calibrated ungated 9/9/9 out of 16 across three seeds.
See the [objective evidence](docs/investigations/task17-objective-results.md).

Task 18 selected Qwen-native exact-text calling: **27/40 base and 31/40 trained**
repairs, versus plain JSON's 19/40 and 21/40 on the same inspected training
families. Normalized diff scored 6/40 and 28/40. New allocation was **1.275
H200-hours**. A restored numerical diagnostic changed one Adam update by 21.4%
versus 0.4% repeat variation; new RL remains disabled pending numerical
qualification. These results do not establish new external generalization.
See [Task 18 results](docs/investigations/task18-results.md), the
[current research decision](docs/investigations/task18-research-decision.md)
and [completion record](docs/task_completion/18-implemented-interface-signal-bottlenecks.md).
Task 19 implements the next study with Qwen3.5-4B, native student thinking,
one shared LoRA, and Dr. GRPO. The planned seed-2026 arms are a Dr. GRPO
control and four OPSD/proposer-reward variants. BF16 cached execution failed
the numerical threshold; uncached BF16 and cached FP32 passed their checks.
FP32 calibration completed; proposer reward discrimination remains a concern
to test in the five arms. The FP32 base repaired 31/41 monitoring tasks;
ID and external baseline evaluation is underway. See the
[Task 19 plan](docs/plan/19-fix-plan.md) and
[execution status](docs/investigations/task19-status.md).

## Getting started

```bash
uv sync --frozen
uv run selfplay doctor --settings configs/local.yaml --isolation
uv run selfplay smoke --output runs/local-smoke
```

See the [operations guide](docs/operations.md) for setup, Yale execution,
checkpoints, evaluation, reproducibility and the preserved base-model snapshot.
