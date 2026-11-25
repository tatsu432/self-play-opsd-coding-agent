# Task 22 mechanistic analysis

The figures currently derive from the verified Task 21 checkpoints. The
[qualification](task22-alignment-qualification.md) records same-batch C/D
gradient measurements; the final figures will add completed C/D runs. The
[aggregate analysis script](../../scripts/analyze_task22.py)
recomputes the table and figures from the frozen evidence.

## Effective distillation strength

A and B used the same nominal coefficient, 0.171717. Their observed
batch-level weighted OPSD-to-RL norm ratios differed. Each median uses
batches with nonzero RL gradients and represents that method's own task
trajectory.

| Seed | A ratio | B ratio | A median RL–OPSD cosine | B median RL–OPSD cosine |
|---:|---:|---:|---:|---:|
| 2026 | 0.674 | 0.177 | +0.058 | −0.020 |
| 2027 | 0.758 | 0.214 | +0.034 | −0.008 |

The matched study varies both token weighting and effective gradient
strength. These recorded cosines describe geometry along the observed
trajectories. They do not estimate the effect of an intervention on learned
repair accuracy.

## Endpoint failures

On seed-2026 primary evaluation, Initial, Control, A and B had respectively
37, 37, 32 and 31 episodes categorized as truncation out of 64. The same
checkpoints verified 21, 22, 26 and 28 repairs. The recorded truncations
identify episodes that reached limits. The data contain no intervention
isolating the effect of extending those limits.

## Presentation figures from verified evidence

- [Research framework](../figures/task22/baseline/01-framework.pdf)
- [Seed-2026 primary accuracy and paired intervals](../figures/task22/baseline/02-primary.pdf)
- [Two-seed development efficiency](../figures/task22/baseline/03-learning-efficiency.pdf)
- [Distillation strength and gradient dynamics](../figures/task22/baseline/05-gradient-dynamics.pdf)
- [Primary evaluation failure categories](../figures/task22/baseline/06-failure-modes.pdf)
