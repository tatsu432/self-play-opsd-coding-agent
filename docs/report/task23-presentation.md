# Five-minute presentation outline

**Demystifying the Effect of Gated On-Policy Self-Distillation on Students
and Proposers in Gradient-Guided Self-Play for Coding Agents**

| Time | Visual | Speaker notes |
| --- | --- | --- |
| 0:00–0:40 | Title and one repair example from the framework | The student gets one binary signal after a multi-turn repair. When all four attempts on a generated task have the same outcome, the group-relative RL advantage is zero. We ask whether privileged feedback helps the student and whether the proposer can use that feedback to choose tasks. |
| 0:40–1:30 | [Study framework](../figures/task23/01-framework.pdf) | A single Qwen3.5-4B LoRA adapter plays both roles. Control uses Dr. GRPO. A and B add ungated or gated distillation. C and D retain B's student objective and change the proposer reward to reference-gradient alignment. C scores the RL gradient. D includes the distillation gradient in that score. Each trained method runs for 512 attempted episodes per seed. |
| 1:30–2:35 | [Primary results](../figures/task23/presentation-primary.pdf) | On the same 64 held-out single-bug tasks, B verifies 28 repairs in each seed, versus A's 26 and 21 and Control's 22 and 21. The observed mean gains are 7.0 and 10.2 percentage points. C beats D by eight repairs in seed 2026 and D beats C by seven in 2027. Keep the two seed points visible. |
| 2:35–3:30 | [Paired gradient sparsity](../figures/task23/presentation-sparsity.pdf) | Recompute C and D on identical candidate tasks, trajectories and weights. The RL-only candidate gradient is near zero on 264 of 512 records. Adding gated distillation removes those near-zero cases. This adds signal on both all-failed and all-successful groups. The repair scores leave its downstream benefit unresolved. |
| 3:30–4:20 | [Development and efficiency](../figures/task23/presentation-development.pdf) | D finishes highest on the repeatedly monitored 24 tasks. The curves fluctuate and training takes just 19–22 optimizer steps. A and B share the nominal coefficient, but B's weighted auxiliary gradient is about a quarter to a third as large relative to RL. Selectivity and strength both changed. |
| 4:20–5:00 | Limitations and takeaway | Two training seeds and one small primary benchmark limit claims about the ranking. The original transfer sets have 8, 1 and 9 tasks. More than half of original evaluations fell into the truncation-failure category. Raising the per-turn cap left aggregate repairs flat and narrowed B’s advantage over Control. The proposer’s gradient became denser, while accuracy depended on the inference conditions. |

See the [report-ready results](task23-results-and-discussion.md) for detailed
tables, inference conditions, and secondary evaluations.
