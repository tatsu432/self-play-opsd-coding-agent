# Task 20 status

**November 19, 2026.** The archive audit and local benchmark preparation are
complete. The CPU regression tests exercise the solver-only and controller
paths. Task 20 has no measured model-learning result.

| Gate | State | Evidence |
|---|---|---|
| Task 19 audit | Complete | [Verified training-signal reconstruction](task20-root-cause.md) |
| A · Benchmark | EvalPlus fallback selected, model difficulty pending | [PDB admission audit and 205 verified tasks](task20-benchmark-decision.md) |
| B · Solver-only | Implementation tested locally, pilot pending | [Positive-control procedure](task20-positive-control.md) |
| C · Shared viability | Pending solver evidence | Shared diagnostics and isolated allocation stage implemented |
| D · Matched Control/A/B | Pending shared evidence and measured cost | Freeze and analysis tools available |
| E · C/D | Deferred | Requires useful alignment evidence and remaining allocation |

Yale authentication and the H200 resource check succeeded. The user authorized
available GPUs without a numeric study cap. Task 20 accounting starts at zero,
and the accepted Task 19 worker was reused. The verified dataset is staged.

Screen job 33943325 / run `1f524c611789ad34141d09dc` completed exit 0
in 9,240 H200-seconds. Its checkpoint was verified. Its four-rollout
histogram was 2/1/2/2/25 for 0/1/2/3/4 successes. The 5/32 mixed rate is
below the 20–30% screening target. On the paired eight-task subset, groups
of four and eight yielded 1 and 3 mixed groups, with 1,907 and 3,836 seconds
respectively. Use groups of eight for the bounded positive control. The
five parent-distinct mixed tasks form the frozen solver curriculum under
`data/task20/evalplus-solver-v1`. The fallback needs further difficulty
calibration before a main self-play claim. Never duplicate the screen.
Solver-only job `33964224` / run `4b2f395e3a71886c44948db5` was submitted
once from `47f9fda212288ac1be35f2a3f37395a9c5a8c8e8` with a 12-hour
reservation. Inspect the existing job and never duplicate it.
The current ledger charges 2.566667 H200-hours.

December 2 remains
the reporting deadline. The feasibility of completing the GPU comparison
depends on restored access, GPU time, and the learning gates.
