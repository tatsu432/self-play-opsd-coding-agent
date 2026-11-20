# Task 20 status

**November 19, 2026.** The archive audit and local benchmark preparation are
complete. The CPU regression tests exercise the solver-only and controller
paths. The solver-only pilot shows training-set improvement on a small fixed curriculum.

| Gate | State | Evidence |
|---|---|---|
| Task 19 audit | Complete | [Verified training-signal reconstruction](task20-root-cause.md) |
| A · Benchmark | EvalPlus fallback selected, model difficulty pending | [PDB admission audit and 205 verified tasks](task20-benchmark-decision.md) |
| B · Solver-only | Reviewed pass with a numerical limitation | [Positive-control procedure](task20-positive-control.md) |
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
Solver-only job `33964224` / run `4b2f395e3a71886c44948db5` completed
exit 0 after eight nonzero updates and 256 episodes. Its checkpoint passed
Yale checksums. Training-only paired results were 9→11/20 and 8→13/20.
The separate probe was 27→27/32 and 27→26/32, and development 25→28/41.
The original gate records false due to 29 combined pre/post one-step
support exits. The reviewed gate accepts the repeated training improvement
and zero on-policy support exits in all eight training batches, retaining the
diagnostic limitation and original marker. See the positive-control report.
Slurm charged 42,513 H200-seconds; the log records worker-step OOM kills
during shutdown. The completion marker and endpoint are verified. Do not
duplicate. Screen plus solver cost is 51,753 H200-seconds (14.375833 hours).
Calibration and shared-pilot work are next, subject to proposer validity.

December 2 remains
the reporting deadline. The feasibility of completing the GPU comparison
depends on restored access, GPU time, and the learning gates.
