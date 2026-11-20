# Task 20 status

**November 20, 2026.** The archive audit and local benchmark preparation are
complete. The CPU regression tests exercise the solver-only and controller
paths. The solver-only pilot shows training-set improvement on a small fixed curriculum.

| Gate | State | Evidence |
|---|---|---|
| Task 19 audit | Complete | [Verified training-signal reconstruction](task20-root-cause.md) |
| A · Benchmark | EvalPlus fallback selected and difficulty screened | [PDB admission audit and 205 verified tasks](task20-benchmark-decision.md) |
| B · Solver-only | Reviewed pass with a numerical limitation | [Positive-control procedure](task20-positive-control.md) |
| C · Shared viability | Pilot running | Shared diagnostics and isolated allocation stage implemented |
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
Calibration job `34022431` / run `f67c877fe83f197929e5adab` completed
exit 0 in 2,716 H200-seconds. Initial adapter, data, execution and worker
identities match the solver and checkpoint hashes verified. The measured
RL/OPSD/proposer norms were 0.0088883/0.0584771/0.1536654. Frozen OPSD and
proposer scales are 0.04559882929803436 and 0.028920942507191107. This
calibration does not establish valid-only reward discrimination. Never
duplicate it. Total allocated time is 54,469 H200-seconds (15.130278 hours).
Shared-control pilot `34027247` / run `da000a22f79187e0dc7fc87e` was
submitted once from `2cfe4c9e11404846c63aa0ddfb912ac8d6874f78`. Its
six-hour reservation is additional. Inspect this job and never duplicate.
Its initial development evaluation verified 25/41, with all 41 task outcomes
matching the solver pilot's initial evaluation. Its first 32-episode update
had two mixed groups of four, a nonzero solver gradient and adapter update,
and zero behavior support exits. Seven of eight proposals were valid, but no
group had distinct rewards among valid proposals; the viability gate remains
pending the second batch.
Local regression on the current task branch passed 483 tests, with 38 skipped.

December 2 remains the reporting deadline. The matched comparison depends on
the shared pilot demonstrating stable updates and reward discrimination, and on
the available time for training and evaluation.
