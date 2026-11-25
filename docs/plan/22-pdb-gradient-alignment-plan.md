# Task 22 plan

Use the completed Task 21 PDB study as the matched comparator. The primary
comparison is seed-2026 C and D at 512 attempted student episodes. Both students
use the B objective, Dr. GRPO plus gated OPSD. C aligns the RL gradient with a
reference repair gradient, and D aligns the actual weighted student gradient.

1. Select the 11 parent- and family-disjoint feasibility examples in stable
   order. Check native edit construction, PDB verifier results, and token
   limits. Fill to eight from training-only examples if needed. Bind the
   resulting private tokenized cache to the unchanged dataset and execution.
2. Replay the seed-2026 B batches immediately following 0 and 128 episodes.
   Add the 256 and 384 snapshots if needed. Use fresh training-only candidate
   groups when compatible archived batches are unavailable. Require finite
   gradients and task-specific proposer signal for both arms. Spend at most
   five H200-hours on qualification.
3. Freeze C/D controls against a controller-verified qualification and launch
   one H200 per arm within the verified Yale allocation, at most 95
   incremental H200-hours. Charge failed and resumed segments. Use the same
   initialization, native tool protocol, FP32 path, and development monitor.
4. Evaluate the frozen 64 primary tasks at 512 episodes. If budget or queue
   timing prevents a complete matched comparison, select 384 or then 256
   based on completion and evaluation feasibility. Use the protected B
   checkpoint at the same exposure. Run the 8/1/9 transfer sets if feasible.
5. If the entire matched continuation fits after primary work or an
   alignment no-go, fork all three verified seed-2026 Control/A/B checkpoints
   to 1,024 or 768 episodes, verifying LoRA, Adam, RNG, and curriculum state.
6. Freeze results by November 30. Report paired family-cluster intervals and
   checkpoint-conditioned uncertainty, alignment geometry, reward diversity,
   curriculum and failure modes, learning efficiency, and actual compute.

See the [spec](../task/22-pdb-c-d-spec.md) and the
[execution record](../investigations/task22-status.md).
