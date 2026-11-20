# Task 20 solver-only Dr. GRPO positive control

Status and live blockers are in the [Task 20 gate record](task20-status.md).
Local tests exercise the fixed curriculum with four and eight rollouts
per group using a small differentiable policy. A unanimous group produces
zero RL gradient and no step. A mixed group changes the parameters, skips
proposer and teacher work, and round-trips the checkpoint and optimizer state.
The local worker separately verified the fallback tasks. Actual Qwen3.5
learnability, optimization health, and transfer remain unmeasured.

## Execution order

1. Authenticate with `uv run selfplay hpc login` and check live resources.
   Reuse the verified single-process worker and pinned Qwen3.5 trainer SIF
   from the authorized Yale workspace. Stage the immutable code and
   `data/task20/evalplus-v1` using the base Task 20 settings.

   ```bash
   uv run selfplay hpc resources --settings configs/task20-yale.yaml \
     --profile single_gpu_80gb --gpu-type h200 --account cpsc4880
   uv run selfplay hpc sync --settings configs/task20-yale.yaml
   uv run selfplay hpc reuse-worker repair-selfplay-2026-task19-v7 \
     --settings configs/task20-yale.yaml
   ```

2. Set `TASK20_AUTHORIZED_HOURS` and `TASK20_SCREEN_MINUTES` from the verified
   allocation and rollout cost. Preview once. When its allocation and
   identities match, submit once with the same arguments and without
   `--dry-run`. Use the existing trainer image flags if the inherited SIF
   is supplied explicitly.

   ```bash
   uv run selfplay hpc submit --settings configs/task20-yale.yaml \
     --operation task20-screen --method control --profile single_gpu_80gb \
     --gpu-type h200 --account cpsc4880 --task20-budget-hours "$TASK20_AUTHORIZED_HOURS" \
     --task20-wall-minutes "$TASK20_SCREEN_MINUTES" --dry-run
   ```

3. Fetch the verified screen summary, inspect four-rollout mixed fractions,
   and compare informative groups per second in the paired eight-rollout
   subset. Freeze a deterministic curriculum from mixed cases. Copy the
   settings and its same-stem JSON control to a solver settings pair, changing
   the data root to the new screened directory and the group size only if
   the paired comparison supports it. Sync and preview `task20-solver` under
   the same authorization. Its intended maximum is twelve batches, stopping
   earlier at eight informative updates. Require repeatable training-set
   success gains over two paired blocks and report the separate training
   probe and development outcomes. The one-step diagnostic restores the
   initial adapter and optimizer between two authentic mixed-batch steps.

   ```bash
   uv run python scripts/task20_data.py \
     --source data/task20/evalplus-v1 --output data/task20/evalplus-solver-v1 \
     --screen-summary "$TASK20_SCREEN_SUMMARY"
   ```

4. After a passing solver gate, run `task20-calibrate` from fresh weights.
   Record its unweighted gradient evidence and choose the measured proposer
   scale for the separate `task20-shared-pilot` control. If a causal question
   remains, diagnose the shared update with the proposer-frozen mode on a
   separately identified pilot. Use the same frozen screened data and
   matching initial weights for the main triad.

5. Reconcile every allocation with `uv run selfplay hpc task20-budget`.
   Preserve a final evaluation and interruption reserve. Use
   `scripts/freeze_task20.py` to derive the common endpoint from the verified
   solver, calibration, and shared-pilot markers and measured costs. Launch
   `task20-main` for Control/A/B with the same frozen configuration. After
   training closes, use `task20-evaluate` with the prespecified checkpoint
   number. The controller accepts base, frozen endpoint, and earliest best
   development by the frozen rule. Keep held-out outcomes out of selection.

6. Use the existing controller's status, fetch, and verified checkpoint resume.
   Reconcile before retrying any interrupted run. Replay offline W&B aggregates
   with the authenticated workspace and check server history. Use
   `scripts/analyze_task20.py` with the immutable dataset, common control,
   completed arm manifest, and reconciled budget. Its source JSON retains the
   observed checkpoints and task-level outcomes for paired analysis and plots.

## Completed seed-2026 pilot

Solver job `33964224` / run `4b2f395e3a71886c44948db5` completed
with eight nonzero RL updates in 256 attempted episodes. The endpoint
checkpoint and its metadata and state checksums were verified on Yale.
Every training batch recorded zero support exits at rollout rescoring.
Training-only paired success changed from 9/20 to 11/20 and 8/20 to 13/20.
The separate probe changed from 27/32 to 27/32 and 27/32 to 26/32.
Development was 25/41 initially and 28/41 at the endpoint. These small
training-only results support proceeding to a shared pilot. They do not
establish held-out generalization.

The original completion marker records `learnable: false`. Its gate
required zero combined pre- and post-update support exits in the one-step
diagnostic. That extra criterion was introduced in the initial implementation
and was stricter than the research specification. The diagnostic recorded
29 combined exits but did not retain separate pre/post counts or the complete
scored batch. It also reported a nonzero gradient and reproducible adapter
displacement of 0.05898. Post-step changes in top-k/top-p support are possible.
The finite-only weighted log-probability summary does not quantify the impact
of the excluded tokens. Preserve that limitation.

The reviewed gate uses both repeated training-only improvements, actual
nonzero optimizer updates, and zero behavior-support exits at on-policy
rescoring in every training iteration. It retains the original marker and
the post-step limitation. This is a documented departure from the originally
coded gate, made after observing the pilot. Further diagnosis is required if
new scoring evidence indicates an execution mismatch.

Slurm charged 42,513 H200-seconds for the pilot. Slurm reported a successful
batch while its log recorded OOM kills in the worker step during allocation
shutdown. The completed endpoint and evaluations were saved before exit.
Task 20 allocation including the screen totals 51,753 H200-seconds.

The Yale job did not receive W&B credentials, so its telemetry stayed local.
The checkpoint-backed aggregate history was replayed from the verified
iteration artifacts. The W&B API confirmed nine rows, including the eight
training updates and the evaluation summary. The replay receipt is retained
privately to prevent duplicate uploads. [Solver W&B history](https://wandb.ai/tatsu432s/self-play-opsd-coding-agent/runs/b16b95a99028b81934a7da4c)
