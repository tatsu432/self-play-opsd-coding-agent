# Task 21 · PDB verification and learning-signal study

Task 21 uses executable PDB repairs to test whether Qwen3.5-4B can receive
useful verifier-driven updates before comparing nominal ungated OPSD, gated
OPSD and Dr. GRPO. The first execution identity is
`repair-selfplay-2026-task21-pdb-v1`.

## Evidence

- The initial CPU pilot admitted **11 parent-disjoint PDB single-bug tasks**,
  eight from LiveCodeBench and three from BigCodeBench, and 24 separate
  LiveCodeBench screening tasks. Every accepted canonical implementation
  passed, and every accepted mutant produced a semantic failure.
- The pinned scorer routines and isolated adapter agreed on **22/22** verdicts
  for the original and canonical pilot sources. This comparison covers those
  sources and tests. Model-edit agreement and support for broader BigCodeBench
  fixtures require separate evidence.
- All 300 matched LiveCodeBench parents in the pinned release were LeetCode
  problems. The adapter's stdin/stdout path has an isolated synthetic test.
  The local verifier uses the full public and private fixtures of each
  admitted LiveCodeBench task. BigCodeBench tasks enter when the full suite
  can be translated or the adapted evaluation is explicitly identified.
- Slurm job `34122424`, controller run `aefab0ada76973689cac5bdf`, completed
  exit 0 from `9ba11127648a1a51c8cfc387a2eeaedb60c6efae`. Checkpoint `000000`
  and its file checksums verified. It used **1,246 H200-seconds** and solved
  **5/11** training-only tasks. Five episodes ended on a token or turn
  limit and one passed its public tests but failed hidden verification.
  [W&B feasibility aggregate](https://wandb.ai/tatsu432s/self-play-opsd-coding-agent/runs/46c174d34df0113c661be084)
- The screening bundle contains 24 training parents, two supplemental
  Multi feasibility parents, eight training-only probe parents, 24 development
  parents, and 64 held-out single-bug ID parents. Transfer comprises eight
  PDB-Multi parents, one adapted BigCodeBench-derived PDB parent, and nine
  separately generated BigCodeBench mutants on additional parent-disjoint
  problems. Keep the two BigCodeBench results distinct.
- Slurm job `34123330`, controller run `f694b7c181e4dc10dde279dd`, failed
  with exit 1 after **22,188 H200-seconds**, with 182 screen episode records
  saved and no completion marker. No Python failure was captured and its
  health record remained `running`, so its immediate cause is unconfirmed.
  The shared project filesystem subsequently reported zero available space
  and rejected the recovery ledger write with `Disk quota exceeded`. Four
  inactive historical method-smoke runs held 6,343,435,272 bytes of
  regenerable `state/rollout_adapters` inference exports. Those exports were
  removed after checking that no account jobs were active. Their checkpoints
  and run records remain. Seven more inactive backend-check and method-smoke
  export directories totaling 2,907,407,833 bytes were removed while the
  solver job was pending. Five older inactive pilot export directories
  totaling 2,378,788,227 bytes were removed after the solver started. Project
  storage reported 8.2 GiB available after the second cleanup; shared usage
  can change independently.
  The controller verified checkpoint `000000` and submitted continuation
  `34150626` for this same logical run. The original failure,
  checkpoint and allocation remain in the ledger. Never duplicate either job.
- The incomplete screen contains 12 mixed groups among 23 complete groups
  of four and 17 mixed groups among 22 complete groups of eight. These are
  interim counts from before recovery.
- Continuation `34150626` completed exit 0 in 1,180 H200-seconds. The final
  24-parent screen had 12 mixed groups of four and 17 groups of eight. Its
  measured four-rollout efficiency was 3.70 mixed groups per H200-hour,
  versus 2.62 for eight, including both screen allocations. A four-rollout
  curriculum was frozen in `data/task21/pdb-solver-v1`. Solver-only job
  `34151943` / controller run `c8e9914126d585fb4685fa27` was submitted once
  from `755174d` and continued once as job `34205327` from checkpoint
  `000008`. Both allocations completed exit 0; the endpoint and checkpoint
  were verified. Never duplicate them. The pilot used 51,056 H200-seconds,
  made eight effective RL updates in 128 training episodes, and produced
  paired training results of 24/48 before and 30/48 after (12 gains, six
  losses). Its independent probe changed from 8/32 to 10/32 (five gains,
  three losses); development changed from 7/24 to 8/24. These are small,
  single-run observations, not evidence of generalization. On November 22,
  shared project storage had about 174 GiB available; usage can change
  independently. Preserve its checkpoint and scientific identity.
  Resampling tasks as clusters (20,000 paired bootstrap draws, seed 21), the
  95% interval for the change in four-sample repair rate is 0 to +27.1
  percentage points on the 12 training tasks and −9.4 to +28.1 points on the
  eight probe tasks. These intervals are conditional on this single trained
  model and small observed task sets; they do not measure training-seed
  uncertainty or establish an effect on unseen task populations.
- Shared-policy calibration job `34225453` / controller run
  `e75f67bcc64df23c764807cb` completed exit 0 in 2,751 H200-seconds from
  `755174d`; its checkpoint and result were verified. On four proposed tasks,
  RL, ungated OPSD, gated OPSD and proposer gradient norms were respectively
  0.01396, 0.07262, 0.02110 and 0.04615. The calibrated proposer scale is
  0.15128449959390794 and gated OPSD scale is 0.19849138589186444. This
  measurement does not establish that proposer rewards remain diverse during
  training. The [calibration aggregate](https://wandb.ai/tatsu432s/self-play-opsd-coding-agent/runs/cd05b24447acd87178bd106a)
  was reconciled. Shared pilot `34229678` / `5a71bfd3b213e7beaf4e3c02` from
  `02ca483` completed exit 0 in 15,045 H200-seconds; its endpoint and
  checkpoint were verified. Never duplicate it. Its initial adapter and dataset hashes match the solver pilot;
  its calibrated shared-policy control has a new scientific identity. Its
  7/24 initial development evaluation matches the solver baseline task by task.
  The [shared pilot aggregate](https://wandb.ai/tatsu432s/self-play-opsd-coding-agent/runs/ec66c45e9ea4983516cb602e)
  has been independently reconciled with the checkpoint-backed history.
  Its first committed batch had two mixed solver groups out of four, a finite
  nonzero RL gradient and an optimizer step. Five of 16 proposals were valid;
  no group had distinct rewards among valid proposals. These are interim
  observations, and the first-batch gradient norms match the earlier calibration
  because they replay the same seeded initial-policy candidates; do not count
  them as independent confirmation. Six attempts failed to produce a
  verifier-confirmed semantic failure, four exceeded the four-line edit limit,
  and one valid candidate was left unscored by the batch's task cap. The
  proposer prompt requests local edits but does not state the enforced
  four-line limit; this is a candidate cause of those four rejections, to be
  assessed against the completed pilot before changing its scientific code.
  The second committed batch also had two mixed solver groups of four and a
  nonzero RL update. Four of eight proposals were valid, with two distinct
  candidates in each of two parent groups; both candidates in each group had
  the same reward. Three proposals lacked a verified semantic failure and one
  exceeded the four-line limit. Across these first two batches no valid-only
  proposer group had differing rewards; the pilot is not yet complete.
  In batch three, one of four solver groups was mixed and the third nonzero
  RL update occurred. Five scored proposals were valid, one additional valid
  candidate exceeded the scoring cap, and neither parent group showed
  valid-only reward differences. Across three batches the totals are five
  mixed solver groups out of 12 and zero proposer groups with valid-only
  reward differences.
  Batch four produced two mixed solver groups, a fourth nonzero RL update and
  a zero proposer gradient. All eight attempts were valid, but four were
  duplicates and neither of the two parent groups had differing valid rewards.
  Across all four batches, seven of 16 solver groups were mixed and zero
  proposer groups had valid-only reward differences. Endpoint development
  changed from 7/24 to 6/24, with three task gains and four losses. Its verified result records 24 valid proposals,
  including two left unscored by the cap, and zero parents with differing
  valid rewards; its prespecified viability check was false. Across the four batch
  timing records, student generation took 7,977 of 8,756 seconds (91.1%);
  this excludes the separate baseline and endpoint evaluations. The revised
  eight-candidate cap can increase generation cost and must be judged against
  its measured signal, not only its number of valid proposals.
  As a rough planning reference, 8,756 seconds for 64 attempted episodes is
  about 137 seconds per training episode; at that observed rate, three arms of
  512 episodes would require roughly 58 H200-hours for training alone, before
  development and final evaluations. This is not a reservation or a forecast
  for the revised eight-candidate design.
  An isolated candidate prompt correction is prepared on
  `feat/21-pdb-proposer-limit` at `d1c43ec` and passed the six Task 21 tests
  and Ruff. Commit `e0c902e` prepares its separate `pdb-prompt-v2` study
  configuration with the same frozen data. With the four pilot batches now
  showing no valid-only reward differences and two valid candidates left
  unscored by the cap, `ef3e76a` also raises the candidate cap from four to
  eight. The new study reuses the dependency-checked worker evidence and
  retains the old study unchanged. Its bounded calibration job `34248044` /
  `89ba014f6f4b9115086dbbd0` completed exit 0 in 4,171 H200-seconds; its
  checkpoint and result were verified. It scored six distinct tasks and found
  nonzero RL, ungated, gated and proposer gradients (norms 0.01874, 0.11913,
  0.03275 and 0.03088); the calibrated proposer and gated scales are
  0.30352814071666645 and 0.1717172194518303. This alone does not establish
  valid-only reward diversity. The [calibration aggregate](https://wandb.ai/tatsu432s/self-play-opsd-coding-agent/runs/afa9d0c0552ada381ad9a3c3)
  was reconciled. Revised shared pilot `34254191` /
  `7346c5d4dca7dbd9240637e1` completed exit 0 from `f07d078` in 23,676
  H200-seconds; its endpoint and checkpoint were verified. Never duplicate it.
  Both the explicit prompt
  and cap change are bundled, so a change in outcome cannot be attributed to
  either one alone. Its 7/24 initial development result matches all 24 original
  pilot baseline outcomes task by task.
  The first revised batch scored six distinct tasks: four of six solver groups
  were mixed and produced an RL update. One parent group contained two distinct
  valid candidates with different rewards; seven of 16 proposal attempts were
  scored as valid, seven lacked verifier-confirmed semantic failures, and two
  exceeded the four-line edit limit. This batch replays the same seeded
  candidates used for revised calibration and is not an independent repeat.
  Its recorded training time was 3,226 seconds for 24 attempted episodes.
  The second batch scored eight distinct tasks: five of eight solver groups
  were mixed and a second RL update occurred. Two parent groups contained
  distinct valid candidates with different rewards. Across both batches,
  three distinct parents have shown valid-only reward differences. This second
  batch took 4,603 seconds for 32 attempted episodes; the pilot continues.
  The third batch added five mixed solver groups out of six and a third
  nonzero RL update, but no additional parent group with differing rewards
  among valid candidates. The three-batch interim totals meet the numerical
  update and parent counts used by the pilot's prespecified check; the fourth
  batch, endpoint development evaluation and full verification remain.
  The fourth batch added three mixed solver groups out of eight, a fourth
  nonzero RL update and differing valid rewards for two more parent groups.
  Across all four committed batches, 17/28 solver groups were mixed and five
  distinct parents had valid-only reward differences. The 112-episode pilot's
  endpoint development score was 8/24, versus 7/24 at baseline. The verified
  result records 38 valid proposals including one unscored due to capacity,
  five valid-reward parents and `viable: true`. The small development change
  is not a claim of generalization or a comparison of student objectives.
  The [revised pilot aggregate](https://wandb.ai/tatsu432s/self-play-opsd-coding-agent/runs/509c7529c8bcb3360249dc5b)
  is reconciled with its checkpoint-backed history.
  A matched shared-policy comparison was frozen at 512 attempted episodes
  per arm from commit `b44d858`, with common initialization, data, objective
  scales and evaluation milestones. Its planning envelope is 246,988 H200-
  seconds of training plus 185,723 seconds reserved for evaluation (about
  120.2 hours total). This is an estimate with margin, not a user-imposed cap
  or guaranteed scheduler capacity. Control `34298915` /
  `237fb774dc01bcce763e228c`, A `34298923` / `e1278a8956cde4249f8c9497`,
  and B `34298926` / `bd6538a8448ad6423b69bead` were each submitted once;
  all are running. Their initial adapter, data and shared scientific hashes
  match, and all three initial development evaluations were 7/24 with
  task-by-task identical outcomes. Inspect their ledgers, never duplicate them, and reconcile actual
  allocations, checkpoints and final evaluations before conclusions. The
  4,560,362-byte `b44d858` transport bundle was removed after verifying its
  deployed checkout; the checkout and records are preserved.
  Seed-2026 Control's first batch committed 24 attempted episodes, four mixed
  solver groups of six, and a nonzero RL update with no support exits; its
  seeded metrics match the revised pilot's first batch, so they are not an
  independent confirmation.
  A second matched seed was frozen at `6a3d3f4`, with seed 2027 the only
  configuration difference and the same calibration coefficients. Its
  Control `34300635` / `09c37760aad43848c1591a61`, A `34300641` /
  `09e87387be4df8df15ec5f17`, and B `34300654` / `f66fd05309d2ffa381c24ace`
  were submitted once and all are running. Their seed-2027 initial adapter,
  data and scientific hashes match within the seed; its initial adapter differs
  from seed 2026 as expected. All three initial development evaluations were
  8/24 with identical task-by-task outcomes. The two-seed planning envelope is
  about 240.4 H200-hours including evaluation reserves; it is an estimate,
  not a cap or guaranteed availability. Held-out evaluations wait until all
  six training endpoints are verified. Never duplicate these jobs. The
  4,569,076-byte transport bundle for `6a3d3f4` was removed after checking the
  deployed checkout; the checkout and scientific records remain.
  Those first two revised batches used about 140 recorded seconds per
  attempted episode. A tentative 512-episode three-arm run would therefore
  take about 60 H200-hours of training alone at that rate. This extrapolation
  is conditional on the pilot and excludes all development and held-out
  evaluation; the subsequently frozen plan above includes an evaluation
  reserve and a training-time safety margin.
  The regenerable 4,596,578-byte transport bundle for `ef3e76a` was removed
  after verifying its deployed checkout; the checkout and data are preserved.
  The same check allowed removal of the 4,565,084-byte `f07d078` transport
  bundle after its pilot checkout was deployed.
  The nine checkpoint-backed
  solver aggregate rows are reconciled in [W&B](https://wandb.ai/tatsu432s/self-play-opsd-coding-agent/runs/03df50eeb5bed38a3395292c),
  as is the [screen aggregate](https://wandb.ai/tatsu432s/self-play-opsd-coding-agent/runs/fb6a2b44ccbcc1b06d31443c).
  Preliminary checkpoint-backed plots and analysis are in the ignored local
  directory `runs/task21/solver-analysis/`.

| Training-only feasibility source | Verified | Public pass | Hidden pass | Accepted edits | Rejected edits |
|---|---:|---:|---:|---:|---:|
| LiveCodeBench | 3/8 | 3/8 | 3/8 | 3 | 0 |
| BigCodeBench | 2/3 | 3/3 | 2/3 | 3 | 0 |

The 11 episodes generated 40,180 tokens. Median measured episode time was
83 seconds and the maximum was 200 seconds. The sample supports executable
native repair on these tasks. An estimate of mixed reward frequency requires
repeated sampling. No transfer or training improvement is claimed here.

## PreciseCoder protocol and scope

The [PreciseCoder paper](https://arxiv.org/html/2610.18755v1) trained on
5,274 PDB-Single-Full tasks and evaluated on 1,205 PDB-Single tasks with
disjoint base problems. Its main single-turn method combines a unit-test
reward with an edit-fidelity F1 reward, normalizing the terms separately;
its reported Qwen3.5-4B PDB test pass rate was 55.3% versus 36.5% for base
and 55.1% for unit-test-only GRPO. The paper reports three 4B seeds and a
group size of six. It also evaluates a separate five-turn agent and reports
agentic RL for its 9B model. Our study uses an eight-turn native tool agent,
four-rollout groups and verifier-driven RL/OPSD on a much smaller adapted
subset; its results are not a direct replication or comparison of that paper's
single-turn or 9B agentic results.

## Reproduction

Reuse the pinned Task 20 release files under `runs/task20/pdb/`. The adapter
joins PDB parent IDs to the six LiveCodeBench fixture files at revision
`0fe84c3912ea0c4d4a78037083943e8f0c4dd505` and to the BigCodeBench fixtures in
PDB scorer commit `fbfa901eeae5b5f6944daea479a4ce31fad0e4a6`. Upstream fixture
inputs, generated data and trajectories stay in ignored local or Yale
storage. The worker receives source and case inputs, with expected outputs and
verdict computation in the trusted verifier. Private case inputs are used
during execution, while the agent's tools expose only designated public tests
and observations. [PDB scorer](https://github.com/Bill1235813/PDB/tree/fbfa901eeae5b5f6944daea479a4ce31fad0e4a6)
· [LiveCodeBench fixtures](https://huggingface.co/datasets/livecodebench/code_generation_lite/tree/0fe84c3912ea0c4d4a78037083943e8f0c4dd505)

```bash
uv run --frozen python scripts/prepare_task21.py \
  --pdb runs/task20/pdb --upstream runs/task21/upstream \
  --output data/task21/pdb-pilot-reproduction --report runs/task21/local-reproduction.json \
  --screen 24
uv run --frozen selfplay hpc status aefab0ada76973689cac5bdf
uv run --frozen selfplay hpc task21-verify aefab0ada76973689cac5bdf
uv run --frozen selfplay hpc status f694b7c181e4dc10dde279dd
uv run --frozen selfplay hpc task21-budget
```

The original run used the first pinned LiveCodeBench fixture file. To recreate
its exact manifest, provide only that file or its saved parent-matched extract
alongside the pinned BigCodeBench fixture file, and execute the preparation
script from the submitted commit. The full six-file fixture join is for the
subsequent study partitions. The `check_task21_parity.py` script records the
independent original-source verdict check. Changes in data or code create a
fresh manifest or scientific identity.
