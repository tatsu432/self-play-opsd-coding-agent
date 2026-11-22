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
- Shared-policy calibration job `34225453` / controller run
  `e75f67bcc64df23c764807cb` completed exit 0 in 2,751 H200-seconds from
  `755174d`; its checkpoint and result were verified. On four proposed tasks,
  RL, ungated OPSD, gated OPSD and proposer gradient norms were respectively
  0.01396, 0.07262, 0.02110 and 0.04615. The calibrated proposer scale is
  0.15128449959390794 and gated OPSD scale is 0.19849138589186444. This
  measurement does not establish that proposer rewards remain diverse during
  training. The [calibration aggregate](https://wandb.ai/tatsu432s/self-play-opsd-coding-agent/runs/cd05b24447acd87178bd106a)
  was reconciled. Shared pilot `34229678` / `5a71bfd3b213e7beaf4e3c02` was
  submitted once from `02ca483` and is running; inspect its ledger and never
  duplicate it.
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
