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
  and rejected the recovery ledger write with `Disk quota exceeded`. No
  continuation was submitted. Once project space is restored, inspect the
  ledger and checkpoint and resume this same run after reviewing the failure.
  Preserve the existing records and do not submit a duplicate screen.
- The incomplete screen contains 12 mixed groups among 23 complete groups
  of four and 17 mixed groups among 22 complete groups of eight. These are
  interim counts. The curriculum and solver-only pilot remain unfrozen.

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
