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
- Slurm job `34122424`, controller run `aefab0ada76973689cac5bdf`, was
  submitted once from `9ba11127648a1a51c8cfc387a2eeaedb60c6efae` using one H200
  and a 45-minute reservation. It started and is generating feasibility
  episodes. Reconcile its terminal record and checkpoint before using results
  or sending a follow-up allocation.

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
uv run --frozen selfplay hpc task21-budget
```

The original run used the first pinned LiveCodeBench fixture file. To recreate
its exact manifest, provide only that file or its saved parent-matched extract
alongside the pinned BigCodeBench fixture file, and execute the preparation
script from the submitted commit. The full six-file fixture join is for the
subsequent study partitions. The `check_task21_parity.py` script records the
independent original-source verdict check. Changes in data or code create a
fresh manifest or scientific identity.
