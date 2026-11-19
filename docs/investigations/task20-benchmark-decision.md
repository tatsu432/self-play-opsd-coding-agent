# Task 20 benchmark decision

Activate the EvalPlus-derived fallback. PDB remains a useful reference for
the experiment design. The current worker and source adapter do not establish
an executable, equivalent PDB evaluation within the bounded migration.

## PDB feasibility

The pinned source-only audit applied the existing static admission policy
to both buggy and canonical code. It performed no candidate execution.

| Release | Pinned revision | Rows | Both sources admitted |
|---|---|---:|---:|
| PDB-Single-Full | `dac1f9aea17f507b2df3f253f06ff545bbdf2ce1` | 7,589 | 45 BigCodeBench |
| PDB-Single | `b3c0b911c5df4444ff3e8160b8dbe4ea297c08ec` | 5,751 | 30 BigCodeBench |
| PDB-Multi | `cb13eb9a160a702745d22129149083c14bedf215` | 256 | 0 |

All 3,892 LiveCodeBench rows in Single-Full were rejected. Of those, 3,860
failed the class-definition restriction and 32 failed the import restriction.
The fallback verifier executes an admitted top-level function with encoded
arguments and expected returns. The official LiveCodeBench handler uses its
custom evaluator and test harness, and the official BigCodeBench handler
runs separate test suites with their dependencies. Static admission of 45
BigCodeBench pairs leaves unit-test parity and eligible parent coverage
unverified. No PDB base score, accuracy, or edit metric was measured.

The release documents list MIT licensing. PDB-Single is a filtered subset
of Single-Full. The 256 PDB-Multi records are part of the broader PDB-Wild
collection. The pinned scorer/import inspection used
`fbfa901eeae5b5f6944daea479a4ce31fad0e4a6`. Public release dates do not
determine contamination of upstream source problems. [Official implementation](https://github.com/Bill1235813/PDB/tree/fbfa901eeae5b5f6944daea479a4ce31fad0e4a6)
· [Dataset releases](https://huggingface.co/Precise-Debugging-Benchmarking)

```bash
uv run --frozen python scripts/audit_task20_pdb.py runs/task20/pdb runs/task20/pdb-admission.json
```

PreciseCoder supplies motivation to test debugging learnability with a
training-only positive control. Its parent-disjoint 5,274/1,205 split and
single-turn results define a different protocol. [PreciseCoder](https://arxiv.org/html/2610.18755v1)

## Fallback

The new manifest reuses verified canonical parents and their family partition
from protocol-v3. Generate separate semantic mutants, including two-line
literal or boundary changes and single comparison/operator mutations, and
require a semantic test failure plus canonical success in the isolated
worker. Allocate a parent-disjoint training-only probe separately from screen
and solver curriculum tasks. Final families are held out from Task 20
training. Many upstream problems and families were used in earlier studies.

The benchmark decision and reward-variation gate have separate evidence.
Worker verification establishes correctness of the fallback oracles. The
Task 20 base screen is still required to establish useful difficulty.

The local isolated Docker worker verified all 205 admitted mutants and
canonical programs. A subsequent pass checked every frozen task and public
test export and replayed one verifier canary per partition. The manifest
identity is `508cdae5ec332991389b2f42df13b0377f466ca143d40a2db1b5458acff629f2`.

| Partition | Tasks | Parents | Families | Two-line mutants |
|---|---:|---:|---:|---:|
| Training screen | 32 | 32 | 32 | 10 |
| Separate solver probe | 8 | 8 | 8 | 4 |
| Development | 41 | 23 | 23 | 23 |
| Primary final | 100 | 60 | 52 | 36 |
| Harder transfer | 24 | 24 | 24 | 24 |

Development came in below its 50-task preparation target, so freeze its
41 available tasks. Retain unique families and report parent-clustered
intervals. Keep the private manifest and task artifacts under
`data/task20/evalplus-v1`. The isolated Yale runner checks two training
canaries against its own worker before model inference.

```bash
uv run --frozen python scripts/task20_data.py \
  --source data/protocol-v3 --output data/task20/evalplus-v1
uv run --frozen python scripts/task20_data.py \
  --source data/task20/evalplus-v1 --validate-only
```
