# Task 19 implementation and execution status

The implementation adds a shared Qwen3.5 LoRA, one AdamW state, Dr. GRPO,
native thinking and exact-text tools, executable semantic mutations, and
training-reference gradient alignment. Task 19 uses a separate dependency
lock and SIF identity. Its initial control has empty calibration and
qualification hashes. The controller requires real evidence before launching
the five-arm study.

Local validation passed 462 tests, with 20 skipped and 18 deselected. Ruff
lint and format, mypy, and the Task 19 lock check passed. CPU tests cover the
native protocol, objective, joint step and checkpoint restoration. An isolated environment with Transformers 5.13.0,
PEFT 0.19.1 and PyTorch 2.10.0 exercised the real Qwen3.5 model classes with
tiny CPU weights and the pinned tokenizer. All 14 frozen reference edits
round-tripped to the original correct source. Five figure templates rendered
to PNG and PDF with synthetic test values. GPU image construction, numerical
qualification, measured costs, evaluation and method comparisons are pending.

Yale authentication succeeded. CPU image check job `33706240` completed exit 0.
Trainer image build job `33706414` completed exit 0. H200 qualification job
`33707814`, controller run `170dd97ea69313f0badb1268`, was submitted once from
commit `122604ebbe432da8225aab3eb4315fbbbeeeff12`. It failed before worker startup because the allocation dispatcher rejected the
Task 19 operation. Slurm charged 2 H200-seconds. The dispatcher is corrected
and its local tests passed.
Replacement qualification job `33720283`, controller run
`01a3a1ca9cbf435ad60584db`, was submitted once from
`2436cedded42bbdd4b224d6272f057cf234906e3` after a passing preview. It is
pending on priority with a four-H200-hour reservation. Inspect this existing
job and its ledger before any submission. The replacement failed after 4,362 H200-seconds because its nine student
groups had eight all-success groups and one all-failure group. Its numerical
check therefore had no informative RL gradient, and did not establish a
cached-scoring mismatch. The revision `task19-v2` reuses the checksum-bound
frozen batch and extends it with up to 20 additional training tasks before
retrying the numerical check. No Task 19 benchmark or training result is
available.

## Yale execution

Run these from the pushed Task 19 branch. After `uv run selfplay hpc login`
and Duo approval, use `uv run selfplay hpc inspect` and the matching resource
query to confirm the current account, jobs and H200 allowance.

```bash
uv run selfplay hpc resources --settings configs/task19-yale.yaml \
  --profile single_gpu_80gb --gpu-type h200 --account cpsc4880
uv run selfplay hpc sync --settings configs/task19-yale.yaml
uv run selfplay hpc reuse-worker repair-selfplay-2026-task18-v1 \
  --settings configs/task19-yale.yaml
uv run selfplay hpc images build --settings configs/task19-yale.yaml \
  --kind task19-trainer --build-mode yale --check-only
```

After the CPU build check finishes, submit the same command with
`--check-only` removed and monitor the existing build to a pinned SIF. If
the exact immutable commit already has a passing build check, reuse it.

Use the following options for the allocated scientific stages.

```bash
uv run selfplay hpc submit --settings configs/task19-yale.yaml \
  --profile single_gpu_80gb --gpu-type h200 --account cpsc4880 \
  --operation task19-qualify --method control --dry-run
```

After a successful preview, submit once with `--dry-run` removed. Track the
returned controller run ID through `hpc status`, `hpc watch`, and `hpc fetch`.
Qualification records actual model compatibility and calibration artifacts.
A recorded cache parity failure exposes the bounded `task19-qualify-uncached`
operation. Use the same worker and model image for either route.

Run `task19-base --method control`, then `task19-pilot` for each of
`control A B C D`. Pass the existing private W&B secrets file with
`--secrets-file` for online aggregate metrics. Keep raw trajectories,
reference repairs, prepared data, and full checkpoints private. For a clean
allocation stop, continue the same controller run with
`uv run selfplay hpc resume RUN_ID --continue-stopped` after inspecting it.
The same completed pilot is the checkpoint source for its main segment.

Once all five pilot markers and their Slurm charges are verified, run
`uv run selfplay hpc task19-freeze --cap VERIFIED_TOTAL_H200_HOURS --export runs/task19/main-control.json`. It writes
one common endpoint from actual pilot costs, the evaluation reserve and a 20%
contingency. Then submit `task19-train` once per arm. These segments continue
the same scientific runs and include the frozen final ID and external
evaluations. Review `hpc task19-budget` after every stage.

## Analysis

Fetch the base and five arms with `hpc fetch RUN_ID --full` and reconcile
`hpc task19-budget` into the private analysis directory. The analysis manifest
maps `base`, `control`, `A`, `B`, `C` and `D` to objects with a local
`path` and a `controller_runs` array. Include each arm's pilot and main run IDs.
Use the controller-published `main-control.json` for the common endpoint.

```bash
uv run python scripts/analyze_task19.py \
  --manifest runs/task19/analysis-manifest.json \
  --control runs/task19/main-control.json \
  --budget runs/task19/budget-final.json \
  --output runs/task19/analysis
```

The final report should record results conditional on seed 2026, paired family
intervals, compute and failure taxonomy. The previously inspected external
benchmark is exploratory. Historical Task 15–18 evidence remains in its own
results records.

Task 19 v2 extended qualification job `33730057` / controller run
`4297c7d19ed430ce68361596` started from immutable commit
`2d58885daf00f6eb747f8e75bd67f536f7fb4d9a`. The original nine groups are
checksum-bound, and the job can add at most 20 training tasks. Its four-hour
reservation and the earlier 4,364 H200-seconds share the 80-hour ledger.
Inspect this job before taking action and never duplicate it.

The v2 qualification job `33730057` failed after 4,631 H200-seconds because
no proposal group contained distinct valid difficulty rewards. Its 29 verified
task groups comprised 23 all-success groups, five all-failure groups, and one
mixed group. Some proposer groups contained both valid and invalid proposals.
The specification requires continuing controlled pilots when proposer
reward discrimination is weak. Task 19 v3 therefore reuses the exact saved
batch, records the lack of valid reward variation, and qualifies numerical
behavior using the mixed group. Loss calibration uses up to four student and
four informative proposer groups, with invalid-proposal penalties retained.
The v1 and v2 jobs used 8,995 H200-seconds in total.

The v3 qualification job `33739684`, controller run
`556bb9e4c2b309660f082527`, was submitted once from immutable commit
`5b1eb075c98411b499d15ba91a31dd917de6fe7e` after a passing preview. It is
pending on priority with a four-hour reservation. Its saved batch source is
v2 `frozen-training-batch.json` SHA-256
`4cd5991768348988e351bbebf43ed59c58bc14c822a5a20dfdb2c2ea30a3194d`.
Inspect this run and the shared ledger before further actions.
