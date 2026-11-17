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
Inspect this existing job and its ledger before any submission. No Task 19
benchmark or training result is available.

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
