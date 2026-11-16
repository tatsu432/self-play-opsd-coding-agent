# Task 18: interface and learning-signal investigation

## Current status (2026-11-16)

Implementation and local validation are underway on
`feat/18-interface-signal-bottlenecks`, from main `48b4839`. The study is
`repair-selfplay-2026-task18-v1`, with a **hard 12 new allocated H200-hour cap**
including failures and continuations. Historical Tasks 15–17 remain complete
and unchanged. New GPU conclusions are pending; training is disabled in the
checked-in control until numerical and interface evidence supports it.

Yale login was refreshed successfully and live inspection found no active
account jobs. All ten files in the existing scratch model snapshot matched the
preserved project's `SHA256SUMS`; no model files were downloaded or restored.

## Saved-data findings

The read-only audit reconciles all 12 Task 17 objective endpoints and their
120 committed iteration events. A sampled-token maximum exceeds 0.2 in 119
batches; the affected-token fraction ranges from zero to 0.9622%. These are
batch statistics, not an estimate of harmful optimization.

The calibrated-2026 training shards were retrieved separately, without copying
checkpoint tensors locally. Direct saved-array analysis reproduces iteration
4's maximum 5.48991494, with 6/3,346 tokens above 0.2. One sampled token changes
the active clipped-surrogate branch relative to a ratio-one diagnostic. This
is not a measured gradient or optimizer-update effect. The exact generating
checkpoint 3 was retired; retained checkpoints 0/5/9 match batches 1/6/10.
Task 18 therefore targets calibrated-2028 batch 10 (a mixed-reward group) and
calibrated-2026 batch 6 (uniform-reward controls) for restored-state GPU replay,
retaining each original objective, coefficient and both optimizer states.
Neither replay can establish the worst batch's causal effect. All 495 saved
calibrated-2026 turn prefixes and token/probability lengths passed a CPU check
with the pinned tokenizer; none of its six >0.2 outliers was an EOS token.

![Historical aggregate numerical differences](../figures/task18/numerical-history.png)

The final Task 17 pool contains 104 independent families and 120 rows including
presentation variants. Only one originally reserved eligible parent remains
unassigned after preparation replacements; it is not a new confirmation set.
All new interface comparisons reuse previously inspected training families.

## Implementation and decisions

- Extend existing numerical helpers with distribution, clipping-branch,
  repeat-controlled gradient/sign and actual Adam displacement diagnostics.
  Restore both adapters, both optimizers, RNG and model mode after every branch
  and after failure. Rescored denominators are diagnostic only.
- Use existing normalized diff for the missing fresh-base/B2026 comparisons,
  with the same strict prompts and observations. No new editor is introduced.
- Add opt-in Qwen-native serialization with the pinned tokenizer's `tools`
  argument, exact sampled IDs, single-call parsing, and existing action/path/
  import/API validation. Tool serialization is separate from editing semantics.
- Task 18-only configuration binds the control checksum and serialization to
  scientific identity. Historical configuration hashes remain unchanged;
  changed runtime code has a new code/prompt identity and is never a historical
  continuation. Historical interface controls require behavioral parity and are
  otherwise contextual evidence.
- Extend the existing controller, worker launch and transactional continuation
  paths. One locked Task 18 budget spans operations, seeds, readiness GPU work,
  failures and continuations. No historical budget is imported.
- Task 18 runtime proof covers data/verifier dependencies, model revision and
  numeric observation/sequence budgets; full code, prompt, image and worker
  identities remain bound. It does not certify loss math or native generation.
  A matching proof can cover interface/objective arms without repeating the
  unchanged full dataset audit for each arm.

The first matching CPU runtime allocation is job `33574512`, controller run
`96daa9183384957cd9ea58b5`, submitted once from `dd068f0`, completed exit 0.
Its ledger remains authoritative; never duplicate it. Its verifier evidence can be reused after
changes confined to learning runners only when every file in the explicit
verifier dependency list and every non-code runtime identity still matches the
archived executed checkout. Changed verifier dependencies require a new check.

Local validation passed **443 tests**, with 20 skipped and 18 deselected.
Ruff lint/format and mypy (55 source files) passed. These checks include actual
small-model Adam restoration and cached pinned-tokenizer prefix/EOS tests; they
are not GPU research results.

## Frozen execution sequence

1. Reconcile Slurm and existing logical-run ledgers, synchronize a clean commit,
   reuse unchanged worker containment, and run one matching CPU runtime check.
2. Replay the two checksum-bound historical batches under their exact complete
   checkpoints. Compare fresh scoring, gradient-mode scoring, repeat updates
   and diagnostic denominator substitution. Relative update difference above
   1% or cosine below 0.99 beyond repeat noise requires investigation; passing
   this screen alone cannot enable training.
3. Run normalized diff on 40 primary families for base and B2026. Run native
   exact-text serialization on eight deterministic families per snapshot.
   Expand native to 40 only if validity improves without fewer repairs and the
   remaining interface budget permits. Select by common-40-family verified
   repairs, token cost, then simplicity; retain plain exact-text on a tie.
4. Train only for a substantive new interface/numerical comparison. Freeze
   GRPO/gated/nominal ungated, two paired seeds, binary reward and an 80-episode
   target; stop all six arms at iteration 4 (32 episodes). Continue all six only
   after credible nonzero updates and a conservative whole-study cost forecast.
   Development performance cannot decide continuation. Otherwise report the
   common pilot endpoint; do not selectively extend favorable arms.

Initial earmarks are 2 numerical / 2 interface / 8 training H200-hours. Unused
budget may move forward, but actual charges plus outstanding scheduler
reservations may never exceed 12. Training, partial-reward training, C/D, SFT,
new editors, observation changes and adapter transfer are not enabled by this
preparation record.

## Reproduction

Private outputs stay ignored under `runs/task18/`. No source, oracle, raw
trajectory or credential is published.

```bash
uv run python scripts/analyze_task18.py \
  --manifest runs/task17/objective-manifest-final.json \
  --pool runs/hpc/yale-hpc-cpsc4880/86e2df49bb04a9321f7d755d/pool.json \
  --output runs/task18/saved-audit-v1
uv run python scripts/analyze_task18_tokens.py \
  --source runs/task18/recovered-calibrated-2026 \
  --output runs/task18/recovered-token-audit-v1
uv run python scripts/plot_task18_numerical.py \
  --summary runs/task18/saved-audit-v1/summary.json --output docs/figures/task18
```

After synchronization and readiness, preview first, then submit the same command
without `--dry-run`. Interface arms use A=base, B=trained B2026; student arms
use D=GRPO, B=gated, A=nominal ungated. These labels do not authorize full C/D.

```bash
uv run selfplay hpc inspect
uv run selfplay hpc sync --settings configs/task18.yaml
uv run selfplay hpc reuse-worker repair-selfplay-2026-task17-interface-v1 --settings configs/task18.yaml
uv run selfplay hpc submit --operation runtime-check --method B \
  --profile single_gpu_80gb --gpu-type h200 --settings configs/task18.yaml
uv run selfplay hpc submit --operation task18-numerical --method A \
  --profile single_gpu_80gb --gpu-type h200 --settings configs/task18.yaml --dry-run
uv run selfplay hpc submit --operation task18-normalized --method A \
  --profile single_gpu_80gb --gpu-type h200 --settings configs/task18.yaml --dry-run
uv run selfplay hpc submit --operation task18-native --method A \
  --profile single_gpu_80gb --gpu-type h200 --settings configs/task18.yaml --dry-run
```

The native model's pinned template is documented in its
[tokenizer configuration](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/cdbee75f17c01a7cc42f958dc650907174af0554/tokenizer_config.json).
Compatibility tests use that locally cached tokenizer with no download.
