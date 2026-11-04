# Task 15: stabilized training and completed A/B study

Task 15 is complete. All six frozen online A/B training arms reached their prescribed 4,096-episode exposure target (B-2028 finished a four-episode group at 4,100), all 14 prescribed held-out evaluations completed, and the paired analysis and cloud telemetry reconciliation completed. The full results, uncertainty, causal limits and allocation accounting are in [the final investigation](../investigations/task15-final-results.md).

On the 115-task final in-distribution set, A/B scores for seeds 2026, 2027 and 2028 were 61/70, 57/68 and 64/71. On the 116-task external set they were 2/3, 1/0 and 1/0. The common base scored 7/115 and 2/116. B's in-distribution advantage is consistent in these three seeds; external repair remained near zero. Family-clustered task uncertainty and training-seed variation are reported separately in the investigation; three seeds are insufficient to establish precise population effects.

## Implementation and validation

The successor preserves actual behavior probabilities for PPO and uses a same-scorer signed privileged/unprivileged gap for distillation. It replaces invisible historical duplicate starvation with structured online proposal generation, bounded refill and explicit replay accounting; checkpoints retain both adapters, both optimizers, RNG, curriculum, supply and episode milestones. A and B both trained their difficulty-reward proposers as well as students. All observed training task groups were attributed to online proposals, with zero fallback groups; historically repeated proposal samples were recorded and are not counted as proof of novelty.

Training-only factorial replays demonstrated a material difference between using behavior versus rescored PPO denominators in one targeted base sample; the cause of the mismatch and its population impact remain unresolved. A changed-state stop/resume check verified checksum and identity, snapshot and exposure continuity, and optimizer-step advancement. An isolated audit found one structurally eligible training parent whose original failed prepared verification. The fresh V2 study excludes that parent from both arms; it does not reuse incompatible V1 checkpoints or change the held-out splits. The root cause of that parent's discrepancy is unresolved. See [the prior diagnostic record](../investigations/task15-status.md) for preserved evidence.

The V2 local suite passed 331 tests with 38 skipped; Ruff passed, and hosted CI [41396322243](https://github.com/tatsu432/self-play-opsd-coding-agent/actions/runs/41396322243) passed for the frozen execution. Local Docker was unavailable. The allocated Yale runtime and H200 backend checks passed; the accepted isolated worker evidence was reused under dependency checks. The last evaluation used six CPUs rather than eight because Yale changed its H200 allocation limit; its isolated worker and GPU ownership checks passed. Operational controller changes for timeout-attributed RPC disconnects and CPU-limit rejection are recorded on branch `fix/task15-timeout-recovery` (commits `d71ccd1` and `6de01c5`); they do not change the frozen training implementation.

## Frozen identity and checkpoints

- Study: `repair-selfplay-2026-task15-ab-v2`; settings `configs/task15-v2.yaml`; frozen execution `0ac77cde4bc5f70ba2b3eb9e457749e3106a2bc5`.
- Model: `Qwen/Qwen3-4B-Instruct-2507`, model and tokenizer revision `cdbee75f17c01a7cc42f958dc650907174af0554`; one H200 per training/evaluation allocation.
- Freeze record SHA-256: `5e0222709f72418dcf2a3c165a6d75bdfc2a6a86198081779ce1a70e493bf3fa`; trainer image SHA-256 `fbe2737a6815644ecaad01df40cfa7883b05e5554d518fd0352129b1f6fd69e2`; isolated worker image SHA-256 `76a43eef3424a6e72d516a73c967d0610e12c13288eba6cbd762d8635e860ec3`.
- Dataset manifest identity: `fd47b9117ffa627531a24b938999d9e063f1e6eee47e4b0b91b1bbf46c4ed714`. Final ID/external task counts were 115/116. All checkpoint files were verified by the controller and again by the CPU telemetry reconciler; hashes below identify each checkpoint's completion manifest.

| Arm | Run ID | Checkpoint | Completion manifest SHA-256 |
| --- | --- | --- | --- |
| A 2026 | `033b9d537c8ee9c23dc09038` | `000241` | `829ef022116e5076d36f8073a4a471cbce6101f412f08ff467e58c72f9db37ac` |
| B 2026 | `50964da5e6b09d580816f8c3` | `000245` | `14cc4179a73056607d8dd3671d1e6e0232fa5a0420b95efc0288aff2252725d0` |
| A 2027 | `d0cbacb4d235f68da22ec9db` | `000239` | `23b9b3882bbb6239ae73cedc140b47b4ccab81cdb9129e75ed19b3400b21a785` |
| B 2027 | `c2f03af1d35aab20132a995d` | `000242` | `6ee7a238539635b8cd3385a1efa60e6744cc847def4e519fb75a7d1f66f3fbf9` |
| A 2028 | `4e7ee388c11f7c301a9f123c` | `000244` | `e613b5f6c234bbd3bf79827aa768c5a07bb178ee077d07930c023e0012b593bf` |
| B 2028 | `e64cf66955c1416b33d66364` | `000236` | `3bd2644eff5b9cffeb43dfe2864ed5d2774e6dd7171a2b3c2dd209c83c0994cf` |

The main training allocations consumed 145.724 H200-hours; held-out evaluations used 14.640; abandoned V1 training used 20.183; and separately identified Task 15 diagnostics, backend and stabilization used 2.300. This totals 182.847 H200-hours, excluding reused earlier-study evidence and CPU allocations. Each held-out ledger has one accepted job and no pending intent. The final Yale scheduler reconciliation showed no active study job. CPU reconciliation job `32037439` completed exit 0 and cloud-verified canonical aggregates through all six checkpoint iterations; B-2028 had 90 earlier delivery errors, retained in local telemetry.

## Reproduction and retained artifacts

The Task 15 analysis commands and frozen evaluation procedure are in [operations](../operations.md#task-15-online-ab-successor). With private Yale records fetched and the immutable prepared data mounted, run:

```bash
uv run python scripts/analyze_task15.py --manifest runs/task15/manifest.json --data data/protocol-v3 --output runs/task15/analysis
uv run python scripts/analyze_task15_health.py --manifest runs/task15/manifest.json --output runs/task15/analysis
```

The private analysis summary SHA-256 is `6c1fe6e719a1a2781a40dec520d9cb4c511b2a29091d28ea047e9fc8e818f3aa`; the private 28-allocation training manifest SHA-256 is `d6c038d223659d1da1458508395b35608af10c1c5a474eba1c4639b199864263`. These and raw events remain ignored; final checkpoints remain checksum-verified on Yale rather than being bulk-copied over the home network. `learning-curves.csv` records episodes, generated tokens, optimizer steps and allocated GPU-hours at evaluations. The additional private health export records per-iteration supply, validity, mixed groups, action errors and probability mismatch. No raw task source, oracle, credentials or trajectory is included in this record.
