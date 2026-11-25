# Task 22 alignment qualification

## Local evidence

The candidate sidecar is content addressed and bound to the seed-2026 Task 21
PDB manifest. The 11 preferred feasibility examples have zero parent and
family overlap with the training curriculum and evaluation splits. A local
native-protocol check constructed one exact `replace_text` action per example
and reconstructed all 11 canonical sources byte for byte. The sidecar retains
the independently verified training-only reserves in stable order. A local
check with tokenizer files from the pinned Qwen revision found a maximum
prompt of 1,267 tokens and a maximum repair target of 104 tokens. All 11 masks
included repair-action tokens. This check used Transformers 4.57.1. The Yale
runner will freeze the verified cache with the qualified runtime.

The archived B snapshots available for replay are:

| Prior training episodes | Protected checkpoint iteration | Subsequent batch |
|---:|---:|---:|
| 0 | 0 | 1 |
| 128 | 6 | 7 |
| 256 | 10 | 11 |
| 384 | 15 | 16 |

Local batch checksums match their committed iteration records. Their policy
signatures match the preceding snapshots, and all nonempty student turns
contain original behavior scores and detached student and teacher scores.
Remote checkpoint files, installed runtime, and worker must match before
these batches can be used.

The local Docker worker failed its containment preflight before reference
execution. The separate Yale qualification passed: job `34554246`, run
`1a8172783b2dbd834a3332af`, completed with exit 0, using 1,433 H200-seconds
(0.3981 hours). Its controller-verified receipt is retained privately at
`runs/task22/qualification-verified.json`; the verified tokenized reference
SHA-256 is `a3876e02743ef1f86fcc5001f966ef4f4d34257abcabe8739d56be1ba11555f5`.
The gate accepted the reference and authorized the frozen main configuration.

| B snapshot episodes | Scored tasks | Reference norm | C near-zero | D near-zero | C proposer norm | D proposer norm |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6 | 5.1292 | 2 | 0 | 0.02847 | 0.03690 |
| 128 | 8 | 4.5527 | 3 | 0 | 0.02289 | 0.00113 |

Both methods produced distinct valid rewards within two parent groups at the
first snapshot and three at the second. These paired replay diagnostics show
that D can retain a directional signal where C is near zero; they do not show
that either reward improves held-out repair performance. The two diagnostic
batches took approximately 643 and 717 seconds, respectively. The local
native edit check is separate from the Yale gate.
