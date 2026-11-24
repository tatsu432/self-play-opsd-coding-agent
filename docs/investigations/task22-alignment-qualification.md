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
execution. The charged qualification on Yale remains
pending. After it executes, record the final verifier and tokenized reference
evidence, runtime, paired reward diagnostics, allocation, and go/no-go here.
The local native edit check is separate from this gate.
