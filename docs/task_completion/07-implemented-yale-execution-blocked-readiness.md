# Task 07 — Yale execution implemented; launch readiness blocked

Status on 2026-10-27: **not eligible to submit main A–D**. Authorized execution
repairs and independent acceptance checks are implemented. The remaining blocker
is the frozen model/prompt combination's zero valid proposal yield. The real D
pilot and A/B/C checks correctly fail their learning-coverage requirements.
No study was frozen and no main training job was submitted. Task 07's launch
objective is incomplete pending a scientific decision.

Specification: [Task 07](../task/07-fix-ycrc-issue-spec.md).
Plan/history: [Task 07 plan](../plan/07-fix-ycrc-issue-plan.md).
Pending decision: [shared diff-format clarification](../task/07-shared-diff-format-decision.md).
The proposal is not applied to the active prompts or parser.

## Verified state

| Gate | Result | Live evidence |
|---|---|---|
| Worker | Passed | 31033392 / `639345f9a3d7a5c3953ece55`; all 19 checks |
| Deployed data | Passed | 31045342 / `6ab6345e274dd617c2f0d9da`; all 306 tasks |
| Single-GPU backend | Passed | 31040984 / `d3a9ac3571ccc56dc379385b` |
| Selected target backend | Passed | Same one-H200 proof; no multi-GPU comparison required |
| Base evaluation | Completed | 31046386 / `46ec4ac904e9611f551f8713`; 0/41 repairs |
| Real D pilot | Failed acceptance | 31049338 then 31050754 / `f597e3f66092a9a9c1265ea0` |
| A/B/C acceptance | Failed acceptance | A 31054795, B 31055413, C 31056122 |
| Freeze | Blocked | No successful pilot/method evidence; no frozen study |

All four actual main previews returned `Study is not eligible for main
submission` (exit 1). **0/4 successful previews.** These dry runs allocated
nothing. The authoritative doctor reports the five gates above as true and
pilot/methods/freeze as false. No jobs or watchers remain active.

## What changed and how it was verified

- Replaced unsupported nested Apptainer cgroup/PID controls with Slurm step
  CPU/memory enforcement and an auditable Linux AMD64 seccomp allowlist installed
  before candidate compilation/imports. Candidate-created processes and threads
  are prohibited, including alternate syscall ABI/creation paths. The contract
  reports one candidate process/thread, not a fictional pids.max or 32-process cap.
- Fresh Apptainer payloads have private filesystem/network namespaces, no GPU,
  credentials, trainer or oracle mounts, read-only root/devices and a 64-MiB
  private temporary store. Only three trusted worker entry files are mounted
  individually read-only. Time/output limits are enforced by the supervisor.
- Real Bouchet checks proved the one-CPU cpuset cannot be expanded, independent
  memory is kernel-bounded at 1 GiB, candidate OOM leaves the broker usable for a
  subsequent correct request, and timeout/output/cancellation cleanup works.
  An actual Slurm KILL of a live contained step left no observed processes
  running. `--oom-kill-step=0` retains the memory bound while keeping the broker
  alive after a candidate OOM. Deliberate OOM witnessing is CPU acceptance only.
  GPU allocations subsequently passed their ordinary worker preflights too.
- Separated operation submission slots while retaining equivalent/uncertain
  submission exclusion. Definitive scheduler rejection now releases its slot;
  known legacy rejected slots can be reclaimed, while empty uncertain ledgers
  and active/pending intents remain blocking. Budget locking/accounting persists.
- Scoped numerical/allocation compatibility to relevant dependencies, normalized
  Python 3.9/3.12 verifier AST differences, and reused checksummed SIFs with full
  original provenance. Controller/docs changes do not force model retests or
  image rebuilds. Trusted backend checks remain independent of generated workers.
- Retained kernel device-cgroup ownership and exact container CUDA UUID/initial
  occupancy verification. Pinned vLLM needs numeric CUDA masks; these are proved
  by actual CUDA UUIDs before model load. Check 31038198 demonstrated NVIDIA
  index 0 with Linux minor 1; equality is never assumed. Publication-only recovery
  of 31030407 preserved the failed job/raw proof without another GPU allocation.
- Fixed PEFT's BF16 proposer versus FP32 student adapter-master mismatch by
  matching master precision before copying; version equality and independent
  storage remain required. BF16 base compute and all objectives are unchanged.
- Removed inherited torchrun elastic-agent/rank/rendezvous state from the separate
  vLLM engine. Real CPU reproduction and H200 acceptance verify independent
  process-group startup. Startup exceptions/EOF now give actionable child errors.
- Made the base no-op guard count local DTensor coordinates and reduce across
  ranks. A real two-rank CPU test makes both ranks reject a nonzero remote B
  weight. The real baseline then completed on Bouchet.
- Reserved seven trainer CPUs/128 GiB plus a separate worker CPU/1 GiB: eight
  CPUs/129 GiB total. Slurm had rejected nine CPUs for one H200. Pilot wall
  ceilings scale to four/eight hours on one GPU while retaining the original
  4+8 allocated GPU-hour ceilings and iteration-two stop/resume.
- Corrected method-smoke's positional launcher argument and allowed explicit
  A/B/C selection; D requests the full suite. Each selected method launches a
  fresh process. Passed evidence remains compatible with these launcher repairs.

## Scientific acceptance failure

The baseline completed 41 fixed monitoring tasks with zero successful repairs.
The D pilot completed ten iterations, stopped cleanly at two, resumed from the
verified joint checkpoint `000002`, and reached checkpoint `000010`.

Across its **160 attempted proposals, zero were valid**. There were 154
required-diff-header failures, one hunk-count failure and five JSON failures.
Consequently there were zero generated student episodes, student/proposer
optimizer updates and alignment scores. The CLI correctly refused to publish a
successful pilot report. No malformed proposal was repaired or refilled.

A, B and C each completed their own one-iteration check, attempted 16 proposals,
produced zero valid tasks/episodes/updates and saved a joint checkpoint. All
method distillation-gate checks passed. C's required alignment coverage failed;
A/B correctly reported no alignment path. None passed learning acceptance.

An offline diagnostic on the first 112 D proposals found 13 Git preambles.
Removing only this metadata still yielded zero valid mutations because of
context/hunk/line-format errors. These counterfactual outputs were never used
in training. Parser, prompts, sampling, rewards, main 200 iterations, seed 2026,
model/tokenizer, mathematical budgets and frozen data boundaries remain unchanged.

The reviewed proposal adds a shared explicit diff-format reference to the
student/proposer descriptions. Its synthetic example passes the current mutation
validator (one AST node/two changed lines). It changes model conditioning, so
Task 07's prompt-preservation requirement makes user authorization indispensable.
Authorization is pending. Do not rerun the unchanged zero-yield checks or weaken
their acceptance gates.

## Code, CI and identities

Execution code: `9980c78c425765068327ffdfe2d249f848cd455c`.
[Hosted CI 40617600830](https://github.com/tatsu432/self-play-opsd-coding-agent/actions/runs/40617600830)
passed **220 tests, one skipped, three deselected**, including Linux/Docker worker
tests, lint, formatting and type checking. Focused local controller regressions
passed 88 tests. Earlier real backend source was 6768da2; the evaluation repair
was 4a393df; subsequent changes preserve their relevant evidence identities.

| Identity | SHA-256 / revision |
|---|---|
| Model and tokenizer | `cdbee75f17c01a7cc42f958dc650907174af0554` |
| Frozen manifest file | `44fe6caffc0548bebb01bf39208f8b2f5de10ab89ee8487dca20dd436c0154f5` |
| Canonical manifest | `633a0ee3a94ff9f7efdfd48c50f1ac22ae565879f7b9739ee00b28dbca9f9658` |
| Prompt | `ec1f46db1df8afe308934c54222a37cb6c9d73d072995699e19ea9419ccfd9c5` |
| Trainer SIF | `fbe2737a6815644ecaad01df40cfa7883b05e5554d518fd0352129b1f6fd69e2` |
| Worker SIF | `76a43eef3424a6e72d516a73c967d0610e12c13288eba6cbd762d8635e860ec3` |
| Backend code | `9a35161e745f6dc007ca4d7761fa096c9af5f21a935fcf6a38f9d9e6cb80fa78` |
| Runtime scientific code | `b086428de1cd2da28daa0dbde0c592fd5aed688821720c5cb0a1c9531c906c9c` |

## Hardware, measurements and allowance

Selected: one H200 in `gpu_h200`, profile `single_gpu_80gb`, tensor parallel one,
rollout concurrency one, microbatch one. H200 was chosen for documented live
scheduling reasons within entitled cpsc4880 resources. No H100 OOM was fabricated
or established. No extra host or worker service was provisioned.

Baseline: 1205 allocated GPU-seconds. D pilot: 406+513=919 GPU-seconds including
startup, reference preparation, checkpoint/continuation and the failed acceptance.
Its empty-workload median iteration was 49.06 s, p90 104.53 s; maximum reported
trainer PyTorch peak was 10.46 GiB. These exclude rollout-child peak memory and
**do not measure valid-task training throughput or learning memory fit**.
A/B/C took 173/170/363 GPU-seconds respectively. A prior launcher-argument failure
cost 46 GPU-seconds and is also charged.

Reconciled cumulative readiness consumption: **1.091111111 GPU-hours of 24**,
including Tasks 03–06 and every Task 07 attempt. Outstanding reservations: zero.
Remaining allowance: 22.908888889 GPU-hours. Use the durable ledger/live Slurm
accounting before more work; never reset it for a new study identity.

A defensible main-run cost or full learning-memory estimate is unavailable
because the pilot had no valid learning workload. Extrapolating its empty
iterations would misrepresent main training. No main-launch commands are valid
until the failed gates, freeze and four previews pass under an authorized setup.

## Review and continuation

```bash
uv run selfplay hpc doctor --gpu-type h200
uv run selfplay hpc inspect
uv run selfplay hpc status f597e3f66092a9a9c1265ea0
uv run selfplay hpc logs f597e3f66092a9a9c1265ea0
uv run selfplay hpc fetch f597e3f66092a9a9c1265ea0
```

Local fetched artifacts live under `runs/hpc/yale-hpc-cpsc4880/`. The pilot's
`acceptance_diagnosis.json` is a derived negative-acceptance summary; the durable
events, checkpoints and Slurm ledgers remain authoritative. Preview refusals are
recorded in `task07-main-preview-refusals.json`. The tested preview commands were:

```bash
uv run selfplay hpc submit --operation train --method A --profile single_gpu_80gb --gpu-type h200 --dry-run
uv run selfplay hpc submit --operation train --method B --profile single_gpu_80gb --gpu-type h200 --dry-run
uv run selfplay hpc submit --operation train --method C --profile single_gpu_80gb --gpu-type h200 --dry-run
uv run selfplay hpc submit --operation train --method D --profile single_gpu_80gb --gpu-type h200 --dry-run
```

Keep the current failed pilot/baseline/method records intact. If the scientific
proposal is authorized, establish a new common prompt/study identity and all
required compatible admission/backend/learning evidence, retaining every frozen
parent/task/split and the same cumulative allowance. Main authorization is still
separate. Standard recovery (`hpc watch RUN_ID`, or `hpc resume RUN_ID
--continue-stopped` for a verified clean stop) applies only to eligible authorized
runs; it must not retry this scientifically completed zero-yield pilot. No YCRC
support response is required to continue the accepted worker path.
