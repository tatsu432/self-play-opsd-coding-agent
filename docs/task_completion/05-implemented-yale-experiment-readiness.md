# 05 — Yale experiment readiness implementation and live checkpoint

Source: `docs/task/05-yale-experiment-readiness-spec.md`; plan:
`docs/plan/05-yale-experiment-readiness-plan.md`. Work is on
`feat/05-yale-experiment-readiness`. Main A–D training remains unauthorized.

## Implemented

- Added `hpc images build --kind trainer --build-mode yale` and `--check-only`.
  The authoritative Dockerfile is converted with uv-locked spython 0.3.14.
  Docker Buildx reads small immutable registry configuration on the controller;
  image layers, extraction, dependency installation and SIF compression run on
  Yale CPU nodes, eliminating the large local export requirement.
- Conversion resolves COPY destinations, enforces supported instructions and
  pinned bases, preserves exec argument quoting and base environment, and
  removes the informational Docker tag while retaining its exact digest.
  Corrected the trainer Dockerfile to use the pinned base's `python3` command.
  Official uv build metadata is accepted without relaxing the version pin.
  The trainer uses `/usr/bin/python3` from its digest-pinned base, verifies
  Python 3.12 and records the exact patch version. This meets the project
  requirement `>=3.12,<3.13`; uv 0.8.15 cannot provision the former 3.12.12
  request from its embedded interpreter catalog. The worker base remains pinned
  to Python 3.12.12.
- Squashfs compression is bounded to eight CPUs and 8 GiB for the trainer
  (two CPUs / 1 GiB for the small check), after a measured compression OOM at
  the 32-GiB allocation limit. `uv sync --no-cache` avoids retaining transient
  package downloads inside the final SIF.
- Builds stage checksummed sources in a private temporary context, require
  scratch/project capacity, validate architecture/package imports/versions/CLI,
  publish atomically and retain provenance/SIF hashes. Completed artifacts are
  checksummed before reuse; interrupted publication cannot advertise readiness.
  Scratch measurement tolerates concurrent extraction cleanup on Yale Python 3.9.
- Added one-iteration full-workload A/B/C acceptance in separate fresh processes
  and study directories, retaining ordinary update behavior and training-only C
  references. Each proof requires both updates, generated episodes, the correct
  distillation/alignment path and an intact fully resumable checkpoint.
- Study freeze/readiness now require all three proofs in addition to the existing
  ten-step D continuation pilot and common image/data/backend/base evidence.
  Generated-code allocation gates remain mandatory. The combined smoke request
  reserves four GPU-hours inside the existing cumulative 24-hour allowance.
- Added the launch runbook with four preview commands and operational recovery.
  Preserved Docker archive support, offline W&B, immutable submissions, duplicate
  prevention and existing scientific settings. No local cache was deleted and no
  additional host was provisioned.

## Support dependency

Sent the authorized request from the Yale account on 2026-10-22. YCRC ticket
79403, Gmail thread `1a1640b41863817b`, awaits technical guidance. Its automated
acknowledgement does not authorize or establish worker isolation. No equivalent
worker probe was repeated; existing failures remain auditable. Follow up on the
same thread after three business days if unanswered.

## Verification

The initial complete local correctness suite passed 135 tests, with 3 workflow
exclusions. Hosted acceptance of implementation commit 6ed8a55 passed 137 tests,
3 exclusions, lint/format, mypy and Docker worker build:
https://github.com/tatsu432/self-play-opsd-coding-agent/actions/runs/39800502289

Subsequent focused image/controller/readiness checks passed 61 tests. Trainer-build
implementation commit `2d17d4d3242cffe1640742e7e23da3c944633df8` passed hosted
CI with **140 tests**, 3 exclusions, lint/format, mypy and Docker worker build:
https://github.com/tatsu432/self-play-opsd-coding-agent/actions/runs/39810403945
The final job log is retained locally at `runs/task05-ci-final.log`.

## Live result

The planned implementation is present; operational acceptance is still in progress. Main
A–D have not been submitted. The existing worker isolation failure requires a
YCRC-supported setup change before runtime validation, base evaluation, D pilot,
A/B/C acceptance, study freeze and four valid main previews can finish.

### Selected Yale CPU acceptance checkpoints

| Job | Purpose | Result |
| --- | --- | --- |
| 30267432 | First Dockerfile-to-SIF feasibility | Passed; exposed scratch sampler cleanup race, subsequently fixed |
| 30267831 | Feasibility after scratch/runtime fixes | Passed |
| 30269780 | Feasibility with preserved base environment | Passed |
| 30272639 | Feasibility with digest-pinned base Python | Passed |
| 30274881 | Feasibility with bounded Squashfs compressor | Passed |
| 30267939 | First complete trainer build | Failed: base supplies `python3`, not `python`; fixed |
| 30270336 | Complete trainer build | Failed: uv 0.8.15 cannot provision requested Python 3.12.12; fixed |
| 30272802 | Complete trainer build | Installed 231 packages; CPU packaging OOM; bounded compressor and disabled wheel cache |
| 30275129 | Complete trainer build with all fixes | Passed: published verified trainer SIF |

The compression OOM occurred in a CPU-only image build. It is not H100 CUDA OOM
and does not authorize H200 fallback. No equivalent worker probe was repeated.

### Published trainer

- Source commit: `2d17d4d3242cffe1640742e7e23da3c944633df8`.
- Build run: `27703ee5494846122fdf02d3`; Slurm job `30275129`, COMPLETED,
  exit `0:0`, 2,122 allocated seconds (about 35 minutes).
- SIF SHA-256: `fbe2737a6815644ecaad01df40cfa7883b05e5554d518fd0352129b1f6fd69e2`.
- SIF: `images/trainer/56e4d2e8a350f9a6b0b985e31562c59e85741b50bcfed4d9ac8ba94f622718bc/trainer.sif`
  under the managed personal course project workspace.
- SIF size: 11,829,288,960 bytes (11.02 GiB). Sampled peak logical scratch
  use: 60,859,672,055 bytes (56.68 GiB); retained temporary/cache bytes before
  cleanup: 12,454,911,971. Sampling is not a guarantee of true physical peak use.
- Apptainer 1.5.3-1.el9; base Python 3.12.11; verified torch 2.8.0,
  transformers 4.57.1, peft 0.17.1, verl 0.7.1, vLLM 0.11.0 and uv 0.8.15.
  All imports and `selfplay --help` passed on a real Yale CPU compute node.
- All twelve Task 05 CPU feasibility/build allocations consumed 11.6556 allocated
  CPU-hours. Their accounting is retained at `runs/task05-accounting.json`.

### GPU readiness checkpoint

The first one-H100 reference attempt, run `043a317d6e2cc705d1a8751e` /
job `30278347`, FAILED after 78 allocated seconds (0.021667 GPU-hours).
The model load reported a full 79.18-GiB device while attributing only 6.94 GiB
to this process. This does not establish genuine model memory fit failure.
The old launcher did not record/verify physical allocation UUIDs, so the exact
cause cannot be proven retrospectively. No H200 fallback was taken.

Added fail-closed Slurm/NVIDIA index validation, explicit allocated UUID selection,
PCI bus ordering and actual container CUDA UUID/initial occupancy proofs before
model load. Missing/ambiguous masks, mismatched UUIDs or more than 2 GiB initial
occupancy stop the job. Only an OOM preceded by a successful matching allocation
proof can authorize H200. This does not change the scientific workload.
See the [Slurm GPU mapping documentation](https://slurm.schedmd.com/gres.html#GPU_Management)
and [Apptainer GPU selection documentation](https://apptainer.org/docs/user/latest/gpu.html#multiple-gpus).

The revised launcher (`cf1e64454dc9d5f7c72d9fa892b2ebd65e54264a`) passed
72 focused tests and the local complete suite (139 passed, 12 local Docker skips,
3 exclusions), lint/format and mypy. Hosted result: [passed: 151 tests, 3 exclusions, lint/format, mypy and Docker worker build](https://github.com/tatsu432/self-play-opsd-coding-agent/actions/runs/39817237872).

A preparation attempt stopped during the read-only context call when SSH
expired, before submission. Normal Duo reauthentication succeeded. Revised
one-H100 job 30287970 then ran and FAILED after 14 seconds: this driver's
`nvidia-smi --query-gpu` rejects `minor_number`. The guard stopped before model
loading. Corrected the inventory to read supported CSV fields and join full XML
minor numbers by UUID, with no reliance on XML ordering. Source
`eebd4c0903a0c42619fe8b73127368113e362e2a` passed 72 focused tests and
[hosted CI: 151 passed, 3 exclusions, lint/format, mypy and Docker worker build](https://github.com/tatsu432/self-play-opsd-coding-agent/actions/runs/39826670302).
The final log is `runs/task05-ci-xml-final.log`.

Exactly one corrected reference check was subsequently submitted:

- Run: **`db32746277268abab0265c7e`**; Slurm job **30289141**.
- One H100, one-hour cap, source `eebd4c0`; last scheduler snapshot PENDING.
- Submission returned successfully; no local watcher or automatic follow-up is
  running. The job will execute allocation/occupancy checks when scheduled.
- No two-GPU backend job or main experiment has been submitted.

At submission, cumulative charged GPU cost was 0.025556 hours (78 + 14 seconds),
with 1 hour reserved for this job: 1.025556 hours committed of 24, leaving
22.974444 hours for additional reservations. Current accounting may change when
this job starts/ends. No valid H100 memory-fit failure has been established.
The last doctor snapshot had all eight production readiness gates false.

Inspect this existing job before further preparation; do not submit a
replacement merely because no watcher is running:

```bash
uv run selfplay hpc status db32746277268abab0265c7e
uv run selfplay hpc logs db32746277268abab0265c7e
uv run selfplay hpc fetch-study repair-selfplay-2026
```

Its allocation proof, once created, is
`readiness/repair-selfplay-2026/gpu-allocation-30289141.json` in managed runs.
After successful reference evidence, synchronize the clean committed checkout
and continue `hpc prepare-study --stage backend --gpu-hours-limit 24 --watch`.
If it fails, inspect its allocation proof/logs and repair the cause before
another attempt. The same run can be monitored in a later session; no new
submission is required to monitor it.

**A–D are not ready to submit.** The remaining evidence is the one-/two-GPU
backend checks, supported worker isolation and target data validation, base
evaluation, ten-step D continuation pilot, A/B/C acceptance, freeze/readiness
and four successful main previews. No main training has started. YCRC still
has only acknowledged ticket 79403; no technical setup change is available.



### Local evidence

Fetched build provenance: `runs/hpc/yale-hpc-cpsc4880/27703ee5494846122fdf02d3/trainer.json`.
Per-attempt submission/status/log records are under `runs/task05-*`; the final
CPU CI log is `runs/task05-ci-final.log`. These generated operational records
are intentionally ignored by Git; the managed Yale workspace is authoritative.


### Continue after the external blocker

Use `docs/yale-launch-runbook.md`. Keep the same cumulative readiness budget and
inspect the existing ledgers before any new submission. After a documented
supported worker setup change, run allocation-time isolation acceptance and the
remaining readiness stages. Readiness must produce a frozen study and four valid
previews before requesting main-training authorization. The main experiments
start fresh; readiness checkpoints are never their initialization.

