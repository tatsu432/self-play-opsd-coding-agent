# 03 — Experiment readiness implementation plan

Source: `docs/task/03-experiment-readiness-spec.md`.

1. Complete the controller: CPU/GPU resource validation, persisted submission
   specifications, logical output-slot locking, operation-specific artifact
   retrieval, study retrieval, dry-run launches and evidence-based readiness.
   Create the freeze record on Yale using a CPU allocation and existing outputs.
2. Build explicit AMD64 Docker archives and convert them to hashed SIFs on Yale
   CPU nodes. Check capacity, preserve archive/image/lock/commit provenance and
   publish atomically. Validate the worker before expensive trainer preparation.
3. Share a standard-library runtime supervisor between the local sandbox and
   host-side authenticated worker. Launch an allocation-scoped loopback broker,
   probe actual Apptainer isolation, and fail closed on site restrictions.
   Validate frozen data against the deployed worker into a separate output report.
4. Add resumable `hpc prepare-study`: CPU runtime validation, one-H100 reference
   (1 GPU-hour), two-H100 comparison (2), base dev-monitor evaluation (4), ten-step
   D pilot (12 total, including a stop at iteration 2 and continuation), freeze
   and four main-run dry runs. Remaining allowance: 5 GPU-hours for bounded
   retries. Lock GPU-hour reservations and reconcile actual Slurm consumption;
   uncertain accounting retains the reservation. Never exceed 24 GPU-hours.
5. Separate planned continuation from the two transient retries, use per-attempt
   stop markers, and resume the original committed payload/checksummed joint
   checkpoint. Slurm COMPLETED does not imply the scientific budget completed.
6. Bind runtime evidence to images, code, model/tokenizer, hardware, data, prompts
   and execution limits. Reject stale evidence and conflicting study freezes.
   Finish code changes before GPU probes. Main A–D remain 200 iterations/seed2026
   and fresh initialization; no main training launches in this task.
7. Test concurrency, budget accounting, interruption, artifact layouts, stale
   evidence, frozen settings and continuation. Run repository checks, perform
   the authorized live stages, and write a completion record with passed gates,
   measured costs and concrete blockers. If Yale isolation is unavailable, stop
   at that gate without provisioning or connecting to another host.
