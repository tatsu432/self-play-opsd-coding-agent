# Project and operating instructions

## Task 14 execution authorization — 2026-11-01

The user explicitly directs uninterrupted completion of Task 14, including
implementation, validation, Yale job submissions, monitoring and analysis.
GPU and CPU allocations needed for the work are authorized; do not ask again
or stop at a plan or controller implementation. Reconcile Slurm and the
ledger before each submission, preserve immutable identities and worker
isolation, and never duplicate an uncertain or completed job. The new study is
`repair-selfplay-2026-task14-v1`. CPU runtime-validation job `31618471`, run
`15932a55fc1f948cd84fcda8`, was submitted once from execution
`2c8b1615cec63f62612af3417347a8b02acb712c`; inspect it before any recovery.
Runtime validation completed exit 0. Prospective 4B diff job `31618956`, run
`276931a7445252e963f37b1b`, and structured job `31618960`, run
`36db89f113049ef37c32b79d`, were each submitted once from execution
`d24c09664c3ec1b83598302bfa3aae1161867617`. Both use the same 40
prespecified training-only parents outside the Task 13 sample. Inspect their
ledgers and live Slurm state before any action; never duplicate either job.
Both completed exit 0. On 160 samples each, diff yielded 58 verified outputs
and 22 unique parent/mutation pairs in 589 H200-seconds; structured yielded
90 and 34 in 257 H200-seconds. The paired bootstrap interval for the unique
difference was -0.025 to +0.60 mutations per parent. Results and reproduction
are in `docs/investigations/task14-informative-selfplay.md`.
Paired base-student editing probes were submitted once from execution
`36e8247937aacef114996e6348e42ac617e7e577`: unified diff job `31618995`,
run `cd5dae1d517b8b78d2535071`, and function-body job `31618998`, run
`6efc9d94e4f5e2e7b0b0fd04`. Both use the same 16 training-only static tasks
and four rollout seeds each. Inspect their ledgers; never duplicate them.
Base function-body student job `31618998` completed exit 0. Its saved
rollouts feed numerical audit job `31619683`, run `5cfc6fa822734449d11d284d`,
submitted once from `fd89d896774ec75e8a028b3890eb80a12ae9bc0c`. Inspect that
ledger before action; do not duplicate it.
The matched base diff/body student comparison completed: 7/64 versus 2/64
repairs, while 7/160 versus 81/140 edit attempts succeeded. B200 student
comparisons remain open. Base diff numerical audit `31619786` / run
`50c0b26f9c555e120db54e90` and training-only diff calibration `31619788` /
`f78ba9f3c0386e1289aafdb3` were submitted once from execution
`5ebd956ed25c8ba28a17113e35795bdbb9ac6d31`; do not duplicate either.
Diff calibration completed exit 0 and selected ungated coefficient
`0.007773662398632436` from eight training-only tasks; its `calibration.json`
SHA-256 is `058fac5830e8fb3114fe3dfc169296628833d15916a136a638b2773be66bc993`.
The frozen static control is `configs/task14-static-control.json`. Seed-2026
matched static arms were submitted once from `61cb728b540a63d26ae92f1929b2b8814a56376d`:
GRPO `31620238` / `b67448ace7296c93663241b6`, gated `31620241` /
`9cbf2d833377ebba0f78c7c9`, ungated `31620251` / `3a6083a657af90615f43cb2b`.
They evaluate at 80 and 160 attempted episodes. Inspect each ledger; never
duplicate. Review health and exposure before launching seeds 2027 and 2028.
Separately, structured online candidate study `repair-selfplay-2026-task14-online-v1`
has CPU runtime-validation job `31620333`, run `9dcb80b41bd00c7dc37ffed5`,
submitted once from worktree branch `feat/14-online-telemetry`, execution
`a3f9d0a446319e6e91bad1071ff89c360c8ec480`. No online training was submitted.
Inspect this job and ledger, never duplicate it, and require the scientific
reviews and target backend check before deciding whether online arms are useful.
The independent online-study H200 backend check was submitted once from the
same execution as job `31620340`, run `07da1772865ca4778a63d2c8`. Inspect its
ledger and do not duplicate it.
The base diff numerical audit completed exit 0: max log-probability mismatch
1.070, sign-sensitive clipping 21/5,608 eligible tokens, combined gradient
cosine 0.825 and Adam-update cosine 0.656 against a diagnostic rescored-
denominator branch. Preserve behavior probabilities. Focused RL/SD and
outlier-only replay job `31620094`, run `84f19b42553244e439b0611e`, was
submitted once from `7cc62d3c22047e9311e1e5fdd8404dae9ec1b4e8`; do not duplicate.
B200 paired student comparison completed: diff 20/64 repairs and 28/168 valid
edit actions, body 1/64 and 99/147, respectively. Diff editing was selected
for static controls using these training-only outcomes. B200 diff numerical
audit job `31619962`, run `aca754b0a57ee5f72b89e53e`, was submitted once from
`a96b3503062eab8c21559f44f6f81da51d03810f`; do not duplicate it.
B200 numerical audit completed exit 0; its eight groups were all uniform.
Targeted B200 mixed/uniform decomposition job `31620455`, run
`02ed9e64e46ef4269b5a3c7a`, was submitted once from
`668f01d46136bcb072d9d05bc65512d333554238`; it includes both mixed groups
from the saved training-only source and must not be treated as a population
estimate. Inspect the ledger and do not duplicate it.
Four student-rollout probes on the already verified novel proposal outputs
were submitted once from `fa1bf8ff5fb2ad9e6022cd3f7232ad716cb0b880`:
base diff `31619815` / `e9b9d60a8fdfa400bfb9864a`, base structured
`31619817` / `6dfde0dc399bc11f09abe923`, B200 diff `31619819` /
`08abe587ab5f63da1cdba245`, and B200 structured `31619820` /
`2b9520fbe8585cd5c453abb8`. They use diff student editing and up to 32
history-novel tasks, four rollouts each; inspect ledgers before action.
Structured base-snapshot reward counterfactual job `31619094`, run
`e74ecdee52efe104590c60b3`, was submitted once from execution
`ec28f24568c34468ff4f4fc67567d1be508e3532`; inspect its ledger and Slurm
state before action and never duplicate it.
That counterfactual completed exit 0: 12 valid tasks, six mixed student groups,
and three parent groups with multiple valid candidates. Pairwise replay job
`31620042`, run `4997792779c7970174458437`, was submitted once from
`4b1d446ffff4050e777ba0100cc9a6686b9f4c41` against the saved samples; inspect
its ledger and do not duplicate it.
The replay completed exit 0. Standardized RL-versus-combined alignment update
cosine was 0.99224, while an identical validity-reward repeat was 0.99542.
Only three multi-valid groups were observed; C/D online arms remain deferred
because their update separation is small relative to repeat variation.
The prospectively paired B200 diagnostics were submitted once from execution
`729e5ebb48b8824c0416fac2406ee824e38346d4`: proposer diff job `31619188` /
run `213649b633cd16eed4c4bb85`, proposer structured `31619189` /
`3e9f7398bbf00cb0fba4e773`, student diff `31619190` /
`63a52b186816085f0a8fe549`, and student body `31619191` /
`9fa54dd8953d44263f8696ed`. These load the checksum-verified corrected B200
snapshot for inference only. Inspect each existing ledger; never duplicate.
Job `31619188` failed because the B200 verifier contained a transcribed hash
missing the digits `307`; jobs `31619189`, `31619190` and `31619191` were
cancelled after discovery and their terminal scheduler states confirmed. The
verifier was corrected and all six expected metadata fields plus model config
were compared directly to the retained checkpoint. Preserve those attempts and
account for any allocation; replacement attempts require the corrected commit.
Corrected B200 diff proposer replacement job `31619234`, run
`d113371007fb12d958f92ccf`, was submitted once from execution
`577445bdc83fbd9be6973f4e9be5edf1240a232a`. Confirm checkpoint loading and
generated records before submitting the remaining B200 replacements.
Checkpoint verification and generation passed. The remaining B200 replacement
arms were submitted once from `56ce3fedd45bad9b291159b52935f5c1c4d47478`:
structured proposer `31619324` / `5c34adbea7983c01775ef5a8`, student diff
`31619326` / `0078cd3bbbb89eb3c5282ddd`, and student body `31619331` /
`bce7b4ed210efe8445488fbd`. Inspect these ledgers; do not duplicate.

## Task 13 CPU header replay — 2026-10-31

CPU-only worker job `31537818`, run `8878e774a2d4480b5f0eba36`, was submitted
from `c187909b9afbff4c59b4fdc250ca01087447f859` with a one-hour cap. It
replayed the checksummed 8B diff outputs after removing one exact optional
header, using the accepted isolated worker and the matching cached tokenizer.
It completed exit 0 and verified 37 of 114 affected samples. With the six
originally accepted outputs, 43/160 were valid and 30 unique under this
post-hoc normalization. The first fetch raced filesystem visibility; a later
fetch succeeded. Inspect the existing ledger; do not duplicate this job.

## Task 13 larger-model inference probes — 2026-10-31

Diff-format inference-only probes were submitted from immutable execution
`662a7b74d91dc5bab7924c5bc911736e7918ea21`: Qwen2.5-Coder-7B job
`31536071`, run `1b5cdead66a593d2b51829d9`; Qwen3-8B non-thinking job
`31536072`, run `a34502e0d48191cbef4a79e6`. Both hung before model loading and
were cancelled after 472 and 399 seconds, respectively. A live Python stack
showed vLLM waiting for a nonexistent rendezvous store because torchrun exports
`TORCHELASTIC_USE_AGENT_STORE=True`. The model-only launcher now starts Python
directly, leaving vLLM to create its own store; regression-test and synchronize
the fix before submitting replacement diff probes. Corrected diff probes were
submitted once from `7f45eaec27986057eac62643e9e55b2cab4dec31`: coder job
`31537201`, run `842b6c405b193624f33a826d`; Qwen3 job `31537246`, run
`66d86331ec30a14afb9548eb`. Inspect their existing ledgers; do not duplicate.
The 7B diff arm initialized and generated proposals; its structured arm was
submitted once from `fd53d2c69276ca1065de8d03f23f820a9e214644`, job
`31537529`, run `e57c317e45dfb110b33e8a30`. The 8B diff arm also initialized
and generated proposals; its structured arm was submitted once from
`3a1619e19c69179c035bbbe14a8823c666535be5`, job `31537580`, run
`09e709fd64972a2f91dcea51`. All four corrected arms completed exit 0 and were
compared on matched parents/seeds. Preserve the failed jobs for accounting.

## Task 13 independent extra proposer samples — 2026-10-31

Four additional slots (4–7) were submitted for the prespecified first 16
parents from execution `f49334e6d02813716019d7312ddd481032cb3f21`:
base diff job `31535931` / run `bbfefdccb9d1a67dfa486e0a`, base structured
`31535971` / `522e881e218fee881e05081c`, and B200 diff `31535981` /
`1e5f64bb179d21b06665614f`. A launcher wiring error failed to pass the extra
slot flags; B's completed 160 records exactly repeat the original base samples
and are not independent. A and D were cancelled when the error was discovered.
Slurm charged 340, 256, and 169 seconds respectively (0.2125 H200-hours total).
Preserve these invalid/partial runs for audit; do not count them as extra samples.
The launcher was corrected and has an explicit command-argument regression test.
Replacement extra-slot runs were submitted once from
`5d92eddf3299a9a0b6ad39bdbdcbca4a0910c559`: base diff job `31536424` /
run `ea1f43c3f981c35d767eb12f`, base structured `31536425` /
`174d4fb895b98bd10a866aeb`, and B200 diff `31536426` /
`346d7294a177f23792416fa7`. All three completed exit 0. Inspect these existing
ledgers; do not duplicate.

## Task 13 C/D counterfactual submitted — 2026-10-31

Job `31535924`, run `913f441b4f52675eb2479c16`, was submitted once from
`f3c290dc3d01af3af8df9b5ad926fd26b8bac850` with one H200 and a two-hour cap.
It completed with exit 0. Of 64 proposals in 16 parent groups, two candidates
in one group were accepted. Both RL candidate gradients were zero; combined
gradients were nonzero. Normalized-advantage L2 difference was 0.13213,
proposer-gradient cosine 0.95646 and Adam-update cosine 0.98893. Both adapters
and optimizers were restored and verified. This is one group, not a population
estimate. Private result: `runs/hpc/yale-hpc-cpsc4880/913f441b4f52675eb2479c16`.
Inspect the existing job and ledger before action; never duplicate it.

## Task 13 static student control submitted — 2026-10-31

Job `31535783`, run `9a44124b61147aee02c685d6`, was submitted from
`8eed8c09aa4035c7e32a9f5c11744d78b3369cc2` with one H200 and a six-hour
cap; it later timed out during the gated final evaluation as described below.
All 32 training-only tasks were
screened: 12/32 mixed over eight rollouts, versus a 27.95% mean mixed fraction
over all four-rollout subsets. GRPO initial development evaluation completed
0/41. At five student steps it had six batches, eight mixed groups in 48
episodes, no proposer steps, finite nonzero gradients, and rollout/rescoring
mismatch plus invalid-action warnings. At ten steps GRPO had 21 batches, 168
episodes, 13 mixed groups, zero proposer steps and finite recorded gradients;
20 scoring-mismatch warnings had maximum log-probability difference 2.535.
Its ten-step development evaluation completed 10/41 versus 0/41 initially;
at 15 steps it had 31 batches, 248 episodes, 19 mixed groups, finite recorded
gradients, zero proposer steps, and maximum scoring difference 3.203. GRPO
reached 20 steps after 53 batches, 424 episodes and 24 mixed groups; its final
development score was 12/41 (initial 0/41, ten-step 10/41). The matched gated
arm's initial evaluation also completed 0/41. The arms used one verified
training-only pool, the existing isolated worker and joint checkpoints. Inspect
the existing jobs and ledgers before action; do not duplicate them.
At five gated steps, five batches yielded 40 episodes, seven mixed groups,
finite recorded gradients and zero proposer steps; five scoring-mismatch
warnings had maximum difference 1.376, plus one invalid-action warning.
At ten gated steps, ten batches yielded 80 episodes, ten mixed groups, finite
recorded gradients and zero proposer steps; ten mismatch warnings had maximum
difference 1.547. Its ten-step development evaluation completed 10/41,
matching GRPO at the same step count but with 80 versus 168 episodes. The gated
arm reached 15 steps in 15 batches with 120 episodes, eleven mixed groups,
finite gradients, zero proposer steps and maximum scoring difference 1.547.
Original job `31535783` reached gated step 20 but timed out during its final
evaluation; the last complete gated joint checkpoint was iteration 19. The
uncheckpointed iteration and partial evaluation must be rolled back. After
reconciliation and a passing preview, continuation job `31555806`, run
`edb71511b52d23183f3593e5`, was submitted once from
`6df71fd4b3cd5fad8200ab23ca76041d249b80a3` and completed exit 0. It verified
the checkpoint on the allocated node, rolled back to iteration 19, replayed the
twentieth step and completed the gated final evaluation at 10/41 (0/41 initial,
10/41 at ten steps). Gated used 160 training episodes; GRPO scored 12/41 at
20 steps after 424 episodes. These are single-seed, unequal-exposure results.
The two static allocations consumed 22,852 H200-seconds (6.3478 GPU-hours).
All Task 13 diagnostic GPU jobs together consumed 30,491 H200-seconds (8.4697
GPU-hours), including 1,636 seconds from launcher mistakes. The scheduler had
no active jobs at final reconciliation; never duplicate completed jobs.

## Task 13 checkpoint probes submitted — 2026-10-31

Five checkpoint probes were submitted from immutable execution
`d1d24c9904fc2e92870ad84412904abc36ad5e88`: A20 diff job `31535565`
(`0d253ac48cd9fbac666560fe`), A200 diff `31535570`
(`ef6ce5271c24ee8de47e2221`), B200 structured `31535572`
(`0a78616509db9372715f7602`), C100 diff `31535578`
(`d406303bf5b4d2eed674cb69`), B200 diff `31535584`
(`f1b225fa1a92372f332f94a5`). All five completed with exit 0 and verified
checkpoint checksums and scientific/data/prompt/model identity on the allocated
node before loading. Inspect existing ledgers; never submit a duplicate.

## Task 13 base proposer probe submitted — 2026-10-31

Diff-format run `0cbb6df656bf96b50a81b19e`, Slurm job `31533595`, and
structured-format run `a5eea7b65ed72d2ff4b3f70d`, job `31533597`, were
submitted once from immutable execution `440451f9e75501b07f70542d96dc0c19a11a16ab`.
Both completed with exit 0. Diff used 592 seconds and produced 43/160 verified
proposals, 20 unique parent/AST pairs; structured used 242 seconds and produced
95/160 verified proposals, 33 unique pairs. Both completed all 160 samples
on the same 40 training parents and model. Private artifacts were fetched to
`runs/hpc/yale-hpc-cpsc4880/<run-id>/`. These are preliminary paired base-model
descriptive results, not checkpoint or larger-model comparisons. Inspect live
scheduler state and these ledgers before acting; never duplicate either job.
The task13 probe uses the accepted isolated worker.

## Execution persistence — 2026-10-31

The user directs the agent to continue through implementation, validation,
submission, monitoring, and scientific review of authorized experiments rather
than stopping at a plan or an unimplemented controller route. GPU time is
authorized as needed for efficient progress; do not ask again for a budget.
Treat planning caps as review points, not permission gates. If a route is missing,
implement and test it, reconcile existing scheduler and ledger state, then
submit and monitor. Preserve the scientific identity, worker isolation,
checkpoint verification, accounting, and no-duplicate safeguards below.
Older statements below that further GPU work is unauthorized or requires a new
user approval record historical decisions and are superseded for the current
Task 13 investigation. Scientific decision points still guide what to run; they
do not require asking the user to restate their authorization.

## Corrected C iteration 100 — latest live state

Job `31469208` completed with exit 0, stopped at iteration 100, and used
12,782 seconds (3.550556 H200 GPU-hours). C development scores at iterations
0/20/40/60/80/100 were 0/12/10/10/12/11 out of 41. Through iteration 100,
116/1,600 proposals were valid and 98/116 candidate alignment scores were
neutral due to zero candidate gradient. CPU checkpoint verification and W&B
reconciliation job `31528917` completed with exit 0 (29 seconds); cloud
verification reached 100, canonical revision
`07e12f64924f324c994551ebb4a8f98037da9561849c9bf2796557cc9971a5e6`.
Task 13 inspection on 2026-10-31 found no active job, new C continuation,
or corrected D main ledger/directory. Do not repeat reconciliation. The CPU
diagnosis and follow-up priorities are in
`docs/investigations/learning-stagnation-diagnosis.md` and
`docs/plan/13-diagnosis-plan.md`; raw outputs remain private under `runs/task13/`.

## Corrected C iteration 40 — latest live state

Continuation job `31455833` completed successfully at the bounded iteration 40
stop (5,063 seconds, 1.406389 H200 GPU-hours). Development evaluation scored
10/41, versus corrected A 12/41 and B 13/41 at the same iteration. The trainer
reports stopped; do not duplicate the completed job. CPU job `31468758`
verified the checkpoint checksums and identity and reconciled W&B through 40;
canonical revision is `688236f3308ee749b7e89a192165fd35fd8b2190a38cee57e9c876ea020cf5a3`.
Continuation job `31469208` was submitted from checkpoint `000040`, with
the scheduler export and ledger confirming the bounded stop at 100. It was
initially pending. Inspect this existing job before any action; do not duplicate
it. Review science and telemetry at 60, 80 and 100. The local controller watcher
was started with `hpc watch 35bfbd1bc074a8963e916f77 --interval 60` and may
retry only its supported transient failures, retaining the stop at 100.

## Standing submission authorization — 2026-10-30

The user authorizes the submissions needed to complete the corrected study,
including main GPU allocations and necessary continuations, without asking for
permission again. Start corrected Experiment A with one H200 allocation through
iteration 20; review scientific health at iterations 5, 10 and 20 and verify the
joint checkpoint before deciding on continuation. Reconcile live scheduler and
ledger state before submissions and never duplicate an uncertain job. This
authorization does not waive scientific identity, isolation, accounting,
checkpoint, failure-handling or held-out-data safeguards. Historical statements
below that main training requires a new authorization record earlier decisions
and are superseded for work needed to complete the corrected study.

Corrected study live checkpoint: A run `747bd2e097258445de647e47` started as
job `31364215`, completed iteration 20 cleanly in 1:34:28, and continued from
the controller-verified checkpoint `000020` in job `31372772`. Its development
result improved from 0/41 to 14/41; 34/320 proposals were valid and 36/136
episodes succeeded. Iteration-5 and -10 health reviews passed, and W&B cloud
history was verified through iteration 20. Fresh corrected B run
`f3271727f5fd25c07466e50d` completed its iteration-20 stop in job `31372774`:
13/41 development successes versus 0/41 initially, 30/320 valid proposals and
20/120 successful episodes. Its iteration-5, -10 and -20 health and cloud
delivery reviews were performed. Controller-verified checkpoint `000020`
continued in job `31381376`. Inspect these existing jobs and ledgers before
any action; do not duplicate them. Monitor development and telemetry for
regression. A's later development results were 12/41 at iteration 40, 11/41
at 60, 10/41 at 80, 11/41 at both 100 and 120, 10/41 at 140, and 11/41
at both 160 and 180,
following 14/41 at 20.
B was 13/41 at both 20 and 40, 12/41 at 60, 11/41 at 80, 9/41 at 100,
10/41 at both 120 and 140, 9/41 at 160, and 11/41 at 180.
A's iterations 98–117
produced only 6 valid proposals out of 320 attempts, with 131 duplicates;
monitor the declining proposal yield. A has not repeated the earlier collapse
to zero. A's W&B delivery became intermittent after iteration 142 with repeated
`OSError` warnings, while training and local scientific records continued; cloud
history later reached 181, but the dashboard can incorrectly show finished. B
also reported delivery errors; its cloud history reached at least 133. After each
trainer writer has stopped, reconcile and cloud-verify checkpoint-backed aggregates
as necessary with the single-writer recovery mechanism. These job
states are a snapshot, not live state.

Corrected A reached iteration 200 and Slurm job `31372772` completed with exit
0; its final development result was 11/41. Including initial job `31364215`,
allocated time was 29,179 seconds (8.1053 H200 GPU-hours). After its writer
exited, CPU job `31397974` verified the final checkpoint checksums and reconciled
the original W&B run; cloud verification reached iteration 200, canonical
revision `292f50c216df0a7bb0bdd680e51dc459e83d88d212d67e67f6ff46637b6f3f7a`.
Corrected B job `31381376` was running at iteration 153 at this snapshot.
An aggregate event analysis of A finds 161/3,200 valid proposals; iterations
141–200 contributed 17/960. Of 98 student optimizer steps, 82 had zero current
group-relative RL signal; earlier optimizer momentum can still contribute. All
saved A trajectory shards were subsequently fetched to the ignored private
workspace and analyzed: at iterations 20/100/200, rejected patches examined
had actual newlines (152/90/93) and none had literal backslash-n. Patch context
and hunk errors remained common. This is observational, not causal evidence.

Corrected B has also completed iteration 200: jobs `31372774` and `31381376`
both exited 0 and used 29,167 allocated seconds (8.1020 H200 GPU-hours).
Final development was 12/41 versus A's 11/41, a single-seed descriptive result.
CPU job `31404487` verified B's final checkpoint and reconciled W&B; canonical
revision `22ae1beb5cd0a9b7d9acfbc31d86f32b4a0e30edd006585ae9fdf5095889b810`
was cloud-verified through 200. B yielded 162/3,200 valid proposals and 84/102
student steps had zero current RL signal; earlier momentum can contribute.
Both complete trajectory sets are in the ignored private local workspace. C/D
have not been submitted; assess the documented alignment numerical sensitivity
and scientific value before any further method. Inspect the live scheduler.

Corrected C bounded stage was submitted after user direction and a passing
preview: controller run `35bfbd1bc074a8963e916f77`, Slurm job `31429026`, one
H200, stop at iteration 20, no automatic continuation. Inspect this existing
job and ledger before acting; never duplicate it. Review health, alignment
coverage, development results and W&B at iterations 5, 10 and 20, then verify
the joint checkpoint and decide scientifically whether to continue. The earlier
statement that C had not been submitted is historical.
At C iteration 5, the completed review found 8/80 valid proposals, 3/32
successful episodes, one mixed group, and five steps per optimizer. Seven of
eight alignment scores were neutral due to zero candidate gradients; the one
nonzero cosine was 0.000988. Cloud W&B history was verified through 5. A
rollout-scoring mismatch warning was recorded; maximum student log-probability
difference through 5 was 2.216, and its causal effect is unverified.
At iteration 10, C had 16/160 valid proposals, 11/64 successful episodes,
three mixed groups and nine steps per optimizer. Thirteen of 16 alignment
scores were neutral; the other cosines were 0.000988, -0.001611 and 0.113171.
Cloud W&B history was verified through 10. Continue the existing bounded job
to its iteration-20 stop and review; do not duplicate it.
The C job `31429026` subsequently stopped cleanly at iteration 20 and Slurm
recorded `COMPLETED` exit 0 after 6,119 seconds (1.699722 H200 GPU-hours).
Its development evaluation was 12/41, versus 14/41 A and 13/41 B at iteration
20. CPU job `31444352` was submitted once to verify checkpoint checksums and
reconcile W&B; inspect it before any further recovery. Complete the 20-iteration
scientific health and alignment review before deciding whether to continue.
CPU job `31444352` completed checksum verification and cloud reconciliation
through 20; canonical revision `904d5540957132887f6432184ef60438f803983f0bdc0e9726f9aa0efe099b04`.
The 20-iteration review found 29/320 valid proposals, 28/116 successful
episodes, six mixed groups, 15 steps per optimizer, and 23/29 neutral
alignments. The six non-neutral cosines were 0.000988, -0.001611, 0.113171,
0.063790, 0.300570 and -0.162341. There were 14 rollout-scoring mismatch
warnings; the maximum sampled student log-probability difference was 2.317.
The stopped checkpoint is fully resumable with matching corrected identity. No
continuation or D job was submitted. Inspect live state before future action.
The user subsequently directed continued investigation. Controller commit
`e194101` added an explicit clean-continuation stop that persists across
transient retries; local tests and hosted CI `41132705970` passed. C continuation
job `31455833` resumed verified checkpoint `000020` with an explicit stop at
iteration 40; the scheduler export and ledger both confirm 40. Inspect this
existing job before any action and never duplicate it. Review the complete
iteration-40 result and alignment health, then verify its joint checkpoint.
The standard local `hpc watch` was started for this run (local process 73899,
2026-10-30) without `--continue-stopped`: it monitors and may retry only
documented transient scheduler failures, preserving the iteration-40 stop.
It does not perform the scientific review; check whether the watcher remains
alive after any interruption and inspect the ledger and Slurm before acting.
The resumed segment reached iteration 25: iterations 21–25 had 5/80 valid
proposals, 7/20 successful episodes and five alignment scores, four neutral;
the non-neutral cosine was -0.040464. Both optimizers stepped three times.
These small-window observations are provisional; continue to the bounded 40.
Through iteration 30, the resumed 21–30 window had 9/160 valid proposals,
15/36 successful episodes, seven steps per optimizer and eight neutral scores
of nine. The job remained active with stop 40 after this review.
Through iteration 35, iterations 21–35 had 15/240 valid proposals, 27/60
successful episodes, eleven steps per optimizer and fourteen neutral alignment
scores of fifteen. The existing continuation remains bounded to 40.

This repository studies controlled Python repair self-play with on-policy self
distillation (OPSD), GRPO and verifiable rewards. The authoritative scientific
specification is `docs/task/01-experiment-implementation-spec.md`; early proposal
documents are historical. Read `README.md`, `docs/methods.md` and
`docs/operations.md` before changing experiments. Related papers are in
`docs/papers`.

## Engineering invariants

- Use the pinned `uv.lock`, digest-pinned Docker images for reproducibility, and
  Apptainer for the trainer on Yale. W&B is offline by default; never put keys in
  YAML, Git, scheduler arguments or logs.
- Keep science, hardware, environment and logging settings separate. Changes to
  method, data, model, prompts or scientific code require new compatibility
  evidence and, after freeze, a new study ID. Never silently modify A–D
  comparisons or move held-out data into training.
- Checkpoints contain both adapters, optimizers, RNG and curriculum state.
  Resume only a checksummed project checkpoint with matching scientific/data/
  prompt identity. A failed partial optimizer step is never resumed directly.
- Keep `README.md` short and link detailed material from `docs/`. For each task
  spec under `docs/task/NN-...-spec.md`, create the matching plan under
  `docs/plan/NN-...-plan.md` and a completion record under
  `docs/task_completion/NN-implemented-....md` after implementation.
- Keep implementation modular and DRY. Fix code errors and run relevant tests
  before declaring readiness. CPU tests do not establish GPU backend, memory fit
  or generated-code isolation on Yale.

## Yale operations

**Authorized initial A run:** the user explicitly authorized one H200 allocation
up to 23 GPU-hours, capped at iteration 20, with no automatic continuation and no
B/C/D. Controller run `a87f3772b284d35a375cf69e`, Slurm job `31219895`, was
submitted with frozen execution `54f087fdc5cf5170800d825bf1ee92b89a43c8ed`.
Inspect this existing run and scheduler state before any action; never duplicate
it. Review live W&B and scientific health at iterations 5, 10 and 20. A clean
checkpoint at 20 requires a new authorization for continuation.


Task 10 online successor is `repair-selfplay-2026-main-online-v1`, settings
`configs/task10-online.yaml`, frozen execution
`54f087fdc5cf5170800d825bf1ee92b89a43c8ed` (CPU freeze 31206522).
Frozen-source CI 40784025848 passed 272 tests. Allocated-node W&B cloud delivery
and same-run cross-process resume passed in CPU job 31208617; authorized entity
is `tatsu432s`, username `tatsu432`. All four actual online main previews passed;
no main training job was submitted, and main training remains unauthorized.
Use `docs/task10-launch.md` and the Task 10 completion record. Credentials remain
outside Git in private files. Login uses isolated W&B 0.22.3 for new-format keys;
the frozen trainer SDK/image/lock remain unchanged and online delivery is proven.
The user accepted the measured proposer numerical sensitivity for a bounded
initial A experiment. Main GPU-hour authorization is still required. Student total
counterfactual gradient cosines were about 0.9986, but the actual-D proposer
counterfactual Adam-update cosine was 0.58968. No estimator/science correction
was adopted. Replays 31208618 and 31212941 completed; the shared locked ledger
is reconciled at **7.79/24 allocated GPU-hours**, zero reservation, no active
Slurm job/watcher. Do not repeat probes to seek a different outcome or treat
controller eligibility as scientific risk acceptance. Task 09 and Task 08
freezes below remain separate preserved baselines.

Task 09 v2 is launch ready with caveats on `feat/09-efficiency-learnability`,
`configs/task09-diagnostics.yaml`, study `repair-selfplay-2026-efficiency-v2`.
Freeze 31197007 and all four actual previews passed; frozen execution is
`93a8533beeae26e54bb1510caa12991fb2471337`. CI 40748915396 passed. Main A–D
training remains unauthorized. The Task 08 frozen study is a separate valid
baseline. See the Task 09 completion record and launch runbook for evidence,
numerical caveats, costs and exact commands. Reconcile live jobs and budget
before operations; last reconciled consumption was 7.478055556/24 GPU-hours,
with no outstanding reservation.

Task 09 freeze 31149887 is incomplete for launch planning: it contains no
resource plan, and controller previews use a 484-second stop lead and one
continuation. The controller reports eligibility, but do not submit this study
until a workload-based resource plan is bound to a correctly frozen study and
the remaining performance validation is complete. Preserve the freeze for audit.

The follow-on mismatch instrumentation changed scientific code identity and
passed its own dependency-scoped validation. Its frozen measured plan uses stop
leads 4616/4540/4717/5211 seconds and 5/5/5/6 continuations for A/B/C/D.
The earlier v1 freeze above remains unsuitable for launch.

The last accepted readiness task is `docs/task/08-task-learnability-probe-spec.md`; its plan is
`docs/plan/08-task-learnability-probe-plan.md`. Task 08 authorizes the common
patch-format clarification, complete protocol hashing, a new study/data revision,
and bounded acceptance windows. Task 07's worker/resource rules remain in force
where Task 08 does not supersede them. Main training remains unauthorized. The cumulative
cpsc4880 readiness allowance remains 24 allocated GPU-hours, including prior
attempts and outstanding reservations. Main A–D training and new infrastructure
remain unauthorized.

Worker safety is outcome-based: bound CPU, memory, process/thread creation,
temporary storage, output and time; isolate trainer/oracles/credentials, network
and GPUs; verify cleanup. Investigate kernel-enforced single-process execution
before subordinate UIDs or a service. Nested user cgroups, pids.max and exactly
32 processes are not requirements when equivalent protection is proven. Never
execute generated candidates before real Bouchet containment acceptance.
Trusted CPU fixtures and backend checks remain independent of worker readiness.
Reuse evidence by relevant dependencies, preserving provenance. Independent
operations must not share a submission slot merely because their report parent
is the same. Prefer H100, but entitled H200 is authorized for a documented memory,
scheduling or runtime reason without first manufacturing an H100 OOM. Preserve
GPU ownership and initial-occupancy checks. Preserve all scientific settings.

The local controller is
`uv run selfplay hpc --help`. Its standard-library remote endpoint is
`scripts/hpc_remote.py`; `scripts/yale.slurm` runs allocated jobs. The default
alias is `yale-hpc-cpsc4880`. The `cpsc4710` and `sds6190` aliases are inspection
only until each account's authorization, course policy and shared limits are
verified. All three reach Bouchet, so never assume independent GPU pools.

Task 03 authorizes CPU setup and a total of 24 allocated GPU-hours for readiness
on `cpsc4880`, starting with H100. It ends before main A–D training. The old
AWS hosts are unavailable: use Yale only and stop if its required worker
isolation cannot be enforced. Use `hpc images build --kind worker`, then
`hpc prepare-study --watch`. The preparation command identifies the next missing
image or failed evidence gate. Never treat a passed CPU test as target readiness.

1. For Task 08, start with `uv run selfplay hpc doctor --settings configs/task08.yaml --gpu-type h200` and `uv run selfplay hpc inspect`.
   The commands are read-only and do not allocate resources. Inspect the live
   account/QOS associations, partitions, GRES, queue, quotas and software.
   YCRC web documentation is guidance; Slurm is the live authority. If SSH says
   keyboard-interactive authentication is needed, run `uv run selfplay hpc login`
   in a terminal and complete the normal MFA prompt. It opens a two-hour idle
   SSH control connection for noninteractive controller calls. Never work around
   Yale authentication or guess authorization.
2. Before a run, require a clean committed Git tree and prepared, validated
   data. `uv run selfplay hpc sync` sends a Git bundle to a commit-specific
   checkout in project storage and copies the frozen dataset to a manifest-hash
   directory. Durable runs/checkpoints live in the user's personal directory
   inside course project storage; model caches live in the corresponding
   personal course scratch directory. The controller rejects a missing,
   unwritable or ambiguous storage directory.
3. A submission needs the actual account, partition, typed GPU resource,
   profile, SIF path and SHA-256. The controller checks live associations and
   node fit; `scripts/yale.slurm` hashes the SIF on the allocated compute node
   before invoking the image. Do not run model, image-hash or sandbox work on
   login nodes. Use `backend-check` before pilot, then base evaluation and a
   measured D pilot before `freeze-study` and main A–D runs.
   Use `hpc resources --profile ...` to validate a proposed request without an
   allocation. Choose GPUs using measured pilot memory, phase time and allocated
   GPU-hours, not a transient `sinfo` snapshot.
4. The current Yale worker is an authenticated loopback broker on the allocated
   compute host, inside an independently limited no-GRES Slurm step. Fresh
   Apptainer payloads install the single-process seccomp policy before candidate
   compilation/imports. No external worker host or SSH tunnel is needed. Require
   real containment, OOM recovery and scheduler-termination acceptance. Keep any
   optional persistent credentials in a user-owned mode-600 file outside Git.
5. Use `hpc status`, `hpc logs` and `hpc fetch` to inspect a submitted run.
   `hpc watch` can resume a previously authorized training run after a confirmed
   PREEMPTED, NODE_FAIL or BOOT_FAIL state, at most twice, from the latest
   verified joint checkpoint. It reconciles the durable ledger with Slurm first.
   If a submission response or scheduler state is uncertain, stop and inspect;
   never create a duplicate. Scientific failures, OOM, authentication, quota
   and unauthorized resource requests are not automatic retries.

## Task 08 verified launch checkpoint — 2026-10-28

Launch readiness is **true**: all ten gates, immutable freeze 31082462 and four
actual A–D controller previews passed. **No main job was submitted; main training
remains unauthorized.** See `docs/task_completion/08-implemented-task-learnability-probe.md`
and the launch runbook for exact reports, hashes, costs and commands. Do not repeat
completed probes/pilots to obtain another stochastic outcome.

Use `configs/task08.yaml`, study `repair-selfplay-2026-protocol-v2`, immutable
`data/protocol-v2`, one H200 (`single_gpu_80gb`), seven trainer CPUs/128 GiB plus
one worker CPU/1 GiB. Seed 2026, 200 main iterations and all mathematical budgets,
objectives, rewards and sampling are unchanged. Main payloads pin frozen execution
`983a6cf57ce5caf5c9aa7ddf53f5ad7f313b4e73`; controller metadata fix `25b07f2`
reports the correct 200-iteration budget. Final settings checkpoint every iteration.
Frozen-source CI 40639028520 passed 242 tests; controller CI 40640124480 passed
244 tests (one skipped, three deselected), including Linux/Docker acceptance.

Prompt hash: `07968cec687ca52d8ce02c4a6d52eaca529f830855690cab1caff71c096f4322`.
Manifest file SHA: `934412b2181ead4377d383130a19ad0d58eaab6bb9f74c5d8005416d09412fd2`;
canonical manifest: `d9140367e20587051b328dda20a7d93916ee5cadc7e72c2564523cf6ab6beade`.
Original parents/tasks/oracles/splits/coverage are byte-identical; all 306 revised
admissions/target validations passed. `data/prepared` remains intact.
`configs/task08-source.yaml` is historical probe/revision setup, not main settings.

Worker 31033392 (19 checks) and both SIFs were reused with provenance. Probe
31069728 passed its fixed 32 proposals / five valid tasks / 20 episodes. Revision
31069732, runtime 31071377 and backend 31071410 passed. Corrected **greedy** base
31072777 completed 0/41; the older 0/41 remains historical sampled evaluation.
D jobs 31075857/31077825 completed ten iterations with verified resume at two,
18 valid tasks, 72 episodes, ten steps per optimizer and 18 alignment scores.
A/B/C 31080094/31080096/31080097 each completed one three-iteration window,
six valid tasks, 24 episodes and three actual steps per optimizer. C scored six
alignments. Fixtures are labelled, state-restored and separate from real yield.

Shared authorization `cpsc4880-readiness-24h-v1`, ledger
`state/budget-readiness-cpsc4880.json`: 3.234722222/24 GPU-hours consumed,
zero outstanding reservation, 31 reconciled records. Shared pilot use is
1.025000/12 hours. No active job or HPC watcher remains. Reconcile live state
before future operations; new study IDs never reset either allowance.

Whole-device measured peak was about 71.81 GiB of 140.40 GiB; host phase RSS
reached 28.45 GiB across actor/descendants. Samples are lower-bound peaks and
include rollout startup. Conditional total main training+monitoring cost is
70.24 hours at observed yield, 304.85 hours with all proposals valid at
observed lengths, or 329.45 planning hours. These are scenarios, not
future guarantees. Final evaluations/diagnostics are additional. Segments are
23 hours with measured stop lead 4,503–4,874 s and 5/5/6/5 clean continuations.
No representation/model/sampling/reward/budget change is authorized by readiness.
Initial probe/base B policy-cache files were archived outside the main namespace;
all four main directories and submission ledgers are absent. Their scientific
records and the freeze remain unchanged; see the closeout record for provenance.

## Task 07 historical checkpoint — 2026-10-27

This section records the 2026-10-27 snapshot. Task 08 above supersedes its
readiness status and pending shared-format authorization.

Execution code `9980c78` passed hosted CI 40617600830 (220 tests). Real worker
acceptance 31033392 passed all 19 checks; runtime 31045342 validated all 306 tasks;
one-H200 backend 31040984 passed numerical/adapter/vLLM/checkpoint acceptance.
Base evaluation 31046386 completed 0/41 repairs. The selected request is one H200,
seven trainer CPUs/128 GiB plus one worker CPU/1 GiB (eight CPUs/129 GiB total).

D jobs 31049338/31050754 completed ten iterations with verified resume at two,
but 0/160 proposals were valid and no episodes, updates or alignment scores were
produced. Independent A/B/C jobs 31054795/31055413/31056122 each committed one
iteration and a joint checkpoint, but also had zero yield/updates and failed
acceptance. **Launch readiness is false.** No freeze or successful main preview
exists; all four dry runs refused eligibility. No main job was submitted.

Cumulative consumption is 1.091111111 of 24 GPU-hours, with no outstanding
reservation, active job or watcher. Do not repeat the unchanged zero-yield pilot
or reinterpret empty-workload timings as training throughput. A concrete shared
format clarification is in `docs/task/07-shared-diff-format-decision.md`; user
authorization is pending because Task 07 preserves prompts. No prompt/parser,
sampling, reward, model or frozen-data change has been applied. A new study
identity must not reset the existing cumulative allowance. Read
`docs/task_completion/07-implemented-yale-execution-blocked-readiness.md` and the
launch runbook for exact evidence and the remaining scientific decision.

## Historical checkpoints (superseded operational rules; inspect live state)

The current Task 08 instructions and retained Task 07 worker/resource rules take
precedence over mechanism and hardware restrictions below. Job states and
accounting here are historical snapshots.

## Task 03 historical checkpoint (2026-10-22)

- Normal Duo MFA established a control session for `yale-hpc-cpsc4880`.
  `cpsc4880_ts2236` has Slurm account `cpsc4880`, with `interactive,normal`
  QOS and a blank partition restriction. Read-only resource checks accepted
  the intended H100, H200 and RTX Pro 6000 Blackwell two-GPU requests. The
  scheduler still decides when a job can start.
- The writable personal course project and scratch directories are available.
  Group quota at the Oct 21 snapshot was 979/4096 GiB project and 515/10240
  GiB scratch. The user had no active jobs; two jobs from other `cpsc4880`
  users were running. The implementation commit `213647a` and validated data
  manifest `44fe6caffc05...` were synchronized to the user's project space.
- Task 03 built and converted a Linux AMD64 worker SIF successfully (job
  30217522; SHA-256 `76a43eef3424...`). Worker probes 30217591 and 30218129
  failed before candidate startup because Apptainer could not apply cgroup
  limits. The second node had cgroups v2 but no `/run/user/35094/bus`; normal
  user-bus discovery could not supply `DBUS_SESSION_BUS_ADDRESS`. This requires
  a YCRC-supported execution setup; do not remove limits or probes to proceed.
- Readiness used zero GPU-hours (approximately 0.148 allocated CPU-hours).
  No trainer SIF, GPU backend proof, base evaluation, pilot or frozen study has
  been established. Main A–D have not started. Local free storage was about
  22 GiB, below the 60 GiB trainer-build preflight minimum.
- `cpsc4710` and `sds6190` noninteractive access was rejected by MFA. A Duo
  prompt for `cpsc4710` timed out, so their permissions and quotas remain unknown.

Read `docs/task_completion/03-implemented-experiment-readiness.md` for the exact
state, fetched diagnostics, remaining gates and a YCRC support-request draft.
Re-check live state before new work. Trusted `backend-check` and
`hpc prepare-study --stage backend` do not require a worker or target execution
validation. Task 05 authorizes bounded GPU readiness after CPU/hosted CI. Backend evidence
uses `compatibility/backend-v2`; legacy evidence cannot satisfy the new contract.
Resolve the supported worker execution setup before generated-code work.
Trainer exports support `hpc images build --build-root PATH --builder NAME`;
the Docker builder also needs sufficient cache storage. Do not provision
another host. Task 05 explicitly authorizes contacting support.
The support request and proposed Yale managed-service design are in
`docs/yale-worker-support.md`.

## Task 05 checkpoint (2026-10-22)

- Yale-native Dockerfile-derived trainer build is verified. CPU job 30275129
  published an 11.02-GiB SIF, SHA-256
  `fbe2737a6815644ecaad01df40cfa7883b05e5554d518fd0352129b1f6fd69e2`,
  from source `2d17d4d`. Base Python 3.12.11, pinned packages/imports and CLI
  passed. Build mode is `hpc images build --kind trainer --build-mode yale`;
  no large local archive is required. CPU attempts consumed 11.6556 CPU-hours.
- The first H100 reference job 30278347 failed after 78 seconds with a nearly
  full device but only about 7 GiB attributed to this process. Allocation mapping
  was not proven, so this does not authorize H200. Source `cf1e644` now resolves
  Slurm GPU UUIDs, verifies actual container UUIDs/initial occupancy and rejects
  ambiguous or occupied devices. Its hosted CI passed 151 tests. Live acceptance
  of this correction is pending.
- The first reference attempt cost 0.021667 GPU-hours of the authorized 24. No main
  A–D runs have started. All production readiness gates remain false. Normal Duo
  reauthentication succeeded. Job 30287970 failed after 14 seconds because
  NVIDIA's CSV query rejects `minor_number`; source `eebd4c0` now reads/join XML
  minor numbers by UUID and passed hosted CI (151 tests). Total charged readiness
  cost is now 0.025556 GPU-hours (92 seconds).
- Corrected one-H100 reference job 30289141 / run `db32746277268abab0265c7e`
  was submitted from `eebd4c0`, last seen PENDING. It has a one-hour cap and
  reserves one GPU-hour. There is no watcher or automatic follow-up. Inspect this
  existing job before further preparation; do not duplicate it. After it passes,
  synchronize the clean committed checkout and resume backend-only preparation.
  If it fails, inspect/repair before another attempt. No two-GPU job is submitted.
- YCRC ticket 79403 / Gmail thread `1a1640b41863817b` contains the authorized
  request. Only an automated acknowledgement is available. No equivalent worker
  probe was repeated; require a documented supported setup change before another.
  Follow up in that thread after three business days (2026-10-27) if unanswered.

Read `docs/task_completion/05-implemented-yale-experiment-readiness.md` and
`docs/yale-launch-runbook.md` for exact artifacts, CI evidence, remaining gates
and continuation commands. The Task 03 absence of a trainer and zero GPU cost
above are historical and have been superseded by this checkpoint.

## Task 06 acceptance in progress

The user requires actual submission readiness, not implementation completion.
Job 30289141 failed before model load after 16 seconds; reconciled cumulative
readiness consumption is 0.03 GPU-hours. No H100 memory-fit failure is proven.
The old pending-job description above is superseded. No equivalent worker
acceptance probe may run until a documented supported setup change.

Task 06 adds `hpc worker-diagnose` (CPU only) and five-minute
`hpc allocation-check --profile single_gpu_80gb|dual_gpu_80gb` operations.
GPU ownership must be proven through the actual Slurm device cgroup and container
UUIDs; NVIDIA index/minor equality is not required. Diagnostics cannot establish
backend compatibility or worker safety. Runtime evidence includes the allocation
code and documented worker setup identity. See the launch runbook for commands.

Continue through all eight gates and four valid previews before recording Task 06
as complete. Existing approved Yale services are the only worker fallback;
new infrastructure, main training and exceeding 24 GPU-hours remain unauthorized.

Task 06 live checkpoint: commit `8bb364b` passed hosted CI (169 tests).
CPU diagnostic 30293989 proved missing user bus, non-writable cgroup controls and
no `pids` delegation below slurmstepd.scope; YCRC updates are recorded in the
support document. Normal-queue diagnostic 30295248 was cancelled before start
(zero GPU time). Corrected diagnostic 30296835 ran for 20 seconds and proved
kernel/container UUID ownership, then rejected an already occupied GPU on
`a1127u05n01` (VLLM worker PID 572771, 73,814 MiB). Reconciled GPU cost is
0.035556 hours. No model was loaded and no H100 memory-fit failure is established.
No controller is currently watching that failed run. The next code revision
adds an optional `scavenge_gpu` route for stateless trusted diagnostics only,
explicit node exclusions and `--no-requeue`; verify its CI before GPU use.
Continue with `hpc prepare-study --stage backend --diagnostic-partition scavenge_gpu
--exclude-node a1127u05n01 --gpu-hours-limit 24 --watch`. Inspect existing ledgers
first. Main jobs retain normal partitions and remain unauthorized.

Task 06 update 2026-10-27: YCRC replied to ticket 79403 (thread
`1a1640b41863817b`, message `1a17e27e17b5f49f`). Users cannot configure cgroups
on Bouchet; YCRC recommends Slurm job arrays/steps for per-task CPU/memory and
Apptainer for filesystem/network isolation. Slurm has no process-count limit,
so this guidance alone does not satisfy the existing per-worker 32-process
contract. Do not execute candidate code yet. A concise follow-up asks whether
Apptainer subordinate UID/GID mappings are available and allowed, to evaluate a
unique mapped UID plus kernel-enforced `RLIMIT_NPROC=32` inside a no-GRES Slurm
step. See `docs/yale-worker-support.md` and the updated Task 06 plan. This path
requires adversarial escape/cleanup checks and real Bouchet acceptance; it is
only a proposal.

Normal SSH now requires interactive Duo MFA. The previous `scavenge_gpu`
watcher exited when its SSH control connection failed; job 30299146 has not
been reconciled since then. Reauthenticate with `uv run selfplay hpc login`,
inspect that job and accounting, then resume monitoring without duplicating it.
The existing 24 GPU-hour cap and H100-first/OOM-only H200 rule still apply.
The user has not answered the separate H200 exception question. No main A–D
jobs have been submitted; Task 06 remains incomplete.

## Task 10 initial main A stage — 2026-10-28

The user authorized one bounded Experiment A H200 allocation through iteration 20. Job 31219895 completed successfully, stopped at iteration 20 and consumed 1.527778 allocated GPU-hours; no continuation or B–D job was submitted. The latest checkpoint is `000020`, with complete manifest and resumable metadata; the controller must verify file checksums and identity before any future resume. W&B run: https://wandb.ai/tatsu432s/self-play-opsd-coding-agent/runs/9c125e660c942541076d5d50. Valid proposals 30/320, successful solver episodes 40/120, development evaluation 12/41 after training versus 0/41 initially. See `docs/task_completion/10-implemented-main-experiment.md` for scientific health, limits and accounting. Further main allocation requires authorization; do not automatically continue.

## Task 10 Experiment A completed — 2026-10-29

The user authorized continuation to the frozen 200-iteration target. Initial job 31219895 and continuation 31232768 used 7.7025 allocated H200 GPU-hours in total. Slurm recorded the continuation `COMPLETED` (exit 0); trainer health is `complete` at iteration 200 and the latest checkpoint pointer is `000200`. No B–D jobs have been submitted. Scientific events report 173/3,200 valid proposals, 162/692 successful episodes, and 5/173 mixed-outcome groups. The 41-task development success was 12 at iterations 20 and 40, then declined to 0 at iteration 100 and remained 0 through iteration 200. Final held-out evaluations have not been run. W&B delivery failed after iteration 139 (86 recorded delivery errors); the cloud dashboard is stale, while local events and checkpoints are preserved. A full checkpoint-inclusive fetch/reconciliation remains outstanding; see the Task 10 completion record and `docs/task10-launch.md`. Do not interpret cloud finished status at 139 as a training stop. Do not submit B–D or new training without authorization.

## Task 14 B200 downstream rollout check — 2026-11-01

Structured student-rollout job `31619820`, run `2b9520fbe8585cd5c453abb8`, completed exit 0 after 1,164 seconds. It produced four mixed-reward groups in 28 structured history-novel tasks (112 episodes), versus zero in two diff tasks (eight episodes, job `31619819`). Including proposal generation, allocations were 1,400 versus 906 H200-seconds. Only one parent was shared; this is descriptive, not a reliable matched-parent comparison. Do not duplicate these jobs. Base structured rollout job `31619817` remains active at this snapshot.

## Task 14 bounded structured online A/B submitted — 2026-11-01

The online protocol has a separate immutable study `repair-selfplay-2026-task14-online-v1`, execution `a3f9d0a446319e6e91bad1071ff89c360c8ec480` on branch `feat/14-online-telemetry`, structured proposer and unified-diff student. CPU runtime validation `31620333` completed exit 0, and H200 backend check `31620340` completed exit 0; both actual A/B previews passed. Existing matched parent schedule is keyed by seed and iteration. The B200 structured proposal probe yielded 28 history-novel verified tasks and four mixed student groups; comparison and numerical caveats remain under review. Bounded 20-iteration A job `31620834`, run `0c1ad526d5433313c6e212ca`, and B job `31620835`, run `fcf4a2f847f261abfd316f37`, were each submitted once, with W&B online and six-hour caps. Inspect the ledgers and scheduler; never duplicate. Review initial adapter hashes, iteration health and diagnostic metrics, including numerical sensitivity, and verify checkpoints and cloud delivery after writers stop. Do not submit C/D without evidence of meaningful optimizer differences. Static controls and numerical decompositions remain active independently.

## Task 14 base numerical decomposition — 2026-11-01

Job `31620094`, run `84f19b42553244e439b0611e`, completed exit 0 in 1,366 H200-seconds. On 32 saved base-diff episodes, replacing behavior log probabilities only at the 84/17,180 tokens with absolute mismatch above 0.2 yielded combined gradient/Adam-update cosines 0.8250/0.6560 versus true behavior, close to replacing every denominator (0.8254/0.6551). RL-only cosines were 0.8195/0.6529 and SD-only 0.9311/0.8058 for outliers. These diagnostic changes also alter the SD gap, so do not infer a proven root cause or change true behavior-policy logging. Targeted B200 replay `31620455` remains active. Do not duplicate this job.

## Task 14 base downstream rollout check — 2026-11-01

Structured job `31619817`, run `6dfde0dc399bc11f09abe923`, completed exit 0 in 2,231 H200-seconds. Both base formats produced seven mixed groups: diff 7/22 tasks, structured 7/32 capped tasks. Including proposer allocations, mixed-group yields were 15.0 versus 10.1 per H200-hour. Across 12 shared parents the structured-minus-diff parent-macro difference was 0.00 (bootstrap 95% interval −0.33 to +0.33). Candidate tasks differ and structured was capped at 32, so do not claim proof of a general mixed-group increase. Do not duplicate this job.

## Task 14 additional matched static seeds submitted — 2026-11-01

After all three seed-2026 arms entered training with finite gradients and mixed groups, the remaining prespecified seeds were submitted once using the same immutable `61cb728b540a63d26ae92f1929b2b8814a56376d` execution, frozen pool, schedule and calibration. Seed 2027: GRPO job `31621256` / run `5cdc4f1ee35e243d0fc63a75`, gated `31621258` / `7fceb25129c8c7fcaee6a52e`, ungated `31621259` / `9eec00179b5e0c745c4d1a35`. Seed 2028: GRPO `31621260` / `d38b3612ed8b4eb508aa7f40`, gated `31621263` / `a9694619af3f317d27080314`, ungated `31621266` / `9b02fd0f80661f5672631b42`. Each is capped at six hours, with 80/160-episode evaluations. Inspect the existing ledgers and scheduler, verify matching initial adapters within each seed and do not duplicate them. They were queued at submission because six GPU jobs were active.

## Task 14 targeted B200 numerical decomposition — 2026-11-01

Job `31620455`, run `02ed9e64e46ef4269b5a3c7a`, completed exit 0 after 835 H200-seconds. It used two available mixed and six uniform groups: six of 2,854 eligible tokens were sign-sensitive clipped; 26/8,870 had absolute mismatch over 0.2. The outlier-only combined gradient and Adam-update cosines were 0.9304 and 0.9969, RL-only 0.9402/0.9971, and distillation-only 0.9454/0.99995. Results depend on checkpoint optimizer state and do not negate base-initialization sensitivity. Do not duplicate this job.

## Task 14 matched control interim at 80 episodes — 2026-11-01

All three seed-2026 static arms completed their 80-episode development evaluation: GRPO 12/41 with seven student steps, gated OPSD 10/41 with ten steps, and matched-strength ungated OPSD 9/41 with ten steps. Initial scores were 0/41; all had zero proposer steps. These are single-seed interim observations, not a final comparative finding. All three jobs continue to 160 attempted episodes; the two extra seeds are submitted and must be monitored, never duplicated.

## Task 14 online initial health review — 2026-11-01

Online A/B jobs `31620834`/`31620835` both initialized student and proposer adapters to SHA `1e30ec499be6c996f0d4889c6bf74490e3fadc8f9b9f850b3eebda35f4c4e55d`, authenticated W&B, and completed six iterations at this snapshot. A had 20/96 valid proposals, 80 student episodes, six student/proposer steps, six parent groups with multiple valid proposals and three with distinct valid rewards. B had 17/96 valid, 68 episodes, six steps per optimizer, three multi-valid groups and two distinct-reward groups. Both emitted health warnings, including the already-investigated rollout-scoring mismatch; inspect their full records and verify gradient/checkpoint/cloud health before final conclusions. Continue the existing 20-iteration jobs; do not duplicate.

## Task 14 seed-2026 matched static completion — 2026-11-01

All three seed-2026 jobs completed exit 0 with 160 attempted episodes and complete final evaluations. GRPO `31620238`: 11/41, eight student steps, 6,565 H200-seconds. Gated `31620241`: 10/41, 20 steps, 6,808 seconds. Ungated `31620251`: 8/41, 20 steps, 6,135 seconds. Baselines were 0/41; 80-episode scores were 12/10/9. Gated-minus-GRPO parent-macro difference at 160 was −0.0435 (parent bootstrap 95% [−0.1304, 0]); ungated-minus-GRPO −0.0870 ([−0.2174, 0]). One-seed task-parent intervals are not between-seed uncertainty. Private reproducible manifest/results are under `runs/task14/static-seed2026*`. All three seed-2027 jobs were running and seed-2028 jobs queued at this snapshot. Never duplicate.

## Task 14 online iteration 10 — 2026-11-01

Online A and B completed iteration 10 and began their 41-task development evaluations. A had 32/160 verified proposals, 128 student episodes, seven mixed student groups, nine parent groups with multiple valid candidates and three with distinct valid rewards. B had 29/160, 116 episodes, seven mixed groups, six multi-valid parent groups and two with distinct rewards. Both had ten student and ten proposer optimizer steps. Matched parent schedules and initial adapters are verified; student exposure differs. Evaluations and bounded jobs continue; inspect before action and do not duplicate.
