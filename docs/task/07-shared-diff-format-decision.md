# Pending scientific decision: clarify the shared patch format

Status: proposal only; the frozen prompts and parser have not been changed.

Task 07's D pilot completed ten iterations with a verified stop/resume at two.
All 160 proposals were invalid, producing zero student episodes, optimizer
updates or alignment scores. Of these, 154 failed the required-header check;
the remainder had JSON or hunk-count errors. Base evaluation completed 41 tasks
with zero successful repairs. These are real model outputs, not synthetic
performance fixtures. Runtime, worker and numerical backend acceptance passed.

The smallest proposed change is a shared format reference in the existing
student tool description and proposer output description. Keep the literal
student/proposer/teacher system prompts, four tool schemas, strict parser,
model/tokenizer, objectives, rewards, seed, splits, mathematical budgets and
invalid-proposal handling unchanged. Every A–D method receives the same new
format reference; no malformed proposal is repaired, replaced or resampled.

Proposed exact reference text:

> A patch must be a single-file unified diff. Begin with `--- a/src/solution.py`
> and `+++ b/src/solution.py`. Every context line begins with one space, every
> removed line with `-`, and every added line with `+`. Hunk line counts and
> context must match the actual source. End every patch line with a newline.
> Omit `diff --git`, `index`, markdown fences and explanatory prose. The following
> example illustrates syntax only; use the actual supplied source and task.

Formatting example, using a synthetic function unrelated to an evaluation task:

```diff
--- a/src/solution.py
+++ b/src/solution.py
@@ -1,2 +1,2 @@
 def f(x):
-    return x + 1
+    return x - 1
```

The patch is still the value of the existing JSON `patch` field. Its newlines
must be JSON-escaped; the existing student `action` field is unchanged.

This changes model conditioning and therefore requires user authorization under
Task 07's explicit prompt-preservation requirement. If authorized, record a new
common prompt hash and study identity, preserve the current failed pilot and
baseline, regenerate only prompt-dependent admission/anchor evidence while
retaining every parent/task/split and fixed token budget, and repeat compatibility
and real readiness checks. Link the new identity to the existing readiness
authorization ledger so cumulative consumption still has the same 24 GPU-hour
ceiling; a new study ID must never create a fresh allowance. Main training remains
outside this authorization. If any frozen task no longer fits, stop for a further
scientific decision rather than dropping it or increasing its budget.

Improved validity is unproven. An alternative decision is to preserve the current
prompts and record launch readiness as blocked by the observed zero-yield pilot.
