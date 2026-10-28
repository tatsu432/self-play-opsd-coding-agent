# Task 08 historical protocol diagnosis

Inspected the complete saved D trajectory shards (160 attempts, ten iterations)
and all 41 monitoring-baseline episodes. This baseline was sampled because the
episode boundary dropped `greedy=True`. Nothing here establishes greedy accuracy.
Raw data remains under `runs/hpc/yale-hpc-cpsc4880/`; the derived report is
`runs/task08/historical-protocol-diagnosis.json`, including source file hashes.
Reproduce it without executing generated code:

```bash
uv run python scripts/diagnose_protocol.py --proposals runs/hpc/yale-hpc-cpsc4880/f597e3f66092a9a9c1265ea0 --students runs/hpc/yale-hpc-cpsc4880/46ec4ac904e9611f551f8713 --output runs/task08/historical-protocol-diagnosis.json
```

| Stage | Observation |
|---|---|
| JSON | Five failures: invalid escape/control character, extra data, missing delimiter/value |
| Patch | 154 required-header failures; one hunk-count mismatch |
| Structure/application | No proposal passed the preceding strict syntax stage |
| Semantic verification/admission | Not reached; semantic mutation quality remains unmeasured |
| Duplicates | None admitted; zero useful tasks or student episodes |
| Proposer budget | Zero outputs reached the 512-token boundary |

Twenty raw outputs contain Git preambles. Other observed conventions include a
bare hunk, raw replacement source inside the patch field, missing line prefixes
and incorrect hunk counts. For example, proposal `0d3302859a6f...` declares five
lines but supplies four removed/four added lines. Proposal `be1e81920dd5...`
starts directly with `@@` and omits both file headers and context prefixes.
Removing Git metadata alone cannot fix these errors. The parser stays strict.

Student trajectories contain 325 actions: 118 valid reads, 112 patch tool errors,
67 invalid actions and 28 valid finishes. There are no valid patch or public-test
actions. Twenty-eight episodes finish; twelve reach their turn limit and one
reaches its token limit. No stored turn history has an observation-truncation
marker. Tool-error history counts in the machine report include repeated prior
observations; they must not be interpreted as independent action counts.

The dominant demonstrated problem is shared JSON/diff protocol compliance.
Students can access files, but successful editing/testing and repair ability
remain unestablished. Proposer token-limit expansion, easier parents, cosmetic
parser tolerance and semantic output repair are not justified by this evidence.
The common syntax clarification is fixed before new evaluation outcomes. It
preserves the four-tool interface, strict parser and all mathematical budgets.
