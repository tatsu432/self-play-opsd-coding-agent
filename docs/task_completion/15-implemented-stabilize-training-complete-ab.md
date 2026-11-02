# Task 15 implementation and execution record

**Task 15 is incomplete.** Candidate code and local validation are being completed; no successor Yale allocation or final held-out evaluation has run. Yale authentication currently requires the user's normal MFA. See [current status](../investigations/task15-status.md).

Implemented candidate behavior includes separate behavior/scorer/teacher probabilities, versioned same-backend signed gaps, independent numerical interventions and no-privilege repeats, bounded structured online task supply with explicit replay/fallback, checkpointed exposure/curriculum state, episode milestones and endpoint-aware recovery, scientific freeze/evaluation guards, and local paired analysis.

Implementation `234d443` passed **326 local tests, 34 skipped, 4 deselected**, Ruff, formatting, mypy and whitespace checks. The local immutable snapshot/data validation passed. Hosted worker integration [41367241355](https://github.com/tatsu432/self-play-opsd-coding-agent/actions/runs/41367241355) is running at this handoff; local Docker is unavailable. No allocated-node acceptance or main training is claimed.

Validation results, execution commits, job IDs, allocated checks, freeze identity, six main arms and final analysis must be recorded here as they actually complete. CPU tests alone do not establish GPU fit, generated-code isolation, teacher utility or improved learning. Do not label controller implementation as experiment completion.
