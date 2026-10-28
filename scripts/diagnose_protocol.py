"""Read saved generations as data; never execute or replace sampled outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from selfplay_opsd.mutations import strict_json_object, validate_mutation
from selfplay_opsd.storage import atomic_json, file_hash


def diagnose(proposals: Path, students: Path) -> dict:
    counts, lengths, conventions = Counter(), Counter(), Counter()
    examples, sources = [], {}
    attempted = 0
    for path in sorted(proposals.glob("trajectories/**/*-proposals.json")):
        sources[str(path)] = file_hash(path)
        for record in json.loads(path.read_text()):
            attempted += 1
            turn = record["turn"]
            text = turn["raw_text"]
            lengths[len(turn["target_ids"])] += 1
            for convention in ("diff --git", "```", "\\n", "\n--- a/", "\n+++ b/"):
                conventions[convention] += convention in text
            try:
                parsed = strict_json_object(text)
                patch = parsed["patch"]
                parent = json.loads(turn["history"][1]["content"])["correct_source"]
                validate_mutation(parent, patch)
                stage = (
                    "syntax_and_structure_passed; see original semantic/admission/duplicate result"
                )
            except (ValueError, SyntaxError, KeyError, TypeError) as exc:
                stage = (
                    "json" if not isinstance(locals().get("parsed"), dict) else "patch_or_structure"
                )
                stage += ": " + str(exc)[:180]
            finally:
                parsed = None
            counts[stage] += 1
            if len(examples) < 4:
                examples.append(
                    {"proposal_id": record["proposal_id"], "stage": stage, "excerpt": text[:650]}
                )
    actions, errors, termination, token_lengths = Counter(), Counter(), Counter(), Counter()
    truncated = 0
    episodes = 0
    for path in sorted(students.glob("trajectories/**/*.json")):
        sources[str(path)] = file_hash(path)
        for episode in json.loads(path.read_text()):
            if "turns" not in episode:
                continue
            episodes += 1
            termination[episode["termination"]] += 1
            for turn in episode["turns"]:
                actions[(turn["action"], turn["parse_status"])] += 1
                token_lengths[len(turn["target_ids"])] += 1
                truncated += any("[OBSERVATION_TRUNCATED]" in h["content"] for h in turn["history"])
                for message in turn["history"]:
                    if message["role"] == "tool" and '"ok":false' in message["content"]:
                        errors[message["content"][:200]] += 1
    return {
        "source_sha256": sources,
        "proposer": {
            "attempted": attempted,
            "failure_stages": dict(counts),
            "sampled_token_lengths": dict(lengths),
            "at_512_token_boundary": lengths[512],
            "conventions": dict(conventions),
            "examples": examples,
        },
        "student": {
            "episodes": episodes,
            "actions": {str(k): v for k, v in actions.items()},
            "termination": dict(termination),
            "sampled_token_lengths": dict(token_lengths),
            "turn_histories_with_truncation": truncated,
            "tool_error_histories": dict(errors),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--students", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic_json(args.output, diagnose(args.proposals, args.students))
