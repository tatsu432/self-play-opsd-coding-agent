"""Saved probabilities only: no model loading, candidate execution, or new GPU results."""

import argparse
import json
from pathlib import Path

from selfplay_opsd.core import EpisodeRecord
from selfplay_opsd.storage import atomic_json, file_hash
from selfplay_opsd.task18_numerical import saved_token_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, help="Local pinned tokenizer; never downloads")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve earlier analyses")
    records = []
    client = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        from selfplay_opsd.policy import HFPolicy

        class Client:
            encode_history = HFPolicy.encode_history

        client = Client()
        client.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    for path in sorted(args.source.glob("trajectories/*/*-episodes.json")):
        iteration = int(path.name[:6])
        episodes = [EpisodeRecord.model_validate(row) for row in json.loads(path.read_text())]
        alignment_turns = 0
        if client is not None:
            from selfplay_opsd.environment import append_context

            for episode in episodes:
                previous = None
                for turn in episode.turns:
                    expected = (
                        client.encode_history(turn.history)
                        if previous is None
                        else append_context(
                            client,
                            previous.history,
                            previous.prompt_ids,
                            previous.target_ids,
                            turn.history[len(previous.history) + 1 :],
                        )
                    )
                    if expected != turn.prompt_ids:
                        raise ValueError("Saved prompt prefix alignment differs")
                    previous = turn
                    alignment_turns += 1
        metadata = args.source / "checkpoints" / f"{iteration - 1:06d}" / "metadata.json"
        matched = (
            metadata.is_file()
            and json.loads(metadata.read_text())["student_version"] == episodes[0].policy_version
        )
        records.append(
            {
                "iteration": iteration,
                "aligned_turns": alignment_turns if client is not None else None,
                "source_sha256": file_hash(path),
                "generating_snapshot": episodes[0].policy_version,
                "matching_checkpoint_metadata_available": matched,
                **saved_token_report(episodes),
            }
        )
    if not records:
        raise ValueError("No saved training episode shards")
    atomic_json(
        args.output / "summary.json",
        {
            "kind": "retrospective saved-token analysis; not a causal update comparison",
            "batches": records,
            "new_allocated_gpu_hours": 0,
        },
    )


if __name__ == "__main__":
    main()
