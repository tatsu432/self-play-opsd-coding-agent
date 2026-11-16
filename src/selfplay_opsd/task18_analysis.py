"""Task 18 metrics retain raw native actions without changing historical analysis."""

from collections import Counter

from selfplay_opsd.core import canonical
from selfplay_opsd.mutations import strict_json_object
from selfplay_opsd.task17_analysis import episode_metrics
from selfplay_opsd.task17_recovery import error_category


def metrics(episode):
    # Normalize only a private analysis copy. Sampled IDs and stored trajectories
    # remain untouched, including malformed native calls and their token costs.
    turns = []
    for turn in episode["turns"]:
        text = turn["raw_text"].strip()
        if text.startswith("<tool_call>") and text.endswith("</tool_call>"):
            try:
                call = strict_json_object(text[len("<tool_call>") : -len("</tool_call>")])
                if isinstance(call.get("arguments"), dict) and isinstance(call.get("name"), str):
                    turn = turn | {
                        "raw_text": canonical({**call["arguments"], "action": call["name"]})
                    }
            except ValueError:
                pass
        turns.append(turn)
    result = episode_metrics(episode | {"turns": turns})
    result["semantic_failure_after_applied_edit"] = result["applied_edit"] and not result["success"]
    result["rejection_taxonomy"] = dict(
        Counter(
            error_category((t.get("observation") or {}).get("error", ""))
            for t in turns
            if t["parse_status"] != "valid"
        )
    )
    return result
