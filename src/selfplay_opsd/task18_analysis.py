"""Task 18 metrics retain raw native actions without changing historical analysis."""

from collections import Counter

from selfplay_opsd.core import canonical
from selfplay_opsd.task17_analysis import episode_metrics
from selfplay_opsd.task17_recovery import error_category
from selfplay_opsd.task18_protocol import native_tool_call


def paired_exposure(records):
    """Summarize only the common tasks used by a paired comparison."""
    keys = (
        "actions",
        "valid_schema_actions",
        "accepted_edits",
        "rejected_edits",
        "tokens",
        "turns",
    )
    result = {key: sum(r[key] for r in records) for key in keys}
    result["repairs"] = sum(r["success"] for r in records)
    result["tasks"] = len(records)
    result["invalid_call_rate"] = (
        1 - result["valid_schema_actions"] / result["actions"] if result["actions"] else None
    )
    return result


def native_expansion_gate(plain, native):
    """Frozen screening rule; an eight-task pass is not an interface selection."""
    if not plain["tasks"] or plain["tasks"] != native["tasks"]:
        raise ValueError("Native screening requires the same nonempty task set")
    valid_rates = plain["invalid_call_rate"] is not None and native["invalid_call_rate"] is not None
    return bool(
        valid_rates
        and native["invalid_call_rate"] < plain["invalid_call_rate"]
        and native["repairs"] >= plain["repairs"]
    )


def metrics(episode):
    # Normalize only a private analysis copy. Sampled IDs and stored trajectories
    # remain untouched, including malformed native calls and their token costs.
    turns = []
    for turn in episode["turns"]:
        text = turn["raw_text"].strip()
        if "<tool_call>" in text:
            try:
                call = native_tool_call(text)
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
