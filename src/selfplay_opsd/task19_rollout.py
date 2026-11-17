"""Collect exact-prefix student and proposer behavior through the isolated worker."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import asdict
from typing import cast

from selfplay_opsd.config import Strict
from selfplay_opsd.core import OracleSpec, ParentProgram, TaskSpec, digest, keyed_seed
from selfplay_opsd.environment import RepairEnvironment
from selfplay_opsd.mutations import make_patch, replace_text, validate_mutation
from selfplay_opsd.policy import HFPolicy
from selfplay_opsd.protocol import budget, teacher_messages
from selfplay_opsd.sandbox import SandboxBackend
from selfplay_opsd.structured_mutations import mutation as ast_mutation
from selfplay_opsd.structured_mutations import sites
from selfplay_opsd.task19 import Task19Control
from selfplay_opsd.task19_policy import PROPOSER_SAMPLING, STUDENT_SAMPLING, SharedPolicy
from selfplay_opsd.task19_protocol import (
    PROPOSER_TOOLS,
    STUDENT_TOOLS,
    assistant_history,
    parse_call,
    proposer_history,
    safe_json,
    student_history,
)


class Turn(Strict):
    prompt: list[int]
    ids: list[int]
    logps: list[float]
    text: str
    token_kinds: list[str]
    stopped: bool
    history: list[dict]
    action: str | None = None
    result: str
    teacher_prompt: list[int] | None = None
    teacher_logps: list[float] | None = None
    frozen_student_logps: list[float] | None = None


class Repair(Strict):
    task_id: str
    parent_id: str
    seed: int
    turns: list[Turn]
    success: bool
    public_pass: bool
    hidden_pass: bool
    termination: str
    failure: str
    verifier_status: str
    final_source_hash: str
    test_calls: int
    timings: dict[str, float]

    def metrics(self) -> dict:
        tokens = Counter(kind for turn in self.turns for kind in turn.token_kinds)
        actions = Counter(turn.action for turn in self.turns if turn.result == "accepted")
        return {
            "success": self.success,
            "public_pass": self.public_pass,
            "hidden_pass": self.hidden_pass,
            "failure": self.failure,
            "termination": self.termination,
            "turns": len(self.turns),
            "tokens": sum(tokens.values()),
            "reasoning_tokens": tokens["reasoning"],
            "tool_tokens": tokens["tool"],
            "other_sampled_tokens": tokens["assistant"],
            "prompt_tokens": sum(len(turn.prompt) for turn in self.turns),
            "truncated_thinking": any(
                not turn.stopped and turn.token_kinds[-1:] == ["reasoning"] for turn in self.turns
            ),
            "invalid_calls": sum(turn.result == "invalid" for turn in self.turns),
            "rejected_edits": sum(
                turn.action == "replace_text" and turn.result == "rejected" for turn in self.turns
            ),
            "accepted_edits": actions["replace_text"],
            "reads": actions["read_file"],
            "test_calls": self.test_calls,
            "timings": self.timings,
        }


def repair(
    policy: SharedPolicy,
    sandbox: SandboxBackend,
    cfg: Task19Control,
    task: TaskSpec,
    oracle: OracleSpec,
    seed: int,
) -> Repair:
    env = RepairEnvironment(task, oracle, sandbox, cast(HFPolicy, policy))
    limits = cfg.episode
    history = student_history(task, limits)
    turns: list[Turn] = []
    tokens = 0
    generation = 0.0
    tools = 0.0
    termination = "turn_limit"
    for index in range(limits.max_turns):
        prompt = policy.encode(history, STUDENT_TOOLS, True)
        maximum = min(
            limits.max_turn_generated_tokens,
            limits.max_generated_tokens - tokens,
            limits.max_student_sequence_tokens - len(prompt),
        )
        if maximum <= 0:
            termination = (
                "episode_tokens" if tokens >= limits.max_generated_tokens else "context_limit"
            )
            break
        started = time.monotonic()
        out = policy.generate(
            prompt,
            maximum,
            keyed_seed(seed, "turn", index),
            thinking=True,
            sampling=STUDENT_SAMPLING,
            cached=cfg.cached_rollout,
        )
        generation += time.monotonic() - started
        call = None
        action = None
        result = "invalid"
        started = time.monotonic()
        try:
            call = parse_call(out.visible, STUDENT_TOOLS)
            name, arguments = call
            action = name
            observation = env.step({"action": name, **arguments})
            result = "accepted"
        except ValueError as exc:
            result = "rejected" if action else "invalid"
            observation = {"ok": False, "error": str(exc)[:300]}
        tools += time.monotonic() - started
        fields = {k: value for k, value in asdict(out).items() if k != "visible"}
        turns.append(Turn(prompt=prompt, history=history, action=action, result=result, **fields))
        tokens += len(out.ids)
        if env.finished:
            termination = "finish"
            break
        if not out.ids:
            termination = "empty_response"
            break
        if call is None and not out.stopped:
            termination = (
                "turn_tokens"
                if maximum == limits.max_turn_generated_tokens
                else (
                    "episode_tokens" if tokens >= limits.max_generated_tokens else "context_limit"
                )
            )
            break
        content = safe_json(observation)
        content_ids = policy.tokenizer.encode(content, add_special_tokens=False)
        if len(content_ids) > limits.max_observation_tokens:
            marker = "\n[OBSERVATION_TRUNCATED]"
            marker_ids = policy.tokenizer.encode(marker, add_special_tokens=False)
            content = (
                policy.tokenizer.decode(
                    content_ids[: max(0, limits.max_observation_tokens - len(marker_ids))],
                    skip_special_tokens=False,
                )
                + marker
            )
        history = [
            *history,
            assistant_history(out.visible, call),
            {"role": "tool", "content": content},
            {"role": "user", "content": safe_json({"budget": budget(limits, index + 1, tokens)})},
        ]

    started = time.monotonic()
    final = env.verify()
    public_count = len(oracle.public_cases)
    public_pass = all(final.details[:public_count]) and len(final.details) >= public_count
    hidden_pass = len(final.details) == public_count + len(oracle.hidden_cases) and all(
        final.details[public_count:]
    )
    accepted = any(t.action == "replace_text" and t.result == "accepted" for t in turns)
    if final.passed:
        failure = "verified"
    elif termination in {"turn_limit", "context_limit", "episode_tokens", "turn_tokens"}:
        failure = "truncation"
    elif public_pass and not hidden_pass:
        failure = "public_pass_hidden_fail"
    elif accepted:
        failure = "semantic_or_runtime_after_edit"
    elif any(t.action == "replace_text" and t.result == "rejected" for t in turns):
        failure = "rejected_edit"
    elif any(t.result == "invalid" for t in turns):
        failure = "invalid_tool"
    else:
        failure = "no_edit"
    return Repair(
        task_id=task.task_id,
        parent_id=task.parent_id,
        seed=seed,
        turns=turns,
        success=final.passed,
        public_pass=public_pass,
        hidden_pass=hidden_pass,
        termination=termination,
        failure=failure,
        verifier_status=final.status.value,
        final_source_hash=digest(env.source),
        test_calls=env.test_calls,
        timings={"generation": generation, "tool": tools, "verifier": time.monotonic() - started},
    )


def privilege(policy: SharedPolicy, cfg: Task19Control, oracle: OracleSpec, roll: Repair) -> None:
    for turn in roll.turns:
        prefix = policy.encode(teacher_messages(turn.history, oracle), STUDENT_TOOLS, True)
        if len(prefix) + len(turn.ids) > cfg.episode.max_teacher_sequence_tokens:
            raise ValueError("Verified task exceeds the reserved teacher context")
        turn.teacher_prompt = prefix
        turn.teacher_logps = policy.score(
            prefix, turn.ids, None, exact_prefix=not cfg.cached_rollout
        )[0].tolist()
        turn.frozen_student_logps = policy.score(
            turn.prompt, turn.ids, None, exact_prefix=not cfg.cached_rollout
        )[0].tolist()


class Proposal(Strict):
    parent: str
    group: int
    seed: int
    turn: Turn
    task: str | None = None
    mutation_key: str | None = None
    family: str | None = None
    representation: str | None = None
    changed_lines: int = 0
    changed_nodes: int = 0
    duplicate: bool = False
    invalid_reason: str | None = None
    reward: float = -1.0
    alignment: dict | None = None


def propose(policy: SharedPolicy, cfg: Task19Control, parent: ParentProgram, group: int, slot: int):
    history = proposer_history(parent)
    prefix = policy.encode(history, PROPOSER_TOOLS, cfg.proposer_thinking)
    seed = keyed_seed(cfg.seed, "proposer", group, parent.parent_id, slot)
    maximum = min(cfg.proposal_tokens, cfg.episode.max_student_sequence_tokens - len(prefix))
    out = policy.generate(
        prefix,
        maximum,
        seed,
        thinking=cfg.proposer_thinking,
        sampling=PROPOSER_SAMPLING,
        cached=cfg.cached_rollout,
    )
    fields = {k: value for k, value in asdict(out).items() if k != "visible"}
    row = Proposal(
        parent=parent.parent_id,
        group=group,
        seed=seed,
        turn=Turn(prompt=prefix, history=history, result="invalid", **fields),
    )
    try:
        name, params = parse_call(out.visible, PROPOSER_TOOLS)
        row.representation = name
        row.turn.action = name
        if name == "mutate_ast":
            legal = sites(parent.source)
            if params["site"] >= len(legal):
                raise ValueError("AST site not in the offered catalog")
            edit = ast_mutation(parent.source, legal[params["site"]]["path"], params["operation"])
        else:
            source = replace_text(parent.source, params["old_text"], params["new_text"])
            edit = validate_mutation(parent.source, make_patch(parent.source, source))
        row.mutation_key = edit.dedup_key
        row.family = edit.family
        row.changed_lines = edit.changed_lines
        row.changed_nodes = edit.changed_nodes
        return row, edit
    except (ValueError, SyntaxError) as exc:
        row.invalid_reason = str(exc)[:300]
        return row, None
