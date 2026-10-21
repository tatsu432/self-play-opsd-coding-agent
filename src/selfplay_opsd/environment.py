"""Four-tool finite-horizon repair episodes with trusted final verification."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from selfplay_opsd.config import EpisodeConfig
from selfplay_opsd.core import (
    EpisodeRecord,
    ExperimentError,
    OracleSpec,
    Status,
    TaskSpec,
    TokenTargetMap,
    TokenTurn,
    canonical,
    digest,
)
from selfplay_opsd.mutations import apply_patch, parse_action, signature, validate_source
from selfplay_opsd.policy import HFPolicy
from selfplay_opsd.protocol import PROMPT_HASH, budget, student_messages, teacher_messages
from selfplay_opsd.sandbox import SandboxBackend, Verification


def verify_with_retry(
    sandbox: SandboxBackend,
    source: str,
    task: TaskSpec,
    oracle: OracleSpec,
    *,
    public: bool = False,
) -> Verification:
    return sandbox.verify(
        source,
        task.entry_point,
        oracle.public_cases if public else oracle.public_cases + oracle.hidden_cases,
        task.public_seconds if public else task.verifier_seconds,
    )


@dataclass
class RepairEnvironment:
    task: TaskSpec
    oracle: OracleSpec
    sandbox: SandboxBackend
    policy: HFPolicy
    source: str = ""
    finished: bool = False
    test_calls: int = 0
    public_seconds: float = 0.0

    def __post_init__(self):
        self.source = self.task.initial_source

    def verify(self, *, public: bool = False) -> Verification:
        result = None
        error = None
        if self.policy.is_coordinator:
            try:
                result = verify_with_retry(
                    self.sandbox, self.source, self.task, self.oracle, public=public
                )
            except ExperimentError as exc:
                error = (str(exc), exc.status.value)
        result, error = self.policy.broadcast((result, error))
        if error:
            raise ExperimentError(error[0], Status(error[1]))
        return result

    def step(self, action: dict) -> dict:
        name = action["action"]
        if self.finished:
            raise ValueError("Episode already finished")
        if name == "read_file":
            files = self.task.student_files() | {"src/solution.py": self.source}
            return {"ok": True, "path": action["path"], "content": files[action["path"]]}
        if name == "apply_patch":
            replacement = apply_patch(self.source, action["patch"])
            tree = validate_source(replacement)
            if signature(tree) != signature(validate_source(self.task.initial_source)):
                raise ValueError("Repair must preserve function APIs")
            self.source = replacement
            return {"ok": True, "source_hash": digest(replacement)}
        if name == "run_tests":
            self.test_calls += 1
            result = self.verify(public=True)
            self.public_seconds += result.seconds
            return {
                "ok": True,
                "public_passed": result.passed,
                "passed": sum(result.details),
                "total": len(result.details),
                "status": result.status.value,
            }
        if name == "finish":
            self.finished = True
            return {"ok": True, "finished": True}
        raise ValueError("Unknown action")


def bounded_observation(policy: HFPolicy, observation: dict, cap: int) -> str:
    text = canonical(observation)
    ids = policy.tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= cap:
        return text
    marker = "\n[OBSERVATION_TRUNCATED]"
    marker_ids = policy.tokenizer.encode(marker, add_special_tokens=False)
    if len(marker_ids) > cap:
        raise ValueError("Observation cap smaller than truncation marker")
    prefix = ids[: cap - len(marker_ids)]
    text = policy.tokenizer.decode(prefix, skip_special_tokens=False) + marker
    while len(policy.tokenizer.encode(text, add_special_tokens=False)) > cap and prefix:
        prefix = prefix[:-1]
        text = policy.tokenizer.decode(prefix, skip_special_tokens=False) + marker
    return text


def check_admission(
    policy: HFPolicy, task: TaskSpec, oracle: OracleSpec, limits: EpisodeConfig
) -> None:
    history = student_messages(task, limits)
    student = policy.encode_history(history)
    teacher = policy.encode_history(teacher_messages(history, oracle))
    if len(teacher) - len(student) > limits.privilege_reserve:
        raise ValueError("Task privilege exceeds reserved tokens")
    if len(student) + limits.max_turn_generated_tokens > limits.max_student_sequence_tokens:
        raise ValueError("Task initial context exceeds admission limit")
    if len(teacher) + limits.max_turn_generated_tokens > limits.max_teacher_sequence_tokens:
        raise ValueError("Task teacher context exceeds admission limit")


def append_context(
    policy: HFPolicy,
    history: list[dict[str, str]],
    actual_prompt: list[int],
    targets: list[int],
    context: list[dict[str, str]],
) -> list[int]:
    """Append protocol tokens without decode/re-tokenizing sampled assistant IDs.

    An empty assistant stub determines only forced closing/context delimiters.
    The teacher uses the same retained suffix after its different initial prefix.
    """
    naive_open = policy.encode_history(history)
    stub = history + [{"role": "assistant", "content": ""}]
    closed = policy.tokenizer.apply_chat_template(stub, tokenize=True, add_generation_prompt=False)
    extended = policy.encode_history(stub + context)
    if closed[: len(naive_open)] != naive_open or extended[: len(closed)] != closed:
        raise ExperimentError(
            "Chat template is not prefix-consistent for the frozen tool serialization"
        )
    closing = closed[len(naive_open) :]
    if targets and closing and targets[-1] == policy.tokenizer.eos_token_id == closing[0]:
        closing = closing[1:]  # A sampled EOS must not become a duplicated forced EOS.
    return actual_prompt + targets + closing + extended[len(closed) :]


def run_episode(
    policy: HFPolicy,
    sandbox: SandboxBackend,
    task: TaskSpec,
    oracle: OracleSpec,
    limits: EpisodeConfig,
    seed: int,
    *,
    greedy: bool = False,
) -> EpisodeRecord:
    started = time.monotonic()
    check_admission(policy, task, oracle, limits)
    history = student_messages(task, limits)
    prompt = policy.encode_history(history)
    env = RepairEnvironment(task, oracle, sandbox, policy)
    turns: list[TokenTurn] = []
    tokens = 0
    generation_seconds = 0.0
    termination = "turn_limit"
    for turn_index in range(limits.max_turns):
        max_tokens = min(
            limits.max_turn_generated_tokens,
            limits.max_generated_tokens - tokens,
            limits.max_student_sequence_tokens - len(prompt),
        )
        if max_tokens <= 0:
            termination = (
                "context_limit"
                if len(prompt) >= limits.max_student_sequence_tokens
                else "token_limit"
            )
            break
        generation_started = time.monotonic()
        output = policy.generate(prompt, max_tokens, seed + turn_index)
        generation_seconds += time.monotonic() - generation_started
        if len(output.ids) > max_tokens or len(output.ids) != len(output.logps):
            raise ExperimentError("Backend violated sampled token budget/mapping")
        action, parse_status, observation = "invalid", "invalid", None
        if not output.ids:
            termination = "empty_response"
        try:
            parsed = parse_action(output.text)
            action = parsed["action"]
            parse_status = "valid"
            observation = env.step(parsed)
        except ValueError as exc:
            if parse_status == "valid":
                parse_status = "tool_error"
            observation = {"ok": False, "error": str(exc)[:300]}
        turns.append(
            TokenTurn(
                prompt_ids=prompt,
                target_ids=output.ids,
                old_logps=output.logps,
                history=[dict(m) for m in history],
                raw_text=output.text,
                action=action,
                parse_status=parse_status,
            )
        )
        tokens += len(output.ids)
        if not output.ids:
            break
        if env.finished:
            termination = "finish"
            break
        context = [
            {
                "role": "tool",
                "content": bounded_observation(policy, observation, limits.max_observation_tokens),
            },
            {
                "role": "user",
                "content": canonical({"budget": budget(limits, turn_index + 1, tokens)}),
            },
        ]
        prompt = append_context(policy, history, prompt, output.ids, context)
        history += [{"role": "assistant", "content": output.text}, *context]
        if tokens >= limits.max_generated_tokens:
            termination = "token_limit"
            break
    final = env.verify()
    n_public = len(oracle.public_cases)
    public_pass = bool(n_public) and all(final.details[:n_public])
    hidden_pass = bool(oracle.hidden_cases) and all(final.details[n_public:])
    status = (
        Status.OK
        if final.passed
        else Status.STUDENT_FAILURE
        if final.runtime_failures
        else Status.BUDGET
        if termination.endswith("limit")
        else Status.STUDENT_FAILURE
    )
    return EpisodeRecord(
        episode_id=digest([task.task_id, seed, policy.version("student"), greedy, PROMPT_HASH]),
        task_id=task.task_id,
        seed=seed,
        policy_version=policy.version("student"),
        turns=turns,
        reward=float(final.passed),
        public_pass=public_pass,
        hidden_pass=hidden_pass,
        termination=termination,
        status=status,
        verifier_status=final.status,
        semantic_failures=final.semantic_failures,
        runtime_failures=final.runtime_failures,
        final_source=env.source,
        final_source_hash=digest(env.source),
        test_calls=env.test_calls,
        seconds=time.monotonic() - started,
        timings={
            "generate": generation_seconds,
            "public_tests": env.public_seconds,
            "reward_eval": final.seconds,
        },
    )


def score_teacher(
    policy: HFPolicy, episode: EpisodeRecord, oracle: OracleSpec, limits: EpisodeConfig
) -> tuple[EpisodeRecord, list[TokenTargetMap], dict]:
    """Per-turn teacher forcing under the actual history, with exact target IDs."""
    scored, mappings, errors = [], [], []
    old_seconds = teacher_seconds = 0.0
    with policy.active("student"):
        if policy.version("student") != episode.policy_version:
            raise ExperimentError("Student snapshot changed before teacher scoring")
        initial_prompt = episode.turns[0].prompt_ids if episode.turns else []
        initial_teacher = (
            policy.encode_history(teacher_messages(episode.turns[0].history, oracle))
            if episode.turns
            else []
        )
        student_requests = [(t.prompt_ids, t.target_ids) for t in episode.turns]
        teacher_requests = [
            (initial_teacher + t.prompt_ids[len(initial_prompt) :], t.target_ids)
            for t in episode.turns
        ]
        for prefix, targets in teacher_requests:
            if len(prefix) + len(targets) > limits.max_teacher_sequence_tokens:
                raise ExperimentError("Teacher context overflow; privilege cannot be dropped")
        started = time.monotonic()
        unprivileged_scores = policy.score_many(student_requests)
        old_seconds += time.monotonic() - started
        started = time.monotonic()
        teacher_scores = policy.score_many(teacher_requests)
        teacher_seconds += time.monotonic() - started
        for index, turn in enumerate(episode.turns):
            if index == 0 and policy.encode_history(turn.history) != turn.prompt_ids:
                raise ExperimentError("Initial chat-template prefix mismatch")
            if index:
                previous = episode.turns[index - 1]
                expected_context = turn.history[len(previous.history) + 1 :]
                expected = append_context(
                    policy,
                    previous.history,
                    previous.prompt_ids,
                    previous.target_ids,
                    expected_context,
                )
                if expected != turn.prompt_ids:
                    raise ExperimentError("Sampled assistant IDs changed across tool turns")
            if turn.prompt_ids[: len(initial_prompt)] != initial_prompt:
                raise ExperimentError("Actual history does not retain the initial prefix")
            teacher_prompt = initial_teacher + turn.prompt_ids[len(initial_prompt) :]
            if len(teacher_prompt) + len(turn.target_ids) > limits.max_teacher_sequence_tokens:
                raise ExperimentError("Teacher context overflow; dropping privilege is forbidden")
            unprivileged = unprivileged_scores[index]
            old = torch.tensor(turn.old_logps, device=unprivileged.device)
            if len(old):
                errors.extend((unprivileged - old).abs().tolist())
            teacher = teacher_scores[index].detach().float().tolist()
            if len(teacher) != len(turn.target_ids):
                raise ExperimentError("Teacher/student target mapping error")
            mappings.append(
                TokenTargetMap(
                    target_ids=turn.target_ids,
                    student_positions=list(
                        range(
                            len(turn.prompt_ids) - 1,
                            len(turn.prompt_ids) - 1 + len(turn.target_ids),
                        )
                    ),
                    teacher_positions=list(
                        range(
                            len(teacher_prompt) - 1, len(teacher_prompt) - 1 + len(turn.target_ids)
                        )
                    ),
                )
            )
            base = (
                policy.score_base(turn.prompt_ids, turn.target_ids).tolist()
                if policy.config.method == "GRPO_KL"
                else []
            )
            scored.append(turn.model_copy(update={"teacher_logps": teacher, "base_logps": base}))
    return (
        episode.model_copy(update={"turns": scored, "token_target_maps": mappings}),
        mappings,
        {
            "mapping_error_count": 0,
            "mapped_tokens": sum(len(t.target_ids) for t in scored),
            "pre_update_logprob_abs_max": max(errors, default=0.0),
            "pre_update_logprob_abs_mean": sum(errors) / max(1, len(errors)),
            "old_student_score_seconds": old_seconds,
            "teacher_score_seconds": teacher_seconds,
        },
    )


def canonical_trajectory(
    policy: HFPolicy,
    sandbox: SandboxBackend,
    task: TaskSpec,
    oracle: OracleSpec,
    limits: EpisodeConfig,
) -> dict:
    check_admission(policy, task, oracle, limits)
    history = student_messages(task, limits)
    prompt = policy.encode_history(history)
    env = RepairEnvironment(task, oracle, sandbox, policy)
    edit = None
    generated = 0
    actions = [
        {"action": "read_file", "path": "src/solution.py"},
        {"action": "apply_patch", "patch": oracle.inverse_patch},
        {"action": "run_tests"},
        {"action": "finish"},
    ]
    for i, action in enumerate(actions):
        content = canonical(action)
        target = policy.tokenizer.encode(content, add_special_tokens=False)
        if (
            len(target) > limits.max_turn_generated_tokens
            or generated + len(target) > limits.max_generated_tokens
        ):
            raise ValueError("Canonical action exceeds fixed episode token budget")
        if action["action"] == "apply_patch":
            edit = {
                "prompt": prompt,
                "targets": target,
                "mask_policy": "entire_edit_JSON_content_including_wrapper_no_forced_EOS",
            }
        observation = env.step(action)
        generated += len(target)
        if action["action"] != "finish":
            context = [
                {
                    "role": "tool",
                    "content": bounded_observation(
                        policy, observation, limits.max_observation_tokens
                    ),
                },
                {
                    "role": "user",
                    "content": canonical({"budget": budget(limits, i + 1, generated)}),
                },
            ]
            prompt = append_context(policy, history, prompt, target, context)
            history += [{"role": "assistant", "content": content}, *context]
        else:
            history += [{"role": "assistant", "content": content}]
    if not env.verify().passed:
        raise ExperimentError("Canonical repair verification failed", Status.ORACLE)
    return {
        "task_id": task.task_id,
        "parent_id": task.parent_id,
        "history": history,
        "edit": edit,
        "final_source_hash": digest(env.source),
    }
