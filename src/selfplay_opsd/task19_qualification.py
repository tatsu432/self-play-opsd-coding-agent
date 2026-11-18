"""Training-only calibration and repeat-controlled numerical qualification."""

from __future__ import annotations

import copy
import math
import os
import statistics
import uuid
from pathlib import Path

import torch

from selfplay_opsd.core import OracleSpec, TaskSpec, digest, keyed_seed, read_json
from selfplay_opsd.objectives import alignment, difficulty
from selfplay_opsd.storage import atomic_json, file_hash, restore_rng, rng_state
from selfplay_opsd.task19 import GROUP, Task19Control, centered_rewards, clipped_sum
from selfplay_opsd.task19_policy import STUDENT_SAMPLING
from selfplay_opsd.task19_rollout import Proposal, Repair, privilege, repair
from selfplay_opsd.task19_train import SegmentStop, Study, joint_step, norm


def qualify(study: Study, should_stop=lambda: False) -> Task19Control:
    output = study.output
    cfg = study.cfg
    references = study.references()
    study.reference_gradient(references)
    if cfg.cached_rollout:
        previous = output.parent.parent / "repair-selfplay-2026-task19-v1/qualification"
        source = (
            output.parent.parent
            / "repair-selfplay-2026-task19-v2/qualification/frozen-training-batch.json"
        )
        expected = "4cd5991768348988e351bbebf43ed59c58bc14c822a5a20dfdb2c2ea30a3194d"
        if file_hash(source) != expected:
            raise ValueError("Prior frozen training batch checksum differs")
        batch = read_json(source)
        if batch["snapshot"] != study.snapshot:
            raise ValueError("Prior batch policy snapshot differs")
        old_control = Task19Control.read(previous / "uncached-control.json")
        cfg = cfg.model_copy(
            update={
                "episode": old_control.episode,
                "proposer_thinking": old_control.proposer_thinking,
                "proposal_tokens": old_control.proposal_tokens,
            }
        )
        study.cfg = cfg
        groups = [[Proposal.model_validate(row) for row in group] for group in batch["proposals"]]
        rolls = {
            key: [Repair.model_validate(row) for row in values]
            for key, values in batch["repairs"].items()
        }
        study.proposal_groups = max(row.group for group in groups for row in group) + 1
        study.seen = {
            digest([row.parent, row.mutation_key])
            for group in groups
            for row in group
            if row.mutation_key and row.task
        }
        proposed = [{"source": str(source), "sha256": expected, "reused_groups": len(rolls)}]
        before_caps = {"reused_episodes": sum(map(len, rolls.values()))}
    elif cfg.study.endswith("-v6"):
        groups, rolls, source, expected = load_v5_batch(study)
        study.proposal_groups = max(row.group for group in groups for row in group) + 1
        study.seen = {
            digest([row.parent, row.mutation_key])
            for group in groups
            for row in group
            if row.mutation_key and row.task
        }
        proposed = [{"source": str(source), "sha256": expected, "reused_groups": len(rolls)}]
        before_caps = {"reused_episodes": sum(map(len, rolls.values()))}
    else:
        groups, rolls = [], {}
        expected = None
        proposed = [{"source": "fresh_uncached", "reused_groups": 0}]
        before_caps = {"reused_episodes": 0}
    progress_path = output / "qualification-progress.json"
    pending_tasks = {}
    completed_waves = 0
    if progress_path.is_file():
        progress = read_json(progress_path)
        if progress["identity"] != study.identity or progress["snapshot"] != study.snapshot:
            raise ValueError("Qualification progress identity differs")
        groups = [[Proposal.model_validate(row) for row in group] for group in progress["groups"]]
        roll_order = progress["roll_order"]
        task_order = progress["task_order"]
        if (
            len(roll_order) != len(progress["repairs"])
            or set(roll_order) != set(progress["repairs"])
            or len(task_order) != len(progress["tasks"])
            or set(task_order) != set(progress["tasks"])
        ):
            raise ValueError("Qualification progress task ordering differs")
        rolls = {
            key: [Repair.model_validate(row) for row in progress["repairs"][key]]
            for key in roll_order
        }
        pending_tasks = {
            key: (
                TaskSpec.model_validate(progress["tasks"][key]["task"]),
                OracleSpec.model_validate(progress["tasks"][key]["oracle"]),
            )
            for key in task_order
        }
        study.seen = set(progress["seen"])
        study.proposal_groups = progress["proposal_groups"]
        completed_waves = progress["completed_waves"]
        if not 0 <= completed_waves <= 5:
            raise ValueError("Qualification progress wave count differs")
        for task_id, rows in rolls.items():
            if len(rows) > GROUP or (len(rows) < GROUP and task_id not in pending_tasks):
                raise ValueError("Qualification progress contains an incomplete group")
            for slot, saved_repair in enumerate(rows):
                seed = keyed_seed(cfg.seed, "student", study.iteration, task_id, slot)
                if saved_repair.task_id != task_id or saved_repair.seed != seed:
                    raise ValueError("Qualification progress episode identity differs")
                if any(
                    turn.teacher_prompt is None
                    or turn.teacher_logps is None
                    or turn.frozen_student_logps is None
                    or len(turn.logps) != len(turn.ids)
                    or len(turn.teacher_logps) != len(turn.ids)
                    or len(turn.frozen_student_logps) != len(turn.ids)
                    for turn in saved_repair.turns
                ):
                    raise ValueError("Qualification progress has incomplete teacher scores")

    def save_progress() -> None:
        atomic_json(
            progress_path,
            {
                "identity": study.identity,
                "snapshot": study.snapshot,
                "groups": [[row.model_dump(mode="json") for row in group] for group in groups],
                "roll_order": list(rolls),
                "task_order": list(pending_tasks),
                "repairs": {
                    key: [row.model_dump(mode="json") for row in rows]
                    for key, rows in rolls.items()
                },
                "tasks": {
                    key: {
                        "task": task.model_dump(mode="json"),
                        "oracle": oracle.model_dump(mode="json"),
                    }
                    for key, (task, oracle) in pending_tasks.items()
                },
                "seen": sorted(study.seen),
                "proposal_groups": study.proposal_groups,
                "completed_waves": completed_waves,
            },
        )

    if not progress_path.is_file():
        save_progress()
    for _ in range(completed_waves, 5):
        mixed = any(0 < sum(row.success for row in values) < GROUP for values in rolls.values())
        if mixed and not pending_tasks:
            break
        if not pending_tasks:
            if should_stop():
                raise SegmentStop("Qualification stopped before proposal collection")
            extra_groups, pending_tasks, _ = study.collect(4)
            groups.extend(extra_groups)
            save_progress()
        for task_id, (task, oracle) in pending_tasks.items():
            rows = rolls.setdefault(task_id, [])
            for slot in range(len(rows), GROUP):
                if should_stop():
                    raise SegmentStop("Qualification stopped between saved episodes")
                seed = keyed_seed(cfg.seed, "student", study.iteration, task_id, slot)
                repair_row = repair(study.policy, study.sandbox, cfg, task, oracle, seed)
                privilege(study.policy, cfg, oracle, repair_row)
                rows.append(repair_row)
                save_progress()
        pending_tasks = {}
        completed_waves += 1
        save_progress()
        atomic_json(
            output / "frozen-training-batch.json",
            {
                "snapshot": study.snapshot,
                "source_sha256": expected,
                "proposals": [[row.model_dump(mode="json") for row in group] for group in groups],
                "repairs": {
                    key: [row.model_dump(mode="json") for row in rows]
                    for key, rows in rolls.items()
                },
            },
        )
    mixed = any(0 < sum(row.success for row in values) < GROUP for values in rolls.values())
    if not mixed:
        raise RuntimeError("Bounded training-only extension did not produce mixed student rewards")
    atomic_json(
        output / "frozen-training-batch.json",
        {
            "snapshot": study.snapshot,
            "source_sha256": expected,
            "proposals": [[row.model_dump(mode="json") for row in group] for group in groups],
            "repairs": {
                key: [row.model_dump(mode="json") for row in rows] for key, rows in rolls.items()
            },
        },
    )

    numerical_path = output / "numerical.json"
    binding_path = output / "numerical-binding.json"
    binding = {
        "identity": study.identity,
        "batch_sha256": file_hash(output / "frozen-training-batch.json"),
    }
    if numerical_path.is_file() and binding_path.is_file():
        if read_json(binding_path) != binding | {"numerical_sha256": file_hash(numerical_path)}:
            raise ValueError("Qualification numerical evidence identity differs")
        diagnostic = read_json(numerical_path)
    else:
        if should_stop():
            raise SegmentStop("Qualification stopped before numerical diagnostics")
        diagnostic = numerical(study, rolls)
        atomic_json(numerical_path, diagnostic)
        atomic_json(binding_path, binding | {"numerical_sha256": file_hash(numerical_path)})
    if not diagnostic["passed"]:
        if diagnostic.get("reason") == "No informative replay gradients":
            raise RuntimeError(
                "Qualification lacked an informative gradient; cached execution remains unassessed"
            )
        if not cfg.cached_rollout:
            raise RuntimeError("Task 19 uncached execution failed numerical qualification")
        atomic_json(
            output / "uncached-control.json",
            cfg.model_copy(update={"cached_rollout": False}).model_dump(mode="json"),
        )
        raise RuntimeError(
            "Cached execution failed qualification; use the emitted uncached control"
        )

    if should_stop():
        raise SegmentStop("Qualification stopped before calibration")
    calibration_rolls = sorted(
        rolls.values(), key=lambda rows: not (0 < sum(row.success for row in rows) < GROUP)
    )[:4]
    for group in groups:
        for row in group:
            if row.task is not None:
                row.reward = difficulty(float(item.success) for item in rolls[row.task])
    informative = [group for group in groups if len({row.reward for row in group}) > 1][:4]
    rl_norms, ratios, proposer_norms = calibration_measurements(
        study, calibration_rolls, informative, binding, should_stop
    )
    proposer_norms = [value for value in proposer_norms if value > 1e-12]
    rl_nonzero = [value for value in rl_norms if value > 1e-12]
    if not ratios or not proposer_norms or not rl_nonzero:
        raise RuntimeError("Training-only batch did not identify both loss scales")
    sd_scale = statistics.median(ratios)
    proposer_scale = 0.5 * statistics.median(rl_nonzero) / statistics.median(proposer_norms)
    episode_metrics = [row.metrics() for group in rolls.values() for row in group]
    report = {
        "identity": study.identity,
        "proposer_screen": proposed,
        "distinct_valid_reward_groups": sum(
            len(
                {
                    difficulty(float(r.success) for r in rolls[p.task])
                    for p in group
                    if p.task in rolls
                }
            )
            > 1
            for group in groups
        ),
        "proposer_groups": len(groups),
        "calibration_group_count": len(calibration_rolls),
        "student_cap_screen": before_caps,
        "final_rollouts": episode_metrics,
        "sd_scale": sd_scale,
        "proposer_scale": proposer_scale,
        "sd_norm_targets": ratios,
        "rl_norms": rl_norms,
        "proposer_norms": proposer_norms,
    }
    atomic_json(output / "calibration.json", report)
    result = cfg.model_copy(
        update={
            "sd_scale": sd_scale,
            "proposer_scale": proposer_scale,
            "calibration_hash": file_hash(output / "calibration.json"),
            "qualification_hash": file_hash(output / "numerical.json"),
        }
    )
    atomic_json(output / "calibrated-control.json", result.model_dump(mode="json"))
    return result


def load_v5_batch(
    study, expected="c3d885680074b9566d279c440729e048200997808cbb07e6afd369fbae3611d7"
):
    """Reuse the immutable uncached v5 training batch with verified behavior data."""
    root = study.output.parent.parent
    source_root = root / "repair-selfplay-2026-task19-v5/qualification"
    source = source_root / "frozen-training-batch.json"
    if file_hash(source) != expected:
        raise ValueError("V5 frozen batch checksum differs")
    binding = read_json(source_root / "numerical-binding.json")
    numerical_path = source_root / "numerical.json"
    progress = read_json(source_root / "qualification-progress.json")
    old = progress["identity"]
    current = study.identity
    package = Path(__file__).resolve().parent
    scientific_files = (
        "task19.py",
        "task19_policy.py",
        "task19_protocol.py",
        "task19_rollout.py",
        "task19_train.py",
    )
    if (
        binding["identity"] != old
        or binding["batch_sha256"] != expected
        or binding["numerical_sha256"] != file_hash(numerical_path)
        or not read_json(numerical_path)["passed"]
        or read_json(numerical_path)["cached_rollout"]
        or old["study"] != "repair-selfplay-2026-task19-v5"
        or old["arm"] != current["arm"]
        or old["execution"]["code"]
        != "3bfd3a64a254325e60ed395e933181d9f0f28a67cc3a15bf68bba55df37246d5"
        or digest({name: file_hash(package / name) for name in scientific_files})
        != "62228b2135f534193de08d47e02d5ac123c2de442302d2b0a8f73f2c13270429"
        or old["data"] != current["data"]
        or old["initial"] != current["initial"]
        or old["worker"] != current["worker"]
        or {k: v for k, v in old["execution"].items() if k != "code"}
        != {k: v for k, v in current["execution"].items() if k != "code"}
    ):
        raise ValueError("V5 batch execution identity differs")
    old_control = study.cfg.model_copy(update={"study": "repair-selfplay-2026-task19-v5"})
    if old_control.cached_rollout or old_control.science != old["science"]:
        raise ValueError("V5 and V6 scientific controls differ")
    batch = read_json(source)
    if batch["snapshot"] != study.snapshot:
        raise ValueError("V5 batch policy snapshot differs")
    groups = [[Proposal.model_validate(row) for row in group] for group in batch["proposals"]]
    order = list(
        dict.fromkeys(row.task for group in groups for row in group if row.task in batch["repairs"])
    )
    if len(order) != len(batch["repairs"]) or set(order) != set(batch["repairs"]):
        raise ValueError("V5 batch task ordering is ambiguous")
    rolls = {key: [Repair.model_validate(row) for row in batch["repairs"][key]] for key in order}
    return groups, rolls, source, expected


def calibration_measurements(study, student_groups, proposer_groups, binding, should_stop):
    """Persist complete group measurements and RNG state at transactional boundaries."""
    path = study.output / "calibration-progress.pt"
    identity = {
        "binding": binding,
        "student_groups": [[(row.task_id, row.seed) for row in group] for group in student_groups],
        "proposer_groups": [
            [(row.group, row.seed, row.reward) for row in group] for group in proposer_groups
        ],
    }
    if path.is_file():
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["identity"] != identity:
            raise ValueError("Calibration progress identity differs")
        if len(state["students"]) > len(student_groups) or len(state["proposers"]) > len(
            proposer_groups
        ):
            raise ValueError("Calibration progress contains too many groups")
        if state["proposers"] and (
            len(state["students"]) != len(student_groups) or state["partial_student"] is not None
        ):
            raise ValueError("Calibration progress stage ordering differs")
        if len(state["students"]) == len(student_groups) and state["partial_student"] is not None:
            raise ValueError("Calibration progress has an unexpected partial group")
        restore_rng(state["rng"])
    else:
        state = {
            "identity": identity,
            "students": [],
            "proposers": [],
            "partial_student": None,
            "rng": rng_state(),
        }

    def save() -> None:
        state["rng"] = rng_state()
        temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        with temp.open("xb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)

    if not path.is_file():
        save()
    for rows in student_groups[len(state["students"]) :]:
        turns = [(i, turn) for i, row in enumerate(rows) for turn in row.turns if turn.ids]
        partial = state["partial_student"]
        if partial is None:
            rl, sd, start_position = study.zeros(), study.zeros(), 0
        else:
            rl, sd = study.zeros(), study.zeros()
            start_position = partial["position"]
            if not 0 <= start_position <= len(turns):
                raise ValueError("Calibration progress turn position differs")
            for target, saved in zip((rl, sd), (partial["rl"], partial["sd"]), strict=True):
                if len(target) != len(saved):
                    raise ValueError("Calibration gradient coordinate count differs")
                for dst, src in zip(target, saved, strict=True):
                    if dst.shape != src.shape or dst.dtype != src.dtype:
                        raise ValueError("Calibration gradient coordinate differs")
                    dst.copy_(src.to(dst.device))
        for position in range(start_position, len(turns)):
            if should_stop():
                raise SegmentStop("Calibration stopped between student turns")
            selected, turn = turns[position]
            isolated = [
                row.model_copy(update={"turns": [turn] if i == selected else []})
                for i, row in enumerate(rows)
            ]
            turn_rl, turn_sd, _ = study.student_gradient(isolated, gated=False, use_sd=True)
            for dst, value in zip(rl, turn_rl, strict=True):
                dst.add_(value)
            for dst, value in zip(sd, turn_sd, strict=True):
                dst.add_(value)
            state["partial_student"] = {
                "position": position + 1,
                "rl": [value.detach().cpu().clone() for value in rl],
                "sd": [value.detach().cpu().clone() for value in sd],
            }
            save()
        rl_norm, sd_norm = norm(rl), norm(sd)
        if not math.isfinite(rl_norm) or not math.isfinite(sd_norm):
            raise FloatingPointError("Nonfinite calibration gradient norm")
        state["students"].append((rl_norm, sd_norm))
        state["partial_student"] = None
        save()
    for group in proposer_groups[len(state["proposers"]) :]:
        if should_stop():
            raise SegmentStop("Calibration stopped between proposer groups")
        gradient, _ = study.proposer_gradient([group])
        value = norm(gradient)
        if not math.isfinite(value):
            raise FloatingPointError("Nonfinite proposer calibration gradient norm")
        state["proposers"].append(value)
        save()
    rl_norms = [rl for rl, _ in state["students"]]
    ratios = [0.3 * rl / sd for rl, sd in state["students"] if min(rl, sd) > 1e-12]
    return rl_norms, ratios, state["proposers"]


def numerical(study: Study, rolls: dict) -> dict:
    policy = study.policy
    cfg = study.cfg
    gradient = study.zeros()
    counterfactual = study.zeros()
    mismatch = []
    sampled_replay_max = 0.0
    clip_changes = 0
    compared = 0
    sampled_support_exits = 0
    selected_groups: dict[str, str] = {}
    for key, episodes in rolls.items():
        rewards = [float(row.success) for row in episodes]
        category = "mixed" if min(rewards) != max(rewards) else "uniform"
        if category in selected_groups:
            continue
        selected_groups[category] = key
        for episode, advantage in zip(episodes, centered_rewards(rewards).tolist(), strict=True):
            # Include the first and last turn. Replaying the actual sampled path
            # includes the prompt, thinking markers, tool/EOS and the late prefix.
            positions = sorted({0, len(episode.turns) - 1}) if episode.turns else []
            for position in positions:
                turn = episode.turns[position]
                if not turn.ids:
                    continue
                if cfg.cached_rollout:
                    replayed = policy.replay(turn.prompt, turn.ids, STUDENT_SAMPLING, cached=True)
                    replay_targets = replayed[-8:]
                else:
                    prefix = turn.prompt + turn.ids[:-8]
                    replay_targets = policy.replay(
                        prefix, turn.ids[-8:], STUDENT_SAMPLING, cached=False
                    )
                prefix = turn.prompt + turn.ids[:-8]
                ids = turn.ids[-8:]
                current = policy.score(
                    prefix,
                    ids,
                    STUDENT_SAMPLING,
                    gradients=True,
                    exact_prefix=not cfg.cached_rollout,
                )[0]
                cached = replay_targets.to(device=current.device).clone()
                old = current.new_tensor(turn.logps[-8:])
                if torch.isfinite(cached).all():
                    sampled_replay_max = max(sampled_replay_max, float((cached - old).abs().max()))
                else:
                    sampled_support_exits += int((~cached.isfinite()).sum())
                prod_ratio, cache_ratio = (current.detach() - old).exp(), (cached - old).exp()
                finite = torch.isfinite(current) & torch.isfinite(cached)
                mismatch.extend((current.detach()[finite] - cached[finite]).abs().tolist())
                if advantage:
                    compared += len(ids)
                    if advantage > 0:
                        clip_changes += int(
                            ((prod_ratio > 1 + cfg.clip) != (cache_ratio > 1 + cfg.clip)).sum()
                        )
                    else:
                        clip_changes += int(
                            ((prod_ratio < 1 - cfg.clip) != (cache_ratio < 1 - cfg.clip)).sum()
                        )
                    scale = 1 / (GROUP * cfg.episode.max_generated_tokens)
                    loss = clipped_sum(current, old, advantage, cfg.clip) * scale
                    study.accumulate(gradient, loss, retain_graph=True)
                    # Change the forward ratio for a sensitivity check while
                    # holding the saved behavior denominator and local Jacobian.
                    stable = current.masked_fill(~finite, 0)
                    measured = cached + (stable - stable.detach())
                    compare = clipped_sum(measured, old, advantage, cfg.clip) * scale
                    study.accumulate(counterfactual, compare)
        if len(selected_groups) == 2:
            break
    if norm(gradient) <= 1e-12:
        return {
            "passed": False,
            "reason": "No informative replay gradients",
            "groups": selected_groups,
        }
    state, rng = copy.deepcopy(policy.state()), rng_state()
    anchor = [p.detach().clone() for p in policy.parameters]

    def displacement(values):
        policy.restore(copy.deepcopy(state))
        restore_rng(rng)
        joint_step(policy, values, study.zeros(), 0.0, cfg.grad_clip)
        return [p.detach() - old for p, old in zip(policy.parameters, anchor, strict=True)]

    try:
        repeats = [displacement(gradient) for _ in range(3)]
        candidate = displacement(counterfactual)
    finally:
        policy.restore(state)
        restore_rng(rng)

    def difference(a, b):
        return {
            "relative": norm([x - y for x, y in zip(a, b, strict=True)]) / max(norm(a), 1e-12),
            "cosine": alignment(a, b).cosine,
        }

    repeat = max(difference(repeats[0], row)["relative"] for row in repeats[1:])
    update, grad = difference(repeats[0], candidate), difference(gradient, counterfactual)
    clip_fraction = clip_changes / max(compared, 1)
    passed = (
        set(selected_groups) == {"mixed", "uniform"}
        and sampled_support_exits == 0
        and sampled_replay_max <= 0.02
        and repeat <= 0.01
        and clip_fraction <= 0.01
        and all(
            row["relative"] <= 0.05 and (row["cosine"] or 0.0) >= 0.99 for row in (update, grad)
        )
        and all(map(math.isfinite, mismatch))
    )
    return {
        "passed": passed,
        "cached_rollout": cfg.cached_rollout,
        "groups": selected_groups,
        "snapshot": study.snapshot,
        "sample_seed": keyed_seed(cfg.seed, "qualification"),
        "comparisons": compared,
        "sampled_replay_max_abs": sampled_replay_max,
        "sampled_support_exits": sampled_support_exits,
        "mean_logprob_difference": statistics.mean(mismatch) if mismatch else None,
        "clip_change_fraction": clip_fraction,
        "repeat_relative": repeat,
        "gradient_difference": grad,
        "update_difference": update,
    }


def verify_evidence(
    cfg: Task19Control, root: Path, execution: dict, dataset: str, initial: str
) -> None:
    from selfplay_opsd.core import read_json

    if not cfg.ready:
        raise ValueError("Task 19 has no executed qualification and calibration")
    if (
        file_hash(root / "numerical.json") != cfg.qualification_hash
        or file_hash(root / "calibration.json") != cfg.calibration_hash
    ):
        raise ValueError("Task 19 evidence checksum differs")
    numerical_report = read_json(root / "numerical.json")
    calibration = read_json(root / "calibration.json")
    identity = calibration["identity"]
    if (
        not numerical_report["passed"]
        or numerical_report["cached_rollout"] != cfg.cached_rollout
        or calibration["sd_scale"] != cfg.sd_scale
        or calibration["proposer_scale"] != cfg.proposer_scale
        or identity["execution"] != execution
        or identity["data"] != dataset
        or identity["initial"] != initial
    ):
        raise ValueError("Task 19 calibration is not bound to this execution and dataset")
    qualified = Task19Control.read(root / "calibrated-control.json")
    if cfg.science != qualified.science:
        raise ValueError("Task 19 scientific settings differ from the qualified control")
