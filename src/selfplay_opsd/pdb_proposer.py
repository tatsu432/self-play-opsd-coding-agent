"""Bounded native proposer edits for PDB parent programs."""

from __future__ import annotations

import ast
import difflib
from dataclasses import asdict

from selfplay_opsd.core import ParentProgram, TaskSpec, digest, keyed_seed
from selfplay_opsd.mutations import make_patch, normalized_ast
from selfplay_opsd.pdb_verifier import PROTOCOL, PdbOracle, entry_signature, verify_pdb
from selfplay_opsd.sandbox import SandboxBackend
from selfplay_opsd.structured_mutations import mutate_source, sites
from selfplay_opsd.task19 import Task19Control
from selfplay_opsd.task19_policy import PROPOSER_SAMPLING, SharedPolicy
from selfplay_opsd.task19_protocol import PROPOSER_TOOLS, parse_call, proposer_history
from selfplay_opsd.task19_rollout import Proposal, Turn

PAIR_TOOL = {
    "type": "function",
    "function": {
        "name": "mutate_pair",
        "description": "Introduce a semantic bug using two disjoint unique exact-text replacements",
        "parameters": {
            "type": "object",
            "properties": {
                key: {"type": "string"}
                for key in ("old_text", "new_text", "old_second", "new_second")
            },
            "required": ["old_text", "new_text", "old_second", "new_second"],
            "additionalProperties": False,
        },
    },
}


def bounded_text(source: str, edits: list[tuple[str, str]]) -> str:
    spans = []
    for before, after in edits:
        if (
            not before
            or before == after
            or max(len(before.encode()), len(after.encode())) > 1024
            or max(len(before.splitlines()), len(after.splitlines())) > 4
        ):
            raise ValueError("Each edit must change at most four local lines")
        start = source.find(before)
        if start < 0 or source.find(before, start + 1) >= 0:
            raise ValueError("Each replacement must match a unique original span")
        spans.append((start, start + len(before), before, after))
    spans.sort()
    if len(spans) == 2 and spans[0][1] > spans[1][0]:
        raise ValueError("Proposer edits must be disjoint")
    result = source
    for start, end, _, after in reversed(spans):
        result = result[:start] + after + result[end:]
    if len(result.encode()) > 65536:
        raise ValueError("Proposed source exceeds the worker input limit")
    return result


def admissible_mutant(parent: ParentProgram, oracle: PdbOracle, source: str) -> str:
    def imports(text: str) -> list[str]:
        return [
            ast.dump(node)
            for node in ast.walk(ast.parse(text))
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]

    before = entry_signature(parent.source, parent.entry_point, oracle.handler)
    after = entry_signature(source, parent.entry_point, oracle.handler)
    if before != after or imports(source) != imports(parent.source):
        raise ValueError("Proposer must preserve entry point and imports")
    key = normalized_ast(source)
    if key == normalized_ast(parent.source):
        raise ValueError("Proposer must introduce a semantic AST change")
    return digest([parent.parent_id, key])


def propose_pdb(
    policy: SharedPolicy,
    cfg: Task19Control,
    parent: ParentProgram,
    oracle: PdbOracle,
    group: int,
    slot: int,
    *,
    flexible: bool,
) -> tuple[Proposal, str | None]:
    history = proposer_history(parent)
    if flexible:
        history[0]["content"] += (
            " Use mutate_pair for two localized changes on disjoint original spans. "
            "Use a pair when the two edits interact and retain the original API and imports."
        )
    tools = [*PROPOSER_TOOLS, PAIR_TOOL] if flexible else PROPOSER_TOOLS
    prefix = policy.encode(history, tools, cfg.proposer_thinking)
    maximum = min(cfg.proposal_tokens, cfg.episode.max_student_sequence_tokens - len(prefix))
    seed = keyed_seed(cfg.seed, "proposer", group, parent.parent_id, slot)
    out = policy.generate(
        prefix,
        maximum,
        seed,
        thinking=cfg.proposer_thinking,
        sampling=PROPOSER_SAMPLING,
        cached=cfg.cached_rollout,
    )
    fields = {key: val for key, val in asdict(out).items() if key != "visible"}
    row = Proposal(
        parent=parent.parent_id,
        group=group,
        seed=seed,
        turn=Turn(prompt=prefix, history=history, result="invalid", **fields),
    )
    try:
        method, params = parse_call(out.visible, tools)
        row.representation = row.turn.action = method
        if method == "mutate_ast":
            legal = sites(parent.source)
            if params["site"] >= len(legal):
                raise ValueError("AST site is outside the offered catalog")
            source = mutate_source(
                parent.source, legal[params["site"]]["path"], params["operation"]
            )
            row.family = params["operation"]
        elif method == "mutate_pair":
            source = bounded_text(
                parent.source,
                [
                    (params["old_text"], params["new_text"]),
                    (params["old_second"], params["new_second"]),
                ],
            )
            row.family = method
        else:
            source = bounded_text(parent.source, [(params["old_text"], params["new_text"])])
            row.family = method
        row.mutation_key = admissible_mutant(parent, oracle, source)
        before = ast.unparse(ast.parse(parent.source)).splitlines()
        after = ast.unparse(ast.parse(source)).splitlines()
        row.changed_lines = sum(
            max(right - left, new_right - new_left)
            for action, left, right, new_left, new_right in difflib.SequenceMatcher(
                None, before, after, autojunk=False
            ).get_opcodes()
            if action != "equal"
        )
        return row, source
    except (ValueError, SyntaxError) as exc:
        row.invalid_reason = str(exc)[:300]
        return row, None


def proposal_task(
    sandbox: SandboxBackend,
    parent_task: TaskSpec,
    oracle: PdbOracle,
    row: Proposal,
    source: str,
) -> tuple[TaskSpec, PdbOracle]:
    source_hash = digest(source)
    task_id = digest([PROTOCOL, parent_task.parent_id, row.mutation_key, source_hash])[:24]
    task = parent_task.model_copy(
        update={
            "task_id": task_id,
            "initial_source": source,
            "source_hash": source_hash,
            "split": "train",
            "mutation_seed": row.seed,
        }
    )
    variant = oracle.model_copy(
        update={
            "task_id": task_id,
            "inverse_patch": make_patch(source, oracle.correct_source),
            "provenance": oracle.provenance
            | {
                "mutation_family": row.family,
                "parent_task": parent_task.task_id,
                "changed_lines": row.changed_lines,
                "proposer_seed": row.seed,
            },
        }
    )
    result = verify_pdb(sandbox, source, task, variant)
    if not result.semantic_failures or result.passed:
        raise ValueError("Candidate needs a verifier-confirmed semantic failure")
    return task, variant
