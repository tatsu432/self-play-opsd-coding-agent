"""Bounded Task 13 proposer probe; run only inside an allocated isolated worker job."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from task13_model_policy import MODELS, ModelProbePolicy
from task13_structured import mutation, sites

from selfplay_opsd.cli import make_policy
from selfplay_opsd.config import load_config
from selfplay_opsd.core import digest, keyed_seed
from selfplay_opsd.data import TaskRepository, build_task
from selfplay_opsd.environment import check_admission
from selfplay_opsd.mutations import strict_json_object, validate_mutation
from selfplay_opsd.protocol import PROMPT_HASH, proposer_messages
from selfplay_opsd.sandbox import SandboxBackend
from selfplay_opsd.storage import CheckpointManager, atomic_json, scientific_code_hash


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--representation", choices=("diff", "structured"), required=True)
    parser.add_argument("--parents", type=int, default=40)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--slot-start", type=int, default=0)
    parser.add_argument("--model-choice", choices=tuple(MODELS))
    args = parser.parse_args()
    if args.model_choice and args.checkpoint:
        raise ValueError("Larger-model inference cannot load a 4B adapter checkpoint")
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SELFPLAY_WORKER_TOKEN"):
        raise RuntimeError("Probe requires an allocated Slurm job and isolated worker")
    if args.output.exists():
        raise RuntimeError("Probe output already exists; preserve previous evidence")
    cfg = load_config(args.settings, method="B", hardware_profile="single_gpu_80gb")
    repo = TaskRepository(cfg.data_root)
    selected = sorted(repo.parents("train"), key=lambda p: digest([2026, p.parent_id]))[
        : args.parents
    ]
    if len(selected) != args.parents or args.slots < 1 or args.slot_start < 0:
        raise ValueError("Insufficient training parents or invalid slot count")
    sandbox = SandboxBackend(cfg.sandbox)
    sandbox.preflight()
    policy = ModelProbePolicy(cfg, args.model_choice) if args.model_choice else make_policy(cfg)
    try:
        checkpoint = None
        if args.checkpoint:
            state, checkpoint = CheckpointManager.load(args.checkpoint)
            if (
                checkpoint["manifests_hash"] != repo.hash
                or checkpoint["prompt_hash"] != PROMPT_HASH
            ):
                raise ValueError("Checkpoint dataset or prompt identity differs")
            if checkpoint["resolved_config"]["model"] != cfg.model.model_dump(mode="json"):
                raise ValueError("Checkpoint model identity differs")
            expected = load_config(
                args.settings,
                method=checkpoint["resolved_config"]["method"],
                hardware_profile="single_gpu_80gb",
            )
            if (
                checkpoint["scientific_hash"] != expected.scientific_hash
                or checkpoint["scientific_code_hash"] != scientific_code_hash()
            ):
                raise ValueError("Checkpoint scientific identity differs")
            policy.load_state_dict(state["policy"])
        manifest = {
            "study": "repair-diagnosis-2026-model-probe-v1"
            if args.model_choice
            else "repair-diagnosis-2026-probe-v1",
            "manifest_hash": repo.hash,
            "model": cfg.model.model_dump(mode="json"),
            "checkpoint": checkpoint,
            "representation": args.representation,
            "chat_template_non_thinking": args.model_choice == "qwen3-8b",
            "inference_backend": "vllm-inference-only" if args.model_choice else cfg.model.backend,
            "adapter": "none" if args.model_choice else "proposer-lora",
            "slots": args.slots,
            "slot_start": args.slot_start,
            "parents": [{"id": p.parent_id, "source_hash": digest(p.source)} for p in selected],
            "sampling": {
                "temperature": 1,
                "top_p": 1,
                "top_k": -1,
                "max_tokens": cfg.proposer_max_tokens,
            },
        }
        args.output.mkdir(parents=True, mode=0o700)
        atomic_json(args.output / "manifest.json", manifest)
        history = state["history"] if args.checkpoint else {}
        records = []
        with policy.active("proposer"):
            for parent in selected:
                available = sites(parent.source) if args.representation == "structured" else []
                for slot in range(args.slot_start, args.slot_start + args.slots):
                    seed = keyed_seed(2026, "task13-proposer", parent.parent_id, slot)
                    messages = proposer_messages(parent)
                    if args.representation == "structured":
                        messages[0]["content"] = (
                            "Introduce a small semantic bug. Choose one listed AST site and operation. Return exactly JSON with fields site (integer index) and operation (string)."
                        )
                        messages[1]["content"] = json.dumps(
                            {
                                "specification": parent.specification,
                                "correct_source": parent.source,
                                "sites": available,
                            }
                        )
                    prompt = policy.encode_history(messages)
                    generated = policy.generate(prompt, cfg.proposer_max_tokens, seed)
                    record = {
                        "parent_id": parent.parent_id,
                        "slot": slot,
                        "seed": seed,
                        "target_ids": generated.ids,
                        "old_logps": generated.logps,
                        "raw_text": generated.text,
                        "finish_reason": getattr(policy, "last_finish_reason", None),
                        "valid": False,
                        "reason": None,
                        "dedup_key": None,
                    }
                    try:
                        parsed = strict_json_object(generated.text)
                        if args.representation == "structured":
                            if (
                                set(parsed) != {"site", "operation"}
                                or type(parsed["site"]) is not int
                                or not isinstance(parsed["operation"], str)
                                or not 0 <= parsed["site"] < len(available)
                            ):
                                raise ValueError("Invalid structured proposal")
                            change = mutation(
                                parent.source,
                                available[parsed["site"]]["path"],
                                parsed["operation"],
                            )
                        else:
                            if set(parsed) != {"patch"} or not isinstance(parsed["patch"], str):
                                raise ValueError("Invalid patch proposal")
                            change = validate_mutation(parent.source, parsed["patch"])
                        record["dedup_key"] = change.dedup_key
                        record["in_training_history"] = change.dedup_key in history.get(
                            parent.parent_id, []
                        )
                        public, hidden = repo.parent_cases(parent.parent_id)
                        task, oracle = build_task(
                            parent, change, public, hidden, "train", seed, sandbox
                        )
                        check_admission(policy, task, oracle, cfg.episode)
                        record.update(valid=True, reason="valid")
                    except (ValueError, SyntaxError, TypeError, KeyError, IndexError) as exc:
                        record["reason"] = str(exc)[:300]
                    records.append(record)
                    atomic_json(args.output / "proposals.json", records)
        atomic_json(
            args.output / "complete.json",
            {"count": len(records), "valid": sum(r["valid"] for r in records)},
        )
    finally:
        policy.close()


if __name__ == "__main__":
    main()
