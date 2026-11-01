"""Verify one fixed set of 8B header-normalized proposals in the isolated worker."""

import argparse
import json
import os
from pathlib import Path

from analyze_task13_headers import HEADER
from transformers import AutoTokenizer

from selfplay_opsd.config import load_config
from selfplay_opsd.data import TaskRepository, build_task
from selfplay_opsd.environment import check_admission
from selfplay_opsd.mutations import strict_json_object, validate_mutation
from selfplay_opsd.sandbox import SandboxBackend
from selfplay_opsd.storage import atomic_json, file_hash

SOURCE = Path("/runs/repair-diagnosis-2026-model-probe-v1/66d86331ec30a14afb9548eb")
HASHES = {
    "manifest.json": "beea9ad8407bac76e3f030fb69ad4a1da2fd9df41935e4abc779ee8334cd4255",
    "proposals.json": "1fc70f4e02448fd759c136de38158e5f5a4b454ac6d7ec43b8a689c64c1660f8",
    "complete.json": "74efd566697d01dbe60854391ad4c53f45d8ea7066b4db72eee0211f33d8a688",
}
MODEL = "Qwen/Qwen3-8B"
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


class TokenizerPolicy:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL, revision=REVISION, local_files_only=True
        )

    def encode_history(self, messages):
        return self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SELFPLAY_WORKER_TOKEN"):
        raise RuntimeError("Replay requires an allocated job and isolated worker")
    if args.output.exists() or any(file_hash(SOURCE / name) != sha for name, sha in HASHES.items()):
        raise RuntimeError("Existing output or changed source evidence")
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    rows = json.loads((SOURCE / "proposals.json").read_text())
    complete = json.loads((SOURCE / "complete.json").read_text())
    cfg = load_config(args.settings, method="B", hardware_profile="single_gpu_80gb")
    repo = TaskRepository(cfg.data_root)
    if (
        repo.hash != manifest["manifest_hash"]
        or manifest["model"]["name"] != MODEL
        or manifest["model"]["revision"] != REVISION
        or manifest["representation"] != "diff"
        or complete["count"] != len(rows)
    ):
        raise RuntimeError("Source probe identity mismatch")
    parents = {p.parent_id: p for p in repo.parents("train")}
    sandbox = SandboxBackend(cfg.sandbox)
    sandbox.preflight()
    policy = TokenizerPolicy()
    results = []
    args.output.mkdir(parents=True, mode=0o700)
    for row in rows:
        try:
            obj = strict_json_object(row["raw_text"])
            if (
                set(obj) != {"patch"}
                or not isinstance(obj["patch"], str)
                or not obj["patch"].startswith(HEADER)
            ):
                continue
            parent = parents[row["parent_id"]]
            result = {"parent_id": parent.parent_id, "slot": row["slot"], "valid": False}
            try:
                change = validate_mutation(parent.source, obj["patch"][len(HEADER) :])
                public, hidden = repo.parent_cases(parent.parent_id)
                task, oracle = build_task(
                    parent, change, public, hidden, "train", row["seed"], sandbox
                )
                check_admission(policy, task, oracle, cfg.episode)
                result.update(valid=True, dedup_key=change.dedup_key, reason="valid")
            except (ValueError, SyntaxError, TypeError, KeyError, IndexError) as exc:
                result["reason"] = str(exc)[:300]
            results.append(result)
            atomic_json(args.output / "results.json", results)
        except (ValueError, SyntaxError, TypeError, KeyError, IndexError):
            continue
    atomic_json(
        args.output / "complete.json",
        {
            "source_hashes": HASHES,
            "count": len(results),
            "valid": sum(r["valid"] for r in results),
            "unique": len({(r["parent_id"], r["dedup_key"]) for r in results if r["valid"]}),
            "change": "remove exactly one leading diff --git header",
            "worker": "accepted allocation worker",
        },
    )


if __name__ == "__main__":
    main()
