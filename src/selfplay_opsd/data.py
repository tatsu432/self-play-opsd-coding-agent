"""Pinned dataset ingestion, immutable splits, oracle caches and frozen tasks."""

from __future__ import annotations

import ast
import importlib.metadata
import re
import shutil
from collections import defaultdict
from collections.abc import Callable
from importlib.resources import files
from pathlib import Path

from selfplay_opsd.codec import decode, encode
from selfplay_opsd.core import (
    Case,
    ExperimentError,
    OracleSpec,
    ParentProgram,
    Status,
    TaskSpec,
    digest,
    keyed_seed,
    read_json,
)
from selfplay_opsd.mutations import (
    Mutation,
    fixed_mutation,
    normalized_family,
    validate_source,
)
from selfplay_opsd.sandbox import SandboxBackend
from selfplay_opsd.storage import atomic_json, file_hash

DATA_VERSIONS = {"mbpp": "v0.2.0", "humaneval": "v0.1.10"}
DATA_SHA256 = {
    "mbpp": "b54e762755248ca411b523c917fa9f93c07b5ff2966bf60b3917b853926a3dad",
    "humaneval": "42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f",
    "original_mbpp": "ca95deaa9a01ef0a6f439f88bcf0dd3db3563d22f22aad6cae04ebb9a8d8c8e9",
}


def preserve_notices(root: Path) -> None:
    target = root / "licenses"
    target.mkdir(parents=True, exist_ok=True)
    for resource in files("selfplay_opsd").joinpath("licenses").iterdir():
        if resource.is_file():
            (target / resource.name).write_bytes(resource.read_bytes())


SPLIT_TARGETS = {
    "train": 200,
    "reference": 32,
    "dev_monitor": 34,
    "dev_diagnostic": 12,
    "final_id": 100,
}


def public_arguments(text: str, entry_point: str) -> list:
    """Extract literal calls appearing in original examples, never invent examples."""
    arguments, seen = [], set()
    for match in re.finditer(rf"\b{re.escape(entry_point)}\s*\(", text):
        start, depth, quote, escape, end = match.start(), 0, None, False, match.end() - 1
        for i in range(match.end() - 1, len(text)):
            char = text[i]
            if quote:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == quote:
                    quote = None
            elif char in "\"'":
                quote = char
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        try:
            node = ast.parse(text[start:end], mode="eval").body
            if not isinstance(node, ast.Call) or node.keywords:
                continue
            args = [ast.literal_eval(arg) for arg in node.args]
            key = digest(encode(args))
            if key not in seen:
                arguments.append(args)
                seen.add(key)
        except (ValueError, SyntaxError):
            continue
    return arguments


def download_inventory(root: Path) -> list[dict]:
    """Network/cache IO only; no benchmark code is executed here."""
    if (root / "manifest.json").exists():
        raise ExperimentError("Cannot replace inventory inside a frozen dataset")
    preserve_notices(root)
    from evalplus.data import get_human_eval_plus, get_mbpp_plus
    from evalplus.data.humaneval import _ready_human_eval_plus_path
    from evalplus.data.mbpp import _ready_mbpp_plus_path, get_mbpp

    raw_dir = root / "private/raw"
    raw_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    mbpp = get_mbpp_plus(version=DATA_VERSIONS["mbpp"])
    human = get_human_eval_plus(version=DATA_VERSIONS["humaneval"])
    original = get_mbpp()
    from evalplus.data.utils import CACHE_DIR

    original_path = Path(CACHE_DIR) / "sanitized-mbpp.json"
    if file_hash(original_path) != DATA_SHA256["original_mbpp"]:
        raise ExperimentError(
            "Original MBPP examples checksum differs from pinned source", Status.ORACLE
        )
    shutil.copyfile(original_path, raw_dir / "original-mbpp.json")
    source_hashes = {}
    for name, source_file in (
        ("mbpp", Path(_ready_mbpp_plus_path(version=DATA_VERSIONS["mbpp"]))),
        ("humaneval", Path(_ready_human_eval_plus_path(version=DATA_VERSIONS["humaneval"]))),
    ):
        target = raw_dir / f"{name}.jsonl"
        shutil.copyfile(source_file, target)
        source_hashes[name] = file_hash(target)
        if source_hashes[name] != DATA_SHA256[name]:
            raise ExperimentError(f"Pinned {name} dataset checksum mismatch", Status.ORACLE)
    records = []
    for name, dataset in (("mbpp", mbpp), ("humaneval", human)):
        for parent_id, raw in dataset.items():
            if name == "mbpp":
                original_task = original[parent_id.split("/")[-1]]
                specification = original_task["prompt"]
                public_text = "\n".join(original_task["test_list"])
                # Setup needed by the authoritative implementation is retained;
                # setup with non-stdlib dependencies will fail admission.
                source = raw["canonical_solution"].strip() + "\n"
                imports = "\n".join(original_task.get("test_imports", []))
                if imports:
                    source = imports + "\n" + source
            else:
                specification = raw["prompt"]
                public_text = raw["prompt"]
                source = raw["prompt"] + raw["canonical_solution"]
                # Remove correct implementation docstrings from source only to
                # avoid redundant context; specification remains the original.
            raw_source_hash = digest(source)
            try:
                source = ast.unparse(ast.parse(source)) + "\n"
            except SyntaxError:
                # Preserve unsupported source for an explicit admission exclusion.
                source = source.rstrip() + "\n"
            records.append(
                {
                    "parent": ParentProgram(
                        parent_id=parent_id,
                        source=source,
                        specification=specification,
                        entry_point=raw["entry_point"],
                        dataset=name,
                        provenance={
                            "dataset_version": DATA_VERSIONS[name],
                            "dataset_sha256": source_hashes[name],
                            "evalplus_version": importlib.metadata.version("evalplus"),
                            "license": "https://github.com/evalplus/evalplus#license",
                            "source_hash": digest(source),
                            "raw_source_hash": raw_source_hash,
                            "source_normalization": "stdlib ast.unparse before any mutations",
                        },
                    ).model_dump(mode="json"),
                    "public_inputs": [
                        encode(a) for a in public_arguments(public_text, raw["entry_point"])
                    ],
                    "public_text": public_text,
                    "base_inputs": [encode(a) for a in raw["base_input"]],
                    "plus_inputs": [encode(a) for a in raw["plus_input"]],
                    "atol": raw["atol"],
                    "contract": raw["contract"],
                }
            )
    atomic_json(root / "private/inventory.json", records)
    atomic_json(
        root / "inventory_report.json",
        {
            "counts": {"mbpp": len(mbpp), "humaneval": len(human)},
            "source_hashes": source_hashes,
            "versions": DATA_VERSIONS,
            "evalplus": importlib.metadata.version("evalplus"),
            "original_mbpp_sha256": DATA_SHA256["original_mbpp"],
            "original_mbpp_canonical_hash": digest(original),
            "public_policy": "literal calls in original public examples only; remaining base_input and plus_input private",
        },
    )
    return records


def family_groups(
    parents: list[ParentProgram], threshold: float = 0.9
) -> tuple[dict[str, str], list[dict]]:
    """Alpha-normalized AST equality + conservative token-5-gram Jaccard.

    Connected components remain together. This cannot establish semantic uniqueness.
    """
    owners = {p.parent_id: p.parent_id for p in parents}

    def find(key):
        while owners[key] != key:
            owners[key] = owners[owners[key]]
            key = owners[key]
        return key

    signatures = {p.parent_id: normalized_family(p.source) for p in parents}
    grams = {}
    for key, text in signatures.items():
        tokens = re.findall(r"\w+|[^\s\w]", text)
        grams[key] = {tuple(tokens[i : i + 5]) for i in range(max(1, len(tokens) - 4))}
    decisions = []
    for i, parent in enumerate(parents):
        for other in parents[:i]:
            a, b = grams[parent.parent_id], grams[other.parent_id]
            exact = signatures[parent.parent_id] == signatures[other.parent_id]
            similarity = len(a & b) / max(1, len(a | b))
            if exact or similarity >= threshold:
                x, y = sorted([find(parent.parent_id), find(other.parent_id)])
                owners[y] = x
                decisions.append(
                    {
                        "a": parent.parent_id,
                        "b": other.parent_id,
                        "rule": "alpha_ast_exact" if exact else "ast_token_5gram_jaccard",
                        "similarity": similarity,
                    }
                )
    return {p.parent_id: find(p.parent_id) for p in parents}, decisions


def assign_splits(
    parents: list[ParentProgram], families: dict[str, str], seed: int
) -> dict[str, str]:
    groups: dict[str, list[ParentProgram]] = defaultdict(list)
    for parent in parents:
        groups[families[parent.parent_id]].append(parent)
    assignments = {}
    counts = dict.fromkeys(SPLIT_TARGETS, 0)
    # Reserve held-out capacity first. A shortage is explicitly borne by training,
    # never filled by moving reserved held-out parents back into training.
    order = ["final_id", "reference", "dev_monitor", "dev_diagnostic", "train"]
    for family in sorted(groups, key=lambda key: keyed_seed(seed, "split", key)):
        group = groups[family]
        if any(p.dataset == "humaneval" for p in group):
            for parent in group:
                assignments[parent.parent_id] = (
                    "external" if parent.dataset == "humaneval" else "excluded_external_overlap"
                )
            continue
        role = next(
            (role for role in order if counts[role] + len(group) <= SPLIT_TARGETS[role]), None
        )
        if role is None:
            role = "excluded_capacity"
        else:
            counts[role] += len(group)
        for parent in group:
            assignments[parent.parent_id] = role
    return assignments


def comparison_for(parent: ParentProgram) -> str:
    from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS, MBPP_OUTPUT_SET_EQ_TASKS

    if parent.dataset == "humaneval" and parent.entry_point == "find_zero":
        return "polynomial_root"
    if parent.dataset == "mbpp":
        if parent.entry_point == "are_equivalent":
            raise ValueError("EvalPlus oracle accepts every output; no useful semantic verifier")
        if parent.entry_point in MBPP_OUTPUT_NOT_NONE_TASKS:
            return "non_none"
        if parent.entry_point in MBPP_OUTPUT_SET_EQ_TASKS:
            return "set"
        return {
            "sum_div": "sum_div",
            "surface_Area": "surface_area",
            "digit_distance_nums": "digit_distance",
        }.get(parent.entry_point, "equal")
    return "equal"


def construct_oracle(
    raw: dict, sandbox: SandboxBackend, cache: Path
) -> tuple[ParentProgram, list[Case], list[Case]]:
    parent = ParentProgram.model_validate(raw["parent"])
    validate_source(parent.source)
    if not raw["public_inputs"]:
        raise ValueError("No literal original public examples supported")
    comparison = comparison_for(parent)
    public_keys = {digest(x) for x in raw["public_inputs"]}
    hidden = {
        digest(x): x
        for x in raw["base_inputs"] + raw["plus_inputs"]
        if digest(x) not in public_keys
    }
    if not hidden:
        raise ValueError("No additional non-public verifier inputs")
    inputs = raw["public_inputs"] + list(hidden.values())
    key = digest(
        {
            "source": parent.source,
            "inputs": inputs,
            "image": sandbox.identity(),
            "contract": raw["contract"],
        }
    )
    path = cache / f"{key}.json"
    if path.exists():
        result = read_json(path)
    else:
        result_obj = sandbox.execute(
            parent.source, parent.entry_point, inputs, sandbox.config.verifier_seconds
        )
        if result_obj.status != Status.OK or any(not r["ok"] for r in result_obj.results):
            raise ExperimentError(
                f"Correct program failed: {parent.parent_id}: {result_obj.detail}", Status.ORACLE
            )
        # Verify the specified preconditions using the same isolated runner.
        if raw["contract"].strip():
            tree = ast.parse(parent.source)
            function = next(
                n
                for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == parent.entry_point
            )
            contract_function = ast.parse("def _contract():\n" + raw["contract"]).body[0]
            assert isinstance(contract_function, ast.FunctionDef)
            function.body = contract_function.body + function.body
            checked = sandbox.execute(
                ast.unparse(tree) + "\n",
                parent.entry_point,
                inputs,
                sandbox.config.verifier_seconds,
            )
            if checked.status != Status.OK or any(not r["ok"] for r in checked.results):
                raise ExperimentError("Reference preconditions failed", Status.ORACLE)
        result = result_obj.results
        atomic_json(path, result)
    cases = []
    for args, outcome in zip(inputs, result, strict=True):
        expected = outcome["value"]
        if comparison == "non_none":
            expected = encode(decode(expected) is not None)
        cases.append(Case(args=args, expected=expected, comparison=comparison, atol=raw["atol"]))
    return parent, cases[: len(raw["public_inputs"])], cases[len(raw["public_inputs"]) :]


def build_task(
    parent: ParentProgram,
    mutation: Mutation,
    public: list[Case],
    hidden: list[Case],
    split: str,
    seed: int,
    sandbox: SandboxBackend,
) -> tuple[TaskSpec, OracleSpec]:
    verification = sandbox.verify(
        mutation.source, parent.entry_point, public + hidden, sandbox.config.verifier_seconds
    )
    if verification.semantic_failures < 1:
        raise ValueError("Mutant has no semantic failure (crash/timeout-only mutants invalid)")
    restored = sandbox.verify(
        parent.source, parent.entry_point, public + hidden, sandbox.config.verifier_seconds
    )
    if not restored.passed:
        raise ExperimentError("Restoring original did not pass", Status.ORACLE)
    task_id = digest([parent.parent_id, mutation.dedup_key, split, seed])
    tests = render_public_tests(parent.entry_point, public)
    task = TaskSpec(
        task_id=task_id,
        parent_id=parent.parent_id,
        split=split,
        specification=parent.specification,
        initial_source=mutation.source,
        entry_point=parent.entry_point,
        public_tests=tests,
        source_hash=digest(mutation.source),
        mutation_seed=seed,
        public_seconds=sandbox.config.public_seconds,
        verifier_seconds=sandbox.config.verifier_seconds,
    )
    oracle = OracleSpec(
        task_id=task_id,
        correct_source=parent.source,
        inverse_patch=mutation.inverse_patch,
        public_cases=public,
        hidden_cases=hidden,
        provenance=parent.provenance
        | {"mutation_key": mutation.dedup_key, "image_digest": sandbox.identity()},
    )
    return task, oracle


def render_public_tests(entry_point: str, cases: list[Case]) -> str:
    """Expose only public examples with the supervisor's comparison semantics."""
    tests = f"from src.solution import {entry_point}\nimport cmath\n\n"
    tests += "def _near(a, b, atol):\n    if a == b:\n        return True\n    if type(a) != type(b):\n        return False\n    if isinstance(b, (list, tuple)):\n        return len(a) == len(b) and all(_near(x, y, atol) for x, y in zip(a, b))\n    try:\n        return cmath.isclose(a, b, abs_tol=atol, rel_tol=1e-7)\n    except (TypeError, ValueError):\n        return False\n\n"
    for index, case in enumerate(cases):
        inputs, expected = decode(case.args), decode(case.expected)
        args = ", ".join(repr(a) for a in inputs)
        comparison = f"result == {expected!r}"
        if case.comparison == "set":
            comparison = f"set(result) == set({expected!r})"
        elif case.comparison == "non_none":
            comparison = (
                f"(result if isinstance(result, bool) else result is not None) == {expected!r}"
            )
        elif case.comparison == "sum_div":
            comparison += " or result == 0"
        elif case.comparison in {"surface_area", "digit_distance"}:
            from evalplus.eval._special_oracle import _digit_distance_nums, _surface_Area

            alternative = (
                _surface_Area(*inputs)
                if case.comparison == "surface_area"
                else _digit_distance_nums(*inputs)
            )
            comparison += f" or abs(result - {alternative!r}) <= {case.atol!r}"
        elif case.comparison == "polynomial_root":
            comparison = (
                f"abs(sum(c * result**i for i, c in enumerate({inputs[0]!r}))) <= {case.atol!r}"
            )
        elif (
            case.atol
            or isinstance(expected, float)
            or (
                isinstance(expected, (list, tuple))
                and expected
                and all(isinstance(x, float) for x in expected)
            )
        ):
            comparison = f"_near(result, {expected!r}, {case.atol or 1e-6!r})"
        tests += f"def test_public_{index}():\n    result = {entry_point}({args})\n    assert {comparison}\n\n"
    return tests


class TaskRepository:
    def __init__(self, root: Path):
        self.root = root
        self.manifest = read_json(root / "manifest.json")
        self.hash = digest(self.manifest)
        if file_hash(root / "private/parents.json") != self.manifest["parents_hash"]:
            raise ExperimentError("Prepared parent storage checksum mismatch", Status.ORACLE)
        self._parents = read_json(root / "private/parents.json")
        self._parent_by_id = {r["parent"]["parent_id"]: r for r in self._parents}

    def parents(self, split: str) -> list[ParentProgram]:
        return [
            ParentProgram.model_validate(x["parent"]) for x in self._parents if x["split"] == split
        ]

    def parent_cases(self, parent_id: str) -> tuple[list[Case], list[Case]]:
        record = self._parent_by_id[parent_id]
        return [Case.model_validate(c) for c in record["public"]], [
            Case.model_validate(c) for c in record["hidden"]
        ]

    def tasks(self, split: str) -> list[tuple[TaskSpec, OracleSpec]]:
        path = self.root / "private" / f"tasks-{split}.json"
        if not path.exists():
            return []
        if file_hash(path) != self.manifest["task_hashes"].get(path.name):
            raise ExperimentError("Frozen task storage checksum mismatch", Status.ORACLE)
        return [
            (TaskSpec.model_validate(x["task"]), OracleSpec.model_validate(x["oracle"]))
            for x in read_json(path)
        ]

    def anchors(self, split: str = "reference") -> list[dict]:
        path = self.root / "private" / f"anchors-{split}.json"
        if file_hash(path) != self.manifest.get("anchor_hashes", {}).get(split):
            raise ExperimentError("Canonical anchor storage checksum mismatch", Status.ORACLE)
        return read_json(path)


def prepare_data(
    root: Path,
    sandbox: SandboxBackend,
    seed: int = 2026,
    attempts: int = 20,
    *,
    allow_reduced: bool = False,
    admit_parent: Callable | None = None,
) -> dict:
    if (root / "manifest.json").exists():
        raise ExperimentError("Prepared manifests are immutable; choose a new output directory")
    sandbox.preflight()
    root.mkdir(parents=True, exist_ok=True)
    (root / "private").mkdir(mode=0o700, exist_ok=True)
    inventory_path = root / "private/inventory.json"
    inventory = read_json(inventory_path) if inventory_path.exists() else download_inventory(root)
    if inventory and "source_normalization" not in inventory[0]["parent"]["provenance"]:
        inventory = download_inventory(root)
    eligible, exclusions = [], []
    for raw in inventory:
        try:
            parent, public, hidden = construct_oracle(raw, sandbox, root / "private/oracle_cache")
            if admit_parent is not None:
                admit_parent(parent, public, hidden)
            eligible.append((parent, public, hidden))
        except (ValueError, SyntaxError) as exc:
            exclusions.append({"parent_id": raw["parent"]["parent_id"], "reason": str(exc)})
        except ExperimentError as exc:
            if exc.status != Status.ORACLE:
                raise
            exclusions.append({"parent_id": raw["parent"]["parent_id"], "reason": str(exc)})
        atomic_json(
            root / "preparation_health.json",
            {
                "eligible": len(eligible),
                "excluded": len(exclusions),
                "processed": len(eligible) + len(exclusions),
                "total": len(inventory),
            },
        )
    families, decisions = family_groups([p for p, _, _ in eligible])
    splits = assign_splits([p for p, _, _ in eligible], families, seed)
    records: list[dict] = []
    counts: dict[str, int] = defaultdict(int)
    for parent, public, hidden in eligible:
        split = splits[parent.parent_id]
        counts[split] += 1
        records.append(
            {
                "parent": parent.model_copy(
                    update={"family_id": families[parent.parent_id]}
                ).model_dump(mode="json"),
                "split": split,
                "public": [c.model_dump(mode="json") for c in public],
                "hidden": [c.model_dump(mode="json") for c in hidden],
            }
        )
    report = {
        "counts": dict(counts),
        "targets": SPLIT_TARGETS,
        "exclusions": exclusions,
        "family_decisions": decisions,
        "split_assignments": splits,
        "seed": seed,
        "image": sandbox.attestation,
        "coverage_policy": "held-out reservations first; deficits remain explicit; no redistribution into training",
    }
    atomic_json(root / "coverage.json", report)
    if not allow_reduced and any(counts[k] != v for k, v in SPLIT_TARGETS.items()):
        raise ExperimentError(
            "Eligible counts differ from requested targets; inspect coverage.json and explicitly acknowledge reduced coverage",
            Status.ORACLE,
        )
    if any(counts[k] == 0 for k in SPLIT_TARGETS):
        raise ExperimentError("A required partition is empty", Status.ORACLE)
    atomic_json(root / "private/parents.json", records)
    task_counts = {}
    for split, target in {
        "reference": 1,
        "dev_monitor": 2,
        "dev_diagnostic": 2,
        "final_id": 2,
        "external": 1,
    }.items():
        tasks = []
        for parent, public, hidden in eligible:
            if splits[parent.parent_id] != split:
                continue
            history: set[str] = set()
            for attempt in range(attempts):
                mutation_seed = keyed_seed(
                    seed, "frozen_mutation", split, parent.parent_id, attempt
                )
                try:
                    mutation = fixed_mutation(parent.source, mutation_seed)
                    if mutation.dedup_key in history:
                        continue
                    task, oracle = build_task(
                        parent, mutation, public, hidden, split, mutation_seed, sandbox
                    )
                except ValueError:
                    continue
                history.add(mutation.dedup_key)
                tasks.append(
                    {"task": task.model_dump(mode="json"), "oracle": oracle.model_dump(mode="json")}
                )
                if len(history) == target:
                    break
            atomic_json(root / "private" / f"tasks-{split}.json", tasks)
        task_counts[split] = {
            "actual": len(tasks),
            "target": counts[split] * target,
            "parents_covered": len({r["task"]["parent_id"] for r in tasks}),
        }
    report["tasks"] = task_counts
    atomic_json(root / "coverage.json", report)
    manifest = {
        "schema_version": 1,
        "seed": seed,
        "dataset_versions": DATA_VERSIONS,
        "inventory_hash": file_hash(inventory_path),
        "parents_hash": file_hash(root / "private/parents.json"),
        "coverage_hash": digest(report),
        "task_hashes": {
            p.name: file_hash(p) for p in sorted((root / "private").glob("tasks-*.json"))
        },
        "protocol": "repair-json-v1",
        "image_digest": sandbox.identity(),
        "execution_limits": sandbox.config.model_dump(
            mode="json", exclude={"backend", "image", "runtime", "remote_url", "remote_token_env"}
        ),
        "ready": False,
    }
    atomic_json(root / "manifest.json", manifest)
    return report
