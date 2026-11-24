"""Prepare Task 22’s private reference sidecar and freeze controls from executed evidence."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from selfplay_opsd.core import digest, read_json
from selfplay_opsd.storage import atomic_json, file_hash
from selfplay_opsd.task21 import PdbRepository
from selfplay_opsd.task22 import Task22Control, reference_sidecar
from selfplay_opsd.task22_contract import (
    BASE_SCIENCE,
    BASE_STUDY,
    EXTENSIONS,
    INITIAL,
    MATCHED_DATA,
    STUDY,
    code_identity,
    common_control,
    reference_path,
)

ROOT = Path(__file__).resolve().parents[1]


def stage_data(source: Path, target: Path) -> str:
    payload = reference_sidecar(PdbRepository(source))
    for name in ("manifest.json", "validation_report.json", "pdb.json"):
        old, copy = source / name, target / name
        if copy.exists():
            if file_hash(old) != file_hash(copy):
                raise ValueError("Task 22 data staging would change the original bundle")
        else:
            shutil.copyfile(old, copy)
            copy.chmod(0o600)
    temporary = target / ".reference.pending"
    atomic_json(temporary, payload)
    checksum = file_hash(temporary)
    temporary.replace(reference_path(target, checksum))
    return checksum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/task21/pdb-solver-v1")
    parser.add_argument("--data", type=Path, default=ROOT / "data/task22/pdb-alignment-v1")
    parser.add_argument("--verified", type=Path)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--extend-to", type=int, choices=(768, 1024))
    parser.add_argument(
        "--parent-artifacts", type=Path, default=ROOT / "runs/hpc/yale-hpc-cpsc4880"
    )
    parser.add_argument("--parent-receipts", type=Path, default=ROOT / "runs/task21")
    args = parser.parse_args()
    if args.extend_to and (args.gate or args.verified):
        parser.error("Freeze extensions from the completed Task 21 parents")

    if bool(args.gate) != bool(args.verified):
        parser.error(
            "Pass both the raw gate and the controller verification to freeze the main runs"
        )
    args.data.mkdir(parents=True, exist_ok=True, mode=0o700)
    checksum = stage_data(args.source, args.data)
    matched = read_json(ROOT / "configs/task21-main.json")
    control = matched | {"study": STUDY, "reference_sha256": checksum}
    phase = "qualification"
    if args.extend_to:
        study = next(key for key, endpoint in EXTENSIONS.items() if endpoint == args.extend_to)
        points = [128, 256, 640] if args.extend_to == 768 else [640, 768, 896]
        for arm in ("control", "A", "B"):
            receipt = read_json(args.parent_receipts / f"matched-{arm}-verified.json")
            marker = args.parent_artifacts / receipt["run_id"] / "complete-main.json"
            parent = read_json(marker)
            identity = parent["identity"]
            if (
                any(receipt.get(key) != value for key, value in parent.items())
                or receipt["scheduler"]["state"] != "COMPLETED"
                or parent["episodes"] != 512
                or identity["study"] != BASE_STUDY
                or identity["science"] != BASE_SCIENCE
                or identity["arm"] != arm
                or identity["data"] != MATCHED_DATA
                or identity["initial"] != INITIAL
            ):
                raise ValueError(f"Verified 512-episode parent differs for {arm}")
            extended = matched | {
                "study": study,
                "endpoint": args.extend_to,
                "dev_milestones": points,
                "parent_marker_sha256": file_hash(marker),
            }
            write_settings(extended, f"extension-{args.extend_to}-{arm}", args.data, matched)
        return
    if args.gate:
        gate, verification = read_json(args.gate), read_json(args.verified)
        report, identity = gate["result"], gate["identity"]
        if (
            any(verification.get(key) != value for key, value in gate.items())
            or verification["scheduler"]["state"] != "COMPLETED"
            or not re.fullmatch(r"[a-f0-9]{24}", verification["run_id"])
            or verification["run_id"] != report["run_id"]
            or report["ready"] is not True
            or report["common_control"] != digest(common_control(matched))
            or identity["initial"] != INITIAL
            or identity["data"] != MATCHED_DATA
            or identity["reference"] != checksum
            or identity["study"] != STUDY
            or identity["execution"]["task22_code"] != code_identity(ROOT)
        ):
            raise ValueError("Controller-verified alignment evidence differs or did not qualify")
        control["alignment_gate_hash"] = file_hash(args.gate)
        phase = "main"
    write_settings(control, phase, args.data, matched)


def write_settings(control: dict, phase: str, data: Path, matched: dict) -> None:
    Task22Control.model_validate(control)
    location = ROOT / f"configs/task22-{phase}"
    original = (ROOT / "configs/task21-main.yaml").read_text()
    settings = original.replace(matched["study"], control["study"]).replace(
        "data/task21/pdb-solver-v1", data.resolve().relative_to(ROOT).as_posix()
    )
    config_path = location.with_suffix(".json")
    yaml_path = location.with_suffix(".yaml")
    if config_path.exists() and read_json(config_path) != control:
        raise ValueError(f"Preserve the already frozen control at {config_path}")
    if yaml_path.exists() and yaml_path.read_text() != settings:
        raise ValueError(f"Preserve the already frozen settings at {yaml_path}")
    atomic_json(config_path, control)
    yaml_path.write_text(settings)
    print(location.relative_to(ROOT).with_suffix(".yaml"))


if __name__ == "__main__":
    main()
