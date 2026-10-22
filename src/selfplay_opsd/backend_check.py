"""Explicit target-hardware acceptance probe, isolated from scientific runs."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import torch

from selfplay_opsd.config import RunConfig
from selfplay_opsd.core import ExperimentError
from selfplay_opsd.diagnostics import _optimizer_digest
from selfplay_opsd.objectives import alignment, sd_coefficients
from selfplay_opsd.policy import LossExample
from selfplay_opsd.readiness import BACKEND_EVIDENCE_DIRECTORY, backend_identity
from selfplay_opsd.storage import (
    CheckpointManager,
    atomic_json,
    code_hash,
    file_hash,
    provenance,
    read_json,
)


def check_backend(config: RunConfig, output: Path, *, reference: Path | None = None) -> dict:
    from selfplay_opsd.cli import make_policy

    config = config.model_copy(deep=True)
    config.study_id += "-backend-check"
    identity = backend_identity(config)
    if config.hardware.gpu_count and config.model.backend != "verl":
        raise ExperimentError("Production backend checks require verl")
    policy = make_policy(config)
    try:
        initial = copy.deepcopy(policy.state_dict())
        initial_version = policy.version("student")
        proposer_version = policy.version("proposer")
        if initial_version != proposer_version:
            raise ExperimentError("Student/proposer initialization differs")
        prompt = policy.encode_history([{"role": "user", "content": "Reply with one integer."}])
        targets = policy.tokenizer.encode("42", add_special_tokens=False)
        example = LossExample(prompt, targets, [], 0.0, [], 1 / len(targets), "ce")
        gradients, _ = policy.gradients([example])

        def gather(gradients):
            full = {}
            if policy.config.model.backend == "verl":
                from torch.distributed.tensor import DTensor

                for (name, parameter), gradient in zip(
                    policy.named_parameters("student"), gradients, strict=True
                ):
                    full[name] = (
                        DTensor.from_local(
                            gradient,
                            parameter.device_mesh,
                            parameter.placements,
                            run_check=False,
                            shape=parameter.shape,
                            stride=parameter.stride(),
                        )
                        .full_tensor()
                        .cpu()
                    )
            else:
                full = {
                    name: gradient.cpu()
                    for (name, _), gradient in zip(
                        policy.named_parameters("student"), gradients, strict=True
                    )
                }
            return full

        full = {"CE/" + name: tensor for name, tensor in gather(gradients).items()}
        with policy.active("student"):
            old = policy.score(prompt, targets).detach().float()
            for method in "ABCD":
                _, _, coefficients = sd_coefficients(
                    old + 0.5,
                    old,
                    method != "A",
                    config.objective.gate_beta,
                    config.objective.gap_clip,
                )
                combined = LossExample(
                    prompt,
                    targets,
                    old.tolist(),
                    1.0,
                    coefficients.tolist(),
                    1 / len(targets),
                    "combined",
                )
                student_gradients, _ = policy.gradients([combined])
                for name, tensor in gather(student_gradients).items():
                    full[f"{method}/student/{name}"] = tensor
                if method in "CD":
                    candidate = LossExample(
                        prompt,
                        targets,
                        old.tolist(),
                        1.0,
                        coefficients.tolist(),
                        1 / len(targets),
                        "rl" if method == "C" else "combined",
                    )
                    candidate_gradients, _ = policy.gradients([candidate])
                    score = alignment(
                        candidate_gradients,
                        gradients,
                        config.objective.near_zero,
                        policy.distributed,
                    )
                    if score.near_zero or score.cosine is None:
                        raise ExperimentError(
                            "Controlled objective alignment probe has zero gradient"
                        )
                    for name, tensor in gather(candidate_gradients).items():
                        full[f"{method}/alignment/{name}"] = tensor
        if not full or any(not torch.isfinite(t).all() for t in full.values()):
            raise ExperimentError("Nonfinite objective probe gradients")
        compared = False
        if reference:
            reference_report = read_json(reference.parent / "backend_report.json")
            if (
                reference_report.get("backend_identity") != identity
                or reference_report.get("gpu_count") != 1
                or reference_report.get("reference_sha256") != file_hash(reference)
            ):
                raise ExperimentError("Single-GPU gradient reference is stale or corrupt")
            expected = torch.load(reference, map_location="cpu", weights_only=True)
            if set(expected) != set(full):
                raise ExperimentError("Reference gradient coordinates differ")
            for key in full:
                torch.testing.assert_close(full[key], expected[key], atol=2e-4, rtol=2e-2)
            reference_square = sum(t.double().square().sum().item() for t in expected.values())
            difference_square = sum(
                (full[k].double() - expected[k].double()).square().sum().item() for k in full
            )
            if reference_square <= 0 or (difference_square / reference_square) ** 0.5 > 0.02:
                raise ExperimentError(
                    "Global distributed gradient differs from single-process reference by >2% L2"
                )
            compared = True
        for step in range(2):
            with policy.active("student"):
                policy.gradients([example], step=True)
                policy.sync_adapter("student")
                output_tokens = policy.generate(prompt, 8, 2026 + step)
                scored = policy.score(prompt, output_tokens.ids)
                torch.testing.assert_close(
                    scored.cpu(), torch.tensor(output_tokens.logps), atol=0.05, rtol=0.01
                )
            if policy.version("proposer") != proposer_version:
                raise ExperimentError("Student update modified proposer")
        state = copy.deepcopy(policy.state_dict())
        # A real atomic save/load and both optimizer identities are checked.
        if policy.is_coordinator:
            output.mkdir(parents=True, exist_ok=True)
            manager = CheckpointManager(output)
            path = manager.save(2, state, {"probe_only": True})
        else:
            path = None
        path = policy.broadcast(path)
        loaded, _ = CheckpointManager.load(path)
        policy.load_state_dict(loaded)
        if _optimizer_digest(policy.state_dict()) != _optimizer_digest(state):
            raise ExperimentError("Backend checkpoint roundtrip changed state")
        policy.load_state_dict(initial)
        if policy.version("student") != initial_version:
            raise ExperimentError("Initial adapter restoration failed")
        report = {
            "ready": True,
            "objective_probes": list("ABCD"),
            "probe_label": "controlled synthetic gradient fixtures; not performance evidence",
            "backend_identity": identity,
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "gpu_count": config.hardware.gpu_count,
            "gpu_type": os.getenv("SELFPLAY_GPU_TYPE", "h100"),
            "hardware_profile": config.hardware.profile,
            "model_revision": config.model.revision,
            "tokenizer_revision": config.model.tokenizer_revision,
            "code_hash": code_hash(),
            "two_adapter_syncs": True,
            "checkpoint_roundtrip": True,
            "all_coordinate_gradients": True,
            "single_process_reference_compared": compared,
            "gradient_precision": "BF16 model compute; FP32 gradient buffers; FP64 dot/norm",
            "provenance": provenance(),
        }
        if policy.is_coordinator:
            torch.save(full, output / "reference_gradients.pt")
            report["reference_sha256"] = file_hash(output / "reference_gradients.pt")
            atomic_json(output / "backend_report.json", report)
            atomic_json(
                config.output_root / BACKEND_EVIDENCE_DIRECTORY / f"{config.hardware.profile}.json",
                report,
            )
        return report
    finally:
        policy.close()
