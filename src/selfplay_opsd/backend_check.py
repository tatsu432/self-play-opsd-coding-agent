"""Explicit target-hardware acceptance probe, isolated from scientific runs."""

from __future__ import annotations

import copy
from pathlib import Path

import torch

from selfplay_opsd.config import RunConfig
from selfplay_opsd.core import ExperimentError
from selfplay_opsd.diagnostics import _optimizer_digest
from selfplay_opsd.policy import LossExample
from selfplay_opsd.storage import CheckpointManager, atomic_json, code_hash, provenance


def check_backend(config: RunConfig, output: Path, *, reference: Path | None = None) -> dict:
    from selfplay_opsd.cli import make_policy

    config = config.model_copy(deep=True)
    config.study_id += "-backend-check"
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
        compared = False
        if reference:
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
            "gpu_count": config.hardware.gpu_count,
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
            atomic_json(output / "backend_report.json", report)
            atomic_json(
                config.output_root / "compatibility" / f"{config.hardware.profile}.json", report
            )
        return report
    finally:
        policy.close()
