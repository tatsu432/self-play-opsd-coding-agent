import os
from pathlib import Path

import pytest
import torch


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or not os.getenv("SELFPLAY_RUN_GPU_TESTS"),
    reason="Explicit allocated CUDA integration run required",
)
def test_production_backend_acceptance(tmp_path):
    from selfplay_opsd.backend_check import check_backend
    from selfplay_opsd.config import load_config

    config = load_config(
        Path("configs/main.yaml"),
        hardware_profile=os.getenv("SELFPLAY_GPU_PROFILE", "single_gpu_80gb"),
    )
    reference = os.getenv("SELFPLAY_GPU_REFERENCE")
    report = check_backend(config, tmp_path, reference=Path(reference) if reference else None)
    assert report["ready"] and report["two_adapter_syncs"] and report["checkpoint_roundtrip"]
