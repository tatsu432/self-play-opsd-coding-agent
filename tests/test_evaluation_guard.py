import socket
import subprocess
import sys


def test_base_adapter_guard_checks_real_dtensor_shards_on_every_rank(tmp_path):
    script = tmp_path / "probe.py"
    script.write_text("""from types import SimpleNamespace
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard
from selfplay_opsd.core import ExperimentError
from selfplay_opsd.evaluation import verify_base_adapter

dist.init_process_group("gloo")
mesh = init_device_mesh("cpu", (2,))
local = torch.zeros(1, 2)
weight = DTensor.from_local(local, mesh, [Shard(0)], run_check=False,
    shape=torch.Size([2, 2]), stride=(2, 1))
policy = SimpleNamespace(device="cpu", distributed=True,
    named_parameters=lambda role: [("layer.lora_A.student.weight", torch.ones(2, 2)),
                                   ("layer.lora_B.student.weight", weight)])
verify_base_adapter(policy)
# Rank zero still has zero local weights, but must also reject the adapter.
if dist.get_rank() == 1:
    local[0, 0] = 1
try:
    verify_base_adapter(policy)
except ExperimentError:
    pass
else:
    raise AssertionError("Nonzero remote shard was accepted")
dist.barrier()
dist.destroy_process_group()
""")
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--master-addr=127.0.0.1",
            f"--master-port={port}",
            "--nproc-per-node=2",
            str(script),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
