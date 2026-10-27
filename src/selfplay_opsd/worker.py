"""Container-only untrusted execution entry point. Do not call in the trainer.

Only source, entry point, and inputs enter this process. Expected outputs never do.
The OS/container boundary, not this harness, is the security boundary.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

# The worker image contains only these two trusted scripts plus stdlib.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from codec import decode, encode  # type: ignore[import-not-found]  # noqa: E402


def main():
    if os.environ.get("SELFPLAY_PROCESS_POLICY", "pids") == "single_process":
        from worker_policy import install  # type: ignore[import-not-found]

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        disk_bytes = int(os.environ["SELFPLAY_DISK_BYTES"])
        resource.setrlimit(resource.RLIMIT_FSIZE, (disk_bytes, disk_bytes))
        install()
    raw = sys.stdin.buffer.read(8 * 1024 * 1024 + 1)
    if len(raw) > 8 * 1024 * 1024:
        raise ValueError("Worker request exceeds 8 MiB")
    request = json.loads(raw)
    if set(request) != {"source", "entry_point", "inputs"}:
        raise ValueError("Invalid worker request")
    namespace: dict[str, Any] = {"__name__": "solution"}
    # Capture references before candidate code executes. This does not make
    # arbitrary Python cryptographically unable to inspect its own process.
    dumps, write, clock, pack, unpack = json.dumps, os.write, time.monotonic, encode, decode
    rows = []
    started = clock()
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec(compile(request["source"], "src/solution.py", "exec"), namespace)
            function = namespace[request["entry_point"]]
            for arguments in request["inputs"]:
                start = clock()
                try:
                    value = function(*unpack(arguments))
                    result = {"ok": True, "value": pack(value), "seconds": clock() - start}
                except BaseException as exc:
                    result = {"ok": False, "error": type(exc).__name__, "seconds": clock() - start}
                rows.append(result)
        output = {"status": "ok", "results": rows, "seconds": clock() - started}
    except BaseException as exc:
        output = {
            "status": "candidate_error",
            "error": type(exc).__name__,
            "results": rows,
            "seconds": clock() - started,
        }
    encoded = dumps(output, allow_nan=False, separators=(",", ":")).encode()
    if len(encoded) > 4 * 1024 * 1024:
        encoded = b'{"status":"output_limit","results":[]}'
    write(1, encoded + b"\n")


if __name__ == "__main__":
    main()
