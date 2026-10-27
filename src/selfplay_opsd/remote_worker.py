"""Authenticated bounded worker service for hosts without safe local isolation.

Run on a dedicated worker host. Bind loopback and use SSH forwarding, or terminate
TLS at an explicitly configured proxy. The service never accepts mounts/commands.
"""

from __future__ import annotations

import argparse
import hmac
import json
import math
import os
import signal
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from selfplay_opsd.worker_runtime import LocalRuntime, RuntimeConfig, canonical, publish_json


def serve(
    config,
    host: str = "127.0.0.1",
    port: int = 8765,
    concurrency: int = 4,
    *,
    ready_file: Path | None = None,
    report: Path | None = None,
    check_only: bool = False,
):
    config = RuntimeConfig.from_values(config.model_dump())
    if config.resource_backend == "slurm" and concurrency != 1:
        raise ValueError("A Slurm worker step must have concurrency one")
    token = os.getenv(config.remote_token_env)
    if not token or len(token) < 24:
        raise ValueError(
            "Worker token must be supplied through environment and contain >=24 characters"
        )
    if host not in {"127.0.0.1", "localhost"} or not 1 <= concurrency <= 32:
        raise ValueError("Worker must bind loopback with concurrency 1 through 32")
    cancelled = threading.Event()
    sandbox = LocalRuntime(config, cancelled=cancelled)
    attestation = sandbox.preflight()
    if report:
        publish_json(report, attestation)
    if not attestation["ready"]:
        raise ValueError("Worker isolation failed: " + canonical(attestation))
    if check_only:
        return attestation
    semaphore = threading.BoundedSemaphore(concurrency)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):
            # Never log request bodies, source, tokens or Authorization headers.
            return None

        def respond(self, code, value):
            data = canonical(value).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def authorized(self):
            header = self.headers.get("Authorization", "")
            return hmac.compare_digest(header, f"Bearer {token}")

        def do_GET(self):
            if not self.authorized():
                self.respond(401, {"error": "unauthorized"})
            elif self.path == "/capabilities":
                self.respond(
                    200,
                    attestation | {"concurrency": concurrency},
                )
            else:
                self.respond(404, {"error": "not_found"})

        def do_POST(self):
            if not self.authorized():
                self.respond(401, {"error": "unauthorized"})
                return
            if self.path != "/execute":
                self.respond(404, {"error": "not_found"})
                return
            if not semaphore.acquire(blocking=False):
                self.respond(503, {"error": "worker_busy"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 8 * 1024 * 1024:
                    raise ValueError("Invalid request length")
                self.connection.settimeout(10)
                body = json.loads(self.rfile.read(length))
                seconds = float(body["seconds"])
                if not math.isfinite(seconds) or not 0 < seconds <= config.verifier_seconds:
                    raise ValueError("Worker time limit exceeded")
                request = body["request"]
                if set(request) != {"source", "entry_point", "inputs"}:
                    raise ValueError("Invalid execution request")
                result = sandbox.execute(**request, seconds=seconds, probe=body.get("probe", False))
                self.respond(
                    200,
                    {
                        "status": result["status"],
                        "results": result["results"],
                        "seconds": result["seconds"],
                        "detail": result["detail"],
                    },
                )
            except (ValueError, KeyError, TypeError) as exc:
                self.respond(400, {"error": str(exc)[:200]})
            finally:
                semaphore.release()

    with ThreadingHTTPServer((host, port), Handler) as server:
        if ready_file:
            publish_json(
                ready_file, {"url": f"http://127.0.0.1:{server.server_port}", "ready": True}
            )
        previous_handlers = {}

        def terminate(*_):
            cancelled.set()
            threading.Thread(target=server.shutdown, daemon=True).start()

        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGTERM, signal.SIGINT):
                previous_handlers[sig] = signal.signal(sig, terminate)
        try:
            server.serve_forever(poll_interval=0.5)
        finally:
            cancelled.set()
            for sig, previous in previous_handlers.items():
                signal.signal(sig, previous)
            if ready_file:
                ready_file.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--image", default="selfplay-worker:local")
    parser.add_argument("--config-json", default="{}")
    parser.add_argument("--backend", choices=("docker", "apptainer"), default="docker")
    parser.add_argument("--runtime", default=None)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    config = RuntimeConfig.from_values(
        json.loads(args.config_json)
        | {"image": args.image, "backend": args.backend, "runtime": args.runtime or args.backend}
    )
    serve(
        config,
        args.host,
        args.port,
        args.concurrency,
        report=args.report,
        ready_file=args.ready_file,
        check_only=args.check_only,
    )


if __name__ == "__main__":
    main()
