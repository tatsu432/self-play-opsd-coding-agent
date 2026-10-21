"""Authenticated bounded worker service for hosts without safe local isolation.

Run on a dedicated worker host. Bind loopback and use SSH forwarding, or terminate
TLS at an explicitly configured proxy. The service never accepts mounts/commands.
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from selfplay_opsd.config import SandboxConfig
from selfplay_opsd.core import canonical
from selfplay_opsd.sandbox import SandboxBackend


def serve(config: SandboxConfig, host: str = "127.0.0.1", port: int = 8765, concurrency: int = 4):
    token = os.getenv(config.remote_token_env)
    if not token or len(token) < 24:
        raise ValueError(
            "Worker token must be supplied through environment and contain >=24 characters"
        )
    if config.backend != "docker":
        raise ValueError("Remote service executes only its configured Docker sandbox")
    sandbox = SandboxBackend(config)
    attestation = sandbox.preflight()
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
                    attestation
                    | {"limits": config.model_dump(mode="json"), "concurrency": concurrency},
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
                if not 0 < seconds <= config.verifier_seconds:
                    raise ValueError("Worker time limit exceeded")
                request = body["request"]
                if set(request) != {"source", "entry_point", "inputs"}:
                    raise ValueError("Invalid execution request")
                result = sandbox.execute(**request, seconds=seconds, probe=body.get("probe", False))
                self.respond(
                    200,
                    {
                        "status": result.status.value,
                        "results": result.results,
                        "seconds": result.seconds,
                        "detail": result.detail,
                    },
                )
            except (ValueError, KeyError, TypeError) as exc:
                self.respond(400, {"error": str(exc)[:200]})
            finally:
                semaphore.release()

    with ThreadingHTTPServer((host, port), Handler) as server:
        server.daemon_threads = True
        server.serve_forever(poll_interval=0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--image", default="selfplay-worker:local")
    args = parser.parse_args()
    serve(SandboxConfig(image=args.image), args.host, args.port, args.concurrency)


if __name__ == "__main__":
    main()
