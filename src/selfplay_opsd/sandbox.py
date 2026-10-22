"""Disposable execution. Candidate code never executes in this module/process."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import httpx
import numpy as np

from selfplay_opsd.codec import decode
from selfplay_opsd.config import SandboxConfig
from selfplay_opsd.core import Case, ExperimentError, Status, canonical, digest
from selfplay_opsd.mutations import validate_source
from selfplay_opsd.worker_runtime import LocalRuntime, isolation_probes
from selfplay_opsd.worker_runtime import bounded_process as bounded_process


@dataclass(frozen=True)
class Execution:
    status: Status
    results: list[dict]
    seconds: float
    detail: str = ""


@dataclass(frozen=True)
class Verification:
    status: Status
    passed: bool
    details: list[bool]
    semantic_failures: int
    runtime_failures: int
    seconds: float


class SandboxBackend:
    def __init__(self, config: SandboxConfig):
        self.config = config
        self._local_runtime = LocalRuntime(config)
        self.attestation: dict | None = None
        self._image_id: str | None = None
        self.infrastructure_retries = 0

    def identity(self) -> str:
        if self._image_id is not None:
            return self._image_id
        if self.config.backend in {"docker", "apptainer"}:
            self._image_id = self._local_runtime.identity()
            return self._image_id
        if not self.config.remote_url:
            raise ExperimentError("Remote worker URL is required")
        response = httpx.get(
            self.config.remote_url.rstrip("/") + "/capabilities",
            headers=self._headers(),
            timeout=10,
        )
        response.raise_for_status()
        return response.json()["image_digest"]

    def _headers(self) -> dict:
        token = os.getenv(self.config.remote_token_env)
        if not token:
            raise ExperimentError(
                f"Missing remote worker token in {self.config.remote_token_env}", Status.DEPENDENCY
            )
        return {"Authorization": f"Bearer {token}"}

    def _command(self, name: str) -> list[str]:
        return self._local_runtime.command(name)

    def execute(
        self, source: str, entry_point: str, inputs: list, seconds: float, *, probe: bool = False
    ) -> Execution:
        for attempt in range(self.config.retries + 1):
            result = self._execute_once(source, entry_point, inputs, seconds, probe=probe)
            if result.status != Status.INFRA or attempt == self.config.retries:
                return result
            self.infrastructure_retries += 1
            time.sleep(0.1 * (attempt + 1))
        raise AssertionError("Unreachable retry state")

    def _execute_once(
        self, source: str, entry_point: str, inputs: list, seconds: float, *, probe: bool = False
    ) -> Execution:
        if not probe:
            validate_source(source)
        payload = {"source": source, "entry_point": entry_point, "inputs": inputs}
        raw = canonical(payload).encode()
        if len(raw) > 8 * 1024 * 1024:
            raise ExperimentError("Worker input exceeds admitted task limit", Status.ORACLE)
        if self.config.backend == "remote":
            if not self.config.remote_url:
                raise ExperimentError("Remote URL required")
            try:
                response = httpx.post(
                    self.config.remote_url.rstrip("/") + "/execute",
                    json={"request": payload, "seconds": seconds, "probe": probe},
                    headers=self._headers(),
                    timeout=seconds + 30,
                )
                response.raise_for_status()
                value = response.json()
                return Execution(
                    Status(value["status"]),
                    value["results"],
                    value["seconds"],
                    value.get("detail", ""),
                )
            except httpx.HTTPError as exc:
                return Execution(Status.INFRA, [], 0, str(exc))
        value = self._local_runtime.execute(source, entry_point, inputs, seconds, probe=probe)
        return Execution(
            Status(value["status"]), value["results"], value["seconds"], value["detail"]
        )

    def verify(
        self, source: str, entry_point: str, cases: list[Case], seconds: float
    ) -> Verification:
        result = self.execute(source, entry_point, [c.args for c in cases], seconds)
        if result.status in {Status.INFRA, Status.DEPENDENCY, Status.ORACLE}:
            raise ExperimentError(result.detail, result.status)
        if result.status != Status.OK:
            return Verification(
                result.status, False, [False] * len(cases), 0, len(cases), result.seconds
            )
        checks = [
            compare(case, row["value"]) if row["ok"] else False
            for case, row in zip(cases, result.results, strict=True)
        ]
        runtime = sum(not row["ok"] for row in result.results)
        return Verification(
            Status.OK if all(checks) else Status.STUDENT_FAILURE,
            bool(cases) and all(checks),
            checks,
            sum(not c for c in checks) - runtime,
            runtime,
            result.seconds,
        )

    def preflight(self) -> dict:
        from selfplay_opsd.codec import encode

        image = self.identity()
        probes = isolation_probes(self.config)
        outcomes = {}
        for name, source in probes.items():
            outcome = self.execute(source, "check", [encode([])], 5, probe=True)
            outcomes[name] = (
                outcome.status == Status.OK
                and bool(outcome.results)
                and outcome.results[0].get("value") is True
            )
        self.attestation = {
            "image_digest": image,
            "config_hash": digest(self.config.model_dump()),
            "checks": outcomes,
            "ready": all(outcomes.values()),
            "timestamp": time.time(),
            "threat_model": "OS isolation; expected outputs remain outside candidate process; hidden inputs are observable while executing",
        }
        if not self.attestation["ready"]:
            raise ExperimentError(f"Sandbox isolation preflight failed: {self.attestation}")
        return self.attestation


def compare(case: Case, encoded_output) -> bool:
    """Trusted comparison over safe native data; never call candidate Python methods."""
    try:
        output, expected, args = decode(encoded_output), decode(case.expected), decode(case.args)
        if case.comparison == "non_none":
            return (output if type(output) is bool else output is not None) == expected
        if case.comparison == "set":
            return set(output) == set(expected)
        if case.comparison == "sum_div":
            return output == expected or output == 0
        if case.comparison == "surface_area":
            from evalplus.eval._special_oracle import _surface_Area

            return output == expected or abs(output - _surface_Area(*args)) <= case.atol
        if case.comparison == "digit_distance":
            from evalplus.eval._special_oracle import _digit_distance_nums

            return output == expected or output == _digit_distance_nums(*args)
        if case.comparison == "polynomial_root":
            return abs(sum(c * output**i for i, c in enumerate(args[0]))) <= case.atol
        if output == expected:
            return True
        floats = isinstance(expected, float) or (
            isinstance(expected, (list, tuple))
            and bool(expected)
            and all(isinstance(x, float) for x in expected)
        )
        atol = case.atol or (1e-6 if floats else 0)
        return bool(
            atol
            and type(output) is type(expected)
            and np.allclose(output, expected, rtol=1e-7, atol=atol)
        )
    except (ValueError, TypeError, OverflowError, ArithmeticError):
        return False
