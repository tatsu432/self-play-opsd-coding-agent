"""Small, verified Dockerfile-to-Apptainer build inputs; no image export on the controller."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import shlex
import subprocess
from pathlib import Path, PurePosixPath

from selfplay_opsd.core import digest
from selfplay_opsd.storage import file_hash

CONVERTER_VERSION = "0.3.14"
BUILD_CHECK_DOCKERFILE = """FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c
USER root
WORKDIR /opt/selfplay
COPY README.md ./
RUN test -s README.md && test "$PYTHON_VERSION" = 3.12.12 && printf 'build-ok' > /opt/selfplay/witness
ENV SELFPLAY_BUILD_WITNESS=verified
ENTRYPOINT ["python", "-c", "import os,pathlib; assert os.environ['SELFPLAY_BUILD_WITNESS']=='verified'; assert pathlib.Path('/opt/selfplay/witness').read_text()=='build-ok'; assert pathlib.Path('/opt/selfplay/README.md').is_file()"]
"""


def base_environment(dockerfile: Path) -> list[str]:
    """Read immutable registry configuration only; never pull large image layers."""
    base = next(
        line.split(maxsplit=1)[1]
        for line in dockerfile.read_text().splitlines()
        if line.startswith("FROM ")
    )
    if "@sha256:" not in base:
        raise ValueError("Base image must be digest pinned")
    result = subprocess.run(
        ["docker", "buildx", "imagetools", "inspect", base, "--format", "{{json .Image}}"],
        text=True,
        capture_output=True,
        check=True,
        timeout=180,
    )
    configuration = json.loads(result.stdout)
    configuration = configuration.get("linux/amd64", configuration)
    if configuration.get("architecture") != "amd64" or configuration.get("os") != "linux":
        raise ValueError("Base registry configuration is not Linux AMD64")
    return configuration["config"]["Env"]


def source_build(
    root: Path, dockerfile: Path, commit: str, *, base_env: list[str], check: bool = False
) -> dict:
    from spython.main.parse.parsers import DockerParser
    from spython.main.parse.writers import SingularityWriter

    if importlib.metadata.version("spython") != CONVERTER_VERSION:
        raise ValueError("Install pinned build tooling: uv sync --frozen --group build")
    instructions = []
    for line in dockerfile.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        instruction, _, value = line.strip().partition(" ")
        if instruction not in {"FROM", "USER", "WORKDIR", "COPY", "RUN", "ENV", "ENTRYPOINT"}:
            raise ValueError(f"Unsupported Dockerfile instruction: {instruction}")
        if line.endswith("\\"):
            raise ValueError("Multiline Dockerfile conversion requires explicit validation")
        instructions.append((instruction, value))
    names = [name for name, _ in instructions]
    if any(names.count(name) != 1 for name in ("FROM", "USER", "WORKDIR", "ENTRYPOINT")):
        raise ValueError("Source builds require one base, root USER, WORKDIR and ENTRYPOINT")
    if next(value for name, value in instructions if name == "USER") != "root":
        raise ValueError(
            "Only root build USER is supported; candidate workers use a separate image"
        )
    # COPY/RUN/ENV must follow the sole WORKDIR; reject scope changes instead of
    # silently relying on spython's lossy representation of Docker instructions.
    if any(name in {"COPY", "RUN", "ENV"} for name in names[: names.index("WORKDIR")]):
        raise ValueError("Build instructions must follow WORKDIR")
    if "ENV" in names and "RUN" in names[names.index("ENV") :]:
        raise ValueError("ENV preceding RUN requires explicit conversion support")
    base = next(value for name, value in instructions if name == "FROM")
    if not re.fullmatch(r"[A-Za-z0-9_./:-]+@sha256:[0-9a-f]{64}", base):
        raise ValueError("Build base must be pinned by digest")
    workdir = PurePosixPath(next(value for name, value in instructions if name == "WORKDIR"))
    if not workdir.is_absolute() or ".." in workdir.parts:
        raise ValueError("Build WORKDIR must be absolute")
    context: dict[str, str] = {}
    copies = []
    for name, value in instructions:
        if name != "COPY":
            continue
        parts = shlex.split(value)
        if len(parts) < 2 or any(not re.fullmatch(r"[A-Za-z0-9_./-]+", p) for p in parts):
            raise ValueError("Unsupported COPY syntax")
        destination = PurePosixPath(parts[-1])
        if ".." in destination.parts:
            raise ValueError("COPY destination escapes build directory")
        destination = destination if destination.is_absolute() else workdir / destination
        for source in parts[:-1]:
            path = root / source
            if Path(source).is_absolute() or ".." in Path(source).parts or not path.exists():
                raise ValueError("COPY source is outside the immutable context or missing")
            files = sorted(path.rglob("*")) if path.is_dir() else [path]
            for item in files:
                if (
                    "__pycache__" in item.parts
                    or item.suffix == ".pyc"
                    or any(p.endswith(".egg-info") for p in item.parts)
                ):
                    continue
                if item.is_symlink() or not item.resolve().is_relative_to(root.resolve()):
                    raise ValueError("Symlinks are forbidden in source build context")
                if item.is_file():
                    context[str(item.relative_to(root))] = file_hash(item)
            # Apptainer %files does not inherit Docker WORKDIR. Resolve each
            # destination explicitly, including Docker's multi-source COPY rule.
            target = destination
            if not path.is_dir() and (
                len(parts) > 2 or parts[-1].endswith("/") or parts[-1] == "."
            ):
                target /= path.name
            copies.append([source, str(target)])
    parser = DockerParser(str(dockerfile))
    recipe = next(iter(parser.recipe.values()))
    # Yale's Apptainer OCI transport rejects tag@digest although Docker accepts
    # it. Remove the informational tag, retaining the exact immutable digest.
    image, pinned_digest = base.rsplit("@", 1)
    if ":" in image.rsplit("/", 1)[-1]:
        image = image.rsplit(":", 1)[0]
    recipe.fromHeader = f"{image}@{pinned_digest}"
    entrypoint = json.loads(next(value for name, value in instructions if name == "ENTRYPOINT"))
    if (
        not isinstance(entrypoint, list)
        or not entrypoint
        or not all(isinstance(arg, str) for arg in entrypoint)
    ):
        raise ValueError("ENTRYPOINT must use the JSON argument-array form")
    # spython joins exec-form arguments without shell quoting. Quote each
    # original argument so spaces/semicolons cannot change runscript semantics.
    recipe.entrypoint = [shlex.quote(arg) for arg in entrypoint]
    recipe.files = copies
    recipe.install = [step for step in recipe.install if not step.startswith("USER ")]
    definition = SingularityWriter(parser.recipe).convert()
    # A failure in an early RUN must never be hidden by a successful later RUN.
    environment = []
    for assignment in base_env:
        name, separator, value = assignment.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError("Invalid base environment assignment")
        environment.append(f"export {name}={shlex.quote(value)}")
    definition = definition.replace(
        "%post\n", "%post\nset -eu\n" + "\n".join(environment) + "\n", 1
    )
    metadata = {
        "schema_version": 2,
        "kind": "build-check" if check else "trainer",
        "build_mode": "yale",
        "platform": "linux/amd64",
        "source_commit": commit,
        "dockerfile_sha256": file_hash(dockerfile),
        "lock_sha256": file_hash(root / "uv.lock"),
        "converter": {"name": "spython", "version": CONVERTER_VERSION},
        "base_image": base,
        "bootstrap_reference": recipe.fromHeader,
        "base_environment": base_env,
        "context_files": context,
        "definition_sha256": hashlib.sha256(definition.encode()).hexdigest(),
    }
    return {"definition": definition, "image_metadata": metadata, "build_key": digest(metadata)}
