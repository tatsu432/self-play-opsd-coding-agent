FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c
RUN python -m pip install --no-cache-dir uv==0.8.15
WORKDIR /opt/selfplay
COPY pyproject.toml uv.lock README.md METHODS.md LICENSE THIRD_PARTY_NOTICES.md ./
COPY src ./src
COPY configs ./configs
RUN uv sync --frozen --no-dev
ENV PATH="/opt/selfplay/.venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/workspace/src
ENTRYPOINT ["selfplay"]
