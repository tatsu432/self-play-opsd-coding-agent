FROM python:3.12.12-slim-bookworm@sha256:2986c55feb36e6cae00fa1fefb454283e4b33f35e75ff8bdd123b134130be301
USER root
WORKDIR /opt/task19
RUN mkdir -p /workspace /data/prepared /runs /cache/huggingface
RUN python -m pip install --no-cache-dir uv==0.8.15
COPY docker/task19/pyproject.toml docker/task19/uv.lock ./
RUN uv sync --frozen --no-cache --python /usr/local/bin/python3
ENV PATH="/opt/task19/.venv/bin:${PATH}" PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=/workspace/src
ENTRYPOINT ["python"]
