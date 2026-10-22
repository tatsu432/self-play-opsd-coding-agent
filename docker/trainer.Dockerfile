FROM vllm/vllm-openai:v0.11.0@sha256:014a95f21c9edf6abe0aea6b07353f96baa4ec291c427bb1176dc7c93a85845c
USER root
WORKDIR /opt/selfplay
RUN mkdir -p /workspace /data/prepared /runs /cache/huggingface
RUN python3 -m pip install --no-cache-dir uv==0.8.15
COPY pyproject.toml uv.lock README.md docs/methods.md LICENSE THIRD_PARTY_NOTICES.md ./
COPY src ./src
COPY configs ./configs
RUN uv sync --frozen --group gpu --no-dev --python 3.12.12
ENV PATH="/opt/selfplay/.venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=/workspace/src
ENTRYPOINT ["selfplay"]
