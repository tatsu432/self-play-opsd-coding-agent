FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY src/selfplay_opsd/worker.py src/selfplay_opsd/codec.py /opt/
USER 65534:65534
WORKDIR /tmp
ENTRYPOINT ["python", "-I", "-u", "/opt/worker.py"]
