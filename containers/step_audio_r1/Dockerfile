FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive TZ=UTC

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && python3 -m pip install --upgrade pip uv

# Install Python dependencies at build time outside the /workspace bind mount.
COPY containers/step_audio_r1/pyproject.toml /opt/step_audio_r1/pyproject.toml
COPY containers/step_audio_r1/uv.lock        /opt/step_audio_r1/uv.lock

ENV VIRTUAL_ENV=/opt/step_audio_r1/.venv \
    PATH="/opt/step_audio_r1/.venv/bin:$PATH"

RUN cd /opt/step_audio_r1 && uv sync --frozen --no-dev

WORKDIR /workspace

ENV PYTHONPATH=/workspace \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_CACHE=/root/.cache/huggingface/hub

EXPOSE 8000
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
