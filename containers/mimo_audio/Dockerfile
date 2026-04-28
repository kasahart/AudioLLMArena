FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive TZ=UTC

# Python 3.12 via deadsnakes PPA (Ubuntu 22.04 ships with 3.10 only)
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common gpg-agent && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3.12-venv \
    git git-lfs curl ffmpeg libsndfile1 libsox-fmt-all sox build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && python3.12 -m ensurepip --upgrade \
    && python3.12 -m pip install --upgrade pip uv

# Install Python dependencies at build time into /opt/venv (outside the
# /workspace volume mount so the host directory does not shadow the venv).
COPY containers/mimo_audio/pyproject.toml /opt/mimo_audio/pyproject.toml

ENV VIRTUAL_ENV=/opt/mimo_audio/.venv \
    PATH="/opt/mimo_audio/.venv/bin:$PATH"

RUN cd /opt/mimo_audio && uv sync --no-dev

WORKDIR /workspace

ENV PYTHONPATH=/workspace \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_CACHE=/root/.cache/huggingface/hub

EXPOSE 8000
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
