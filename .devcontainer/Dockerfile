# NVIDIA CUDA 12.8 base image with cuDNN for GPU support
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    python3.11-distutils \
    sudo \
    git \
    git-lfs \
    curl \
    wget \
    ffmpeg \
    libsndfile1 \
    libsox-fmt-all \
    sox \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Create a non-root user for the devcontainer session.
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} \
    && usermod -aG sudo,audio,video ${USERNAME} \
    && echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

# Upgrade pip and install uv
RUN python3 -m pip install --upgrade pip setuptools wheel uv

# Note: the project virtual environment is provisioned at container startup
# to allow using `/workspace/venv` (bind-mounted workspace). The postCreateCommand
# in devcontainer.json will run `uv sync` to create `/workspace/venv` after the
# workspace is mounted.

# Set working directory
WORKDIR /workspace

# Set environment variables for GPU inference
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONPATH=/workspace

USER ${USERNAME}
