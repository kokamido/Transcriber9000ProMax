FROM nvidia/cuda:13.1.2-cudnn-devel-ubuntu24.04
RUN apt update && apt install -y \
    curl \
    ffmpeg \
    git \
    python3-dev
RUN curl -LsSf https://astral.sh/uv/install.sh | sh