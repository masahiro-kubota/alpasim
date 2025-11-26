# Example command (run from repo root):
#   docker build --secret id=netrc,src=$HOME/.netrc -t alpasim_base:latest -f Dockerfile .

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY . /repo

# Configure uv
ENV UV_LINK_MODE=copy

# Compile protos
WORKDIR /repo/src/grpc
RUN --mount=type=secret,id=netrc,target=/root/.netrc \
    --mount=type=cache,target=/root/.cache/uv \
    NETRC=/root/.netrc uv sync
RUN uv run compile-protos --no-sync

WORKDIR /repo

RUN --mount=type=secret,id=netrc,target=/root/.netrc \
    --mount=type=cache,target=/root/.cache/uv \
    NETRC=/root/.netrc uv sync --all-packages

# Note: maglev.av has name collisions with PyAV (both use `import av`).
# Patch torchvision to trigger its "av not available" fallback path.
RUN for f in .venv/lib/python*/site-packages/torchvision/io/video.py \
             .venv/lib/python*/site-packages/torchvision/io/video_reader.py; do \
        [ -f "$f" ] && sed -i 's/import av$/raise ImportError("maglev.av collision")/' "$f"; \
    done || true

ENV UV_CACHE_DIR=/tmp/uv-cache
ENV UV_NO_SYNC=1


#ENTRYPOINT ["uv", "run", "python", "-m", "alpasim_controller.server"]
