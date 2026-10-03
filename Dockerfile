# syntax=docker/dockerfile:1.7
# Tovi GPU worker = FastVideo CUDA image + nunchaku (SVDQuant W4A4) + this repo.
#
# 1) FastVideo base for Blackwell, from a FastVideo checkout at the commit the
#    pod runs (CUDA 13 base, torch 2.12 cu130 -- sm_120 needs CUDA >= 12.8):
#      docker build -f docker/Dockerfile --build-arg TORCH_CUDA_ARCH_LIST=12.0a -t fastvideo:sm120 .
# 2) This image (no GPU at build time, so nunchaku is compiled for all its
#    archs incl. sm_120a; ~1 h on 8 cores, set NUNCHAKU_INSTALL_MODE=FAST only
#    when building on a GPU host):
#      docker build --build-arg FASTVIDEO_IMAGE=fastvideo:sm120 -t tovi-gpu-worker .
# 3) Run, models on the /workspace volume as on the pod:
#      docker run --gpus all -v /workspace:/workspace -e QUANT_MODE=svdq -e GPU_WORKER_MODE=t2va \
#        --network host tovi-gpu-worker
ARG FASTVIDEO_IMAGE=fastvideo:sm120
FROM ${FASTVIDEO_IMAGE}

ARG NUNCHAKU_REF=302e0e97024ebd68688fe890e5df83731edf7b54
ARG NUNCHAKU_INSTALL_MODE=ALL
ARG MAX_JOBS=8

ENV PATH=/opt/venv/bin:${PATH} \
    VIRTUAL_ENV=/opt/venv \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-quant.txt ./
RUN uv pip install --python /opt/venv/bin/python -r requirements.txt -r requirements-quant.txt

# --no-deps: keep FastVideo's diffusers/transformers pins (see requirements-quant.txt).
RUN git clone https://github.com/nunchaku-tech/nunchaku.git /tmp/nunchaku && \
    cd /tmp/nunchaku && git checkout ${NUNCHAKU_REF} && \
    git submodule update --init --recursive --depth 1 && \
    NUNCHAKU_INSTALL_MODE=${NUNCHAKU_INSTALL_MODE} MAX_JOBS=${MAX_JOBS} \
      /opt/venv/bin/python -m pip install --no-build-isolation --no-deps . && \
    rm -rf /tmp/nunchaku

COPY tovi_quant ./tovi_quant
COPY scripts ./scripts
COPY gpu_worker.py main.py ./

# QUANT_MODE=bf16 keeps the current production behaviour.
ENV QUANT_MODE=bf16 \
    GPU_WORKER_MODE=t2va \
    QUANT_CACHE_DIR=/workspace/quant_cache
CMD ["python", "gpu_worker.py"]
