#!/usr/bin/env bash
# Build nunchaku (SVDQuant W4A4 kernels) into the GPU-worker venv on the pod.
# Run on the pod with the GPU visible: the build targets only the local GPU
# arch (sm_120a on RTX PRO 6000), which takes ~10-20 min instead of ~1 h.
#
#   PYTHON_BIN=/workspace/venv_max/bin/python bash scripts/install_quant.sh
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/workspace/venv_max/bin/python}
NUNCHAKU_REF=${NUNCHAKU_REF:-302e0e97024ebd68688fe890e5df83731edf7b54}
SRC_DIR=${SRC_DIR:-/workspace/src/nunchaku}
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)

echo "== python: $PYTHON_BIN"
"$PYTHON_BIN" - <<'PY'
import torch
assert torch.cuda.is_available(), "no CUDA device visible"
cap = torch.cuda.get_device_capability()
print(f"torch {torch.__version__} cuda {torch.version.cuda} gpu {torch.cuda.get_device_name()} sm_{cap[0]}{cap[1]}")
if cap >= (12, 0):
    major, minor = map(int, torch.version.cuda.split(".")[:2])
    assert (major, minor) >= (12, 8), "Blackwell needs a torch build with CUDA >= 12.8"
PY

NVCC=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
command -v "$NVCC" >/dev/null || NVCC=$(command -v nvcc)
echo "== nvcc: $("$NVCC" --version | tail -1)"

if [ ! -d "$SRC_DIR/.git" ]; then
  git clone https://github.com/nunchaku-tech/nunchaku.git "$SRC_DIR"
fi
git -C "$SRC_DIR" fetch --depth 1 origin "$NUNCHAKU_REF"
git -C "$SRC_DIR" checkout --force "$NUNCHAKU_REF"
git -C "$SRC_DIR" submodule update --init --recursive --depth 1

echo "== building nunchaku @ $NUNCHAKU_REF (NUNCHAKU_INSTALL_MODE=${NUNCHAKU_INSTALL_MODE:-FAST})"
(cd "$SRC_DIR" && \
  CUDA_HOME=$(dirname "$(dirname "$NVCC")") \
  NUNCHAKU_INSTALL_MODE=${NUNCHAKU_INSTALL_MODE:-FAST} \
  MAX_JOBS=${MAX_JOBS:-$(nproc)} \
  "$PYTHON_BIN" -m pip install --no-build-isolation --no-deps -v .)

"$PYTHON_BIN" -m pip install -r "$REPO_DIR/requirements-quant.txt"

echo "== smoke import"
(cd "$REPO_DIR" && "$PYTHON_BIN" -c "from tovi_quant import nunchaku_backend as nb; nb._ops(); print('nunchaku kernels OK')")
echo "== next: (cd $REPO_DIR && $PYTHON_BIN -m tovi_quant.doctor)"
