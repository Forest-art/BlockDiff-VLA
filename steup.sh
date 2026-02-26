#!/usr/bin/env bash
set -euo pipefail

# BlockDiff-VLA one-shot environment setup script.
# Usage:
#   bash steup.sh
#
# Optional env vars:
#   PYTHON_BIN=python3.10
#   VENV_DIR=venv_blockdiff
#   TORCH_CUDA_TAG=cu121        # cu121 | cu118 | cpu
#   INSTALL_DEEPSPEED=0         # 1 to install deepspeed
#   INSTALL_CALVIN_ENV=0        # 1 to install calvin_env/tacto for online rollout
#   CALVIN_REPO_DIR=$HOME/src/calvin
#   RUN_SMOKE=0                 # 1 to run minimal smoke test after install

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-venv_blockdiff}"
TORCH_CUDA_TAG="${TORCH_CUDA_TAG:-cu121}"
INSTALL_DEEPSPEED="${INSTALL_DEEPSPEED:-0}"
INSTALL_CALVIN_ENV="${INSTALL_CALVIN_ENV:-0}"
CALVIN_REPO_DIR="${CALVIN_REPO_DIR:-$HOME/src/calvin}"
RUN_SMOKE="${RUN_SMOKE:-0}"

echo "[INFO] repo_root=$REPO_ROOT"
echo "[INFO] python_bin=$PYTHON_BIN"
echo "[INFO] venv_dir=$VENV_DIR"
echo "[INFO] torch_cuda_tag=$TORCH_CUDA_TAG"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[ERROR] Python not found: $PYTHON_BIN"
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[INFO] creating venv: $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

case "$TORCH_CUDA_TAG" in
  cu121)
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
    ;;
  cu118)
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu118"
    ;;
  cpu)
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
    ;;
  *)
    echo "[ERROR] Unsupported TORCH_CUDA_TAG=$TORCH_CUDA_TAG (expected: cu121|cu118|cpu)"
    exit 1
    ;;
esac

echo "[INFO] installing torch from $TORCH_INDEX_URL"
python -m pip install torch==2.2.1 torchvision==0.17.1 --index-url "$TORCH_INDEX_URL"

echo "[INFO] installing minimal dependencies"
python -m pip install -r requirements_minimal.txt \
  omegaconf \
  accelerate==0.21.0 \
  tensorboard \
  lightning \
  hydra-core \
  termcolor \
  gym==0.26.2 \
  sentencepiece

echo "[INFO] installing common runtime packages"
python -m pip install \
  moviepy \
  imageio \
  imageio-ffmpeg \
  wandb \
  opencv-python-headless \
  pybullet

if [[ "$INSTALL_DEEPSPEED" == "1" ]]; then
  echo "[INFO] installing deepspeed"
  python -m pip install "deepspeed>=0.18,<0.19" colorama packaging
fi

if [[ "$INSTALL_CALVIN_ENV" == "1" ]]; then
  echo "[INFO] installing CALVIN env into current venv"
  mkdir -p "$(dirname "$CALVIN_REPO_DIR")"
  if [[ ! -d "$CALVIN_REPO_DIR/.git" ]]; then
    git clone --depth 1 --recurse-submodules https://github.com/mees/calvin.git "$CALVIN_REPO_DIR"
  fi
  python -m pip install -e "$CALVIN_REPO_DIR/calvin_env/tacto"
  python -m pip install -e "$CALVIN_REPO_DIR/calvin_env"
fi

if [[ "$RUN_SMOKE" == "1" ]]; then
  echo "[INFO] running minimal smoke test"
  PYTHONPATH="$REPO_ROOT" python scripts/smoke_upvla_minimal.py \
    --dataset-path ./dummy_data/calvin_processed_training
fi

echo
echo "[DONE] Environment setup complete."
echo "Activate with:"
echo "  source \"$VENV_DIR/bin/activate\""
echo
echo "Recommended runtime env:"
echo "  export PYTHONPATH=\"$REPO_ROOT\""
echo "  export TOKENIZERS_PARALLELISM=false"
if [[ "$INSTALL_CALVIN_ENV" == "1" ]]; then
  echo "  export EGL_VISIBLE_DEVICES=0"
fi
