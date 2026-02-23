#!/usr/bin/env bash
#SBATCH --job-name=blockdiff_quick_ar_ds
#SBATCH --partition=preempt
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=/project/peilab/luxiaocheng/projects/BlockDiff-VLA/logs/blockdiff_quick_ar_ds_%j.out
#SBATCH --error=/project/peilab/luxiaocheng/projects/BlockDiff-VLA/logs/blockdiff_quick_ar_ds_%j.err

set -euo pipefail

REPO=/project/peilab/luxiaocheng/projects/BlockDiff-VLA
VENV=${VENV:-/home/xlubl/.venv_blockdiff_pip}
ENV_MODE=${ENV_MODE:-auto}        # auto | venv | system
PIP_USER=${PIP_USER:-1}           # only used with system env
INSTALL_COLORAMA=${INSTALL_COLORAMA:-1}
INSTALL_DEEPSPEED=${INSTALL_DEEPSPEED:-1}
DEEPSPEED_SPEC=${DEEPSPEED_SPEC:-deepspeed>=0.18,<0.19}
TRAIN_STEPS=${TRAIN_STEPS:-3}
DATASET_ROOT=${DATASET_ROOT:-/project/peilab/luxiaocheng/projects/calvin/dataset/calvin_debug_dataset}
RUN_ROOT=${RUN_ROOT:-/home/xlubl/blockdiff_quick_ar_ds_${SLURM_JOB_ID}}
ACCEL_CONFIG=${ACCEL_CONFIG:-$REPO/accelerate_configs/1_gpu_deepspeed_zero2.yaml}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29666}
TAG=quick_arvla_ds
OUT_DIR="$RUN_ROOT/outputs/$TAG"
MODEL_CFG="$RUN_ROOT/policy_rollout/${TAG}_model.yaml"
EVAL_JSON="$RUN_ROOT/results/${TAG}_offline_eval.json"

mkdir -p "$RUN_ROOT" "$RUN_ROOT/outputs" "$RUN_ROOT/logs" "$RUN_ROOT/results" "$RUN_ROOT/policy_rollout"

USED_VENV=0
if [[ "$ENV_MODE" == "venv" ]]; then
  if [[ ! -f "$VENV/bin/activate" ]]; then
    echo "[ERROR] ENV_MODE=venv but missing $VENV/bin/activate"
    exit 1
  fi
  source "$VENV/bin/activate"
  USED_VENV=1
elif [[ "$ENV_MODE" == "auto" && -f "$VENV/bin/activate" ]]; then
  source "$VENV/bin/activate"
  USED_VENV=1
else
  echo "[INFO] using current python env (ENV_MODE=$ENV_MODE)"
fi

export PYTHONPATH="$REPO"
export TOKENIZERS_PARALLELISM=false

# Some clusters expose non-standard nvcc output at /usr/bin/nvcc.
# Provide a local CUDA_HOME shim so DeepSpeed can parse `nvcc -V`.
TORCH_CUDA_VERSION=$(python - <<'PY'
import torch
print(torch.version.cuda or "12.0")
PY
)
TORCH_CUDA_MAJOR="${TORCH_CUDA_VERSION%%.*}"
TORCH_CUDA_MINOR="$(echo "$TORCH_CUDA_VERSION" | cut -d. -f2)"
FAKE_CUDA_HOME="$RUN_ROOT/fake_cuda_home"
mkdir -p "$FAKE_CUDA_HOME/bin"
cat > "$FAKE_CUDA_HOME/bin/nvcc" <<EOF
#!/usr/bin/env bash
if [[ "\${1:-}" == "-V" || "\${1:-}" == "--version" ]]; then
  echo "Cuda compilation tools, release ${TORCH_CUDA_MAJOR}.${TORCH_CUDA_MINOR}, V${TORCH_CUDA_MAJOR}.${TORCH_CUDA_MINOR}.0"
  exit 0
fi
exec /usr/bin/nvcc "\$@"
EOF
chmod +x "$FAKE_CUDA_HOME/bin/nvcc"
export CUDA_HOME="$FAKE_CUDA_HOME"

PIP_FLAGS=()
if [[ "$USED_VENV" == "0" && "$PIP_USER" == "1" ]]; then
  PIP_FLAGS+=(--user)
  export PATH="$HOME/.local/bin:$PATH"
fi

if ! python - <<'PY'
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("colorama") else 1)
PY
then
  if [[ "$INSTALL_COLORAMA" != "1" ]]; then
    echo "[ERROR] colorama is not installed and INSTALL_COLORAMA=0"
    exit 1
  fi
  python -m pip install "${PIP_FLAGS[@]}" colorama
fi

if ! python - <<'PY'
import sys
try:
    import deepspeed
    from packaging.version import Version
except Exception:
    sys.exit(1)
v = Version(deepspeed.__version__)
sys.exit(0 if Version("0.18.0") <= v < Version("0.19.0") else 1)
PY
then
  if [[ "$INSTALL_DEEPSPEED" != "1" ]]; then
    echo "[ERROR] deepspeed is missing/incompatible and INSTALL_DEEPSPEED=0"
    exit 1
  fi
  python -m pip install "${PIP_FLAGS[@]}" "${DEEPSPEED_SPEC}"
fi

if [[ ! -f "$REPO/debug_data/calvin_debug_processed_training/dataset_info.json" ]]; then
  python "$REPO/scripts/prepare_calvin_debug_processed.py" \
    --source-split-dir "$DATASET_ROOT/training" \
    --output-dir "$REPO/debug_data/calvin_debug_processed_training" \
    --instruction "debug instruction" \
    --max-frames 256 \
    --start-index 0
fi

accelerate launch \
  --config_file "$ACCEL_CONFIG" \
  --main_process_port "$MAIN_PROCESS_PORT" \
  "$REPO/train_upvla.py" \
  "config=$REPO/config/debug_arvla_1step.yaml" \
  "experiment.tracker=tensorboard" \
  "training.max_train_steps=$TRAIN_STEPS" \
  "experiment.output_dir=$OUT_DIR" \
  "experiment.name=$TAG" \
  "dataset.params.train_pre_shards_path_or_url=$REPO/debug_data/calvin_debug_processed_training" \
  "dataset.params.train_mmu_shards_path_or_url=$REPO/dummy_data/llava_tuning_665k_data"

CKPT_DIR="$(ls -d "$OUT_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -n 1)"
if [[ -z "${CKPT_DIR:-}" ]]; then
  echo "[ERROR] no checkpoint found under $OUT_DIR"
  exit 1
fi

python - <<PY
from omegaconf import OmegaConf
cfg = OmegaConf.load("$REPO/policy_rollout/debug_arvla_model.yaml")
cfg.model.showo.tuned_model_path = "$CKPT_DIR"
OmegaConf.save(cfg, "$MODEL_CFG")
print("wrote", "$MODEL_CFG")
PY

python "$REPO/scripts/eval_upvla_offline_actions.py" \
  --dataset-root "$DATASET_ROOT" \
  --split validation \
  --model-config "$MODEL_CFG" \
  --device cuda:0 \
  --start-only \
  --max-samples 1 \
  --clip-action \
  --output-json "$EVAL_JSON"

python "$REPO/scripts/summarize_offline_eval_json.py" --inputs "$EVAL_JSON" | tee "$RUN_ROOT/results/summary.md"

echo "RUN_ROOT=$RUN_ROOT"
echo "DONE"
