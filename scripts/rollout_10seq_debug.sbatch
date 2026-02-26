#!/usr/bin/env bash
#SBATCH --job-name=bdvla_roll10
#SBATCH --partition=preempt
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/project/peilab/luxiaocheng/projects/BlockDiff-VLA/logs/rollout10_%j.out
#SBATCH --error=/project/peilab/luxiaocheng/projects/BlockDiff-VLA/logs/rollout10_%j.err

set -euo pipefail

REPO=/project/peilab/luxiaocheng/projects/BlockDiff-VLA
VENV=${VENV:-$REPO/venv_rollout}
DATASET_ROOT=${DATASET_ROOT:-/project/peilab/luxiaocheng/projects/calvin/dataset/calvin_debug_dataset}
MODEL_CFG=${MODEL_CFG:-$REPO/policy_rollout/arvla_model_rollout_probe.yaml}
NUM_SEQUENCES=${NUM_SEQUENCES:-10}
NUM_VIDEOS=${NUM_VIDEOS:-0}

if [[ ! -f "$VENV/bin/activate" ]]; then
  echo "[ERROR] missing rollout venv: $VENV"
  exit 1
fi

cd "$REPO"
source "$VENV/bin/activate"

export PYTHONPATH="$REPO"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export EGL_VISIBLE_DEVICES=${EGL_VISIBLE_DEVICES:-0}

echo "[INFO] host=$(hostname)"
echo "[INFO] model_cfg=$MODEL_CFG"
echo "[INFO] dataset_root=$DATASET_ROOT"
echo "[INFO] num_sequences=$NUM_SEQUENCES num_videos=$NUM_VIDEOS"

python -m policy_rollout.calvin_evaluate_upvla \
  dataset_path="$DATASET_ROOT" \
  root_data_dir="$DATASET_ROOT" \
  model_config="$MODEL_CFG" \
  device=0 \
  num_sequences="$NUM_SEQUENCES" \
  num_videos="$NUM_VIDEOS" \
  save_rollout_as_video=False \
  log_wandb=False
