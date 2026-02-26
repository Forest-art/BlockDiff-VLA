#!/usr/bin/env bash
#SBATCH --job-name=bdvla_rollout_ddp
#SBATCH --partition=preempt
#SBATCH --account=peilab
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/project/peilab/luxiaocheng/projects/BlockDiff-VLA/logs/rollout_ddp_%j.out
#SBATCH --error=/project/peilab/luxiaocheng/projects/BlockDiff-VLA/logs/rollout_ddp_%j.err

set -euo pipefail

REPO=/project/peilab/luxiaocheng/projects/BlockDiff-VLA
VENV=${VENV:-$REPO/venv_rollout}
DATASET_ROOT=${DATASET_ROOT:-/project/peilab/luxiaocheng/projects/calvin/dataset/calvin_debug_dataset}
MODEL_CFG=${MODEL_CFG:-$REPO/policy_rollout/arvla_model_rollout_probe.yaml}
NUM_SEQUENCES=${NUM_SEQUENCES:-1000}
NUM_VIDEOS=${NUM_VIDEOS:-0}
DIST_MASTER_PORT=${DIST_MASTER_PORT:-29517}
SEQUENCE_WORKERS=${SEQUENCE_WORKERS:-12}
DIST_BACKEND=${DIST_BACKEND:-gloo}

if [[ ! -f "$VENV/bin/activate" ]]; then
  echo "[ERROR] missing rollout venv: $VENV"
  exit 1
fi

cd "$REPO"
source "$VENV/bin/activate"

export PYTHONPATH="$REPO"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT="$DIST_MASTER_PORT"
export WORLD_SIZE="$SLURM_NTASKS"

echo "[INFO] job=$SLURM_JOB_ID nodes=$SLURM_JOB_NUM_NODES world_size=$WORLD_SIZE"
echo "[INFO] master_addr=$MASTER_ADDR master_port=$MASTER_PORT"
echo "[INFO] model_cfg=$MODEL_CFG dataset_root=$DATASET_ROOT"
echo "[INFO] num_sequences=$NUM_SEQUENCES num_videos=$NUM_VIDEOS"
echo "[INFO] dist_backend=$DIST_BACKEND"

srun --ntasks="$SLURM_NTASKS" --gpus-per-task=1 \
  python -m policy_rollout.calvin_evaluate_upvla \
    dataset_path="$DATASET_ROOT" \
    root_data_dir="$DATASET_ROOT" \
    model_config="$MODEL_CFG" \
    device=0 \
    num_sequences="$NUM_SEQUENCES" \
    num_videos="$NUM_VIDEOS" \
    save_rollout_as_video=False \
    log_wandb=False \
    dist_backend="$DIST_BACKEND" \
    dist_master_port="$DIST_MASTER_PORT" \
    sequence_num_workers="$SEQUENCE_WORKERS" \
    'hydra.job.chdir=False' \
    'hydra.run.dir=outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}_rollout_r${oc.env:SLURM_PROCID,0}'
