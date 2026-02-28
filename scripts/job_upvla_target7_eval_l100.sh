#!/bin/bash
#SBATCH --job-name=upvla_t7_l100
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=/project/peilab/luxiaocheng/projects/BlockDiff-VLA/logs/upvla_t7_l100_%j.out
#SBATCH --error=/project/peilab/luxiaocheng/projects/BlockDiff-VLA/logs/upvla_t7_l100_%j.err

set -euo pipefail
cd /project/peilab/luxiaocheng/projects/BlockDiff-VLA
source venv_rollout/bin/activate
export NUMEXPR_MAX_THREADS=128
export TQDM_DISABLE=1
export PYTHONPATH=/project/peilab/luxiaocheng/projects/lmms-eval-codex:$PWD:${PYTHONPATH:-}
export HF_HOME=/project/peilab/luxiaocheng/.cache/huggingface
export HF_DATASETS_CACHE=/project/peilab/luxiaocheng/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/project/peilab/luxiaocheng/.cache/huggingface/transformers
export TMPDIR=/project/peilab/luxiaocheng/.cache/tmp
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$TMPDIR" lmms_outputs/upvla_eval_target7

python scripts/run_upvla_lmms_eval.py \
  --model-config policy_rollout/arvla_model_rollout_probe.yaml \
  --tasks mme,mmbench_en_dev,mmstar,realworldqa,mmerealworld,embspatial,erqa \
  --output-dir lmms_outputs/upvla_eval_target7 \
  --model-tag upvla_target7_l100 \
  --device cuda \
  --dtype bf16 \
  --temperature 1.0 \
  --top-k 1 \
  --limit 100
