#!/usr/bin/env bash
#SBATCH --job-name=blockdiff_home
#SBATCH --partition=preempt
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/project/peilab/luxiaocheng/projects/BlockDiff-VLA/logs/blockdiff_home_%j.out
#SBATCH --error=/project/peilab/luxiaocheng/projects/BlockDiff-VLA/logs/blockdiff_home_%j.err

set -euo pipefail

REPO=/project/peilab/luxiaocheng/projects/BlockDiff-VLA
RUN_ROOT="${RUN_ROOT:-/home/xlubl/blockdiff_debug_run_${SLURM_JOB_ID}}"
VENV="${VENV:-/home/xlubl/.venv_blockdiff_pip}"
KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-0}"
DATASET_ROOT="${DATASET_ROOT:-/project/peilab/luxiaocheng/projects/calvin/dataset/calvin_debug_dataset}"
TRAIN_SPLIT_DIR="${TRAIN_SPLIT_DIR:-$DATASET_ROOT/training}"
export RUN_ROOT

mkdir -p "$RUN_ROOT" "$RUN_ROOT/logs" "$RUN_ROOT/outputs" "$RUN_ROOT/results" "$RUN_ROOT/policy_rollout" "$RUN_ROOT/debug_data"

rm -rf "$VENV"
python3 -m venv "$VENV"
source "$VENV/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "$REPO/requirements_minimal.txt" \
  omegaconf accelerate==0.21.0 tensorboard lightning hydra-core termcolor gym==0.26.2 sentencepiece

export PYTHONPATH="$REPO"
export TOKENIZERS_PARALLELISM=false

python "$REPO/scripts/prepare_calvin_debug_processed.py" \
  --source-split-dir "$TRAIN_SPLIT_DIR" \
  --output-dir "$RUN_ROOT/debug_data/calvin_debug_processed_training" \
  --instruction "debug instruction" \
  --max-frames 256 \
  --start-index 0

run_one_baseline() {
  local cfg="$1"
  local tag="$2"
  local model_tpl="$3"

  local out_dir="$RUN_ROOT/outputs/${tag}"
  local train_log="$RUN_ROOT/logs/${tag}_train.log"
  local eval_log="$RUN_ROOT/logs/${tag}_eval.log"
  local eval_json="$RUN_ROOT/results/${tag}_offline_eval.json"
  local model_cfg="$RUN_ROOT/policy_rollout/${tag}_model.yaml"

  echo "[TRAIN] ${tag} using ${cfg}"
  set +e
  python "$REPO/train_upvla.py" \
    "config=${cfg}" \
    "experiment.tracker=tensorboard" \
    "experiment.output_dir=${out_dir}" \
    "experiment.name=${tag}" \
    "dataset.params.pre_path=${RUN_ROOT}/debug_data/calvin_debug_processed_training" \
    "dataset.params.mmu_path=${RUN_ROOT}/debug_data/calvin_debug_processed_training" \
    2>&1 | tee "$train_log"
  local train_rc=${PIPESTATUS[0]}
  set -e

  local ckpt="$out_dir/checkpoint-1/unwrapped_model/pytorch_model.bin"
  if [[ ! -f "$ckpt" ]]; then
    echo "[ERROR] ${tag} missing checkpoint: $ckpt"
    return 1
  fi
  if [[ "$train_rc" -ne 0 ]]; then
    echo "[WARN] ${tag} training exited non-zero (${train_rc}) but checkpoint exists, continue eval"
  fi

  python - <<PY
from omegaconf import OmegaConf
cfg = OmegaConf.load("$model_tpl")
cfg.model.showo.tuned_model_path = "$out_dir/checkpoint-1"
OmegaConf.save(cfg, "$model_cfg")
print("wrote", "$model_cfg")
PY

  echo "[EVAL] ${tag} using ${model_cfg}"
  python "$REPO/scripts/eval_upvla_offline_actions.py" \
    --dataset-root "$DATASET_ROOT" \
    --split validation \
    --model-config "$model_cfg" \
    --device cuda:0 \
    --start-only \
    --max-samples 1 \
    --clip-action \
    --output-json "$eval_json" \
    2>&1 | tee "$eval_log"

  if [[ "$KEEP_CHECKPOINTS" != "1" ]]; then
    rm -rf "$out_dir"
  fi
}

run_one_baseline "$REPO/config/debug_arvla_1step.yaml" "debug_arvla" "$REPO/policy_rollout/debug_arvla_model.yaml"
run_one_baseline "$REPO/config/debug_mdmvla_1step.yaml" "debug_mdmvla" "$REPO/policy_rollout/debug_mdmvla_model.yaml"
run_one_baseline "$REPO/config/debug_bdvla_1step.yaml" "debug_bdvla" "$REPO/policy_rollout/debug_bdvla_model.yaml"

python "$REPO/scripts/summarize_offline_eval_json.py" \
  --run-root "$RUN_ROOT" \
  --tags debug_arvla debug_mdmvla debug_bdvla \
  --require-all | tee "$RUN_ROOT/results/summary.md"

echo "RUN_ROOT=$RUN_ROOT"
echo "DONE"
