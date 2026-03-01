#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <checkpoint_dir>"
  exit 1
fi

CKPT_DIR="$(realpath "$1")"
if [[ ! -d "$CKPT_DIR" ]]; then
  echo "Error: checkpoint directory not found: $CKPT_DIR"
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
LMMS_DIR="${LMMS_EVAL_DIR:-$(realpath "$REPO_ROOT/../lmms-eval-codex")}"
LMMS_REPO_URL="${LMMS_REPO_URL:-https://github.com/EvolvingLMMs-Lab/lmms-eval.git}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/lmms_outputs/arvla_pope_textvqa}"
MODEL_TAG="${MODEL_TAG:-$(basename "$CKPT_DIR")}"
TEMPLATE_CFG="${TEMPLATE_CFG:-$REPO_ROOT/policy_rollout/arvla_model_local.yaml}"
TMP_CFG="${TMP_CFG:-$REPO_ROOT/policy_rollout/.tmp_eval_${MODEL_TAG}_$(date +%Y%m%d_%H%M%S).yaml}"

if [[ ! -d "$LMMS_DIR" ]]; then
  echo "[Info] lmms-eval not found at $LMMS_DIR, cloning..."
  git clone "$LMMS_REPO_URL" "$LMMS_DIR"
fi

echo "[Info] Installing lmms-eval and required deps..."
"$PYTHON_BIN" -m pip install -e "$LMMS_DIR"
"$PYTHON_BIN" -m pip install colorama omegaconf hydra-core jaxtyping typeguard 'chardet<6'

POPE_YAML="$LMMS_DIR/lmms_eval/tasks/pope/pope.yaml"
if [[ -f "$POPE_YAML" ]]; then
  sed -i 's/token: True/token: False/' "$POPE_YAML"
fi

if [[ ! -f "$TEMPLATE_CFG" ]]; then
  echo "Error: template config not found: $TEMPLATE_CFG"
  exit 1
fi

cp "$TEMPLATE_CFG" "$TMP_CFG"
sed -i "s|^\([[:space:]]*tuned_model_path:\).*|\1 \"$CKPT_DIR\"|" "$TMP_CFG"

mkdir -p "$OUTPUT_DIR"

export PYTHONPATH="$LMMS_DIR:$REPO_ROOT:${PYTHONPATH:-}"
export LMMS_EVAL_PLUGINS="upvla_lmms_plugin"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-256}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-32}"

echo "[Info] Running POPE + TextVQA eval..."
echo "[Info] model_config=$TMP_CFG"
echo "[Info] output_dir=$OUTPUT_DIR model_tag=$MODEL_TAG"

"$PYTHON_BIN" "$REPO_ROOT/scripts/run_upvla_lmms_eval.py" \
  --model-config "$TMP_CFG" \
  --tasks "pope,textvqa_val" \
  --output-dir "$OUTPUT_DIR" \
  --model-tag "$MODEL_TAG" \
  --device cuda \
  --dtype bf16 \
  --max-new-tokens 32 \
  --temperature 0 \
  --top-k 1

RESULT_ROOT="$OUTPUT_DIR/$MODEL_TAG"
RESULT_JSON="$("$PYTHON_BIN" - <<PY
import pathlib
root = pathlib.Path("$RESULT_ROOT")
files = sorted(root.rglob("*_results.json"), key=lambda p: p.stat().st_mtime)
print(files[-1] if files else "")
PY
)"

if [[ -z "$RESULT_JSON" || ! -f "$RESULT_JSON" ]]; then
  echo "[Warn] No *_results.json found. Please check logs above."
  exit 1
fi

echo "[Info] Result file: $RESULT_JSON"
"$PYTHON_BIN" - <<PY
import json
p = "$RESULT_JSON"
with open(p, "r", encoding="utf-8") as f:
    d = json.load(f)
pope = d["results"]["pope"]
textvqa = d["results"]["textvqa_val"]
print(f"POPE accuracy: {pope['pope_accuracy,none']:.4f}")
print(f"POPE f1      : {pope['pope_f1_score,none']:.4f}")
print(f"TextVQA EM   : {textvqa['exact_match,none']:.4f}")
PY
