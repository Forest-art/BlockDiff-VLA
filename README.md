<div align="center">
<h2><center>👉 BlockDiff-VLA: Block-Diffusion Text Objective for Vision-Language-Action</h2>

</div>
This repository contains the `BlockDiff-VLA` training and evaluation suite.

This codebase now supports three text-objective baselines while keeping **vision/image and action objectives unchanged**:
- `AR-VLA`: text objective is autoregressive LM (`text_objective=ar`)
- `MDM-VLA`: text objective is masked denoising (`text_objective=mdm`)
- `BD-VLA`: text objective is native block diffusion (`text_objective=bd`)

All three keep the same image objective and continuous action regression path by default.


##  Installation 🛠️
Default setup (minimal, no extra GitHub repositories required, no `python -m venv` needed):

```bash
git clone <your_repo_url>/BlockDiff-VLA.git
cd BlockDiff-VLA
pip install --user -r requirements_minimal.txt \
  omegaconf accelerate==0.21.0 tensorboard lightning hydra-core termcolor gym==0.26.2 sentencepiece
export PATH="$HOME/.local/bin:$PATH"
```

If you prefer virtualenv, you can still use:
```bash
python3 -m venv blockdiff_vla_env
source blockdiff_vla_env/bin/activate
pip install -r requirements_minimal.txt \
  omegaconf accelerate==0.21.0 tensorboard lightning hydra-core termcolor gym==0.26.2 sentencepiece
```

If you need full training stack instead of minimal stack:
```bash
pip install --user -r requirements.txt
```

Default tracker backend is TensorBoard across provided configs. Launch dashboard with:
```bash
tensorboard --logdir ./outputs
```

### Quick Start Training (No CALVIN Env / No Extra GitHub)
Minimal train-step smoke test (uses only repo-provided dummy data):
```bash
python scripts/smoke_upvla_minimal.py \
  --dataset-path ./dummy_data/calvin_processed_training
```

Quick 3-step debug training + offline eval (Slurm 1 GPU, still no `calvin_env` dependency):
```bash
sbatch scripts/quick_debug_ar_eval.sbatch
TRAIN_STEPS=5 sbatch scripts/quick_debug_ar_eval.sbatch
```
If your cluster does not support venv activation in job scripts:
```bash
ENV_MODE=system TRAIN_STEPS=5 sbatch scripts/quick_debug_ar_eval.sbatch
```

If you want to smoke-test with DeepSpeed (show-o large model path):
```bash
TRAIN_STEPS=1 sbatch scripts/quick_debug_ar_eval_deepspeed.sbatch
```
This script uses `accelerate + deepspeed` and auto-installs `deepspeed>=0.18,<0.19` when missing (`INSTALL_DEEPSPEED=1` by default).
You can override the DeepSpeed pip spec if needed:
```bash
DEEPSPEED_SPEC='deepspeed>=0.18,<0.19' TRAIN_STEPS=1 sbatch scripts/quick_debug_ar_eval_deepspeed.sbatch
```
DeepSpeed smoke outputs are saved under: `/home/xlubl/blockdiff_quick_ar_ds_<jobid>/`

Outputs are saved under:
`/home/xlubl/blockdiff_quick_ar_<jobid>/`

Key artifacts:
- `outputs/quick_arvla/checkpoint-*`
- `results/quick_arvla_offline_eval.json`
- `results/summary.md`

### Config Quick Setup (What to fill to run fast)
Use `config/debug_arvla_1step.yaml` as the fastest base config. You only need to make sure the following keys are valid:

| Config key | Required | Example value | Notes |
|---|---|---|---|
| `experiment.tracker` | Yes | `"tensorboard"` | Keep TensorBoard-only environment |
| `experiment.output_dir` | Yes | `"outputs/quick_arvla"` | Checkpoints + logs |
| `model.vq_model.vq_model_name` | Yes | `"/path/to/magvitv2"` | Local MagVITv2 checkpoint dir |
| `model.showo.pretrained_model_path` | Yes | `"/path/to/show-o-w-clip-vit-512x512"` | Show-o pretrained dir |
| `model.showo.llm_model_path` | Yes | `"/path/to/phi-1_5"` | Phi/Qwen tokenizer+weights dir |
| `dataset.params.train_pre_shards_path_or_url` | Yes | `"/path/to/*processed*/training"` | Must contain `dataset_info.json` |
| `dataset.params.train_mmu_shards_path_or_url` | Yes | `"./dummy_data/llava_tuning_665k_data"` | Keep dummy MMU for smoke |
| `training.max_train_steps` | Yes | `3` or `5` | Debug run only |

Fastest no-CALVIN-env debug flow (train a few steps + offline eval):

```bash
# 1) (optional) build a tiny processed split from CALVIN debug frames
python scripts/prepare_calvin_debug_processed.py \
  --source-split-dir /path/to/calvin_debug_dataset/training \
  --output-dir ./debug_data/calvin_debug_processed_training \
  --instruction "debug instruction" \
  --max-frames 256

# 2) quick training (override paths from command line)
python train_upvla.py \
  "config=$(pwd)/config/debug_arvla_1step.yaml" \
  "experiment.tracker=tensorboard" \
  "training.max_train_steps=5" \
  "experiment.output_dir=$(pwd)/outputs/quick_arvla" \
  "dataset.params.train_pre_shards_path_or_url=$(pwd)/debug_data/calvin_debug_processed_training" \
  "dataset.params.train_mmu_shards_path_or_url=$(pwd)/dummy_data/llava_tuning_665k_data" \
  "model.vq_model.vq_model_name=/path/to/magvitv2" \
  "model.showo.pretrained_model_path=/path/to/show-o-w-clip-vit-512x512" \
  "model.showo.llm_model_path=/path/to/phi-1_5"

# 3) point eval model config to your new checkpoint
export CKPT_DIR=$(ls -d outputs/quick_arvla/checkpoint-* | sort -V | tail -n 1)
python - <<'PY'
import os
from omegaconf import OmegaConf
cfg = OmegaConf.load("policy_rollout/debug_arvla_model.yaml")
cfg.model.showo.tuned_model_path = os.environ["CKPT_DIR"]
OmegaConf.save(cfg, "policy_rollout/debug_arvla_model.yaml")
print("updated policy_rollout/debug_arvla_model.yaml")
PY

# 4) offline eval (no calvin_env / no rollout)
python scripts/eval_upvla_offline_actions.py \
  --dataset-root /path/to/calvin_debug_dataset \
  --split validation \
  --model-config $(pwd)/policy_rollout/debug_arvla_model.yaml \
  --device cuda:0 \
  --start-only \
  --max-samples 1 \
  --output-json outputs/quick_arvla_offline_eval.json
```

If you only need one command on SuperPOD, use:
```bash
TRAIN_STEPS=5 sbatch scripts/quick_debug_ar_eval.sbatch
```

### SuperPOD one-job debug pipeline (optional)
For HKUST SuperPOD users, an end-to-end Slurm smoke pipeline is provided:
```bash
sbatch scripts/blockdiff_debug_pip_home.sbatch
```
This script supports both env modes:
- `ENV_MODE=system` (default): use current python + `pip` (`PIP_USER=1` by default)
- `ENV_MODE=venv`: create and use `$VENV` via `python3 -m venv`

This submits AR/MDM/BD 1-step training + offline eval in one job and writes a final summary table to:
`/home/xlubl/blockdiff_debug_run_<jobid>/results/summary.md`

If you want to perform experiments in [Calvin](https://arxiv.org/pdf/2112.03227), you need also prepare with the calvin environment following the official repo of [Calvin](https://github.com/mees/calvin.git).

Download [showlab/show-o-w-clip-vit-512x512](https://huggingface.co/showlab/show-o-w-clip-vit-512x512), [phi-1.5](https://huggingface.co/microsoft/phi-1_5), or another supported backbone from Hugging Face. Prepare the backbone checkpoints under `./showlab`.

## Data Preparation
### Embodied Data
(1) Choice one

Download [Calvin](https://github.com/mees/calvin.git) dataset and [Bridge](https://docs.google.com/spreadsheets/d/1rPBD77tk60AEIGZrGSODwyyzs5FgCU9Uz3h-3_t2A9g/edit?gid=0#gid=0) dataset (you can skip the bridge dataset during pretraining), and process the raw data with script in `./preprocess_data`:
```bash
cd preprocess_data
# modify the path in scripts
python process_calvin.py
python process_bridge.py
```

(2) Choice two (better for using your own robot data or dataloader):

See the implementation of `DataProvider` class in `training/future_view_predction_w_action_dataset.py` and reimplement the dataloader class fit your dataset.

### MMU Data
We also use the [llava_tuning_665k_data](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json) for cotraining to maintain model's multimodal understanding capability. If you don't want to cotrain with MMU dataset for training, you can modify the config file and exclude the mmu dataloader in `train_upvla.py`.

## Train Baselines 🛸
### 🛸 Training requirements
Our experiments are run on 4 A800 80G GPU. Under this setting, the training process takes ~70G GPU memory. 

If you have limited GPU memory, you can modify the batchsize setting in `config/yaml` config files. 

### 🛸 Training Pipeline
(1) Prediction and Understanding Pretraining：
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file ./accelerate_configs/4_gpus_deepspeed_zero2.yaml --main_process_port=8888 train_upvla.py config=./config/upvla_pred_tuning.yaml
```

(2) Action-stage training (AR text baseline):
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --config_file ./accelerate_configs/4_gpus_deepspeed_zero2.yaml \
  --main_process_port=8888 \
  train_upvla.py config=./config/arvla_action_tuning.yaml
```

(3) Stage-1 training (MDM text baseline):
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --config_file ./accelerate_configs/4_gpus_deepspeed_zero2.yaml \
  --main_process_port=8888 \
  train_mdm_vla.py config=./config/mdmvla_stage1_pred_tuning.yaml
```

(4) Stage-2 training (MDM text baseline):
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --config_file ./accelerate_configs/4_gpus_deepspeed_zero2.yaml \
  --main_process_port=8888 \
  train_mdm_vla.py config=./config/mdmvla_stage2_action_tuning.yaml
```

(5) Action-stage training (Block Diffusion text baseline):
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --config_file ./accelerate_configs/4_gpus_deepspeed_zero2.yaml \
  --main_process_port=8888 \
  train_blockdiff_vla.py config=./config/bdvla_action_tuning.yaml
```

(6) Action-stage training (Qwen3 + Elastic Block Diffusion text baseline):
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --config_file ./accelerate_configs/4_gpus_deepspeed_zero2.yaml \
  --main_process_port=8888 \
  train_blockdiff_vla.py config=./config/bdvla_action_tuning_qwen3_elastic.yaml
```

### 🛸 DeepSpeed Launch for AR-UPVLA Baseline (copy-ready)
Single GPU debug/start command:
```bash
cd /path/to/BlockDiff-VLA
export PYTHONPATH=$(pwd)
CUDA_VISIBLE_DEVICES=0 accelerate launch \
  --config_file ./accelerate_configs/1_gpu_deepspeed_zero2.yaml \
  --main_process_port 29666 \
  train_upvla.py \
  "config=$(pwd)/config/upvla_action_tuning.yaml" \
  "experiment.tracker=tensorboard" \
  "experiment.output_dir=$(pwd)/outputs/upvla_ar_ds" \
  "model.framework=upvla" \
  "training.text_objective=ar" \
  "dataset.params.train_pre_shards_path_or_url=/path/to/calvin_processed_training" \
  "dataset.params.train_mmu_shards_path_or_url=/path/to/llava_tuning_665k_data" \
  "model.vq_model.vq_model_name=/path/to/magvitv2" \
  "model.showo.pretrained_model_path=/path/to/show-o-w-clip-vit-512x512" \
  "model.showo.llm_model_path=/path/to/phi-1_5"
```

4-GPU training command:
```bash
cd /path/to/BlockDiff-VLA
export PYTHONPATH=$(pwd)
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --config_file ./accelerate_configs/4_gpus_deepspeed_zero2.yaml \
  --main_process_port 29666 \
  train_upvla.py \
  "config=$(pwd)/config/upvla_action_tuning.yaml" \
  "experiment.tracker=tensorboard" \
  "experiment.output_dir=$(pwd)/outputs/upvla_ar_ds" \
  "model.framework=upvla" \
  "training.text_objective=ar" \
  "dataset.params.train_pre_shards_path_or_url=/path/to/calvin_processed_training" \
  "dataset.params.train_mmu_shards_path_or_url=/path/to/llava_tuning_665k_data" \
  "model.vq_model.vq_model_name=/path/to/magvitv2" \
  "model.showo.pretrained_model_path=/path/to/show-o-w-clip-vit-512x512" \
  "model.showo.llm_model_path=/path/to/phi-1_5"
```

8-GPU training command:
```bash
cd /path/to/BlockDiff-VLA
export PYTHONPATH=$(pwd)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --config_file ./accelerate_configs/8_gpus_deepspeed_zero2.yaml \
  --main_process_port 29666 \
  train_upvla.py \
  "config=$(pwd)/config/upvla_action_tuning.yaml" \
  "experiment.tracker=tensorboard" \
  "experiment.output_dir=$(pwd)/outputs/upvla_ar_ds" \
  "model.framework=upvla" \
  "training.text_objective=ar" \
  "dataset.params.train_pre_shards_path_or_url=/path/to/calvin_processed_training" \
  "dataset.params.train_mmu_shards_path_or_url=/path/to/llava_tuning_665k_data" \
  "model.vq_model.vq_model_name=/path/to/magvitv2" \
  "model.showo.pretrained_model_path=/path/to/show-o-w-clip-vit-512x512" \
  "model.showo.llm_model_path=/path/to/phi-1_5"
```

Parameter mapping you usually need to replace:
- `dataset.params.train_pre_shards_path_or_url`: your processed CALVIN-style training data root (must contain `dataset_info.json`)
- `dataset.params.train_mmu_shards_path_or_url`: your MMU json/image data root (for smoke you can keep repo dummy)
- `model.vq_model.vq_model_name`: MagVITv2 checkpoint path or HF id
- `model.showo.pretrained_model_path`: Show-o base checkpoint path or HF id
- `model.showo.llm_model_path`: LLM tokenizer/model path (e.g. Phi-1.5)
- `experiment.output_dir`: where checkpoints/logs are saved

If you only want smoke training for a few steps, append:
```bash
"training.max_train_steps=5"
```

Text-objective switches:
- `training.text_objective: "ar" | "mdm" | "bd"`
- `training.mmu_coeff`: weight for selected text objective
- `training.pre_coeff`: vision prediction loss weight
- `training.act_coeff`: action loss weight

`AR-VLA` is the default `UP-VLA` setting (`model.framework: "upvla"` + `training.text_objective: "ar"`).

Block diffusion text knobs:
- `block_diffusion.text_enabled`
- `block_diffusion.block_size`
- `block_diffusion.elastic_block_enabled`
- `block_diffusion.block_size_candidates`
- `block_diffusion.mask_eps`
- `block_diffusion.complementary_mask`

Qwen3 backbone knobs:
- `model.showo.llm_backbone: "auto"`
- `model.showo.trust_remote_code: true`
- `model.showo.auto_set_llm_vocab_size: true` (recommended for non-Phi backbones)

MDM text knobs:
- `text_mdm.min_mask_ratio`
- `text_mdm.max_mask_ratio`

Action is kept on original regression by default (`block_diffusion.action_enabled: false`, `block_diffusion.action_infer_enabled: false`).


## Evaluation 📊

### 📊 Minimal-dependency offline action evaluation (no `calvin_env`)
If you cannot install full CALVIN simulation dependencies, you can run offline action metrics:
```bash
PYTHONPATH=$(pwd) python scripts/eval_upvla_offline_actions.py \
  --dataset-root /path/to/calvin_debug_dataset \
  --split validation \
  --model-config $(pwd)/policy_rollout/arvla_model.yaml \
  --device cuda:0 \
  --start-only \
  --max-samples 100 \
  --output-json outputs/offline_eval_100.json
```
This script supports `arvla`, `mdmvla`, and `bdvla` via `model.framework` in the model yaml.

### 📊 Rollout on Calvin benchmark
Optional advanced path (requires additional CALVIN GitHub environment install). Then set:
- `policy_conf/calvin_evaluate_upvla.yaml`: `dataset_path` (and `root_data_dir`) to your Calvin `task_ABC_D` root.
- `model_config` to one of:
  - `policy_rollout/arvla_model.yaml` (AR text baseline)
  - `policy_rollout/mdmvla_model.yaml` (MDM text baseline)
  - `policy_rollout/bdvla_model.yaml` (BD text baseline)
- `tuned_model_path` in the selected model yaml to your checkpoint directory.

Make sure rollout dependencies are installed in your env:
```bash
pip install hydra-core termcolor gym==0.26.2

# install CALVIN (needed by policy_models.wrappers.hulc_wrapper.HulcWrapper)
git clone --depth 1 --recurse-submodules https://github.com/mees/calvin.git /path/to/calvin
pip install -e /path/to/calvin/calvin_env/tacto
pip install -e /path/to/calvin/calvin_env
```

`dataset_path` must point to an official CALVIN split root (e.g. `task_ABC_D`) that contains:
- `training/`
- `validation/`
- `validation/.hydra/merged_config.yaml`

Copy-ready online rollout command (AR-VLA / UP-VLA baseline):
```bash
cd /path/to/BlockDiff-VLA
source ./venv_rollout/bin/activate
export PYTHONPATH=$(pwd)

# 1) Fill required paths
export CALVIN_DATA_ROOT=/path/to/calvin/task_ABC_D
export MODEL_YAML=$(pwd)/policy_rollout/arvla_model.yaml
export CKPT_DIR=/path/to/your/checkpoint-20000
export ROLLOUT_MODEL_YAML=$(pwd)/rollout_outputs/rollout_model_$(date +%Y%m%d_%H%M%S).yaml
# if EGL auto-detection is unstable on your cluster, pin it manually
export EGL_VISIBLE_DEVICES=0

# 2) Create a temporary rollout config (do not overwrite source yaml)
python - <<'PY'
import os
from pathlib import Path
from omegaconf import OmegaConf
cfg = OmegaConf.load(os.environ["MODEL_YAML"])
cfg.model.showo.tuned_model_path = os.environ["CKPT_DIR"]
Path(os.environ["ROLLOUT_MODEL_YAML"]).parent.mkdir(parents=True, exist_ok=True)
OmegaConf.save(cfg, os.environ["ROLLOUT_MODEL_YAML"])
print("updated", os.environ["ROLLOUT_MODEL_YAML"])
PY

# 3) Launch online rollout
python -m policy_rollout.calvin_evaluate_upvla \
  dataset_path=$CALVIN_DATA_ROOT \
  root_data_dir=$CALVIN_DATA_ROOT \
  model_config=$ROLLOUT_MODEL_YAML \
  device=0 \
  num_sequences=100 \
  num_videos=0 \
  save_rollout_as_video=False \
  log_wandb=False
```
For MDM/BD rollout, switch `MODEL_YAML` to `$(pwd)/policy_rollout/mdmvla_model.yaml` or `$(pwd)/policy_rollout/bdvla_model.yaml`.
Outputs are saved under `log_dir` (default `./rollout_outputs/<timestamp>/`) including `results.json`, rollout videos (`.gif` by default, `.mp4` if `save_rollout_as_video=True`), and `input_predict_truth/*.png`.

Distributed rollout (multi-node/multi-GPU via Slurm + `srun`):
```bash
cd /project/peilab/luxiaocheng/projects/BlockDiff-VLA

# Edit MODEL_CFG / DATASET_ROOT / NUM_SEQUENCES as needed
sbatch scripts/rollout_distributed_calvin.sbatch
```
The script launches one process per GPU and the evaluator will:
- initialize `torch.distributed` from `WORLD_SIZE/RANK/LOCAL_RANK` (or `SLURM_*`),
- split `num_sequences` evenly across ranks,
- gather results across all ranks,
- save a single merged `results.json` on rank 0.

### 📊 Rollout in your own embodiments
For your own data, you should first train the model with your own dataloader. For rollout, we provide a script `./policy_rollout/policy_upvla.py` as a reference, which can be directly used in Franka Emika Robotarm.

## CheckPoints 📷
Release checkpoints will be published in a dedicated BlockDiff-VLA model repository.

## Bibtex 
🌟 If you find our work helpful, please leave us a star and cite our paper. Thank you!
```
@misc{blockdiffvla2026,
  title={BlockDiff-VLA},
  author={BlockDiff-VLA Team},
  year={2026},
  note={Project repository}
}
```
## Acknowledgments
This project builds on open-source model and tooling ecosystems including Show-o, Phi, and LLaVA.
