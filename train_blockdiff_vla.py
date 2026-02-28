# coding=utf-8
# Copyright 2024 HuggingFace, NUS Show Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

# from numpy.distutils.system_info import accelerate_info

os.environ["TOKENIZERS_PARALLELISM"] = "true"
import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import List, Optional, Union, Tuple

import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from lightning.pytorch.utilities import CombinedLoader

from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed

# from parquet import RefinedWebDataset
from transformers import CLIPImageProcessor
from models import Upvla, MAGVITv2, CLIPVisionTower, get_mask_chedule
from training.prompting_utils import UniversalPrompting_w_action, \
    create_attention_mask_predict_next_for_future_prediction, create_attention_mask_for_mmu_vit
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_info, set_verbosity_error

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from training.utils import get_config, flatten_omega_conf, mask_or_random_replace_tokens_for_future_prediction, \
    AverageMeter, image_transform
from training.block_diffusion_utils import (
    build_block_diffusion_attention_mask,
    build_native_block_diffusion_batch,
    build_prefixed_block_diffusion_attention_mask,
    block_diffusion_token_shift_loss,
    quantize_actions_to_tokens,
)
from training.future_view_prediction_w_action_dataset import get_future_view_prediction_w_action_data_loader
import warnings
import tqdm
from llava.llava import conversation as conversation_lib

warnings.filterwarnings('ignore')
try:
    import wandb
except Exception:
    wandb = None

try:
    import apex

    is_apex_available = True
except ImportError:
    is_apex_available = False

logger = get_logger(__name__, log_level="INFO")


def _replace_vla_with_vl(text):
    if not isinstance(text, str):
        return text
    return text.replace("VLA", "VL").replace("vla", "vl")


class _VlaToVlLogFilter(logging.Filter):
    def filter(self, record):
        record.name = _replace_vla_with_vl(record.name)
        record.msg = _replace_vla_with_vl(record.msg)
        if isinstance(record.args, tuple):
            record.args = tuple(_replace_vla_with_vl(arg) for arg in record.args)
        elif isinstance(record.args, dict):
            record.args = {k: _replace_vla_with_vl(v) for k, v in record.args.items()}
        return True


def _resolve_train_logger_name(framework: str) -> str:
    env_name = os.environ.get("TRAIN_VL_LOGGER_NAME")
    if env_name:
        return env_name
    framework_to_name = {
        "mdmvla": "train_mdm_vl",
        "bdvla": "train_blockdiff_vl",
        "arvla": "train_ar_vl",
        "upvla": "train_up_vl",
    }
    return framework_to_name.get(framework, "train_vl")


def _install_vla_log_sanitizer():
    if os.environ.get("TRAIN_VL_SANITIZE_VLA", "1").strip().lower() in {"0", "false", "no", "off"}:
        return

    def _attach_filter(target_logger):
        if not isinstance(target_logger, logging.Logger):
            return
        has_filter = any(isinstance(f, _VlaToVlLogFilter) for f in target_logger.filters)
        if not has_filter:
            target_logger.addFilter(_VlaToVlLogFilter())
        for handler in target_logger.handlers:
            has_handler_filter = any(isinstance(f, _VlaToVlLogFilter) for f in handler.filters)
            if not has_handler_filter:
                handler.addFilter(_VlaToVlLogFilter())

    root_logger = logging.getLogger()
    _attach_filter(root_logger)
    for existing_logger in logging.root.manager.loggerDict.values():
        _attach_filter(existing_logger)


def _get_active_tracker_name(accelerator: Accelerator) -> Optional[str]:
    if len(accelerator.trackers) == 0:
        return None
    names = {t.name for t in accelerator.trackers}
    if "wandb" in names:
        return "wandb"
    if "tensorboard" in names:
        return "tensorboard"
    return None


def _log_images(
    accelerator: Accelerator,
    key: str,
    pil_images: List[Image.Image],
    step: int,
    captions: Optional[List[str]] = None,
):
    if not accelerator.is_main_process or len(pil_images) == 0:
        return

    tracker_name = _get_active_tracker_name(accelerator)
    if tracker_name == "wandb" and wandb is not None:
        wandb_images = []
        for idx, image in enumerate(pil_images):
            caption = captions[idx] if captions is not None and idx < len(captions) else None
            wandb_images.append(wandb.Image(image, caption=caption))
        wandb.log({key: wandb_images}, step=step)
        return

    if tracker_name == "tensorboard":
        tb_tracker = accelerator.get_tracker("tensorboard")
        writer = getattr(tb_tracker, "writer", None)
        if writer is None:
            return
        for idx, image in enumerate(pil_images):
            image_np = np.asarray(image)
            writer.add_image(f"{key}/{idx}", image_np, global_step=step, dataformats="HWC")
            if captions is not None and idx < len(captions):
                writer.add_text(f"{key}/{idx}_caption", captions[idx], global_step=step)


def _log_scalars(accelerator: Accelerator, values: dict, step: int):
    if not accelerator.is_main_process or len(values) == 0:
        return
    if len(accelerator.trackers) == 0:
        return
    accelerator.log(values, step=step)


def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    else:
        raise ValueError(f"model_type {model_type} not supported.")


def build_text_mdm_batch(
    clean_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    mask_token_id: int,
    min_mask_ratio: float = 0.15,
    max_mask_ratio: float = 0.60,
    ignore_id: int = -100,
):
    """
    Build text masked-denoising inputs:
      - input_ids: clean text with random valid tokens replaced by mask token
      - labels: original tokens only on masked positions, ignore elsewhere
    """
    if clean_ids.ndim != 2 or valid_mask.ndim != 2:
        raise ValueError("clean_ids and valid_mask must be [B, L].")
    if clean_ids.shape != valid_mask.shape:
        raise ValueError("clean_ids and valid_mask shape mismatch.")

    bsz, _ = clean_ids.shape
    min_mask_ratio = float(max(0.0, min_mask_ratio))
    max_mask_ratio = float(min(1.0, max_mask_ratio))
    if max_mask_ratio < min_mask_ratio:
        max_mask_ratio = min_mask_ratio

    ratios = torch.empty((bsz, 1), device=clean_ids.device).uniform_(min_mask_ratio, max_mask_ratio)
    sampled = torch.rand(clean_ids.shape, device=clean_ids.device)
    masked_pos = (sampled < ratios) & valid_mask

    # Keep at least one masked token for each sample that has any valid token.
    for b in range(bsz):
        if bool(valid_mask[b].any()) and not bool(masked_pos[b].any()):
            valid_idx = torch.nonzero(valid_mask[b], as_tuple=False).squeeze(-1)
            pick = valid_idx[torch.randint(0, valid_idx.numel(), (1,), device=clean_ids.device)]
            masked_pos[b, pick] = True

    input_ids = clean_ids.clone()
    input_ids[masked_pos] = int(mask_token_id)

    labels = torch.full_like(clean_ids, int(ignore_id))
    labels[masked_pos] = clean_ids[masked_pos]
    return input_ids, labels


def _parse_block_size_candidates(raw_value, default_block_size: int) -> List[int]:
    if raw_value is None:
        return [int(default_block_size)]
    if isinstance(raw_value, (list, tuple)):
        vals = [int(v) for v in raw_value]
    else:
        vals = [int(raw_value)]
    vals = [v for v in vals if v > 0]
    if len(vals) == 0:
        return [int(default_block_size)]
    return sorted(set(vals))


def _sample_block_sizes_for_batch(
    batch_size: int,
    device: torch.device,
    default_block_size: int,
    elastic_enabled: bool,
    candidates: List[int],
) -> Optional[torch.Tensor]:
    if batch_size <= 0:
        return None
    if not elastic_enabled:
        return None
    if len(candidates) == 0:
        return torch.full((batch_size,), int(default_block_size), dtype=torch.long, device=device)
    candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=device)
    idx = torch.randint(low=0, high=candidate_tensor.shape[0], size=(batch_size,), device=device)
    return candidate_tensor[idx]


def main():
    global logger
    #########################
    # SETUP Accelerator     #
    #########################
    config = get_config()
    framework = str(config.model.get("framework", "upvla")).lower()

    # Enable TF32 on Ampere GPUs
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = str(Path(config.experiment.output_dir) / "logs")
    tracker_backend = str(config.experiment.get("tracker", "tensorboard")).lower()
    if tracker_backend not in {"tensorboard", "wandb", "none"}:
        raise ValueError(
            f"Unsupported experiment.tracker='{tracker_backend}', expected one of: tensorboard|wandb|none"
        )
    if tracker_backend == "wandb" and wandb is None:
        logger.warning("wandb is not installed; falling back to tensorboard tracker.")
        tracker_backend = "tensorboard"

    log_with = None if tracker_backend == "none" else tracker_backend
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with=log_with,
        project_dir=config.experiment.logging_dir,
        split_batches=True,
    )

    total_batch_size_per_gpu = (config.training.batch_size_pre + config.training.batch_size_mmu)
    total_batch_size = ((config.training.batch_size_pre + config.training.batch_size_mmu) * accelerator.num_processes *
                        config.training.gradient_accumulation_steps)

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = (
            total_batch_size_per_gpu)

    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    _install_vla_log_sanitizer()
    logger = get_logger(_resolve_train_logger_name(framework), log_level="INFO")
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process and tracker_backend != "none":
        tracker_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        tracker_config.pop("experiment.resume_from_checkpoint", None)

        if tracker_backend == "wandb":
            resume_wandb_run = config.wandb.resume
            run_id = config.wandb.get("run_id", None)
            if run_id is None:
                resume_wandb_run = False
                run_id = wandb.util.generate_id()
                config.wandb.run_id = run_id

            wandb_init_kwargs = dict(
                name=config.experiment.name,
                id=run_id,
                resume=resume_wandb_run,
                entity=config.wandb.get("entity", None),
                config_exclude_keys=[],
            )
            accelerator.init_trackers(
                config.experiment.project,
                config=tracker_config,
                init_kwargs={"wandb": wandb_init_kwargs},
            )
        else:
            accelerator.init_trackers(
                config.experiment.project,
                config=tracker_config,
            )

    if accelerator.is_main_process:
        os.makedirs(config.experiment.output_dir, exist_ok=True)
        config_path = Path(config.experiment.output_dir) / "config.yaml"
        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed)

    entry_script = Path(sys.argv[0]).name
    if framework == "mdmvla" and entry_script == "train_blockdiff_vla.py":
        raise ValueError(
            "MDM training is isolated to the MDM entry. "
            "Please launch: python train_mdm_vl.py ..."
        )
    is_bdvla = framework == "bdvla"
    text_objective_defaults = {
        "arvla": "ar",
        "mdmvla": "mdm",
        "bdvla": "bd",
    }
    text_objective = str(config.training.get("text_objective", text_objective_defaults.get(framework, "ar"))).lower()
    if text_objective not in {"ar", "mdm", "bd"}:
        raise ValueError(f"Unsupported training.text_objective='{text_objective}', expected one of: ar|mdm|bd")

    bd_cfg = config.get("block_diffusion", OmegaConf.create({}))
    bd_text_enabled = bool(bd_cfg.get("text_enabled", text_objective == "bd"))
    bd_action_enabled = bool(bd_cfg.get("action_enabled", is_bdvla))
    bd_block_size = int(bd_cfg.get("block_size", 32))
    bd_mask_eps = float(bd_cfg.get("mask_eps", 1e-3))
    bd_complementary_mask = bool(bd_cfg.get("complementary_mask", True))
    bd_action_num_bins = int(bd_cfg.get("action_num_bins", 256))
    bd_elastic_enabled = bool(bd_cfg.get("elastic_block_enabled", False))
    bd_block_size_candidates = _parse_block_size_candidates(
        bd_cfg.get("block_size_candidates", [bd_block_size]),
        default_block_size=bd_block_size,
    )
    text_mdm_cfg = config.get("text_mdm", OmegaConf.create({}))
    text_mdm_min_mask_ratio = float(text_mdm_cfg.get("min_mask_ratio", 0.15))
    text_mdm_max_mask_ratio = float(text_mdm_cfg.get("max_mask_ratio", 0.60))
    text_mdm_enabled = text_objective == "mdm"
    text_denoise_enabled = bd_text_enabled or text_mdm_enabled
    if bd_action_num_bins > int(config.model.showo.codebook_size):
        raise ValueError("block_diffusion.action_num_bins must be <= model.showo.codebook_size")
    if bd_elastic_enabled:
        logger.info(
            f"Elastic block diffusion enabled with candidates={bd_block_size_candidates} "
            f"(default={bd_block_size})"
        )

    #########################
    # MODELS and OPTIMIZER  #
    #########################
    logger.info("Loading models and optimizer")

    tokenizer_kwargs = {"padding_side": "left"}
    if bool(config.model.showo.get("trust_remote_code", False)):
        tokenizer_kwargs["trust_remote_code"] = True
    tokenizer = AutoTokenizer.from_pretrained(config.model.showo.llm_model_path, **tokenizer_kwargs)
    tokenizer_base_vocab = int(len(tokenizer))
    if bool(config.model.showo.get("auto_set_llm_vocab_size", False)):
        config.model.showo.llm_vocab_size = tokenizer_base_vocab
        config.model.showo.vocab_size = int(
            tokenizer_base_vocab + int(config.model.showo.codebook_size) +
            int(config.model.showo.num_new_special_tokens) + 1
        )
    elif int(config.model.showo.llm_vocab_size) != tokenizer_base_vocab:
        logger.warning(
            f"Configured llm_vocab_size={int(config.model.showo.llm_vocab_size)} but tokenizer has "
            f"{tokenizer_base_vocab} tokens. Consider setting model.showo.auto_set_llm_vocab_size=true."
        )
    action_token_offset = int(config.model.showo.llm_vocab_size + config.model.showo.num_new_special_tokens)

    # unified prompting for show-o
    uni_prompting = UniversalPrompting_w_action(
        tokenizer,
        max_text_len=config.dataset.preprocessing.max_seq_length,
        special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>",
                        "<|lvg|>"),
        ignore_id=-100,
        cond_dropout_prob=config.training.cond_dropout_prob,
        future_steps=config.act_step)
    # "<|t2v|>", "<|v2v|>", "<|lvg|>" are not used
    # use "<|lvg|>" as begin of action output
    print('special tokens : \n', uni_prompting.sptids_dict)
    print(len(uni_prompting.text_tokenizer))
    # VQ model for processing image into discrete tokens
    vq_model = get_vq_model_class(config.model.vq_model.type)
    if config.model.vq_model.get("pretrained_model_path", None):
        vq_model = vq_model().to(accelerator.device)
        state_dict = torch.load(config.model.vq_model.pretrained_model_path)['model']
        vq_model.load_state_dict(state_dict)
    else:
        vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name).to(accelerator.device)
    vq_model.eval()
    vq_model.requires_grad_(False)

    # Initialize Show-o model
    if config.model.showo.load_from_showo:
        model = Upvla.from_pretrained(
            config.model.showo.pretrained_model_path, low_cpu_mem_usage=False,
            act_step=config.act_step).to(accelerator.device)
        if config.model.showo.vocab_size != model.vocab_size:
            model.showo.resize_token_embeddings(config.model.showo.vocab_size)
            model.config.codebook_size = config.model.showo.codebook_size
            model.config.vocab_size = config.model.showo.vocab_size
            model.vocab_size = config.model.showo.vocab_size
            model.output_size = config.model.showo.vocab_size
            model.config.mask_token_id = model.config.vocab_size - 1
            model.mask_token_id = model.config.vocab_size - 1
    else:
        model = Upvla(**config.model.showo).to(accelerator.device)
    mask_id = model.mask_token_id

    # PRE-TRAIN: ONLY TRAIN MM PROJECTOR
    if config.dataset.und_type == "llava_pretrain":
        if hasattr(model, 'module'):
            for n, p in model.module.named_parameters():
                if 'mm_projector' in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
        else:
            for n, p in model.named_parameters():
                if 'mm_projector' in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False

    vision_tower = CLIPVisionTower("openai/clip-vit-large-patch14-336").to(accelerator.device)
    clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    vision_tower.eval()
    for p in vision_tower.parameters():
        p.requires_grad = False

    ##################################
    #   Optimizer and LR scheduler   #
    #################################
    optimizer_config = config.optimizer.params

    # no decay on bias and layernorm and embedding
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    # Create mask scheduler
    if config.get("mask_schedule", None) is not None:
        schedule = config.mask_schedule.schedule
        args = config.mask_schedule.get("params", {})
        mask_schedule = get_mask_chedule(schedule, **args)
    else:
        mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
    )

    # ################################
    #         DATALOADER             #
    # ################################
    logger.info("Creating dataloaders and lr_scheduler")

    preproc_config = config.dataset.preprocessing
    dataset_config = config.dataset.params

    # Data for generation
    if config.dataset.gen_type == "future_view_prediction_w_action":
        train_dataloader_pre = get_future_view_prediction_w_action_data_loader(
            dataset_path=dataset_config.train_pre_shards_path_or_url,
            batch_size=config.training.batch_size_pre,
            num_workers=dataset_config.num_workers,
            world_size=accelerator.num_processes,
            local_rank=accelerator.process_index,
            resolution=preproc_config.resolution,
            future_step=config.act_step)
        num_update_steps_per_epoch = math.ceil(len(train_dataloader_pre) / config.training.gradient_accumulation_steps)
        num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)
    else:
        raise ValueError(f"Unsupported dataset type {config.dataset.type}")

    if config.dataset.und_type == "llava_tuning":
        from llava.llava_instruct_data import get_instruct_data_loader
        train_dataloader_mmu, SYSTEM_PROMPT, SYSTEM_PROMPT_LEN = get_instruct_data_loader(
            tokenizer=tokenizer,
            dataset_path=dataset_config.train_mmu_shards_path_or_url,
            batch_size=config.training.batch_size_mmu,
            num_workers=dataset_config.num_workers,
            world_size=accelerator.num_processes,
            local_rank=accelerator.process_index,
            max_length=preproc_config.max_seq_length - (576 - config.model.showo.num_vq_tokens) + config.act_step,
            # future steps |lvg|*10
            processor=clip_processor,
            return_system_prompt=True)
        # SYSTEM_PROMPT_LEN = 28
    else:
        raise NotImplementedError(f"Unsupported dataset type {config.dataset.und_type}")

    iterables = {
        "pre_flow": train_dataloader_pre,
        "mmu_flow": train_dataloader_mmu,
    }

    combined_dataloader = CombinedLoader(iterables, mode=config.dataset.combined_loader_mode)

    #################################
    #         MODEL RESUME          #
    #################################
    global_step = 0
    first_epoch = 0
    # num_update_steps_per_epoch = math.ceil(
    #     len(train_dataloader_pre) / config.training.gradient_accumulation_steps)
    # num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

    resume_cfg = config.experiment.resume_from_checkpoint
    resume_cfg_str = resume_cfg.strip().lower() if isinstance(resume_cfg, str) else None
    resume_requested = bool(resume_cfg) and resume_cfg_str not in {"", "false", "none", "null", "off", "0", "no"}
    if resume_requested:
        eval_path = config.get("eval_path", None)
        output_dir = Path(config.experiment.output_dir)
        resume_path = None

        if config.eval and eval_path is not None:
            resume_path = Path(eval_path)
        elif isinstance(resume_cfg, str) and resume_cfg_str not in {"latest", "auto", "true", "yes", "on"}:
            maybe_path = Path(resume_cfg)
            candidates = []
            if maybe_path.is_absolute():
                candidates.append(maybe_path)
            else:
                candidates.extend(
                    [
                        output_dir / resume_cfg,
                        output_dir / f"checkpoint-{resume_cfg}",
                    ]
                )
            for candidate in candidates:
                if candidate.exists():
                    resume_path = candidate
                    break
        else:
            dirs = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            if len(dirs) > 0:
                resume_path = output_dir / dirs[-1]

        if resume_path is None:
            raise FileNotFoundError(
                f"Resume requested (experiment.resume_from_checkpoint={resume_cfg}) "
                f"but no checkpoint found in {output_dir}."
            )

        model_candidates = [
            resume_path / "unwrapped_model" / "pytorch_model.bin",
            resume_path / "pytorch_model.bin",
            resume_path / "unwrapped_model" / "model.safetensors",
            resume_path / "model.safetensors",
        ]
        model_path = next((p for p in model_candidates if p.exists()), None)
        if model_path is None:
            raise FileNotFoundError(f"Cannot find model weights under checkpoint path: {resume_path}")

        if resume_path.name.startswith("checkpoint-"):
            try:
                global_step = int(resume_path.name.split("-")[1])
            except Exception:
                global_step = 0
        else:
            global_step = 0

        metadata_path = resume_path / "metadata.json"
        if metadata_path.exists():
            try:
                metadata = json.load(metadata_path.open("r"))
                global_step = int(metadata.get("global_step", global_step))
            except Exception:
                pass

        first_epoch = global_step // num_update_steps_per_epoch
        accelerator.print(f"Resuming from checkpoint {model_path}")
        if model_path.suffix == ".safetensors":
            from safetensors.torch import load_file as safe_load_file

            state_dict = safe_load_file(str(model_path), device="cpu")
        else:
            state_dict = torch.load(str(model_path), map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        del state_dict

    ##################################
    #       Prepare accelerator     #
    #################################
    logger.info("Preparing model, optimizer and dataloaders")
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    vq_model.to(device=accelerator.device)

    if hasattr(model, 'module'):
        mask_dtype = model.module.showo.model.embed_tokens.weight.dtype
    else:
        mask_dtype = model.showo.model.embed_tokens.weight.dtype

    ##################################
    #             Training          #
    #################################
    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {total_batch_size_per_gpu}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")

    @torch.no_grad()
    def prepare_inputs_and_labels(
        pixel_values_or_image_ids,
        texts: Union[str, str],
        min_masking_rate: float = 0.0,
        is_train: bool = True,
    ):
        image_tokens_0 = vq_model.get_code(pixel_values_or_image_ids[0]) + len(uni_prompting.text_tokenizer)
        target_image_tokens_0 = vq_model.get_code(pixel_values_or_image_ids[2]) + len(uni_prompting.text_tokenizer)
        if config.model.vla.num_view == 1:
            image_tokens = image_tokens_0
            target_image_tokens = target_image_tokens_0
        elif config.model.vla.num_view == 2:
            image_tokens_1 = vq_model.get_code(pixel_values_or_image_ids[1]) + len(uni_prompting.text_tokenizer)
            target_image_tokens_1 = vq_model.get_code(pixel_values_or_image_ids[3]) + len(uni_prompting.text_tokenizer)
            image_tokens = torch.cat((image_tokens_0, image_tokens_1), dim=1)
            target_image_tokens = torch.cat((target_image_tokens_0, target_image_tokens_1), dim=1)
        else:
            raise NotImplementedError(f"Num-view {config.model.vla.num_view} not supported")

        # create MLM mask and labels
        input_ids, labels, loss_weight, mask_prob = mask_or_random_replace_tokens_for_future_prediction(
            input_image_tokens=image_tokens,
            target_image_tokens=target_image_tokens,
            mask_id=mask_id,
            config=config,
            mask_schedule=mask_schedule,
            is_train=is_train)
        input_ids, masks, labels = uni_prompting((texts, input_ids, labels), 'pre')

        return input_ids, labels, mask_prob, image_tokens, target_image_tokens

    if accelerator.mixed_precision == "fp16":
        images_feat_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        images_feat_dtype = torch.bfloat16
    else:
        images_feat_dtype = torch.float32
    if accelerator.num_processes == 1:
        images_feat_dtype = torch.float32  # for one gpu training

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    for epoch in range(first_epoch, num_train_epochs):
        model.train()
        if accelerator.is_main_process:
            mmu_generate(
                model,
                vq_model,
                uni_prompting,
                accelerator,
                config,
                global_step + 1,
                clip_processor=clip_processor,
                vision_tower=vision_tower,
                SYSTEM_PROMPT=SYSTEM_PROMPT,
                SYSTEM_PROMPT_LEN=SYSTEM_PROMPT_LEN,
            )
        for batch, batch_idx, dataloader_idx in combined_dataloader:
            # for loss calculation

            batch_size_pre = len(batch["pre_flow"]["input_ids"])
            batch_size_mmu = batch["mmu_flow"]["input_ids"].shape[0]

            pixel_values_static, pixel_values_gripper, texts, pixel_values_static_future, pixel_values_gripper_future = (
                batch["pre_flow"]["images_static"],
                batch["pre_flow"]["images_gripper"],
                batch["pre_flow"]["input_ids"],
                batch["pre_flow"]["images_static_future"],
                batch["pre_flow"]["images_gripper_future"],
            )
            actions = batch["pre_flow"]["actions"].to(
                accelerator.device, non_blocking=True).to(images_feat_dtype)  # (b, timestep, 7) (b,10,7)
            pixel_values_static = pixel_values_static.to(accelerator.device, non_blocking=True)
            pixel_values_gripper = pixel_values_gripper.to(accelerator.device, non_blocking=True)
            pixel_values_static_future = pixel_values_static_future.to(accelerator.device, non_blocking=True)
            pixel_values_gripper_future = pixel_values_gripper_future.to(accelerator.device, non_blocking=True)
            data_time_m.update(time.time() - end)
            pixel_values = (pixel_values_static, pixel_values_gripper, pixel_values_static_future,
                            pixel_values_gripper_future)
            # Encode images to image tokens, mask them and create input and labels
            (
                input_ids,
                labels,
                mask_prob,
                image_tokens_ori,
                target_image_tokens,
            ) = prepare_inputs_and_labels(pixel_values, texts, config.training.min_masking_rate)

            pre_text_clean_ids = None
            pre_text_valid_mask = None
            if text_denoise_enabled:
                text_prefix_len = int(config.dataset.preprocessing.max_seq_length) + 1
                pad_id = int(uni_prompting.sptids_dict['<|pad|>'])
                pre_text_clean_ids = input_ids[:, :text_prefix_len].clone()
                pre_text_valid_mask = pre_text_clean_ids != pad_id
                # keep task token (<|t2i|>) unmasked
                pre_text_valid_mask[:, 0] = False

            attention_mask = create_attention_mask_predict_next_for_future_prediction(
                input_ids,
                pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                rm_pad_in_image=True,
                return_inverse_mask=True)
            attention_mask = attention_mask.to(mask_dtype)
            mmu_text_clean_ids = None
            mmu_text_valid_mask = None
            if config.dataset.und_type == "llava_tuning":
                pixel_values_mmu, input_ids_mmu_raw, labels_mmu_raw, input_ids_system = (
                    batch["mmu_flow"]["images"],
                    batch["mmu_flow"]["input_ids"],
                    batch["mmu_flow"]["labels"],
                    batch["mmu_flow"]["input_ids_system"],
                )

                pixel_values_mmu = pixel_values_mmu.to(accelerator.device, non_blocking=True)
                input_ids_mmu_raw = input_ids_mmu_raw.to(accelerator.device, non_blocking=True)
                labels_mmu_raw = labels_mmu_raw.to(accelerator.device, non_blocking=True)
                input_ids_system = input_ids_system.to(accelerator.device, non_blocking=True)

                input_ids_mmu = torch.cat([
                    (torch.ones(input_ids_mmu_raw.shape[0], 1) * uni_prompting.sptids_dict['<|mmu|>']).to(
                        accelerator.device),
                    input_ids_system,
                    (torch.ones(input_ids_mmu_raw.shape[0], 1) * uni_prompting.sptids_dict['<|soi|>']).to(
                        accelerator.device),
                    (torch.ones(input_ids_mmu_raw.shape[0], 1) * uni_prompting.sptids_dict['<|eoi|>']).to(
                        accelerator.device),
                    input_ids_mmu_raw,
                ],
                                          dim=1).long()
                mmu_text_clean_ids = None
                mmu_text_valid_mask = None
                if text_denoise_enabled:
                    mmu_text_clean_ids = input_ids_mmu.clone()
                    mmu_prefix_len = 1 + input_ids_system.shape[1] + 2  # <|mmu|> + system + <|soi|><|eoi|>
                    mmu_text_valid_mask = torch.zeros_like(input_ids_mmu, dtype=torch.bool)
                    mmu_text_valid_mask[:, mmu_prefix_len:] = labels_mmu_raw != uni_prompting.ignore_id

                images_feat = vision_tower(pixel_values_mmu)
                if hasattr(model, 'module'):
                    mm_projector = model.module.mm_projector
                    text_embed_tokens = model.module.showo.model.embed_tokens
                else:
                    mm_projector = model.mm_projector
                    text_embed_tokens = model.showo.model.embed_tokens
                # Keep projector input dtype aligned with projector weights for torchrun non-DeepSpeed training.
                projector_dtype = next(mm_projector.parameters()).dtype
                images_embeddings = mm_projector(images_feat.to(projector_dtype))
                text_embeddings = text_embed_tokens(input_ids_mmu).to(images_embeddings.dtype)
                split_idx = 1 + input_ids_system.shape[1] + 1  # <|mmu|> + system + <|soi|>
                part1 = text_embeddings[:, :split_idx, :]
                part2 = text_embeddings[:, split_idx:, :]

                input_embeddings = torch.cat((part1, images_embeddings, part2), dim=1)

                labels_mmu = torch.cat(
                    [
                        (torch.ones(input_ids_mmu.shape[0], 1) * uni_prompting.ignore_id).to(accelerator.device),  # mmu
                        torch.ones_like(input_ids_system) * uni_prompting.ignore_id,  # ignore system prompt
                        (torch.ones(input_ids_mmu.shape[0], 1) * uni_prompting.ignore_id).to(accelerator.device),  # soi
                        torch.ones_like(images_embeddings[:, :, 0]) * uni_prompting.ignore_id,  # ignore image embedding
                        (torch.ones(input_ids_mmu.shape[0], 1) * uni_prompting.ignore_id).to(accelerator.device),  # eoi
                        labels_mmu_raw.to(accelerator.device)
                    ],
                    dim=1).long()

            else:
                raise NotImplementedError
            attention_mask_mmu = create_attention_mask_for_mmu_vit(
                input_embeddings, system_prompt_len=SYSTEM_PROMPT_LEN)
            attention_mask_mmu = attention_mask_mmu.to(mask_dtype)
            attention_mask = torch.cat([attention_mask, attention_mask_mmu], dim=0)

            if hasattr(model, 'module'):
                text_embeddings_img_text = model.module.showo.model.embed_tokens(input_ids)
            else:
                text_embeddings_img_text = model.showo.model.embed_tokens(input_ids)
            input_embeddings = torch.cat([text_embeddings_img_text, input_embeddings], dim=0)

            labels = torch.cat((labels, labels_mmu.to(input_ids.device)), dim=0)
            text_bd_pre_batch = None
            text_bd_mmu_batch = None
            text_mdm_pre_batch = None
            text_mdm_mmu_batch = None
            if bd_text_enabled and pre_text_clean_ids is not None and pre_text_valid_mask is not None:
                text_pre_block_sizes = _sample_block_sizes_for_batch(
                    batch_size=pre_text_clean_ids.shape[0],
                    device=pre_text_clean_ids.device,
                    default_block_size=bd_block_size,
                    elastic_enabled=bd_elastic_enabled,
                    candidates=bd_block_size_candidates,
                )
                text_bd_pre_batch = build_native_block_diffusion_batch(
                    clean_ids=pre_text_clean_ids,
                    valid_mask=pre_text_valid_mask,
                    mask_token_id=mask_id,
                    block_size=bd_block_size,
                    block_sizes=text_pre_block_sizes,
                    eps=bd_mask_eps,
                    ignore_id=-100,
                    complementary_mask=bd_complementary_mask,
                )
            if bd_text_enabled and mmu_text_clean_ids is not None and mmu_text_valid_mask is not None:
                text_mmu_block_sizes = _sample_block_sizes_for_batch(
                    batch_size=mmu_text_clean_ids.shape[0],
                    device=mmu_text_clean_ids.device,
                    default_block_size=bd_block_size,
                    elastic_enabled=bd_elastic_enabled,
                    candidates=bd_block_size_candidates,
                )
                text_bd_mmu_batch = build_native_block_diffusion_batch(
                    clean_ids=mmu_text_clean_ids,
                    valid_mask=mmu_text_valid_mask,
                    mask_token_id=mask_id,
                    block_size=bd_block_size,
                    block_sizes=text_mmu_block_sizes,
                    eps=bd_mask_eps,
                    ignore_id=-100,
                    complementary_mask=bd_complementary_mask,
                )
            if text_mdm_enabled and pre_text_clean_ids is not None and pre_text_valid_mask is not None:
                text_mdm_pre_batch = build_text_mdm_batch(
                    clean_ids=pre_text_clean_ids,
                    valid_mask=pre_text_valid_mask,
                    mask_token_id=mask_id,
                    min_mask_ratio=text_mdm_min_mask_ratio,
                    max_mask_ratio=text_mdm_max_mask_ratio,
                    ignore_id=-100,
                )
            if text_mdm_enabled and mmu_text_clean_ids is not None and mmu_text_valid_mask is not None:
                text_mdm_mmu_batch = build_text_mdm_batch(
                    clean_ids=mmu_text_clean_ids,
                    valid_mask=mmu_text_valid_mask,
                    mask_token_id=mask_id,
                    min_mask_ratio=text_mdm_min_mask_ratio,
                    max_mask_ratio=text_mdm_max_mask_ratio,
                    ignore_id=-100,
                )

            action_bd_batch = None
            action_bd_prefix = None
            if bd_action_enabled:
                action_tokens = quantize_actions_to_tokens(
                    actions=actions.float(),
                    num_bins=bd_action_num_bins,
                    token_offset=action_token_offset,
                ).reshape(actions.shape[0], -1)
                action_valid_mask = torch.ones_like(action_tokens, dtype=torch.bool)
                action_block_sizes = _sample_block_sizes_for_batch(
                    batch_size=action_tokens.shape[0],
                    device=action_tokens.device,
                    default_block_size=bd_block_size,
                    elastic_enabled=bd_elastic_enabled,
                    candidates=bd_block_size_candidates,
                )
                action_bd_batch = build_native_block_diffusion_batch(
                    clean_ids=action_tokens,
                    valid_mask=action_valid_mask,
                    mask_token_id=mask_id,
                    block_size=bd_block_size,
                    block_sizes=action_block_sizes,
                    eps=bd_mask_eps,
                    ignore_id=-100,
                    complementary_mask=bd_complementary_mask,
                )
                action_bd_prefix = input_ids if action_bd_batch.repeats == 1 else torch.cat([input_ids, input_ids], dim=0)

            if global_step == 0 and epoch == 0:
                logger.info("Input ids: {}".format(input_ids))
                logger.info("Labels: {}".format(labels))
            # raise NotImplementedError
            with accelerator.accumulate(model):
                logits, loss_pre, loss_mmu, loss_act = model(
                    input_ids=input_ids,  # input_ids is not used (input_embeddings are provided)
                    input_embeddings=input_embeddings,
                    attention_mask=attention_mask,
                    labels=labels,
                    label_smoothing=config.training.label_smoothing,
                    batch_size_pre=batch_size_pre,
                    batch_size_mmu=batch_size_mmu,
                    max_seq_length=config.dataset.preprocessing.max_seq_length,
                    actions=actions,
                    clip_pad_tokens=config.training.clip_pad_tokens)
                if bd_text_enabled:
                    text_bd_terms = []
                    if text_bd_pre_batch is not None:
                        text_pre_mask = build_block_diffusion_attention_mask(
                            seq_len=text_bd_pre_batch.seq_len,
                            block_size=text_bd_pre_batch.block_sizes,
                            batch_size=text_bd_pre_batch.input_ids.shape[0],
                            dtype=mask_dtype,
                            device=text_bd_pre_batch.input_ids.device,
                        )
                        logits_text_pre = model(input_ids=text_bd_pre_batch.input_ids, attention_mask=text_pre_mask)
                        text_bd_terms.append(
                            block_diffusion_token_shift_loss(
                                logits=logits_text_pre,
                                labels_first_half=text_bd_pre_batch.labels,
                                first_half_len=text_bd_pre_batch.seq_len,
                                ignore_id=-100,
                            ))
                    if text_bd_mmu_batch is not None:
                        text_mmu_mask = build_block_diffusion_attention_mask(
                            seq_len=text_bd_mmu_batch.seq_len,
                            block_size=text_bd_mmu_batch.block_sizes,
                            batch_size=text_bd_mmu_batch.input_ids.shape[0],
                            dtype=mask_dtype,
                            device=text_bd_mmu_batch.input_ids.device,
                        )
                        logits_text_mmu = model(input_ids=text_bd_mmu_batch.input_ids, attention_mask=text_mmu_mask)
                        text_bd_terms.append(
                            block_diffusion_token_shift_loss(
                                logits=logits_text_mmu,
                                labels_first_half=text_bd_mmu_batch.labels,
                                first_half_len=text_bd_mmu_batch.seq_len,
                                ignore_id=-100,
                            ))
                    if len(text_bd_terms) > 0:
                        loss_text_bd = torch.stack(text_bd_terms).mean()
                    else:
                        loss_text_bd = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
                else:
                    loss_text_bd = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

                if text_mdm_enabled:
                    text_mdm_terms = []
                    if text_mdm_pre_batch is not None:
                        text_pre_ids, text_pre_labels = text_mdm_pre_batch
                        text_pre_mask = torch.zeros(
                            (text_pre_ids.shape[0], 1, text_pre_ids.shape[1], text_pre_ids.shape[1]),
                            dtype=mask_dtype,
                            device=text_pre_ids.device,
                        )
                        logits_text_pre_mdm = model(input_ids=text_pre_ids, attention_mask=text_pre_mask)
                        text_mdm_terms.append(
                            block_diffusion_token_shift_loss(
                                logits=logits_text_pre_mdm,
                                labels_first_half=text_pre_labels,
                                first_half_len=text_pre_ids.shape[1],
                                ignore_id=-100,
                            ))
                    if text_mdm_mmu_batch is not None:
                        text_mmu_ids, text_mmu_labels = text_mdm_mmu_batch
                        text_mmu_mask = torch.zeros(
                            (text_mmu_ids.shape[0], 1, text_mmu_ids.shape[1], text_mmu_ids.shape[1]),
                            dtype=mask_dtype,
                            device=text_mmu_ids.device,
                        )
                        logits_text_mmu_mdm = model(input_ids=text_mmu_ids, attention_mask=text_mmu_mask)
                        text_mdm_terms.append(
                            block_diffusion_token_shift_loss(
                                logits=logits_text_mmu_mdm,
                                labels_first_half=text_mmu_labels,
                                first_half_len=text_mmu_ids.shape[1],
                                ignore_id=-100,
                            ))
                    if len(text_mdm_terms) > 0:
                        loss_text_mdm = torch.stack(text_mdm_terms).mean()
                    else:
                        loss_text_mdm = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
                else:
                    loss_text_mdm = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

                if text_objective == "ar":
                    loss_text_obj = loss_mmu
                elif text_objective == "mdm":
                    loss_text_obj = loss_text_mdm
                else:  # text_objective == "bd"
                    loss_text_obj = loss_text_bd

                if bd_action_enabled and action_bd_batch is not None and action_bd_prefix is not None:
                    action_bd_inputs = torch.cat([action_bd_prefix, action_bd_batch.input_ids], dim=1)
                    prefix_len = action_bd_prefix.shape[1]
                    action_bd_mask = build_prefixed_block_diffusion_attention_mask(
                        prefix_len=prefix_len,
                        seq_len=action_bd_batch.seq_len,
                        block_size=action_bd_batch.block_sizes,
                        batch_size=action_bd_inputs.shape[0],
                        dtype=mask_dtype,
                        device=action_bd_inputs.device,
                    )
                    logits_action_bd = model(input_ids=action_bd_inputs, attention_mask=action_bd_mask)
                    action_labels_half = torch.full(
                        (action_bd_inputs.shape[0], prefix_len + action_bd_batch.seq_len),
                        -100,
                        dtype=action_bd_batch.labels.dtype,
                        device=action_bd_inputs.device,
                    )
                    action_labels_half[:, prefix_len:] = action_bd_batch.labels
                    loss_action_bd = block_diffusion_token_shift_loss(
                        logits=logits_action_bd,
                        labels_first_half=action_labels_half,
                        first_half_len=prefix_len + action_bd_batch.seq_len,
                        ignore_id=-100,
                    )
                else:
                    loss_action_bd = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
                # loss_act: action loss, loss_pre: image prediction loss, loss_mmu: original mmu lm loss
                avg_loss_pre = accelerator.gather(loss_pre.repeat(config.training.batch_size_pre)).mean()
                avg_loss_act = accelerator.gather(loss_act.repeat(config.training.batch_size_pre)).mean()
                # avg_loss_mmu = accelerator.gather(loss_mmu.repeat(config.training.batch_size_mmu)).mean()
                avg_loss_mmu = accelerator.gather(loss_mmu.repeat(config.training.batch_size_mmu)).mean()
                avg_loss_text_bd = accelerator.gather(loss_text_bd.repeat(config.training.batch_size_pre)).mean()
                avg_loss_text_mdm = accelerator.gather(loss_text_mdm.repeat(config.training.batch_size_pre)).mean()
                avg_loss_text_obj = accelerator.gather(loss_text_obj.repeat(config.training.batch_size_pre)).mean()
                avg_loss_action_bd = accelerator.gather(loss_action_bd.repeat(config.training.batch_size_pre)).mean()

                # Keep the same 3-term skeleton as UPVLA/ARVLA:
                #   total = pre + action + text
                # while text objective itself is selected by training.text_objective.
                loss_action_total = loss_act + loss_action_bd
                avg_loss_a_total = accelerator.gather(loss_action_total.repeat(config.training.batch_size_pre)).mean()
                loss = (
                    config.training.pre_coeff * loss_pre
                    + config.training.act_coeff * loss_action_total
                    + config.training.mmu_coeff * loss_text_obj
                )
                # avg_masking_rate = accelerator.gather(mask_prob.repeat(config.training.batch_size_pre)).mean()

                accelerator.backward(loss)

                if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()

                # log gradient norm before zeroing it
                if (accelerator.sync_gradients and (global_step + 1) % config.experiment.log_grad_norm_every == 0 and
                        accelerator.is_main_process):
                    log_grad_norm(model, accelerator, global_step + 1)

                optimizer.zero_grad(set_to_none=True)
                # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
                # just eval a checkpoint
                # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
                if config.eval and accelerator.is_main_process:
                    generate_images(
                        model,
                        vq_model,
                        uni_prompting,
                        accelerator,
                        config,
                        global_step + 1,
                        mask_schedule=mask_schedule,
                        input_image_tokens=image_tokens_ori,
                        clip_processor=clip_processor,
                        vision_tower=vision_tower,
                    )

                    visualize_predictions(
                        accelerator=accelerator,
                        model=model,
                        vq_model=vq_model,
                        uni_prompting=uni_prompting,
                        config=config,
                        global_step=global_step + 1,
                        input_ids=input_ids,
                        target_image_tokens=target_image_tokens,
                        batch_images=pixel_values,
                        texts=texts,
                        logits=logits,
                    )

                    mmu_generate(
                        model,
                        vq_model,
                        uni_prompting,
                        accelerator,
                        config,
                        global_step + 1,
                        clip_processor=clip_processor,
                        vision_tower=vision_tower,
                        SYSTEM_PROMPT=SYSTEM_PROMPT,
                        SYSTEM_PROMPT_LEN=SYSTEM_PROMPT_LEN,
                    )

                    evaluate_validation(
                        model=model,
                        vq_model=vq_model,
                        uni_prompting=uni_prompting,
                        accelerator=accelerator,
                        config=config,
                        global_step=global_step + 1,
                        clip_processor=clip_processor,
                        vision_tower=vision_tower,
                    )
                    raise NotImplementedError("Evaluation done")
                # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
                # just eval a checkpoint
                # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:

                batch_time_m.update(time.time() - end)
                end = time.time()

                # Log metrics
                if (global_step + 1) % config.experiment.log_every == 0:
                    samples_per_second_per_gpu = (
                        config.training.gradient_accumulation_steps * total_batch_size_per_gpu / batch_time_m.val)
                    logs = {
                        "step_loss_pre": avg_loss_pre.item(),
                        "step_loss_mmu": avg_loss_mmu.item(),
                        "step_loss_a": avg_loss_act.item(),
                        "step_loss_a_total": avg_loss_a_total.item(),
                        "step_loss_text_bd": avg_loss_text_bd.item(),
                        "step_loss_text_mdm": avg_loss_text_mdm.item(),
                        "step_loss_text_obj": avg_loss_text_obj.item(),
                        "step_loss_action_bd": avg_loss_action_bd.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        # "avg_masking_rate": avg_masking_rate.item(),
                        "samples/sec/gpu": samples_per_second_per_gpu,
                        "data_time": data_time_m.val,
                        "batch_time": batch_time_m.val,
                    }
                    accelerator.log(logs, step=global_step + 1)

                    logger.info(f"Step: {global_step + 1} "
                                f"Loss_pre: {avg_loss_pre.item():0.4f} "
                                f"Loss_mmu: {avg_loss_mmu.item():0.4f} "
                                f"Loss_a: {avg_loss_act.item():0.4f} "
                                f"Loss_a_total: {avg_loss_a_total.item():0.4f} "
                                f"Loss_text_bd: {avg_loss_text_bd.item():0.4f} "
                                f"Loss_text_mdm: {avg_loss_text_mdm.item():0.4f} "
                                f"Loss_text_obj({text_objective}): {avg_loss_text_obj.item():0.4f} "
                                f"Loss_action_bd: {avg_loss_action_bd.item():0.4f} "
                                f"Data (t): {data_time_m.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                                f"Batch (t): {batch_time_m.val:0.4f} "
                                f"LR: {lr_scheduler.get_last_lr()[0]:0.6f}")

                    # resetting batch / data time meters per log window
                    batch_time_m.reset()
                    data_time_m.reset()

                # Save model checkpoint
                if (global_step + 1) % config.experiment.save_every == 0:
                    save_checkpoint(model, config, accelerator, global_step + 1)

                if (global_step + 1) % config.experiment.generate_every == 0 and accelerator.is_main_process:
                    generate_images(
                        model,
                        vq_model,
                        uni_prompting,
                        accelerator,
                        config,
                        global_step + 1,
                        mask_schedule=mask_schedule,
                        input_image_tokens=image_tokens_ori,
                        clip_processor=clip_processor,
                        vision_tower=vision_tower,
                    )

                    visualize_predictions(
                        accelerator=accelerator,
                        model=model,
                        vq_model=vq_model,
                        uni_prompting=uni_prompting,
                        config=config,
                        global_step=global_step + 1,
                        input_ids=input_ids,
                        target_image_tokens=target_image_tokens,
                        batch_images=pixel_values,
                        texts=texts,
                        logits=logits,
                    )

                    mmu_generate(
                        model,
                        vq_model,
                        uni_prompting,
                        accelerator,
                        config,
                        global_step + 1,
                        clip_processor=clip_processor,
                        vision_tower=vision_tower,
                        SYSTEM_PROMPT=SYSTEM_PROMPT,
                        SYSTEM_PROMPT_LEN=SYSTEM_PROMPT_LEN,
                    )

                    evaluate_validation(
                        model=model,
                        vq_model=vq_model,
                        uni_prompting=uni_prompting,
                        accelerator=accelerator,
                        config=config,
                        global_step=global_step + 1,
                        clip_processor=clip_processor,
                        vision_tower=vision_tower,
                    )

                global_step += 1

            # Stop training if max steps is reached
            if global_step >= config.training.max_train_steps:
                break
            # End for

    accelerator.wait_for_everyone()

    # Evaluate and save checkpoint at the end of training
    save_checkpoint(model, config, accelerator, global_step)

    # Save the final trained checkpoint
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        model.save_pretrained(config.experiment.output_dir, safe_serialization=False)

    accelerator.end_training()


@torch.no_grad()
def visualize_predictions(
    accelerator,
    model,
    vq_model,
    uni_prompting,
    config,
    global_step,
    input_ids,
    target_image_tokens,
    batch_images,
    texts,
    logits,
):
    logger.info("Visualizing predictions...")
    model.eval()
    predictions = logits[:config.training.batch_size_pre,
                         -(config.model.showo.num_vq_tokens * config.model.vla.num_view + 1) - config.act_step:-1 -
                         config.act_step:,
                         config.model.showo.llm_vocab_size + config.model.showo.num_new_special_tokens:-1]
    predictions = predictions.argmax(axis=-1)

    mask_token_id = config.model.showo.vocab_size - 1 - len(uni_prompting.text_tokenizer)
    input_ids = input_ids[:config.training.batch_size_pre,
                          -(config.model.showo.num_vq_tokens * config.model.vla.num_view + 1) - config.act_step:-1 -
                          config.act_step:] - len(uni_prompting.text_tokenizer)
    mask_ratio = list((torch.where(input_ids == mask_token_id, 1, 0).sum(dim=-1) /
                       config.model.showo.num_vq_tokens).cpu().numpy())  # should be zero
    # predicted_images = torch.where(input_ids == mask_token_id, input_ids,predictions)
    predicted_images = predictions
    if config.model.vla.num_view == 1:
        target_tokens = [target_image_tokens]
        ori_images = [batch_images[0]]
        future_images = [batch_images[2]]
        predicted_images = [predicted_images]
    elif config.model.vla.num_view == 2:
        image_tokens_ori_static, image_tokens_ori_gripper = target_image_tokens.chunk(2, dim=1)
        target_tokens = [image_tokens_ori_static, image_tokens_ori_gripper]
        ori_images = [batch_images[0], batch_images[1]]
        future_images = [batch_images[2], batch_images[3]]
        predicted_static, predicted_gripper = predicted_images.chunk(2, dim=1)
        predicted_images = [predicted_static, predicted_gripper]
    else:
        raise NotImplementedError(f"Num-view {config.model.vla.num_view} not supported")
    for i, (target_tokens_i, ori_images_i, future_images_i,
            predicted_images_i) in enumerate(zip(target_tokens, ori_images, future_images, predicted_images)):
        recons_images = vq_model.decode_code(target_tokens_i - len(uni_prompting.text_tokenizer))
        recons_images = torch.clamp((recons_images + 1.0) / 2.0, min=0.0, max=1.0)
        recons_images *= 255.0
        recons_images = recons_images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

        images = torch.clamp((ori_images_i + 1.0) / 2.0, min=0.0, max=1.0)
        images *= 255.0
        images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

        groundtruth_images = torch.clamp((future_images_i + 1.0) / 2.0, min=0.0, max=1.0)
        groundtruth_images *= 255.0
        groundtruth_images = groundtruth_images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

        predicted_images_ = vq_model.decode_code(predicted_images_i)
        predicted_images_ = torch.clamp((predicted_images_ + 1.0) / 2.0, min=0.0, max=1.0)
        predicted_images_ *= 255.0
        predicted_images_ = predicted_images_.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        predicted_images_ = np.concatenate((images, groundtruth_images, recons_images, predicted_images_), 2)
        pil_images = [Image.fromarray(image) for image in predicted_images_]

        image_captions = [
            f"mask ratio: {r:0.2f} (should be 0.0)\ncaption: {texts[j]}"
            for j, r in enumerate(mask_ratio)
        ]
        _log_images(
            accelerator=accelerator,
            key=(
                f"(view {i + 1}/{config.model.vla.num_view}): "
                "Input original images v.s. Future images v.s Ground truth (Recon) v.s. Predicted images"
            ),
            pil_images=pil_images,
            captions=image_captions,
            step=global_step,
        )

    model.train()


@torch.no_grad()
def generate_images(
    model,
    vq_model,
    uni_prompting,
    accelerator,
    config,
    global_step,
    mask_schedule,
    input_image_tokens=None,
    vision_tower=None,
    clip_processor=None,
):
    logger.info("Generating images...")
    model.eval()

    # read validation prompts from file
    with open(config.dataset.params.validation_prompts_file, "r") as f:
        validation_prompts = f.read().splitlines()

    if hasattr(model, 'module'):
        mask_dtype = model.module.showo.model.embed_tokens.weight.dtype
    else:
        mask_dtype = model.showo.model.embed_tokens.weight.dtype

    mask_token_id = config.model.showo.vocab_size - 1
    if input_image_tokens is not None:
        random_indices = torch.randint(0, input_image_tokens.shape[0], (len(validation_prompts) - 1,))
        image_tokens = input_image_tokens[random_indices]
        new_image_1 = Image.open("./validation_samples/frame_static.jpg").convert('RGB')
        new_image_1 = image_transform(
            new_image_1, resolution=config.dataset.preprocessing.resolution).to(accelerator.device).unsqueeze(0)
        new_image_1 = vq_model.get_code(new_image_1) + len(uni_prompting.text_tokenizer)
        if config.model.vla.num_view == 1:
            image_tokens = torch.cat((image_tokens, new_image_1), dim=0)
        elif config.model.vla.num_view == 2:
            new_image_2 = Image.open("./validation_samples/frame_gripper.png").convert('RGB')
            new_image_2 = image_transform(
                new_image_2, resolution=config.dataset.preprocessing.resolution).to(accelerator.device).unsqueeze(0)
            new_image_2 = vq_model.get_code(new_image_2) + len(uni_prompting.text_tokenizer)
            new_image_tokens = torch.cat((new_image_1, new_image_2), dim=1)
            image_tokens = torch.cat((image_tokens, new_image_tokens), dim=0)
        else:
            raise NotImplementedError(f"Num-view {config.model.vla.num_view} not supported")
    else:
        image_tokens = torch.ones(
            (len(validation_prompts), config.model.showo.num_vq_tokens), dtype=torch.long,
            device=accelerator.device) * mask_token_id
        assert NotImplementedError

    input_ids, _ = uni_prompting((validation_prompts, image_tokens), 'pre_gen')
    if config.training.guidance_scale > 0:
        uncond_input_ids, _ = uni_prompting(([''] * len(validation_prompts), image_tokens), 'pre_gen')
        attention_mask = create_attention_mask_predict_next_for_future_prediction(
            torch.cat([input_ids, uncond_input_ids], dim=0),
            pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
            soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
            eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
            rm_pad_in_image=True).to(mask_dtype)
        assert NotImplementedError
    else:
        attention_mask = create_attention_mask_predict_next_for_future_prediction(
            input_ids,
            pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
            soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
            eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
            rm_pad_in_image=True).to(mask_dtype)
        uncond_input_ids = None

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    with torch.autocast("cuda", dtype=weight_dtype, enabled=accelerator.mixed_precision != "no"):
        # Generate images
        gen_token_ids = accelerator.unwrap_model(model).pre_pad_predict(
            input_ids=input_ids,
            uncond_input_ids=uncond_input_ids,
            attention_mask=attention_mask,
            guidance_scale=config.training.guidance_scale,
            temperature=config.training.get("generation_temperature", 1.0),
            timesteps=config.training.generation_timesteps,
            noise_schedule=mask_schedule,
            noise_type=config.training.get("noise_type", "mask"),
            predict_all_tokens=config.training.get("predict_all_tokens", False),
            seq_len=config.model.showo.num_vq_tokens,
            uni_prompting=uni_prompting,
            config=config,
        )
    # In the beginning of training, the model is not fully trained and the generated token ids can be out of range
    # so we clamp them to the correct range.
    if config.model.vla.num_view == 1:
        image_tokens = [image_tokens]
        gen_token_ids = [gen_token_ids]
    elif config.model.vla.num_view == 2:
        image_tokens_ori_static, image_tokens_ori_gripper = image_tokens.chunk(2, dim=1)
        image_tokens = [image_tokens_ori_static, image_tokens_ori_gripper]
        gen_token_ids_static, gen_token_ids_gripper = gen_token_ids.chunk(2, dim=1)
        gen_token_ids = [gen_token_ids_static, gen_token_ids_gripper]
    else:
        raise NotImplementedError(f"Num-view {config.model.vla.num_view} not supported")
    for i, (image_tokens_i, gen_token_ids_i) in enumerate(zip(image_tokens, gen_token_ids)):
        gen_token_ids_i = torch.clamp(
            gen_token_ids_i, max=accelerator.unwrap_model(model).config.codebook_size - 1, min=0)
        images = vq_model.decode_code(gen_token_ids_i)
        # Convert to PIL images
        gen_images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
        gen_images *= 255.0
        gen_images = gen_images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        # print(image_tokens_i- len(uni_prompting.text_tokenizer))
        recons_images = vq_model.decode_code(image_tokens_i - len(uni_prompting.text_tokenizer))

        recons_images = torch.clamp((recons_images + 1.0) / 2.0, min=0.0, max=1.0)
        recons_images *= 255.0
        recons_images = recons_images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        target_images = np.concatenate([recons_images, gen_images], axis=2)
        pil_images = [Image.fromarray(image) for image in target_images]
        _log_images(
            accelerator=accelerator,
            key=f"(view {i + 1}/{config.model.vla.num_view}): Generated images (left input recon/right predict future)",
            pil_images=pil_images,
            captions=validation_prompts,
            step=global_step,
        )
    model.train()


def save_checkpoint(model, config, accelerator, global_step):
    output_dir = config.experiment.output_dir
    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)

    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
    if accelerator.is_main_process and checkpoints_total_limit is not None:
        checkpoints = os.listdir(output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
        if len(checkpoints) >= checkpoints_total_limit:
            num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]

            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)

    save_path = Path(output_dir) / f"checkpoint-{global_step}"

    # retrieve the model on all processes for deepspeed stage 3 to work then save on one process (we are not using stage 3 yet)
    # XXX: could also make this conditional on deepspeed
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            save_path / "unwrapped_model",
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=False)
        json.dump({"global_step": global_step}, (save_path / "metadata.json").open("w+"))
        logger.info(f"Saved state to {save_path}")


def log_grad_norm(model, accelerator, global_step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / grads.numel()).item()
            accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)


@torch.no_grad()
def mmu_generate(
    model,
    vq_model,
    uni_prompting,
    accelerator,
    config,
    global_step,
    vision_tower=None,
    clip_processor=None,
    SYSTEM_PROMPT="",
    SYSTEM_PROMPT_LEN=0,
):
    logger.info("Generating mmu answers...")
    model.eval()
    if hasattr(model, 'module'):
        mask_dtype = model.module.showo.model.embed_tokens.weight.dtype
    else:
        mask_dtype = model.showo.model.embed_tokens.weight.dtype
    temperature = 0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k = 1  # retain only the top_k most likely tokens, clamp others to have 0 probability

    file_list = os.listdir(config.mmu_image_root)
    file_list = [file for file in file_list if file.endswith('.jpg') or file.endswith('.png')]
    responses = ['' for i in range(len(file_list))]
    images = []
    questions = config.question.split(' *** ')
    if accelerator.mixed_precision == "fp16":
        images_feat_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        images_feat_dtype = torch.bfloat16
    else:
        images_feat_dtype = torch.float32
    if accelerator.num_processes == 1:
        images_feat_dtype = torch.float32  # for one gpu training
    for i, file_name in enumerate(file_list):
        image_path = os.path.join(config.mmu_image_root, file_name)
        image_ori = Image.open(image_path).convert("RGB")
        image = image_transform(image_ori, resolution=config.dataset.params.resolution).to(accelerator.device)
        image = image.unsqueeze(0)  # 1,3,512,512
        images.append(image)
        pixel_values = clip_processor.preprocess(image_ori, return_tensors="pt")["pixel_values"][0]  # 3,336,336
        # image_tokens = vq_model.get_code(image) + len(uni_prompting.text_tokenizer)
        batch_size = 1
        for question in questions:
            conv = conversation_lib.default_conversation.copy()
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()

            question_input = []
            question_input.append(prompt_question.strip())
            # question_input=[question]
            # print(question_input)

            input_ids_system = [
                uni_prompting.text_tokenizer(SYSTEM_PROMPT, return_tensors="pt", padding="longest").input_ids
                for _ in range(batch_size)
            ]
            input_ids_system = torch.stack(input_ids_system, dim=0)
            # assert input_ids_system.shape[-1] == 28
            input_ids_system = input_ids_system.to(accelerator.device)
            input_ids_system = input_ids_system[0]

            input_ids = [
                uni_prompting.text_tokenizer(prompt, return_tensors="pt", padding="longest").input_ids
                for prompt in question_input
            ]

            input_ids = torch.stack(input_ids)
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=uni_prompting.text_tokenizer.pad_token_id)
            input_ids = torch.tensor(input_ids).to(accelerator.device).squeeze(0)  # 1, 13
            input_ids_llava = torch.cat(
                [
                    (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|mmu|>']).to(accelerator.device),
                    input_ids_system,
                    (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|soi|>']).to(accelerator.device),
                    # place your img embedding here
                    (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|eoi|>']).to(accelerator.device),
                    input_ids,
                ],
                dim=1).long()  # 1, 44
            # print(input_ids_llava)

            images_embeddings = vision_tower(pixel_values[None]).to(images_feat_dtype)
            # print(images_embeddings[0][1])
            if accelerator.mixed_precision == "fp16":
                weight_dtype = torch.float16
            elif accelerator.mixed_precision == "bf16":
                weight_dtype = torch.bfloat16
            else:
                weight_dtype = torch.float32
            with torch.autocast("cuda", dtype=weight_dtype, enabled=accelerator.mixed_precision != "no"):
                unwrap_model = accelerator.unwrap_model(model)
                images_embeddings = unwrap_model.mm_projector(images_embeddings)
                # print(images_embeddings[0][1])
                text_embeddings = unwrap_model.showo.model.embed_tokens(input_ids_llava)
                # print(text_embeddings[0][1])
                # Full input seq
                part1 = text_embeddings[:, :2 + SYSTEM_PROMPT_LEN, :]
                part2 = text_embeddings[:, 2 + SYSTEM_PROMPT_LEN:, :]
                input_embeddings = torch.cat((part1, images_embeddings, part2), dim=1)  # 1, 620, 2048
                attention_mask_llava = create_attention_mask_for_mmu_vit(
                    input_embeddings, system_prompt_len=SYSTEM_PROMPT_LEN).to(mask_dtype)  # 1,1, 620,620
                # print(attention_mask_llava)
                cont_toks_list = unwrap_model.mmu_generate(
                    input_embeddings=input_embeddings,
                    attention_mask=attention_mask_llava[0].unsqueeze(0),
                    max_new_tokens=config.max_new_tokens,
                    top_k=top_k,
                    eot_token=uni_prompting.text_tokenizer.eos_token_id)

            cont_toks_list = torch.stack(cont_toks_list).squeeze()[None]

            text = uni_prompting.text_tokenizer.batch_decode(cont_toks_list, skip_special_tokens=True)
            print(text)
            responses[i] += f'User: ' + question + f'\n Answer : ' + text[0] + '\n'

    images = torch.cat(images, dim=0)
    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
    images *= 255.0
    images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    pil_images = [Image.fromarray(image) for image in images]

    _log_images(
        accelerator=accelerator,
        key="multimodal understanding",
        pil_images=pil_images,
        captions=responses,
        step=global_step,
    )

    model.train()


@torch.no_grad()
def evaluate_validation(
    model,
    vq_model,
    uni_prompting,
    accelerator,
    config,
    global_step,
    max_step=50,  # random test 200 steps
    clip_processor=None,
    vision_tower=None,
):
    logger.info("Evaluating validation")
    model.eval()
    batch_size_eval = config.training.batch_size_pre * 4
    if 'bridge' in config.dataset.params.train_pre_shards_path_or_url:
        path = config.dataset.params.train_pre_shards_path_or_url
    else:
        path = config.dataset.params.train_pre_shards_path_or_url + "_validation"
    valid_dataloader_pre = get_future_view_prediction_w_action_data_loader(
        dataset_path=path,
        batch_size=batch_size_eval,
        num_workers=1,
        world_size=1,
        local_rank=accelerator.process_index,
        resolution=config.dataset.preprocessing.resolution,
        future_step=config.act_step,
    )

    @torch.no_grad()
    def prepare_inputs_and_labels(
        pixel_values_or_image_ids,
        texts: Union[str, str],
        min_masking_rate: float = 0.0,
        is_train: bool = True,
    ):
        image_tokens_0 = vq_model.get_code(pixel_values_or_image_ids[0]) + len(uni_prompting.text_tokenizer)
        target_image_tokens_0 = vq_model.get_code(pixel_values_or_image_ids[2]) + len(uni_prompting.text_tokenizer)
        if config.model.vla.num_view == 1:
            image_tokens = image_tokens_0
            target_image_tokens = target_image_tokens_0
        elif config.model.vla.num_view == 2:
            image_tokens_1 = vq_model.get_code(pixel_values_or_image_ids[1]) + len(uni_prompting.text_tokenizer)
            target_image_tokens_1 = vq_model.get_code(pixel_values_or_image_ids[3]) + len(uni_prompting.text_tokenizer)
            image_tokens = torch.cat((image_tokens_0, image_tokens_1), dim=1)
            target_image_tokens = torch.cat((target_image_tokens_0, target_image_tokens_1), dim=1)
        else:
            raise NotImplementedError(f"Num-view {config.model.vla.num_view} not supported")

        # create MLM mask and labels
        input_ids, labels, loss_weight, mask_prob = mask_or_random_replace_tokens_for_future_prediction(
            input_image_tokens=image_tokens,
            target_image_tokens=target_image_tokens,
            # below is useless arguments
            mask_id=None,
            config=config,
            mask_schedule=None,
            is_train=is_train)
        input_ids, masks, labels = uni_prompting((texts, input_ids, labels), 'pre')

        return input_ids, labels, mask_prob, image_tokens, target_image_tokens

    if accelerator.mixed_precision == "fp16":
        images_feat_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        images_feat_dtype = torch.bfloat16
    else:
        images_feat_dtype = torch.float32
    if accelerator.num_processes == 1:
        images_feat_dtype = torch.float32  # for one gpu training

    loss_meter_act = AverageMeter()
    loss_meter_pre = AverageMeter()
    batch_iter = tqdm.tqdm(valid_dataloader_pre)
    counter = 0
    for batch in batch_iter:
        batch_size_pre = len(batch["input_ids"])
        pixel_values_static, pixel_values_gripper, texts, pixel_values_static_future, pixel_values_gripper_future = (
            batch["images_static"],
            batch["images_gripper"],
            batch["input_ids"],
            batch["images_static_future"],
            batch["images_gripper_future"],
        )
        actions = batch["actions"].to(accelerator.device, non_blocking=True)
        pixel_values_static = pixel_values_static.to(accelerator.device, non_blocking=True)
        pixel_values_gripper = pixel_values_gripper.to(accelerator.device, non_blocking=True)
        pixel_values_static_future = pixel_values_static_future.to(accelerator.device, non_blocking=True)
        pixel_values_gripper_future = pixel_values_gripper_future.to(accelerator.device, non_blocking=True)

        pixel_values = (pixel_values_static, pixel_values_gripper, pixel_values_static_future,
                        pixel_values_gripper_future)
        # Encode images to image tokens, mask them and create input and labels
        (
            input_ids,
            labels,
            mask_prob,
            image_tokens_ori,
            target_image_tokens,
        ) = prepare_inputs_and_labels(pixel_values, texts, config.training.min_masking_rate)
        attention_mask = create_attention_mask_predict_next_for_future_prediction(
            input_ids,
            pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
            soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
            eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
            rm_pad_in_image=True,
            return_inverse_mask=True)

        if hasattr(model, 'module'):
            mask_dtype = model.module.showo.model.embed_tokens.weight.dtype
        else:
            mask_dtype = model.showo.model.embed_tokens.weight.dtype
        attention_mask = attention_mask.to(mask_dtype)
        if hasattr(model, 'module'):
            text_embeddings_img_text = model.module.showo.model.embed_tokens(input_ids)
        else:
            text_embeddings_img_text = model.showo.model.embed_tokens(input_ids)
        input_embeddings = text_embeddings_img_text

        logits, loss_pre, loss_mmu, loss_act = model(
            input_ids=input_ids,  # input_ids is not used (input_embeddings are provided)
            input_embeddings=input_embeddings,
            attention_mask=attention_mask,
            labels=labels,
            label_smoothing=config.training.label_smoothing,
            batch_size_pre=batch_size_pre,
            batch_size_mmu=0,
            max_seq_length=config.dataset.preprocessing.max_seq_length,
            actions=actions,
            clip_pad_tokens=config.training.clip_pad_tokens)
        batch_iter.set_description(
            f"# loss_a: {loss_act.item():.4f} pre_loss: {loss_pre.item():.4f} mmu_loss: {loss_mmu.item():.4f}")
        loss_meter_pre.update(loss_pre.item(), batch_size_pre)
        loss_meter_act.update(loss_act.item(), batch_size_pre)
        counter += 1
        if counter >= max_step:
            break
    _log_scalars(
        accelerator=accelerator,
        values={
            "validation_loss_pre": loss_meter_pre.avg,
            "validation_loss_a": loss_meter_act.avg,
        },
        step=global_step,
    )
    logger.info(f"validation_loss_a: {loss_meter_act.avg:0.4f}"
                f"validation_loss_pre: {loss_meter_pre.avg:0.4f}")
    model.train()


if __name__ == "__main__":
    main()
