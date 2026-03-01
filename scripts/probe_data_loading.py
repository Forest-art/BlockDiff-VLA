#!/usr/bin/env python3
# coding=utf-8

"""
Probe data loading only (no model forward/backward), while keeping loader settings
aligned with training config (batch size / num_workers / combined loader mode).

Usage examples:
  python scripts/probe_data_loading.py --config config/mdmvla_stage2_action_tuning.yaml --max-steps 27500
  torchrun --nproc_per_node=8 scripts/probe_data_loading.py --config config/mdmvla_stage2_action_tuning.yaml --max-steps 27500
  python scripts/probe_data_loading.py --config config/debug_mdmvla_1step.yaml --flow pre --max-steps 20
"""

import argparse
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from lightning.pytorch.utilities import CombinedLoader
from omegaconf import OmegaConf
from transformers import AutoTokenizer, CLIPImageProcessor

from llava.llava_instruct_data import get_instruct_data_loader
from training.prompting_utils import UniversalPrompting_w_action
from training.future_view_prediction_w_action_dataset import (
    get_future_view_prediction_w_action_data_loader,
)


def _parse_args():
    parser = argparse.ArgumentParser("Probe data loading without training")
    parser.add_argument("--config", required=True, help="Path to yaml config")
    parser.add_argument("--max-steps", type=int, default=27500, help="How many global steps to iterate")
    parser.add_argument("--target-step", type=int, default=27500, help="Special step to highlight in logs")
    parser.add_argument("--report-every", type=int, default=100, help="Progress logging interval")
    parser.add_argument(
        "--flow",
        choices=["pre", "mmu", "both"],
        default="both",
        help="Which dataloader flow(s) to probe",
    )
    parser.add_argument(
        "--set-epoch",
        action="store_true",
        help="Call sampler.set_epoch(epoch) on each epoch boundary (off by default to mimic current training code)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override")
    args, unknown = parser.parse_known_args()
    return args, unknown


def _load_config(config_path: str, overrides: List[str]):
    yaml_conf = OmegaConf.load(config_path)
    cli_conf = OmegaConf.from_dotlist(overrides) if overrides else OmegaConf.create({})
    return OmegaConf.merge(yaml_conf, cli_conf)


def _dist_env():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    return rank, world_size, local_rank


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _set_loader_epoch(loader, epoch: int):
    sampler = getattr(loader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def _summarize_pre_batch(batch: Dict):
    return (
        f"pre: static={tuple(batch['images_static'].shape)} "
        f"gripper={tuple(batch['images_gripper'].shape)} "
        f"future={tuple(batch['images_static_future'].shape)} "
        f"actions={tuple(batch['actions'].shape)}"
    )


def _summarize_mmu_batch(batch: Dict):
    return (
        f"mmu: images={tuple(batch['images'].shape)} "
        f"input_ids={tuple(batch['input_ids'].shape)} "
        f"labels={tuple(batch['labels'].shape)}"
    )


def main():
    args, unknown_overrides = _parse_args()
    cfg = _load_config(args.config, unknown_overrides)

    rank, world_size, local_rank = _dist_env()
    seed = args.seed if args.seed is not None else int(cfg.training.get("seed", 0))
    _set_seed(seed + rank)

    dataset_cfg = cfg.dataset.params
    preproc_cfg = cfg.dataset.preprocessing

    print(
        f"[rank {rank}] start probe | world_size={world_size} local_rank={local_rank} "
        f"flow={args.flow} max_steps={args.max_steps} target_step={args.target_step} seed={seed}",
        flush=True,
    )
    print(
        f"[rank {rank}] config={args.config} combined_loader_mode={cfg.dataset.combined_loader_mode} "
        f"bs_pre={cfg.training.batch_size_pre} bs_mmu={cfg.training.batch_size_mmu} "
        f"num_workers={dataset_cfg.num_workers}",
        flush=True,
    )

    iterables = {}
    pre_loader = None
    mmu_loader = None

    if args.flow in {"pre", "both"}:
        pre_loader = get_future_view_prediction_w_action_data_loader(
            dataset_path=dataset_cfg.train_pre_shards_path_or_url,
            batch_size=cfg.training.batch_size_pre,
            num_workers=dataset_cfg.num_workers,
            world_size=world_size,
            local_rank=rank,
            resolution=preproc_cfg.resolution,
            future_step=cfg.act_step,
        )
        iterables["pre_flow"] = pre_loader
        print(f"[rank {rank}] pre_loader_len={len(pre_loader)}", flush=True)

    if args.flow in {"mmu", "both"}:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.showo.llm_model_path, padding_side="left")
        # Keep tokenizer setup aligned with training code path.
        _ = UniversalPrompting_w_action(
            tokenizer,
            max_text_len=cfg.dataset.preprocessing.max_seq_length,
            special_tokens=(
                "<|soi|>",
                "<|eoi|>",
                "<|sov|>",
                "<|eov|>",
                "<|t2i|>",
                "<|mmu|>",
                "<|t2v|>",
                "<|v2v|>",
                "<|lvg|>",
            ),
            ignore_id=-100,
            cond_dropout_prob=cfg.training.cond_dropout_prob,
            future_steps=cfg.act_step,
        )
        clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        max_length = preproc_cfg.max_seq_length - (576 - cfg.model.showo.num_vq_tokens) + cfg.act_step
        mmu_loader = get_instruct_data_loader(
            tokenizer=tokenizer,
            dataset_path=dataset_cfg.train_mmu_shards_path_or_url,
            batch_size=cfg.training.batch_size_mmu,
            num_workers=dataset_cfg.num_workers,
            world_size=world_size,
            local_rank=rank,
            max_length=max_length,
            processor=clip_processor,
            return_system_prompt=False,
        )
        iterables["mmu_flow"] = mmu_loader
        print(f"[rank {rank}] mmu_loader_len={len(mmu_loader)}", flush=True)

    if len(iterables) == 0:
        raise ValueError("No flow selected. Use --flow pre|mmu|both")

    combined = CombinedLoader(iterables, mode=cfg.dataset.combined_loader_mode)

    global_step = 0
    epoch = 0
    fetch_time_total = 0.0
    fetch_time_max = 0.0
    start = time.time()
    iterator = iter(combined)

    while global_step < args.max_steps:
        if args.set_epoch:
            if pre_loader is not None:
                _set_loader_epoch(pre_loader, epoch)
            if mmu_loader is not None:
                _set_loader_epoch(mmu_loader, epoch)

        while global_step < args.max_steps:
            fetch_start = time.time()
            try:
                batch, batch_idx, dataloader_idx = next(iterator)
            except StopIteration:
                epoch += 1
                iterator = iter(combined)
                break
            except Exception as exc:
                print(
                    f"[rank {rank}] ERROR at global_step={global_step + 1} epoch={epoch} "
                    f"batch_idx={locals().get('batch_idx', 'NA')} dataloader_idx={locals().get('dataloader_idx', 'NA')}: {exc}",
                    flush=True,
                )
                traceback.print_exc()
                raise

            fetch_time = time.time() - fetch_start
            fetch_time_total += fetch_time
            fetch_time_max = max(fetch_time_max, fetch_time)
            global_step += 1

            should_report = (
                global_step == 1
                or global_step % args.report_every == 0
                or global_step in {args.target_step - 1, args.target_step, args.target_step + 1}
                or global_step == args.max_steps
            )
            if should_report:
                parts = [f"[rank {rank}] step={global_step} epoch={epoch} fetch_t={fetch_time:.4f}s"]
                if "pre_flow" in batch:
                    parts.append(_summarize_pre_batch(batch["pre_flow"]))
                if "mmu_flow" in batch:
                    parts.append(_summarize_mmu_batch(batch["mmu_flow"]))
                print(" | ".join(parts), flush=True)

            if global_step == args.target_step:
                print(f"[rank {rank}] TARGET STEP {args.target_step} reached successfully.", flush=True)

    elapsed = time.time() - start
    avg_fetch = fetch_time_total / max(global_step, 1)
    print(
        f"[rank {rank}] done. steps={global_step} epochs={epoch + 1} elapsed={elapsed:.2f}s "
        f"avg_fetch_t={avg_fetch:.4f}s max_fetch_t={fetch_time_max:.4f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
