#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Tuple

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torch
from transformers import AutoTokenizer

# Allow running from repo root without installing as a package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import MAGVITv2, Upvla
from training.prompting_utils import (
    UniversalPrompting_w_action,
    create_attention_mask_predict_next_for_future_prediction,
)
from training.utils import image_transform


@dataclass
class EvalSample:
    task: str
    instruction: str
    start_frame: int
    horizon: int


class FrameStore:
    def __init__(self, split_dir: Path, action_key: str):
        episode_files = sorted(split_dir.glob("episode_*.npz"))
        if not episode_files:
            raise FileNotFoundError(f"No episode_*.npz found in {split_dir}")

        self.path_by_frame_id: Dict[int, Path] = {}
        self.frame_ids: List[int] = []
        for p in episode_files:
            frame_id = int(p.stem.split("_")[1])
            self.path_by_frame_id[frame_id] = p
            self.frame_ids.append(frame_id)
        self.frame_ids.sort()

        self.actions_by_frame_id: Dict[int, np.ndarray] = {}
        for frame_id in self.frame_ids:
            with np.load(self.path_by_frame_id[frame_id], allow_pickle=True) as data:
                if action_key not in data:
                    raise KeyError(f"{action_key} not found in {self.path_by_frame_id[frame_id]}")
                self.actions_by_frame_id[frame_id] = data[action_key].astype(np.float32)

    def get_rgb(self, frame_id: int, key: str) -> np.ndarray:
        if frame_id not in self.path_by_frame_id:
            raise KeyError(f"Frame {frame_id} does not exist in split.")
        with np.load(self.path_by_frame_id[frame_id], allow_pickle=True) as data:
            return data[key]

    def get_action(self, frame_id: int) -> np.ndarray:
        if frame_id not in self.actions_by_frame_id:
            raise KeyError(f"Frame {frame_id} does not exist in split.")
        return self.actions_by_frame_id[frame_id]


class ActionMetrics:
    def __init__(self) -> None:
        self.samples = 0
        self.steps = 0
        self.sum_abs = np.zeros(7, dtype=np.float64)
        self.sum_sq = np.zeros(7, dtype=np.float64)
        self.gripper_correct = 0

    def update(self, pred: np.ndarray, gt: np.ndarray) -> None:
        # pred, gt: (T, 7)
        if pred.shape != gt.shape:
            raise ValueError(f"Shape mismatch: pred={pred.shape}, gt={gt.shape}")
        self.samples += 1
        self.steps += pred.shape[0]
        err = pred - gt
        self.sum_abs += np.abs(err).sum(axis=0)
        self.sum_sq += np.square(err).sum(axis=0)
        pred_gripper = np.where(pred[:, -1] >= 0.0, 1, -1)
        gt_gripper = np.where(gt[:, -1] >= 0.0, 1, -1)
        self.gripper_correct += int((pred_gripper == gt_gripper).sum())

    def summary(self) -> Dict[str, float]:
        if self.steps == 0:
            return {
                "samples": 0,
                "steps": 0,
                "mae_all": 0.0,
                "rmse_all": 0.0,
                "mae_pos": 0.0,
                "mae_rot": 0.0,
                "mae_gripper": 0.0,
                "gripper_sign_acc": 0.0,
            }
        mae = self.sum_abs / self.steps
        rmse = np.sqrt(self.sum_sq / self.steps)
        return {
            "samples": int(self.samples),
            "steps": int(self.steps),
            "mae_all": float(mae.mean()),
            "rmse_all": float(rmse.mean()),
            "mae_pos": float(mae[:3].mean()),
            "mae_rot": float(mae[3:6].mean()),
            "mae_gripper": float(mae[6]),
            "gripper_sign_acc": float(self.gripper_correct / self.steps),
        }


def resolve_from_config_dir(config_path: Path, maybe_path: str) -> str:
    p = Path(maybe_path).expanduser()
    if p.is_absolute():
        return str(p)
    return str((config_path.parent / p).resolve())


def load_model_bundle(model_config_path: Path, device: torch.device):
    cfg = OmegaConf.load(model_config_path)

    # Resolve local/relative paths against model config directory.
    cfg.model.vq_model.vq_model_name = resolve_from_config_dir(model_config_path, cfg.model.vq_model.vq_model_name)
    for key in ("llm_model_path", "pretrained_model_path", "tuned_model_path"):
        cfg.model.showo[key] = resolve_from_config_dir(model_config_path, cfg.model.showo[key])

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.showo.llm_model_path, padding_side="left")
    uni_prompting = UniversalPrompting_w_action(
        tokenizer,
        max_text_len=cfg.dataset.preprocessing.max_seq_length,
        special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
        ignore_id=-100,
        cond_dropout_prob=cfg.training.cond_dropout_prob,
        future_steps=cfg.act_step,
    )

    vq_model = MAGVITv2.from_pretrained(cfg.model.vq_model.vq_model_name).to(device)
    vq_model.requires_grad_(False)
    vq_model.eval()

    model = Upvla.from_pretrained(
        cfg.model.showo.pretrained_model_path,
        low_cpu_mem_usage=False,
        act_step=cfg.act_step,
        framework=str(cfg.model.get("framework", "upvla")),
    ).to(device)
    ckpt = Path(cfg.model.showo.tuned_model_path) / "unwrapped_model" / "pytorch_model.bin"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    state_dict = torch.load(str(ckpt), map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    del state_dict
    model.eval()

    return cfg, model, uni_prompting, vq_model


def load_lang_annotations(dataset_root: Path, split: str):
    ann_file = dataset_root / split / "lang_annotations" / "auto_lang_ann.npy"
    if not ann_file.exists():
        raise FileNotFoundError(f"Language annotation not found: {ann_file}")
    ann_obj = np.load(ann_file, allow_pickle=True).item()
    anns = ann_obj["language"]["ann"]
    tasks = ann_obj["language"]["task"]
    ranges = ann_obj["info"]["indx"]
    if not (len(anns) == len(tasks) == len(ranges)):
        raise ValueError("Invalid auto_lang_ann.npy content (length mismatch).")
    return anns, tasks, ranges


def build_samples(
    anns: List[str],
    tasks: List[str],
    ranges: List[Tuple[int, int]],
    act_step: int,
    start_only: bool,
    stride: int,
    max_samples: int,
) -> List[EvalSample]:
    samples: List[EvalSample] = []
    stride = max(1, stride)
    for ann, task, idx_range in zip(anns, tasks, ranges):
        start, end = int(idx_range[0]), int(idx_range[1])
        if end <= start:
            continue
        start_frames: Iterable[int]
        if start_only:
            start_frames = [start]
        else:
            start_frames = range(start, end, stride)
        for frame_id in start_frames:
            horizon = min(act_step, end - frame_id)
            if horizon <= 0:
                continue
            samples.append(EvalSample(task=task, instruction=ann, start_frame=frame_id, horizon=horizon))
            if max_samples > 0 and len(samples) >= max_samples:
                return samples
    return samples


def predict_actions_for_frame(
    model_cfg,
    model,
    uni_prompting,
    vq_model,
    rgb_static: np.ndarray,
    rgb_gripper: np.ndarray,
    instruction: str,
) -> np.ndarray:
    resolution = int(model_cfg.dataset.preprocessing.resolution)
    with torch.no_grad():
        pixel_static = image_transform(Image.fromarray(np.uint8(rgb_static)), resolution=resolution).to(model.device).unsqueeze(0)
        image_tokens = vq_model.get_code(pixel_static) + len(uni_prompting.text_tokenizer)

        num_view = int(model_cfg.model.vla.num_view)
        if num_view == 2:
            if rgb_gripper is None:
                raise ValueError("model.vla.num_view=2 but rgb_gripper is missing.")
            pixel_gripper = image_transform(Image.fromarray(np.uint8(rgb_gripper)), resolution=resolution).to(model.device).unsqueeze(0)
            image_tokens_gripper = vq_model.get_code(pixel_gripper) + len(uni_prompting.text_tokenizer)
            image_tokens = torch.cat([image_tokens, image_tokens_gripper], dim=1)
        elif num_view != 1:
            raise ValueError(f"Unsupported model.vla.num_view={num_view}")

        input_ids, _ = uni_prompting(([instruction], image_tokens), "pre_gen")
        attention_mask = create_attention_mask_predict_next_for_future_prediction(
            input_ids,
            pad_id=int(uni_prompting.sptids_dict["<|pad|>"]),
            soi_id=int(uni_prompting.sptids_dict["<|soi|>"]),
            eoi_id=int(uni_prompting.sptids_dict["<|eoi|>"]),
            rm_pad_in_image=True,
        )
        _, actions = model.pre_pad_predict(
            input_ids=input_ids,
            uncond_input_ids=None,
            attention_mask=attention_mask,
            guidance_scale=None,
            temperature=None,
            timesteps=None,
            noise_schedule=None,
            noise_type=None,
            predict_all_tokens=None,
            seq_len=model_cfg.model.showo.num_vq_tokens,
            uni_prompting=uni_prompting,
            config=model_cfg,
            return_actions=True,
        )
    return actions.squeeze(0).detach().cpu().numpy().astype(np.float32)


def print_table(title: str, rows: List[Dict[str, float]]) -> None:
    print(f"\n{title}")
    header = ["task", "samples", "steps", "mae_all", "rmse_all", "mae_pos", "mae_rot", "mae_gripper", "grip_acc"]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        print(
            "| "
            + " | ".join(
                [
                    str(row["task"]),
                    str(row["samples"]),
                    str(row["steps"]),
                    f"{row['mae_all']:.4f}",
                    f"{row['rmse_all']:.4f}",
                    f"{row['mae_pos']:.4f}",
                    f"{row['mae_rot']:.4f}",
                    f"{row['mae_gripper']:.4f}",
                    f"{row['gripper_sign_acc']:.4f}",
                ]
            )
            + " |"
        )


def main():
    parser = argparse.ArgumentParser(description="Offline BlockDiff-VLA action evaluation on CALVIN dataset (no calvin_env rollout).")
    parser.add_argument("--dataset-root", required=True, help="Path to CALVIN dataset root, e.g. calvin_debug_dataset")
    parser.add_argument("--split", default="validation", choices=["training", "validation"])
    parser.add_argument("--model-config", required=True, help="Path to BlockDiff-VLA model yaml config")
    parser.add_argument("--device", default="cuda:0", help="torch device, e.g. cuda:0 or cpu")
    parser.add_argument("--action-key", default="rel_actions", choices=["rel_actions", "actions"])
    parser.add_argument("--start-only", action="store_true", help="Use only the first frame of each language segment")
    parser.add_argument("--stride", type=int, default=1, help="Stride over start frames in each segment")
    parser.add_argument("--max-samples", type=int, default=0, help="Evaluate at most N start frames (0 means all)")
    parser.add_argument("--clip-action", action="store_true", help="Clip predicted actions to [-1, 1] before scoring")
    parser.add_argument("--output-json", default="", help="Optional path to save metrics json")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    model_config_path = Path(args.model_config).expanduser().resolve()
    split_dir = dataset_root / args.split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split dir not found: {split_dir}")
    if not model_config_path.exists():
        raise FileNotFoundError(f"Model config not found: {model_config_path}")

    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA is unavailable. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = requested_device

    model_cfg, model, uni_prompting, vq_model = load_model_bundle(model_config_path, device)
    anns, tasks, ranges = load_lang_annotations(dataset_root, args.split)
    samples = build_samples(
        anns=anns,
        tasks=tasks,
        ranges=ranges,
        act_step=int(model_cfg.act_step),
        start_only=args.start_only,
        stride=args.stride,
        max_samples=args.max_samples,
    )
    if not samples:
        raise RuntimeError("No valid samples found to evaluate.")

    store = FrameStore(split_dir=split_dir, action_key=args.action_key)
    overall = ActionMetrics()
    per_task: Dict[str, ActionMetrics] = defaultdict(ActionMetrics)

    for i, sample in enumerate(samples, start=1):
        rgb_static = store.get_rgb(sample.start_frame, "rgb_static")
        rgb_gripper = None
        if int(model_cfg.model.vla.num_view) == 2:
            rgb_gripper = store.get_rgb(sample.start_frame, "rgb_gripper")

        pred = predict_actions_for_frame(
            model_cfg=model_cfg,
            model=model,
            uni_prompting=uni_prompting,
            vq_model=vq_model,
            rgb_static=rgb_static,
            rgb_gripper=rgb_gripper,
            instruction=sample.instruction,
        )
        pred = pred[: sample.horizon]
        if args.clip_action:
            pred = np.clip(pred, -1.0, 1.0)

        gt = np.stack([store.get_action(sample.start_frame + step) for step in range(sample.horizon)], axis=0)
        overall.update(pred, gt)
        per_task[sample.task].update(pred, gt)

        if i % 20 == 0 or i == len(samples):
            print(f"progress: {i}/{len(samples)}")

    overall_row = {"task": "overall", **overall.summary()}
    task_rows = [{"task": task, **metrics.summary()} for task, metrics in sorted(per_task.items(), key=lambda x: x[0])]
    print_table("Offline Action Metrics", [overall_row])
    print_table("Per-Task Offline Action Metrics", task_rows)

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset_root": str(dataset_root),
            "split": args.split,
            "model_config": str(model_config_path),
            "device": str(device),
            "action_key": args.action_key,
            "start_only": bool(args.start_only),
            "stride": int(args.stride),
            "max_samples": int(args.max_samples),
            "clip_action": bool(args.clip_action),
            "overall": overall_row,
            "per_task": task_rows,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"saved_json={output_path}")


if __name__ == "__main__":
    main()
