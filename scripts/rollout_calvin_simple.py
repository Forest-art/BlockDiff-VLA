#!/usr/bin/env python3
import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]


def _bool_to_hydra(value: bool) -> str:
    return "True" if value else "False"


def build_rollout_model_yaml(model_yaml: Path, output_root: Path, checkpoint_dir: Path | None) -> Path:
    cfg = OmegaConf.load(model_yaml)
    if checkpoint_dir is not None:
        cfg.model.showo.tuned_model_path = str(checkpoint_dir)

    output_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_root / f"rollout_model_{ts}.yaml"
    OmegaConf.save(cfg, out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simple CALVIN rollout launcher: evaluate and save results.json with minimal arguments."
    )
    parser.add_argument("--dataset-root", required=True, help="CALVIN dataset root (contains training/validation)")
    parser.add_argument(
        "--model-yaml",
        default=str(REPO_ROOT / "policy_rollout" / "arvla_model.yaml"),
        help="Model yaml template for rollout.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="",
        help="Optional tuned checkpoint directory. If set, overrides model.showo.tuned_model_path.",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device id for single-process rollout.")
    parser.add_argument("--num-sequences", type=int, default=10, help="How many evaluation sequences to run.")
    parser.add_argument("--num-videos", type=int, default=0, help="How many rollout videos to save.")
    parser.add_argument("--ep-len", type=int, default=0, help="Override ep_len when > 0.")
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "rollout_outputs"),
        help="Output root directory for rollout artifacts and temporary yaml.",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save mp4 videos instead of gif (requires ffmpeg/libx264).",
    )
    parser.add_argument(
        "--debug-render",
        action="store_true",
        help="Enable debug rendering path (uses env.render and OpenCV windows).",
    )
    parser.add_argument(
        "--egl-visible-devices",
        default="",
        help="Optional EGL_VISIBLE_DEVICES value (e.g. 0).",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    model_yaml = Path(args.model_yaml).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve() if args.checkpoint_dir else None

    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset root not found: {dataset_root}")
    if not model_yaml.exists():
        raise FileNotFoundError(f"model yaml not found: {model_yaml}")
    if checkpoint_dir is not None and not checkpoint_dir.exists():
        raise FileNotFoundError(f"checkpoint dir not found: {checkpoint_dir}")

    rollout_model_yaml = build_rollout_model_yaml(model_yaml, output_root, checkpoint_dir)

    cmd = [
        sys.executable,
        "-m",
        "policy_rollout.calvin_evaluate_upvla",
        f"dataset_path={dataset_root}",
        f"root_data_dir={dataset_root}",
        f"model_config={rollout_model_yaml}",
        f"device={args.device}",
        f"num_sequences={args.num_sequences}",
        f"num_videos={args.num_videos}",
        f"save_rollout_as_video={_bool_to_hydra(args.save_video)}",
        f"debug={_bool_to_hydra(args.debug_render)}",
        "log_wandb=False",
        "hydra.job.chdir=False",
    ]
    if args.ep_len > 0:
        cmd.append(f"ep_len={args.ep_len}")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")
    if args.egl_visible_devices:
        env["EGL_VISIBLE_DEVICES"] = args.egl_visible_devices

    print("Temporary rollout model yaml:", rollout_model_yaml)
    print("Launching command:")
    print(" ".join(cmd))

    subprocess.run(cmd, env=env, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
