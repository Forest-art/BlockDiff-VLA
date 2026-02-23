#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List

import numpy as np


def build_dataset_info(
    source_split_dir: Path,
    instruction: str,
    max_frames: int,
    start_index: int,
) -> List[dict]:
    episode_files = sorted(source_split_dir.glob("episode_*.npz"))
    if not episode_files:
        raise FileNotFoundError(f"No episode_*.npz files found in {source_split_dir}")

    selected = episode_files[start_index:]
    if max_frames > 0:
        selected = selected[:max_frames]

    if not selected:
        raise RuntimeError("No frames selected after applying start-index/max-frames.")

    frames = []
    for p in selected:
        with np.load(p, allow_pickle=True) as data:
            if "rel_actions" not in data:
                raise KeyError(f"rel_actions missing in {p}")
            rel_action = data["rel_actions"].astype(np.float32).tolist()
        frames.append(
            {
                "dir": str(p.resolve()),
                "rel_action": rel_action,
            }
        )

    return [{"instruction": instruction, "frames": frames}]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create minimal BlockDiff-VLA processed dataset_info.json from CALVIN debug split."
    )
    parser.add_argument("--source-split-dir", required=True, help="Path to CALVIN split dir, e.g. .../calvin_debug_dataset/training")
    parser.add_argument("--output-dir", required=True, help="Output dir name should contain 'processed'")
    parser.add_argument("--instruction", default="do the task", help="Single instruction for all frames")
    parser.add_argument("--max-frames", type=int, default=512, help="How many frames to include (0 means all)")
    parser.add_argument("--start-index", type=int, default=0, help="Start frame index in sorted episode list")
    args = parser.parse_args()

    source_split_dir = Path(args.source_split_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if "processed" not in str(output_dir):
        raise ValueError("output-dir must contain substring 'processed' for current DataProvider assertions.")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_info = build_dataset_info(
        source_split_dir=source_split_dir,
        instruction=args.instruction,
        max_frames=int(args.max_frames),
        start_index=int(args.start_index),
    )

    out_file = output_dir / "dataset_info.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(dataset_info, f)

    print(f"wrote={out_file}")
    print(f"episodes={len(dataset_info)} frames={len(dataset_info[0]['frames'])}")


if __name__ == "__main__":
    main()

