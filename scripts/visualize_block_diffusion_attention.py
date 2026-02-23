#!/usr/bin/env python3
from pathlib import Path
import argparse
from typing import List, Optional
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

# Allow running from repo root without installing as a package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.block_diffusion_utils import (
    build_block_diffusion_attention_mask,
    build_block_diffusion_mask_bool,
    build_native_block_diffusion_batch,
    build_prefixed_block_diffusion_attention_mask,
    quantize_actions_to_tokens,
)


def get_avg_attention(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        out = model.showo(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            use_cache=False,
        )
    attn_list = out["attentions"]
    if attn_list is None or len(attn_list) == 0:
        raise RuntimeError("Model did not return attentions. Use eager/sdpa attention for visualization.")
    stacked = torch.stack(attn_list, dim=0)  # [Nlayer, B, H, L, L]
    avg = stacked.mean(dim=(0, 2))[0]  # [L, L], sample 0
    return avg.detach().cpu().numpy()


def save_heatmap(
    mat: np.ndarray,
    out_file: Path,
    title: str,
    split_lines: Optional[List[int]] = None,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    plt.figure(figsize=(7, 6))
    plt.imshow(mat, cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax, aspect="auto")
    plt.colorbar(fraction=0.046, pad=0.04)
    if split_lines is not None:
        for s in split_lines:
            plt.axvline(s - 0.5, color="red", linestyle="--", linewidth=1.0)
            plt.axhline(s - 0.5, color="red", linestyle="--", linewidth=1.0)
    plt.title(title)
    plt.xlabel("Key Index")
    plt.ylabel("Query Index")
    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=180)
    plt.close()


def save_layout(layout: np.ndarray, out_file: Path, title: str, split_lines: Optional[List[int]] = None):
    # layout values:
    # 0 = prefix, 1 = x_t keep, 2 = x_t masked, 3 = x_0 clean
    cmap = plt.get_cmap("Set2", 4)
    plt.figure(figsize=(10, 1.5))
    plt.imshow(layout[None, :], cmap=cmap, interpolation="nearest", aspect="auto", vmin=0, vmax=3)
    plt.yticks([])
    plt.xticks([])
    if split_lines is not None:
        for s in split_lines:
            plt.axvline(s - 0.5, color="black", linestyle="--", linewidth=1.0)
    plt.title(title + " (0=prefix, 1=x_t keep, 2=x_t masked, 3=x_0)")
    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=180)
    plt.close()


def build_text_case(
    batch_size: int,
    clean_len: int,
    vocab_size: int,
    mask_token_id: int,
    block_size: int,
    mask_eps: float,
    complementary: bool,
):
    clean = torch.randint(0, vocab_size, (batch_size, clean_len), dtype=torch.long)
    valid_mask = torch.ones_like(clean, dtype=torch.bool)
    pack = build_native_block_diffusion_batch(
        clean_ids=clean,
        valid_mask=valid_mask,
        mask_token_id=mask_token_id,
        block_size=block_size,
        eps=mask_eps,
        complementary_mask=complementary,
    )
    return pack


def build_action_case(
    batch_size: int,
    act_step: int,
    action_dim: int,
    bins: int,
    token_offset: int,
    mask_token_id: int,
    block_size: int,
    mask_eps: float,
    complementary: bool,
):
    actions = torch.rand(batch_size, act_step, action_dim, dtype=torch.float32) * 2 - 1
    clean = quantize_actions_to_tokens(
        actions=actions,
        num_bins=bins,
        token_offset=token_offset,
    ).reshape(batch_size, -1)
    valid_mask = torch.ones_like(clean, dtype=torch.bool)
    pack = build_native_block_diffusion_batch(
        clean_ids=clean,
        valid_mask=valid_mask,
        mask_token_id=mask_token_id,
        block_size=block_size,
        eps=mask_eps,
        complementary_mask=complementary,
    )
    return pack


def main():
    parser = argparse.ArgumentParser(description="Visualize Fast-dLLM-v2 style native block-diffusion attention.")
    parser.add_argument("--mode", choices=["text", "action"], default="text")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--clean-len", type=int, default=64, help="for text mode")
    parser.add_argument("--act-step", type=int, default=10, help="for action mode")
    parser.add_argument("--action-dim", type=int, default=7, help="for action mode")
    parser.add_argument("--bins", type=int, default=256, help="for action mode")
    parser.add_argument("--llm-vocab-size", type=int, default=32000)
    parser.add_argument("--prefix-len", type=int, default=128, help="prefix length for action mode")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--mask-eps", type=float, default=1e-3)
    parser.add_argument("--no-complementary", action="store_true")
    parser.add_argument("--out-dir", default="outputs/block_diffusion_attention")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-model-attention", action="store_true", help="Only plot theoretical masks/layout.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model = None
    if not args.skip_model_attention:
        try:
            from scripts.smoke_upvla_minimal import build_tiny_upvla

            tiny_cfg = Path("./dummy_data/tiny_phi_config_attn_vis")
            model, _ = build_tiny_upvla(
                phi_cfg_dir=tiny_cfg,
                act_step=max(2, args.act_step),
                llm_vocab_size=args.llm_vocab_size,
                codebook_size=max(args.bins, 64),
                num_vq_tokens=64,
            )
            model.eval()
        except Exception as exc:
            print(f"[warn] failed to build tiny model ({exc}); falling back to theoretical-only plots.")
            model = None

    mask_token_id = int(model.mask_token_id) if model is not None else int(args.llm_vocab_size - 1)
    complementary = not args.no_complementary

    token_offset = args.llm_vocab_size + 10
    out_dir = Path(args.out_dir)

    if args.mode == "text":
        pack = build_text_case(
            batch_size=args.batch_size,
            clean_len=args.clean_len,
            vocab_size=args.llm_vocab_size,
            mask_token_id=mask_token_id,
            block_size=args.block_size,
            mask_eps=args.mask_eps,
            complementary=complementary,
        )
        input_ids = pack.input_ids[:1]
        labels = pack.labels[:1]
        seq_len = pack.seq_len
        attention_mask = build_block_diffusion_attention_mask(
            seq_len=seq_len,
            block_size=args.block_size,
            batch_size=1,
            dtype=torch.float32,
            device=input_ids.device,
        )
        allow = build_block_diffusion_mask_bool(seq_len=seq_len, block_size=args.block_size, device=input_ids.device)
        allow = allow.to(torch.float32).cpu().numpy()

        layout = np.ones(2 * seq_len, dtype=np.int64)  # x_t keep
        layout[:seq_len][(labels[0] != -100).cpu().numpy()] = 2  # x_t masked
        layout[seq_len:] = 3  # x_0
        split_lines = [seq_len]
        tag = "text_native"
    else:
        pack = build_action_case(
            batch_size=args.batch_size,
            act_step=args.act_step,
            action_dim=args.action_dim,
            bins=args.bins,
            token_offset=token_offset,
            mask_token_id=mask_token_id,
            block_size=args.block_size,
            mask_eps=args.mask_eps,
            complementary=complementary,
        )
        seq_len = pack.seq_len
        prefix = torch.randint(0, args.llm_vocab_size, (1, args.prefix_len), dtype=torch.long)
        input_ids = torch.cat([prefix, pack.input_ids[:1]], dim=1)
        labels = pack.labels[:1]
        attention_mask = build_prefixed_block_diffusion_attention_mask(
            prefix_len=args.prefix_len,
            seq_len=seq_len,
            block_size=args.block_size,
            batch_size=1,
            dtype=torch.float32,
            device=input_ids.device,
        )
        allow = (attention_mask[0, 0] == 0).to(torch.float32).cpu().numpy()

        total = args.prefix_len + 2 * seq_len
        layout = np.zeros(total, dtype=np.int64)  # prefix
        xt_start = args.prefix_len
        x0_start = args.prefix_len + seq_len
        layout[xt_start:x0_start] = 1
        layout[xt_start:x0_start][(labels[0] != -100).cpu().numpy()] = 2
        layout[x0_start:] = 3
        split_lines = [args.prefix_len, args.prefix_len + seq_len]
        tag = "action_native"

    avg_attn = None
    if model is not None:
        avg_attn = get_avg_attention(model, input_ids=input_ids, attention_mask=attention_mask)

    layout_file = out_dir / f"{tag}_layout.png"
    mask_file = out_dir / f"{tag}_mask.png"
    attn_file = out_dir / f"{tag}_avg_attention.png"

    save_layout(layout, layout_file, f"{tag.upper()} token layout", split_lines=split_lines)
    save_heatmap(
        allow,
        mask_file,
        f"{tag.upper()} theoretical native BD mask",
        split_lines=split_lines,
        cmap="Greys",
        vmin=0.0,
        vmax=1.0,
    )
    if avg_attn is not None:
        save_heatmap(
            avg_attn,
            attn_file,
            f"{tag.upper()} model avg attention",
            split_lines=split_lines,
        )

    print(f"saved: {layout_file}")
    print(f"saved: {mask_file}")
    if avg_attn is not None:
        print(f"saved: {attn_file}")
    print(f"seq_len={seq_len}, block_size={args.block_size}, complementary={complementary}")


if __name__ == "__main__":
    main()
