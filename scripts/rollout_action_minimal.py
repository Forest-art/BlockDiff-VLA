#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

# Allow running from repo root without installing as a package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.smoke_upvla_minimal import build_tiny_upvla, fake_vq_tokens
from training.future_view_prediction_w_action_dataset import (
    get_future_view_prediction_w_action_data_loader,
)


def build_fake_text_prefix(texts, max_seq_length, llm_vocab_size, device):
    """Create deterministic pseudo text tokens to avoid tokenizer dependencies."""
    prefix = torch.zeros(len(texts), max_seq_length + 1, dtype=torch.long, device=device)
    for i, text in enumerate(texts):
        ids = [ord(ch) % llm_vocab_size for ch in text][: max_seq_length + 1]
        if not ids:
            ids = [0]
        prefix[i, -len(ids) :] = torch.tensor(ids, dtype=torch.long, device=device)
    return prefix


def main():
    parser = argparse.ArgumentParser(
        description="Minimal action rollout on preprocessed Calvin-style data (no Calvin env dependency)."
    )
    parser.add_argument("--dataset-path", default="./dummy_data/calvin_processed_training")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-seq-length", type=int, default=32)
    parser.add_argument("--num-vq-tokens", type=int, default=64)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--llm-vocab-size", type=int, default=32000)
    parser.add_argument("--future-step", type=int, default=10)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    dataloader = get_future_view_prediction_w_action_data_loader(
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        num_workers=0,
        world_size=1,
        local_rank=0,
        resolution=args.resolution,
        future_step=args.future_step,
    )

    model, num_special_tokens = build_tiny_upvla(
        phi_cfg_dir=Path("./dummy_data/tiny_phi_config"),
        act_step=args.future_step,
        llm_vocab_size=args.llm_vocab_size,
        codebook_size=args.codebook_size,
        num_vq_tokens=args.num_vq_tokens,
    )
    model.to(device)
    model.eval()

    action_token_id = args.llm_vocab_size + num_special_tokens - 1
    token_offset = args.llm_vocab_size + num_special_tokens

    mse_meter = []
    for step, batch in enumerate(dataloader):
        if step >= args.max_batches:
            break

        images_static = batch["images_static"].to(device)
        actions_gt = batch["actions"].to(device).float()
        texts = batch["input_ids"]
        bsz = images_static.shape[0]

        image_tokens = fake_vq_tokens(images_static, args.num_vq_tokens, args.codebook_size)
        prefix = build_fake_text_prefix(texts, args.max_seq_length, args.llm_vocab_size, device)
        action_suffix = torch.full(
            (bsz, args.future_step), action_token_id, dtype=torch.long, device=device
        )
        input_ids = torch.cat([prefix, image_tokens + token_offset, action_suffix], dim=1)
        input_embeddings = model.showo.model.embed_tokens(input_ids)

        with torch.no_grad():
            actions_pred = model(
                input_ids=input_ids,
                input_embeddings=input_embeddings,
                attention_mask=None,
                labels=None,
                output_mode="action",
            )

        mse = F.mse_loss(actions_pred, actions_gt).item()
        mse_meter.append(mse)
        print(f"batch={step} mse={mse:.6f} shape_pred={tuple(actions_pred.shape)}")
        print(f"  text='{texts[0]}'")
        print(f"  pred_action_t0={actions_pred[0, 0].cpu().tolist()}")
        print(f"  gt_action_t0={actions_gt[0, 0].cpu().tolist()}")

    if mse_meter:
        mean_mse = sum(mse_meter) / len(mse_meter)
        print("rollout_ok=True")
        print(f"num_batches={len(mse_meter)}")
        print(f"mean_mse={mean_mse:.6f}")
    else:
        print("rollout_ok=False")
        print("reason=no_batches")


if __name__ == "__main__":
    main()
