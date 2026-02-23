#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch import nn
from transformers import PhiConfig
from einops.layers.torch import Rearrange

# Allow running from repo root without installing as a package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.modeling_upvla import Upvla
from models.map_block import MAPBlock
from training.future_view_prediction_w_action_dataset import get_future_view_prediction_w_action_data_loader


def fake_vq_tokens(images: torch.Tensor, num_vq_tokens: int, codebook_size: int) -> torch.Tensor:
    side = int(num_vq_tokens**0.5)
    if side * side != num_vq_tokens:
        raise ValueError(f"num_vq_tokens must be a perfect square, got {num_vq_tokens}")
    x = images.float().mean(dim=1, keepdim=True)
    x = F.interpolate(x, size=(side, side), mode="bilinear", align_corners=False)
    x = ((x + 1.0) * 0.5 * (codebook_size - 1)).round().long().clamp(0, codebook_size - 1)
    return x.squeeze(1).reshape(images.shape[0], -1)


def build_tiny_upvla(phi_cfg_dir: Path, act_step: int, llm_vocab_size: int, codebook_size: int, num_vq_tokens: int):
    phi_cfg_dir.mkdir(parents=True, exist_ok=True)
    phi_cfg = PhiConfig(
        vocab_size=llm_vocab_size,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
        embd_pdrop=0.0,
        resid_pdrop=0.0,
        attention_dropout=0.0,
        rope_theta=10000.0,
        partial_rotary_factor=0.5,
        qk_layernorm=False,
        rope_scaling=None,
    )
    phi_cfg.save_pretrained(phi_cfg_dir)

    num_special_tokens = 10
    model = Upvla(
        w_clip_vit=False,
        vocab_size=llm_vocab_size + num_special_tokens + codebook_size + 1,
        llm_vocab_size=llm_vocab_size,
        llm_model_path=str(phi_cfg_dir),
        codebook_size=codebook_size,
        num_vq_tokens=num_vq_tokens,
        load_from_showo=True,
        act_step=act_step,
    )
    hidden_size = model.showo.config.hidden_size
    if hidden_size != 2048:
        model.token_learner = MAPBlock(n_latents=1, embed_dim=hidden_size, n_heads=4)
        model.to_logits = nn.Sequential(nn.Linear(hidden_size, act_step * 7), Rearrange("... (a b) -> ... a b", b=7))
    return model, num_special_tokens


def main():
    parser = argparse.ArgumentParser(description="Minimal BlockDiff-VLA smoke test with dummy data")
    parser.add_argument("--dataset-path", default="./dummy_data/calvin_processed_training")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-seq-length", type=int, default=32)
    parser.add_argument("--num-vq-tokens", type=int, default=64)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--llm-vocab-size", type=int, default=32000)
    parser.add_argument("--future-step", type=int, default=10)
    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device("cpu")

    dataloader = get_future_view_prediction_w_action_data_loader(
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        num_workers=0,
        world_size=1,
        local_rank=0,
        resolution=64,
        future_step=args.future_step,
    )
    batch = next(iter(dataloader))

    model, num_special_tokens = build_tiny_upvla(
        phi_cfg_dir=Path("./dummy_data/tiny_phi_config"),
        act_step=args.future_step,
        llm_vocab_size=args.llm_vocab_size,
        codebook_size=args.codebook_size,
        num_vq_tokens=args.num_vq_tokens,
    )
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=1e-4)

    images_static = batch["images_static"].to(device)
    images_static_future = batch["images_static_future"].to(device)
    actions = batch["actions"].to(device).float()
    bsz = images_static.shape[0]

    image_tokens = fake_vq_tokens(images_static, args.num_vq_tokens, args.codebook_size)
    target_tokens = fake_vq_tokens(images_static_future, args.num_vq_tokens, args.codebook_size)
    token_offset = args.llm_vocab_size + num_special_tokens

    prefix = torch.randint(
        low=0,
        high=args.llm_vocab_size,
        size=(bsz, args.max_seq_length + 1),
        device=device,
    )
    action_token_id = args.llm_vocab_size + num_special_tokens - 1
    action_suffix = torch.full((bsz, args.future_step), action_token_id, dtype=torch.long, device=device)

    input_ids = torch.cat([prefix, image_tokens + token_offset, action_suffix], dim=1)
    labels = torch.full_like(input_ids, -100)
    labels[:, args.max_seq_length + 1: args.max_seq_length + 1 + args.num_vq_tokens] = target_tokens + token_offset

    input_embeddings = model.showo.model.embed_tokens(input_ids)

    _, loss_pre, _, loss_act = model(
        input_ids=input_ids,
        input_embeddings=input_embeddings,
        attention_mask=None,
        labels=labels,
        actions=actions,
        batch_size_pre=bsz,
        batch_size_mmu=0,
        max_seq_length=args.max_seq_length,
        clip_pad_tokens=False,
    )

    loss = loss_pre + loss_act
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    print("smoke_ok=True")
    print(f"loss_pre={loss_pre.item():.6f}")
    print(f"loss_act={loss_act.item():.6f}")
    print(f"loss_total={loss.item():.6f}")
    print(f"input_shape={tuple(input_ids.shape)}")
    print(f"actions_shape={tuple(actions.shape)}")


if __name__ == "__main__":
    main()
