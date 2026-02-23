from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
import torch.nn.functional as F


@dataclass
class NativeBlockDiffusionBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    primary_mask: torch.Tensor
    complementary_mask: Optional[torch.Tensor]
    seq_len: int
    repeats: int
    block_sizes: torch.Tensor


def _require_2d_pair(clean_ids: torch.Tensor, valid_mask: torch.Tensor) -> None:
    if clean_ids.ndim != 2 or valid_mask.ndim != 2:
        raise ValueError("clean_ids and valid_mask must be 2D tensors [B, L].")
    if clean_ids.shape != valid_mask.shape:
        raise ValueError("clean_ids and valid_mask must have the same shape.")


def _safe_attention_min(dtype: torch.dtype) -> float:
    if dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
        return float(torch.finfo(dtype).min)
    return float(torch.finfo(torch.float32).min)


def _normalize_block_sizes(
    block_size: Union[int, Sequence[int], torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")
    if isinstance(block_size, int):
        if block_size <= 0:
            raise ValueError("block_size must be > 0.")
        return torch.full((batch_size,), int(block_size), dtype=torch.long, device=device)
    if isinstance(block_size, torch.Tensor):
        if block_size.ndim != 1:
            raise ValueError("block_size tensor must be 1D [B].")
        if block_size.shape[0] != batch_size:
            raise ValueError("block_size tensor length must match batch_size.")
        bs = block_size.to(device=device, dtype=torch.long)
    else:
        if len(block_size) != batch_size:
            raise ValueError("block_size sequence length must match batch_size.")
        bs = torch.tensor(list(block_size), device=device, dtype=torch.long)
    if bool((bs <= 0).any()):
        raise ValueError("All block sizes must be > 0.")
    return bs


def build_block_diffusion_mask_bool(seq_len: int, block_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Fast-dLLM v2 style block-diffusion attention mask (M_BD + M_OBC + M_BC).
    Returns [2L, 2L] bool, where True means "can attend".
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be > 0.")
    if block_size <= 0:
        raise ValueError("block_size must be > 0.")

    n = int(seq_len)
    total = 2 * n
    idx = torch.arange(total, device=device)

    is_x0 = idx >= n
    block_idx = torch.where(is_x0, (idx - n) // int(block_size), idx // int(block_size))

    q_x0 = is_x0[:, None]
    k_x0 = is_x0[None, :]
    q_block = block_idx[:, None]
    k_block = block_idx[None, :]

    block_diagonal = (q_block == k_block) & (q_x0 == k_x0)  # M_BD
    offset_block_causal = (q_block > k_block) & (~q_x0) & k_x0  # M_OBC
    block_causal = (q_block >= k_block) & q_x0 & k_x0  # M_BC

    return block_diagonal | offset_block_causal | block_causal


def build_block_diffusion_attention_mask(
    seq_len: int,
    block_size: Union[int, Sequence[int], torch.Tensor],
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Dense additive mask for Phi attention: [B, 1, 2L, 2L], 0 for allowed and -inf for blocked.
    """
    block_sizes = _normalize_block_sizes(block_size=block_size, batch_size=batch_size, device=device)
    out = torch.empty((batch_size, 2 * seq_len, 2 * seq_len), dtype=dtype, device=device)
    for bs in torch.unique(block_sizes).tolist():
        allow = build_block_diffusion_mask_bool(seq_len=seq_len, block_size=int(bs), device=device)
        mask = torch.zeros((2 * seq_len, 2 * seq_len), dtype=dtype, device=device)
        mask = mask.masked_fill(~allow, _safe_attention_min(dtype))
        idx = torch.nonzero(block_sizes == int(bs), as_tuple=False).squeeze(-1)
        out[idx] = mask
    return out.unsqueeze(1)


def build_prefixed_block_diffusion_attention_mask(
    prefix_len: int,
    seq_len: int,
    block_size: Union[int, Sequence[int], torch.Tensor],
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Add a fixed prefix before [x_t, x_0], producing [B,1,P+2L,P+2L] additive mask.
    Prefix is causal internally and fully visible to all x_t/x_0 queries.
    """
    if prefix_len < 0:
        raise ValueError("prefix_len must be >= 0.")
    if seq_len <= 0:
        raise ValueError("seq_len must be > 0.")

    total = prefix_len + 2 * seq_len
    block_sizes = _normalize_block_sizes(block_size=block_size, batch_size=batch_size, device=device)
    out = torch.empty((batch_size, total, total), dtype=dtype, device=device)

    for bs in torch.unique(block_sizes).tolist():
        allow = torch.zeros((total, total), dtype=torch.bool, device=device)
        if prefix_len > 0:
            prefix_tri = torch.tril(torch.ones((prefix_len, prefix_len), dtype=torch.bool, device=device))
            allow[:prefix_len, :prefix_len] = prefix_tri
            allow[prefix_len:, :prefix_len] = True
        local_allow = build_block_diffusion_mask_bool(seq_len=seq_len, block_size=int(bs), device=device)
        allow[prefix_len:, prefix_len:] = local_allow

        mask = torch.zeros((total, total), dtype=dtype, device=device)
        mask = mask.masked_fill(~allow, _safe_attention_min(dtype))
        idx = torch.nonzero(block_sizes == int(bs), as_tuple=False).squeeze(-1)
        out[idx] = mask
    return out.unsqueeze(1)


def sample_blockwise_mask_indices(
    valid_mask: torch.Tensor,
    block_size: int,
    eps: float = 1e-3,
    block_sizes: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Fast-dLLM v2 style random masking:
      1) split sequence into blocks
      2) sample one mask probability per block
      3) sample Bernoulli mask per token inside each block
    """
    if valid_mask.ndim != 2:
        raise ValueError("valid_mask must be [B, L].")
    bsz, seqlen = valid_mask.shape
    if block_sizes is None:
        if block_size <= 0:
            raise ValueError("block_size must be > 0.")
        n_blocks = (seqlen + int(block_size) - 1) // int(block_size)
        block_ids = torch.arange(seqlen, device=valid_mask.device) // int(block_size)

        t = torch.rand((bsz, n_blocks), device=valid_mask.device)
        p_mask = (1.0 - float(eps)) * t + float(eps)  # in (eps, 1)
        p_per_token = p_mask[:, block_ids]
        sampled = torch.rand((bsz, seqlen), device=valid_mask.device) < p_per_token
        return sampled & valid_mask.bool()

    if block_sizes.ndim != 1 or block_sizes.shape[0] != bsz:
        raise ValueError("block_sizes must be [B].")
    if bool((block_sizes <= 0).any()):
        raise ValueError("All block_sizes must be > 0.")

    sampled = torch.zeros((bsz, seqlen), dtype=torch.bool, device=valid_mask.device)
    for b in range(bsz):
        bs = int(block_sizes[b].item())
        n_blocks = (seqlen + bs - 1) // bs
        block_ids = torch.arange(seqlen, device=valid_mask.device) // bs
        t = torch.rand((n_blocks,), device=valid_mask.device)
        p_mask = (1.0 - float(eps)) * t + float(eps)
        p_per_token = p_mask[block_ids]
        sampled[b] = torch.rand((seqlen,), device=valid_mask.device) < p_per_token
    return sampled & valid_mask.bool()


def build_native_block_diffusion_batch(
    clean_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    mask_token_id: int,
    block_size: int = 32,
    block_sizes: Optional[torch.Tensor] = None,
    eps: float = 1e-3,
    ignore_id: int = -100,
    complementary_mask: bool = True,
) -> NativeBlockDiffusionBatch:
    """
    Build Fast-dLLM v2 style training batch:
      - primary:    [x_t, x_0], labels on masked tokens in x_t
      - complementary (optional): invert masked/unmasked on valid positions
    """
    _require_2d_pair(clean_ids, valid_mask)
    bsz, seqlen = clean_ids.shape
    primary_block_sizes = _normalize_block_sizes(
        block_size=block_sizes if block_sizes is not None else int(block_size),
        batch_size=bsz,
        device=clean_ids.device,
    )

    primary_mask = sample_blockwise_mask_indices(
        valid_mask=valid_mask,
        block_size=block_size,
        eps=eps,
        block_sizes=primary_block_sizes,
    )
    noisy_primary = clean_ids.clone()
    noisy_primary[primary_mask] = int(mask_token_id)

    labels_primary = torch.full((bsz, seqlen), int(ignore_id), dtype=clean_ids.dtype, device=clean_ids.device)
    labels_primary[primary_mask] = clean_ids[primary_mask]
    inputs_primary = torch.cat([noisy_primary, clean_ids], dim=1)

    if not complementary_mask:
        return NativeBlockDiffusionBatch(
            input_ids=inputs_primary,
            labels=labels_primary,
            primary_mask=primary_mask,
            complementary_mask=None,
            seq_len=seqlen,
            repeats=1,
            block_sizes=primary_block_sizes,
        )

    comp_mask = valid_mask.bool() & (~primary_mask)
    noisy_comp = clean_ids.clone()
    noisy_comp[comp_mask] = int(mask_token_id)

    labels_comp = torch.full((bsz, seqlen), int(ignore_id), dtype=clean_ids.dtype, device=clean_ids.device)
    labels_comp[comp_mask] = clean_ids[comp_mask]
    inputs_comp = torch.cat([noisy_comp, clean_ids], dim=1)

    return NativeBlockDiffusionBatch(
        input_ids=torch.cat([inputs_primary, inputs_comp], dim=0),
        labels=torch.cat([labels_primary, labels_comp], dim=0),
        primary_mask=primary_mask,
        complementary_mask=comp_mask,
        seq_len=seqlen,
        repeats=2,
        block_sizes=torch.cat([primary_block_sizes, primary_block_sizes], dim=0),
    )


def block_diffusion_token_shift_loss(
    logits: torch.Tensor,
    labels_first_half: torch.Tensor,
    first_half_len: int,
    ignore_id: int = -100,
) -> torch.Tensor:
    """
    Token-shift loss used by Fast-dLLM v2:
      - predict token i from logit at i-1 (causal shift)
      - logits are computed on [x_t, x_0], but loss is on the first half only.
    """
    if logits.ndim != 3:
        raise ValueError("logits must be [B, T, V].")
    if labels_first_half.ndim != 2:
        raise ValueError("labels_first_half must be [B, L_half].")
    if logits.shape[0] != labels_first_half.shape[0]:
        raise ValueError("Batch size mismatch between logits and labels.")
    if first_half_len <= 1:
        return logits.new_zeros(())

    logits_half = logits[:, :first_half_len, :]
    if logits_half.shape[1] != labels_first_half.shape[1]:
        raise ValueError("first_half_len must equal labels_first_half.shape[1].")

    shift_logits = logits_half[:, :-1, :].contiguous()
    shift_labels = labels_first_half[:, 1:].contiguous()

    valid = shift_labels.ne(int(ignore_id))
    denom = valid.sum()
    if int(denom.item()) == 0:
        return shift_logits.new_zeros(())

    loss_sum = F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=int(ignore_id),
        reduction="sum",
    )
    return loss_sum / denom.to(loss_sum.dtype)


def build_concat_block_diffusion_inputs(
    clean_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    vocab_size: int,
    ignore_id: int = -100,
    min_mask_ratio: float = 0.15,
    max_mask_ratio: float = 0.6,
):
    """
    Backward-compatible wrapper that preserves the old API shape:
      input  = [clean, noisy]
      labels = [-100 on clean half, masked targets on noisy half]
    """
    _require_2d_pair(clean_ids, valid_mask)
    mask_token_id = max(0, int(vocab_size) - 1)
    eps = max(1e-3, float(min_mask_ratio))
    _ = max_mask_ratio

    native = build_native_block_diffusion_batch(
        clean_ids=clean_ids,
        valid_mask=valid_mask,
        mask_token_id=mask_token_id,
        block_size=32,
        eps=eps,
        ignore_id=ignore_id,
        complementary_mask=False,
    )
    bsz, seqlen = clean_ids.shape
    noisy = native.input_ids[:, :seqlen]
    labels = torch.full((bsz, 2 * seqlen), int(ignore_id), dtype=clean_ids.dtype, device=clean_ids.device)
    labels[:, seqlen:] = native.labels
    inputs = torch.cat([clean_ids, noisy], dim=1)
    return inputs, labels, native.primary_mask


def quantize_actions_to_tokens(
    actions: torch.Tensor,
    num_bins: int,
    token_offset: int,
    min_value: float = -1.0,
    max_value: float = 1.0,
):
    """
    actions: [B, T, D] -> token ids [B, T, D] in [token_offset, token_offset + num_bins - 1]
    """
    if actions.ndim != 3:
        raise ValueError("actions must be [B, T, D].")
    if num_bins <= 1:
        raise ValueError("num_bins must be > 1.")

    x = actions.clamp(min_value, max_value)
    x = (x - min_value) / (max_value - min_value)
    x = torch.round(x * (num_bins - 1)).long()
    x = x + int(token_offset)
    return x


def dequantize_action_tokens(
    tokens: torch.Tensor,
    num_bins: int,
    token_offset: int,
    min_value: float = -1.0,
    max_value: float = 1.0,
):
    """
    token ids [... ] -> continuous actions [... ] in [min_value, max_value]
    """
    x = (tokens.long() - int(token_offset)).clamp(0, int(num_bins) - 1).float()
    x = x / float(num_bins - 1)
    x = x * (max_value - min_value) + min_value
    return x
