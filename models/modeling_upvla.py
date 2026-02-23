# coding=utf-8
# Copyright 2024 NUS Show Lab, HuggingFace.
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

import torch
import torch.nn.functional as F
from numpy import dtype
from transformers import AutoConfig, AutoModelForCausalLM
from .modeling_utils import ConfigMixin, ModelMixin, register_to_config
from .sampling import cosine_schedule, mask_by_random_topk
from .phi import PhiForCausalLM
from .map_block import MAPBlock
from torch import nn
from einops.layers.torch import Rearrange


def _resolve_llm_backend(llm_backbone: str, hf_config: AutoConfig) -> str:
    requested = str(llm_backbone or "auto").lower()
    if requested in {"phi", "auto"}:
        if requested == "phi":
            return "phi"
        model_type = str(getattr(hf_config, "model_type", "")).lower()
        if model_type.startswith("phi"):
            return "phi"
        return "auto"
    raise ValueError(f"Unsupported llm_backbone='{llm_backbone}'. Expected 'auto' or 'phi'.")


def _auto_causallm_from_pretrained(
    llm_model_path: str,
    trust_remote_code: bool,
    attn_implementation: str,
):
    kwargs = {}
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    try:
        return AutoModelForCausalLM.from_pretrained(llm_model_path, **kwargs)
    except TypeError:
        kwargs.pop("attn_implementation", None)
        return AutoModelForCausalLM.from_pretrained(llm_model_path, **kwargs)


def _auto_causallm_from_config(hf_config: AutoConfig, trust_remote_code: bool):
    kwargs = {}
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    try:
        return AutoModelForCausalLM.from_config(hf_config, **kwargs)
    except TypeError:
        kwargs.pop("trust_remote_code", None)
        return AutoModelForCausalLM.from_config(hf_config, **kwargs)


class Upvla(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        w_clip_vit,
        vocab_size,
        llm_vocab_size,
        llm_model_path='',
        codebook_size=8192,
        num_vq_tokens=256,
        load_from_showo=True,
        llm_backbone="auto",
        trust_remote_code=False,
        attn_implementation="sdpa",
        act_step=10,
        framework="upvla",
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.register_to_config(mask_token_id=vocab_size - 1)
        self.framework = framework
        hf_config = AutoConfig.from_pretrained(llm_model_path, trust_remote_code=bool(trust_remote_code))
        if hasattr(hf_config, "_attn_implementation") and attn_implementation:
            hf_config._attn_implementation = attn_implementation
        backend = _resolve_llm_backend(llm_backbone=llm_backbone, hf_config=hf_config)
        if backend == "phi":
            if load_from_showo:
                self.showo = PhiForCausalLM(hf_config)
            else:
                self.showo = PhiForCausalLM.from_pretrained(
                    llm_model_path,
                    attn_implementation=attn_implementation,
                )
        else:
            if load_from_showo:
                self.showo = _auto_causallm_from_config(hf_config, trust_remote_code=bool(trust_remote_code))
            else:
                self.showo = _auto_causallm_from_pretrained(
                    llm_model_path=llm_model_path,
                    trust_remote_code=bool(trust_remote_code),
                    attn_implementation=attn_implementation,
                )
        self.showo.resize_token_embeddings(self.vocab_size)
        self.output_size = self.vocab_size
        if self.w_clip_vit:
            self.mm_projector = torch.nn.Sequential(
                torch.nn.Linear(1024, 2048), torch.nn.GELU(), torch.nn.Linear(2048, 2048))
        # action head
        self.token_learner = MAPBlock(n_latents=1, embed_dim=2048, n_heads=4)
        self.act_step = act_step
        self.to_logits = nn.Sequential(nn.Linear(2048, self.act_step * 7), Rearrange('... (a b) -> ... a b', b=7))

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = True

    def forward(
        self,
        input_ids,
        input_embeddings=None,
        attention_mask=None,
        labels=None,
        actions=None,
        clip_pad_tokens=False,
        label_smoothing=0.0,
        batch_size_pre=0,
        batch_size_mmu=0,
        max_seq_length=128,
        labels_mask_text=None,
        labels_mask_image=None,
        output_mode="action",
        action_loss_mask=None,
        **kwargs,
    ):
        if input_embeddings is None:
            logits = self.showo(input_ids=input_ids, attention_mask=attention_mask)['logits']
            return logits
        else:
            output = self.showo(
                inputs_embeds=input_embeddings, attention_mask=attention_mask, output_hidden_states=True)
        if labels is not None:
            logits = output['logits']
            if batch_size_pre > 0:
                loss_pre = F.cross_entropy(
                    logits[:batch_size_pre, max_seq_length + 1:-self.act_step].contiguous().view(-1, self.output_size),
                    labels[:batch_size_pre, max_seq_length + 1:-self.act_step].contiguous().view(-1),
                    ignore_index=-100,
                )
                tokens_vla = output['hidden_states'][-1][:batch_size_pre]
                tokens_vla = tokens_vla[:, -self.act_step:, :]  # tokens of future steps * <lvg>
                learned_tokens_vla = self.token_learner(tokens_vla)  # (b,hidden_size)
                logits_vla = self.to_logits(learned_tokens_vla)
                criterion = torch.nn.MSELoss(reduction="none")
                loss_act_map = criterion(logits_vla, actions)
                if action_loss_mask is not None:
                    mask = action_loss_mask.to(loss_act_map.dtype)
                    denom = mask.sum().clamp(min=1.0)
                    loss_act = (loss_act_map * mask).sum() / denom
                else:
                    loss_act = loss_act_map.mean()
            else:
                loss_pre = torch.tensor(0, dtype=logits.dtype, device=logits.device)
                loss_act = torch.tensor(0, dtype=logits.dtype, device=logits.device)

            if batch_size_mmu > 0:
                loss_mmu = F.cross_entropy(
                    logits[-batch_size_mmu:, :-1].contiguous().view(-1, self.output_size),
                    labels[-batch_size_mmu:, 1:].contiguous().view(-1),
                    ignore_index=-100,
                )
            else:
                loss_mmu = torch.tensor(0, dtype=logits.dtype, device=logits.device)

            return logits, loss_pre, loss_mmu, loss_act
        else:
            if output_mode == "action":
                tokens_vla = output['hidden_states'][-1]
                tokens_vla = tokens_vla[:, -self.act_step:, :]
                learned_tokens_vla = self.token_learner(tokens_vla)  # (b,hidden_size)
                logits_vla = self.to_logits(learned_tokens_vla)
                return logits_vla
            elif output_mode == "mmu":
                return output["logits"]
            else:
                raise ValueError(f"Invalid output_mode: {output_mode}")

    def pre_pad_predict(
        self,
        input_ids: torch.LongTensor = None,
        uncond_input_ids: torch.LongTensor = None,
        attention_mask=None,
        temperature=1.0,
        timesteps=18,  # ideal number of steps is 18 in maskgit paper
        guidance_scale=0,
        noise_schedule=cosine_schedule,
        generator: torch.Generator = None,
        config=None,
        return_actions=False,
        **kwargs,
    ):
        """
        Generate 1:1 similar to the original MaskGit repo
        https://github.com/google-research/maskgit/blob/main/maskgit/libml/parallel_decode.py#L79
        """
        # begin with all image token ids masked
        num_vq_tokens = config.model.showo.num_vq_tokens * config.model.vla.num_view
        num_new_special_tokens = config.model.showo.num_new_special_tokens
        output = self.showo(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=return_actions)
        logits = output['logits']
        logits = logits[:, -(num_vq_tokens + 1) - self.act_step:-1 - self.act_step,
                        config.model.showo.llm_vocab_size + num_new_special_tokens:-1]
        sampled_ids = torch.argmax(logits, dim=-1)

        # for infer actions
        if return_actions:
            framework = str(getattr(config.model, "framework", "upvla")).lower()
            bd_cfg = config.get("block_diffusion", {})
            use_bd_action_infer = framework == "bdvla" and bool(bd_cfg.get("action_infer_enabled", True))
            if use_bd_action_infer:
                action_bins = int(bd_cfg.get("action_num_bins", 256))
                infer_steps = int(bd_cfg.get("action_infer_steps", 6))
                conf_threshold = float(bd_cfg.get("action_conf_threshold", 0.9))
                block_size = int(bd_cfg.get("block_size", 32))
                infer_steps = max(1, infer_steps)
                token_offset = int(config.model.showo.llm_vocab_size + config.model.showo.num_new_special_tokens)
                mask_token_id = int(self.config.mask_token_id)

                bsz = input_ids.shape[0]
                action_len = int(self.act_step * 7)
                prefix_len = input_ids.shape[1]

                # Native BD-style iterative refinement on x_t with x_0 mirror.
                action_xt = torch.full((bsz, action_len), mask_token_id, dtype=torch.long, device=input_ids.device)
                action_x0 = torch.full_like(action_xt, mask_token_id)

                total = prefix_len + 2 * action_len
                dtype_mask = output["logits"].dtype
                allow = torch.zeros((total, total), dtype=torch.bool, device=input_ids.device)
                if prefix_len > 0:
                    allow[:prefix_len, :prefix_len] = torch.tril(
                        torch.ones((prefix_len, prefix_len), dtype=torch.bool, device=input_ids.device))
                    allow[prefix_len:, :prefix_len] = True

                local_n = action_len
                local_total = 2 * local_n
                local_idx = torch.arange(local_total, device=input_ids.device)
                local_is_x0 = local_idx >= local_n
                local_block = torch.where(local_is_x0, (local_idx - local_n) // max(1, block_size),
                                          local_idx // max(1, block_size))
                q_x0 = local_is_x0[:, None]
                k_x0 = local_is_x0[None, :]
                q_block = local_block[:, None]
                k_block = local_block[None, :]
                local_allow = (
                    ((q_block == k_block) & (q_x0 == k_x0)) |
                    ((q_block > k_block) & (~q_x0) & k_x0) |
                    ((q_block >= k_block) & q_x0 & k_x0)
                )
                allow[prefix_len:, prefix_len:] = local_allow

                attention_mask_bd = torch.zeros((total, total), dtype=dtype_mask, device=input_ids.device)
                attention_mask_bd = attention_mask_bd.masked_fill(
                    ~allow,
                    torch.finfo(dtype_mask).min if dtype_mask.is_floating_point else torch.finfo(torch.float32).min,
                )
                attention_mask_bd = attention_mask_bd.unsqueeze(0).unsqueeze(0).expand(bsz, 1, total, total)

                last_pred_tokens = None
                for _ in range(infer_steps):
                    action_input_ids = torch.cat([input_ids, action_xt, action_x0], dim=1)
                    action_logits = self.showo(input_ids=action_input_ids, attention_mask=attention_mask_bd)["logits"]

                    logits_xt = action_logits[:, prefix_len:prefix_len + action_len, :]
                    logits_xt_aligned = torch.cat([logits_xt[:, :1, :], logits_xt[:, :-1, :]], dim=1)
                    probs = torch.softmax(logits_xt_aligned, dim=-1)
                    pred_ids = torch.argmax(logits_xt_aligned, dim=-1)
                    pred_bins = (pred_ids - token_offset).clamp(min=0, max=action_bins - 1)
                    pred_tokens = pred_bins + token_offset
                    last_pred_tokens = pred_tokens

                    pred_conf = probs.gather(-1, pred_ids.unsqueeze(-1)).squeeze(-1)
                    masked = action_xt.eq(mask_token_id)
                    if not masked.any():
                        break

                    update = (pred_conf >= conf_threshold) & masked
                    for b in range(bsz):
                        if masked[b].any() and not update[b].any():
                            cand = pred_conf[b].masked_fill(~masked[b], float("-inf"))
                            best = torch.argmax(cand)
                            update[b, best] = True

                    action_xt = torch.where(update, pred_tokens, action_xt)
                    action_x0 = action_xt.clone()

                if last_pred_tokens is not None:
                    action_xt = torch.where(action_xt.eq(mask_token_id), last_pred_tokens, action_xt)
                else:
                    fallback = torch.full_like(action_xt, token_offset + action_bins // 2)
                    action_xt = torch.where(action_xt.eq(mask_token_id), fallback, action_xt)

                action_bins_pred = (action_xt - token_offset).clamp(min=0, max=action_bins - 1).float()
                actions = action_bins_pred / float(action_bins - 1)
                actions = actions * 2.0 - 1.0
                actions = actions.reshape(bsz, self.act_step, 7)
                return sampled_ids, actions

            tokens_vla = output['hidden_states'][-1]
            tokens_vla = tokens_vla[:, -self.act_step:, :]
            learned_tokens_vla = self.token_learner(tokens_vla)  # (b,hidden_size)
            logits_vla = self.to_logits(learned_tokens_vla)
            return sampled_ids, logits_vla

        return sampled_ids

    def pre_generate(
        self,
        input_ids: torch.LongTensor = None,
        uncond_input_ids: torch.LongTensor = None,
        attention_mask=None,
        temperature=1.0,
        timesteps=18,  # ideal number of steps is 18 in maskgit paper
        guidance_scale=0,
        noise_schedule=cosine_schedule,
        generator: torch.Generator = None,
        config=None,
        **kwargs,
    ):
        """
        Generate 1:1 similar to the original MaskGit repo
        https://github.com/google-research/maskgit/blob/main/maskgit/libml/parallel_decode.py#L79
        """
        # begin with all image token ids masked
        mask_token_id = self.config.mask_token_id
        num_vq_tokens = config.model.showo.num_vq_tokens
        num_new_special_tokens = config.model.showo.num_new_special_tokens

        input_ids_minus_lm_vocab_size = input_ids[:, -(num_vq_tokens + 1):-1].clone()
        input_ids_minus_lm_vocab_size = torch.where(
            input_ids_minus_lm_vocab_size == mask_token_id, mask_token_id,
            input_ids_minus_lm_vocab_size - config.model.showo.llm_vocab_size - num_new_special_tokens)

        # for classifier-free guidance
        if uncond_input_ids is not None:
            uncond_prefix = uncond_input_ids[:, :config.dataset.preprocessing.max_seq_length + 1]

        for step in range(timesteps):
            if uncond_input_ids is not None and guidance_scale > 0:
                uncond_input_ids = torch.cat(
                    [uncond_prefix, input_ids[:, config.dataset.preprocessing.max_seq_length + 1:]], dim=1)
                model_input = torch.cat([input_ids, uncond_input_ids])
                cond_logits, uncond_logits = self(model_input, attention_mask=attention_mask).chunk(2)
                # logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
                # it seems that muse has a different cfg setting
                logits = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits
                logits = logits[:, -(num_vq_tokens + 1):-1,
                                config.model.showo.llm_vocab_size + num_new_special_tokens:-1]
            else:
                logits = self(input_ids, attention_mask=attention_mask)
                logits = logits[:, -(num_vq_tokens + 1):-1,
                                config.model.showo.llm_vocab_size + num_new_special_tokens:-1]

            probs = logits.softmax(dim=-1)
            sampled = probs.reshape(-1, logits.size(-1))
            sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1])

            unknown_map = input_ids_minus_lm_vocab_size == mask_token_id
            sampled_ids = torch.where(unknown_map, sampled_ids, input_ids_minus_lm_vocab_size)
            # Defines the mask ratio for the next round. The number to mask out is
            # determined by mask_ratio * unknown_number_in_the_beginning.
            ratio = 1.0 * (step + 1) / timesteps
            mask_ratio = noise_schedule(torch.tensor(ratio))
            # Computes the probabilities of each selected tokens.
            selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])
            selected_probs = selected_probs.squeeze(-1)

            # Ignores the tokens given in the input by overwriting their confidence.
            selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
            # Gets mask lens for each sample in the batch according to the mask ratio.
            mask_len = (num_vq_tokens * mask_ratio).floor().unsqueeze(0).to(logits.device)
            # Keeps at least one of prediction in this round and also masks out at least
            # one and for the next iteration
            mask_len = torch.max(
                torch.tensor([1], device=logits.device), torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len))
            # Adds noise for randomness
            temperature = temperature * (1.0 - ratio)
            masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)
            # Masks tokens with lower confidence.
            input_ids[:, -(num_vq_tokens + 1):-1] = torch.where(
                masking, mask_token_id, sampled_ids + config.model.showo.llm_vocab_size + num_new_special_tokens)
            input_ids_minus_lm_vocab_size = torch.where(masking, mask_token_id, sampled_ids)

        return sampled_ids

    @torch.no_grad()
    def mmu_generate(self,
                     idx=None,
                     input_embeddings=None,
                     attention_mask=None,
                     max_new_tokens=100,
                     temperature=1.0,
                     top_k=None,
                     eot_token=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        try:
            device = idx.device
        except:
            device = input_embeddings.device

        result = []
        for _ in range(max_new_tokens):
            logits = self(idx, input_embeddings=input_embeddings, attention_mask=attention_mask, output_mode="mmu")
            # print(logits)
            L = attention_mask.shape[-1]
            attention_mask = attention_mask.squeeze()
            attention_mask_a = torch.hstack([
                attention_mask,  # L, L
                torch.zeros((L, 1)).to(device) + torch.finfo(logits.dtype).min,
            ])
            attention_mask_b = torch.vstack([
                attention_mask_a,  # L, L+1
                torch.hstack([attention_mask[-1, :], torch.tensor([0]).to(device)]).unsqueeze(0),
            ])
            # attention_mask = attention_mask_b # L+1, L+1 , from origin code but get bug
            attention_mask = attention_mask_b.unsqueeze(0).unsqueeze(0)  # 1,1, L+1, L+1, fix bug by upvla

            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            result.append(idx_next[0][0])
            # append sampled index to the running sequence and continue
            if self.config.w_clip_vit:
                idx_next_embeddings = self.showo.model.embed_tokens(idx_next)
                input_embeddings = torch.cat([input_embeddings, idx_next_embeddings], dim=1)
            else:
                idx = torch.cat((idx, idx_next), dim=1)

            if eot_token is not None and idx_next.cpu() == eot_token:
                break

        return result
