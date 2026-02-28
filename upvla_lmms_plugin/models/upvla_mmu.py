import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPImageProcessor

from llava.llava import conversation as conversation_lib
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from models import CLIPVisionTower, Upvla
from training.prompting_utils import UniversalPrompting_w_action, create_attention_mask_for_mmu_vit


def _resolve_from_config_dir(config_path: Path, maybe_path):
    if maybe_path is None:
        return maybe_path
    raw_path = str(maybe_path).strip()
    if not raw_path:
        return raw_path
    path_obj = Path(raw_path).expanduser()
    if path_obj.is_absolute():
        return str(path_obj)
    return str((config_path.parent / path_obj).resolve())


def _resolve_state_dict_path(tuned_model_path: str):
    tuned_path = Path(tuned_model_path).expanduser()
    if tuned_path.is_file():
        if tuned_path.suffix in {".bin", ".safetensors"}:
            return tuned_path
        raise ValueError(f"Unsupported checkpoint file format: {tuned_path}")

    candidates = [
        tuned_path / "unwrapped_model" / "pytorch_model.bin",
        tuned_path / "pytorch_model.bin",
        tuned_path / "model.safetensors",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _to_dtype(dtype: str):
    key = str(dtype).strip().lower()
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.float32


@register_model("upvla_mmu")
class UPVLAMMU(lmms):
    def __init__(
        self,
        model_config: str,
        batch_size: int = 1,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        top_k: int = 1,
        dtype: str = "bf16",
        device: str = "cuda",
        system_prompt: str = "",
        system_prompt_len: int = 0,
        **kwargs,
    ):
        super().__init__()
        if not model_config:
            raise ValueError("model_config is required, e.g. model_config=/path/to/arvla_model.yaml")

        conversation_lib.default_conversation = conversation_lib.conv_templates["phi1.5"]
        self.system_prompt = system_prompt
        self.default_max_new_tokens = int(max_new_tokens)
        self.default_temperature = float(temperature)
        self.default_top_k = int(top_k)
        self.batch_size_per_gpu = max(1, int(batch_size))

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ], "Unsupported distributed type for lmms-eval"
        self.accelerator = accelerator
        self._rank = accelerator.local_process_index
        self._world_size = accelerator.num_processes

        if device == "cuda":
            self.device = accelerator.device
        else:
            self.device = torch.device(device)
        self.weight_dtype = _to_dtype(dtype)
        if self.device.type != "cuda":
            self.weight_dtype = torch.float32

        model_config_path = Path(model_config).expanduser()
        if not model_config_path.is_absolute():
            model_config_path = model_config_path.resolve()
        cfg = OmegaConf.load(model_config_path)
        cfg.model.vq_model.vq_model_name = _resolve_from_config_dir(model_config_path, cfg.model.vq_model.vq_model_name)
        for key in ("llm_model_path", "pretrained_model_path", "tuned_model_path"):
            if key in cfg.model.showo:
                cfg.model.showo[key] = _resolve_from_config_dir(model_config_path, cfg.model.showo[key])

        self.config = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.showo.llm_model_path, padding_side="left")
        self.uni_prompting = UniversalPrompting_w_action(
            self.tokenizer,
            max_text_len=cfg.dataset.preprocessing.max_seq_length,
            special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
            ignore_id=-100,
            cond_dropout_prob=0.0,
            future_steps=cfg.act_step,
        )
        self.system_prompt_len = int(system_prompt_len)

        self.model = Upvla.from_pretrained(
            cfg.model.showo.pretrained_model_path,
            low_cpu_mem_usage=False,
            act_step=cfg.act_step,
        ).to(self.device)
        tuned_model_path = str(getattr(cfg.model.showo, "tuned_model_path", "") or "").strip()
        if tuned_model_path:
            state_dict_path = _resolve_state_dict_path(tuned_model_path)
            if state_dict_path is None:
                raise FileNotFoundError(
                    f"Could not find tuned checkpoint under: {tuned_model_path}. "
                    "Expected one of: unwrapped_model/pytorch_model.bin, pytorch_model.bin, model.safetensors"
                )
            if state_dict_path.suffix == ".safetensors":
                from safetensors.torch import load_file as load_safetensors

                state_dict = load_safetensors(str(state_dict_path), device="cpu")
            else:
                state_dict = torch.load(str(state_dict_path), map_location="cpu")
            self.model.load_state_dict(state_dict, strict=True)
            del state_dict

        self.model.eval()
        if not hasattr(self.model, "mm_projector"):
            raise RuntimeError("Loaded UPVLA model has no mm_projector. Expected w_clip_vit=True checkpoint.")
        if self.device.type == "cuda":
            self.model = self.model.to(dtype=self.weight_dtype)

        self.vision_tower = CLIPVisionTower("openai/clip-vit-large-patch14-336").to(self.device)
        self.vision_tower.eval()
        for p in self.vision_tower.parameters():
            p.requires_grad = False
        if self.device.type == "cuda":
            self.vision_tower = self.vision_tower.to(dtype=self.weight_dtype)
        self.clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

        if self._rank == 0:
            eval_logger.info(f"Loaded upvla_mmu from config={model_config_path}")

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def eot_token_id(self):
        return int(self.tokenizer.eos_token_id)

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        if isinstance(tokens, int):
            tokens = [tokens]
        return self.tokenizer.decode(tokens)

    @staticmethod
    def _flatten_visuals(visuals):
        flat = []
        for x in visuals:
            if isinstance(x, list):
                flat.extend(x)
            else:
                flat.append(x)
        return flat

    @staticmethod
    def _to_pil_image(x):
        if isinstance(x, Image.Image):
            return x.convert("RGB")
        if isinstance(x, str):
            return Image.open(x).convert("RGB")
        if isinstance(x, np.ndarray):
            arr = x.astype(np.uint8)
            return Image.fromarray(arr).convert("RGB")
        raise TypeError(f"Unsupported visual type: {type(x)}")

    def _generate_one(self, question: str, image: Image.Image, gen_kwargs: dict) -> str:
        conv = conversation_lib.default_conversation.copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt().strip()

        input_ids_question = self.tokenizer(prompt_question, return_tensors="pt", padding="longest").input_ids.to(self.device)
        input_ids_system = self.tokenizer(self.system_prompt, return_tensors="pt", padding="longest").input_ids.to(self.device)
        if input_ids_system.ndim == 2:
            input_ids_system = input_ids_system[0].unsqueeze(0)

        mmu_id = int(self.uni_prompting.sptids_dict["<|mmu|>"])
        soi_id = int(self.uni_prompting.sptids_dict["<|soi|>"])
        eoi_id = int(self.uni_prompting.sptids_dict["<|eoi|>"])
        prefix = torch.tensor([[mmu_id]], device=self.device, dtype=torch.long)
        soi = torch.tensor([[soi_id]], device=self.device, dtype=torch.long)
        eoi = torch.tensor([[eoi_id]], device=self.device, dtype=torch.long)
        input_ids_llava = torch.cat([prefix, input_ids_system, soi, eoi, input_ids_question], dim=1).long()

        pixel_values = self.clip_processor.preprocess(image, return_tensors="pt")["pixel_values"].to(self.device)
        pixel_values = pixel_values.to(dtype=self.vision_tower.dtype)

        with torch.no_grad():
            image_features = self.vision_tower(pixel_values).to(dtype=self.model.mm_projector[0].weight.dtype)
            image_embeddings = self.model.mm_projector(image_features)
            text_embeddings = self.model.showo.model.embed_tokens(input_ids_llava).to(dtype=image_embeddings.dtype)
            split_idx = 1 + self.system_prompt_len + 1
            part1 = text_embeddings[:, :split_idx, :]
            part2 = text_embeddings[:, split_idx:, :]
            input_embeddings = torch.cat((part1, image_embeddings, part2), dim=1)

            mask_dtype = input_embeddings.dtype
            attention_mask = create_attention_mask_for_mmu_vit(
                input_embeddings, system_prompt_len=self.system_prompt_len
            ).to(mask_dtype)

            max_new_tokens = int(gen_kwargs.get("max_new_tokens", self.default_max_new_tokens))
            top_k = gen_kwargs.get("top_k", self.default_top_k)
            top_k = None if top_k is None else int(top_k)
            cont_toks_list = self.model.mmu_generate(
                input_embeddings=input_embeddings,
                attention_mask=attention_mask[0].unsqueeze(0),
                max_new_tokens=max_new_tokens,
                temperature=float(gen_kwargs.get("temperature", self.default_temperature)),
                top_k=top_k,
                eot_token=self.tokenizer.eos_token_id,
            )

        if len(cont_toks_list) == 0:
            return ""
        cont_tokens = torch.stack(cont_toks_list).reshape(1, -1)
        text = self.tokenizer.batch_decode(cont_tokens, skip_special_tokens=True)[0].strip()
        until = gen_kwargs.get("until", [])
        if isinstance(until, str):
            until = [until]
        for stop in until:
            if stop and stop in text:
                text = text.split(stop)[0]
        return text.strip()

    def generate_until(self, requests: List[Instance]) -> List[str]:
        outputs = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="UPVLA MMU")
        for req in requests:
            contexts, gen_kwargs, doc_to_visual, doc_id, task, split = req.args
            doc = self.task_dict[task][split][doc_id]
            try:
                visuals = self._flatten_visuals(doc_to_visual(doc))
                if len(visuals) == 0:
                    raise ValueError(f"No visuals returned for task={task}, split={split}, doc_id={doc_id}")
                image = self._to_pil_image(visuals[0])
                answer = self._generate_one(str(contexts), image, dict(gen_kwargs))
            except Exception as exc:  # noqa: BLE001
                eval_logger.error(
                    f"Failed on task={task} split={split} doc_id={doc_id}: {type(exc).__name__}: {exc}"
                )
                answer = ""
            outputs.append(answer)
            pbar.update(1)
        pbar.close()
        return outputs

    def generate_until_multi_round(self, requests) -> List[str]:
        return self.generate_until(requests)

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("UPVLAMMU currently supports generate_until tasks only.")
