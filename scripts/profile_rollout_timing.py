#!/usr/bin/env python3
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import time

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from pytorch_lightning import seed_everything


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from policy_evaluation.multistep_sequences import get_sequences
from policy_evaluation.utils import get_default_beso_and_env, get_env_state_for_initial_condition
import policy_rollout.calvin_evaluate_upvla as eval_mod
from training.utils import image_transform


def _add(timers, key, dt):
    timers[key] += float(dt)


def rollout_timed(env, model_bundle, task_oracle, cfg, subtask, lang_embeddings, val_annotations, timers):
    obs = env.get_obs()
    lang_annotation = val_annotations[subtask][0]
    goal = lang_embeddings.get_lang_goal(lang_annotation)
    goal["lang_text"] = lang_annotation

    model_config, model, uni_prompting, vq_model, _ = model_bundle
    start_info = env.get_info()
    action_buffer = None

    loop_start = time.perf_counter()
    for step in range(int(cfg.ep_len)):
        if step % int(model_config.act_step) == 0:
            t0 = time.perf_counter()
            img_static = obs["rgb_obs"]["rgb_static"].squeeze().permute(1, 2, 0).cpu().numpy()
            image_ori_static = (img_static + 1.0) / 2.0
            image_ori_static *= 255.0
            pixel_values_static = image_transform(
                eval_mod.Image.fromarray(np.uint8(image_ori_static)),
                resolution=model_config.dataset.preprocessing.resolution,
            ).to(model.device).unsqueeze(0)
            image_tokens = vq_model.get_code(pixel_values_static) + len(uni_prompting.text_tokenizer)
            if int(model_config.model.vla.num_view) == 2:
                img_gripper = obs["rgb_obs"]["rgb_gripper"].squeeze().permute(1, 2, 0).cpu().numpy()
                image_ori_gripper = (img_gripper + 1.0) / 2.0
                image_ori_gripper *= 255.0
                pixel_values_gripper = image_transform(
                    eval_mod.Image.fromarray(np.uint8(image_ori_gripper)),
                    resolution=model_config.dataset.preprocessing.resolution,
                ).to(model.device).unsqueeze(0)
                image_tokens_gripper = vq_model.get_code(pixel_values_gripper) + len(uni_prompting.text_tokenizer)
                image_tokens = torch.cat([image_tokens, image_tokens_gripper], dim=1)
            t1 = time.perf_counter()
            _add(timers, "vision_encode_s", t1 - t0)

            instruction = goal["lang_text"]
            input_ids, _ = uni_prompting(([instruction], image_tokens), "pre_gen")
            attention_mask = eval_mod.create_attention_mask_predict_next_for_future_prediction(
                input_ids,
                pad_id=int(uni_prompting.sptids_dict["<|pad|>"]),
                soi_id=int(uni_prompting.sptids_dict["<|soi|>"]),
                eoi_id=int(uni_prompting.sptids_dict["<|eoi|>"]),
                rm_pad_in_image=True,
            )
            t2 = time.perf_counter()
            _add(timers, "text_prep_s", t2 - t1)

            with torch.no_grad():
                gen_token_ids, actions = model.pre_pad_predict(
                    input_ids=input_ids,
                    uncond_input_ids=None,
                    attention_mask=attention_mask,
                    guidance_scale=None,
                    temperature=None,
                    timesteps=None,
                    noise_schedule=None,
                    noise_type=None,
                    predict_all_tokens=None,
                    seq_len=model_config.model.showo.num_vq_tokens,
                    uni_prompting=uni_prompting,
                    config=model_config,
                    return_actions=True,
                )
            action_buffer = actions.squeeze().detach()
            t3 = time.perf_counter()
            _add(timers, "action_model_s", t3 - t2)

            # Keep consistent with original rollout implementation:
            # image decode branch runs every act step regardless of video recording.
            if int(model_config.model.vla.num_view) == 1:
                image_tokens_list = [image_tokens]
                gen_token_ids_list = [gen_token_ids]
            elif int(model_config.model.vla.num_view) == 2:
                image_tokens_ori_static, image_tokens_ori_gripper = image_tokens.chunk(2, dim=1)
                image_tokens_list = [image_tokens_ori_static, image_tokens_ori_gripper]
                gen_token_ids_static, gen_token_ids_gripper = gen_token_ids.chunk(2, dim=1)
                gen_token_ids_list = [gen_token_ids_static, gen_token_ids_gripper]
            else:
                raise NotImplementedError

            for image_tokens_i, gen_token_ids_i in zip(image_tokens_list, gen_token_ids_list):
                gen_token_ids_i = torch.clamp(
                    gen_token_ids_i,
                    max=model_config.model.showo.codebook_size - 1,
                    min=0,
                )
                _ = vq_model.decode_code(gen_token_ids_i)
                _ = vq_model.decode_code(image_tokens_i - len(uni_prompting.text_tokenizer))
            t4 = time.perf_counter()
            _add(timers, "vision_decode_s", t4 - t3)
            timers["act_calls"] += 1

        e0 = time.perf_counter()
        obs, _, _, current_info = env.step(action_buffer[step % int(model_config.act_step)])
        e1 = time.perf_counter()
        _add(timers, "env_step_s", e1 - e0)

        c0 = time.perf_counter()
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        c1 = time.perf_counter()
        _add(timers, "task_check_s", c1 - c0)
        timers["env_steps"] += 1

        if len(current_task_info) > 0:
            _add(timers, "rollout_loop_s", time.perf_counter() - loop_start)
            return True, step + 1

    _add(timers, "rollout_loop_s", time.perf_counter() - loop_start)
    return False, int(cfg.ep_len)


def evaluate_one_sequence_timed(env, model_bundle, lang_embeddings, cfg, sequence, timers):
    task_oracle = hydra.utils.instantiate(cfg.tasks)
    val_annotations = cfg.annotations
    initial_state, eval_sequence = sequence
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    subtask_steps = []
    for subtask in eval_sequence:
        ok, steps = rollout_timed(env, model_bundle, task_oracle, cfg, subtask, lang_embeddings, val_annotations, timers)
        subtask_steps.append({"subtask": subtask, "success": bool(ok), "steps": int(steps)})
        if ok:
            success_counter += 1
        else:
            break
    return success_counter, subtask_steps


def compose_eval_cfg(dataset_root: Path, overrides):
    conf_dir = REPO_ROOT / "policy_conf"
    with hydra.initialize_config_dir(config_dir=str(conf_dir), version_base=None):
        cfg = hydra.compose(config_name="calvin_evaluate_upvla", overrides=overrides)
    cfg.dataset_path = str(dataset_root)
    cfg.root_data_dir = str(dataset_root)
    cfg.log_wandb = False
    cfg.num_videos = 0
    cfg.save_rollout_as_video = False
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Profile one CALVIN rollout sequence by text/vision/action time.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--num-sequences", type=int, default=1)
    parser.add_argument("--sequence-index", type=int, default=0)
    parser.add_argument("--sequence-workers", type=int, default=4)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--ep-len", type=int, default=360)
    parser.add_argument("--dist-backend", default="gloo")
    parser.add_argument("--master-port", type=int, default=29529)
    parser.add_argument("--seed", type=int, default=242)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    model_config_path = Path(args.model_config).expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(dataset_root)
    if not model_config_path.exists():
        raise FileNotFoundError(model_config_path)

    overrides = [
        f"dataset_path={dataset_root}",
        f"root_data_dir={dataset_root}",
        f"model_config={model_config_path}",
        f"device={args.device}",
        f"ep_len={args.ep_len}",
        "log_wandb=False",
        "num_videos=0",
        "save_rollout_as_video=False",
        "hydra.job.chdir=False",
        f"dist_backend={args.dist_backend}",
        f"dist_master_port={args.master_port}",
    ]
    cfg = compose_eval_cfg(dataset_root, overrides)

    runtime = eval_mod.setup_distributed(cfg)
    seed_everything(int(args.seed) + runtime["rank"], workers=True)  # type: ignore

    model_cfg = OmegaConf.load(model_config_path)
    model_cfg = eval_mod.resolve_model_paths(model_cfg, model_config_path)

    env, _, lang_embeddings = get_default_beso_and_env(
        dataset_path=str(dataset_root),
        env=None,
        lang_embeddings=None,
        device_id=runtime["device_id"],
        cfg=cfg,
    )
    model_bundle, loaded_from = eval_mod.get_upvla_agent(model_cfg, runtime["device_id"])

    # Deterministic sequence set; run exactly one sequence by index.
    seqs = get_sequences(args.num_sequences, num_workers=args.sequence_workers)
    if not seqs:
        raise RuntimeError("No sequences generated.")
    seq_idx = int(args.sequence_index)
    if seq_idx < 0 or seq_idx >= len(seqs):
        raise IndexError(f"sequence-index {seq_idx} out of range (n={len(seqs)})")
    sequence = seqs[seq_idx]

    timers = defaultdict(float)
    wall0 = time.perf_counter()
    success_counter, subtask_steps = evaluate_one_sequence_timed(env, model_bundle, lang_embeddings, cfg, sequence, timers)
    wall1 = time.perf_counter()
    timers["wall_total_s"] = float(wall1 - wall0)

    text_s = float(timers["text_prep_s"])
    vision_s = float(timers["vision_encode_s"] + timers["vision_decode_s"])
    action_s = float(timers["action_model_s"])
    major_total = text_s + vision_s + action_s

    result = {
        "loaded_checkpoint": loaded_from,
        "sequence_index": seq_idx,
        "sequence": [str(x) for x in sequence[1]],
        "success_counter": int(success_counter),
        "subtasks": subtask_steps,
        "timers": {
            "wall_total_s": float(timers["wall_total_s"]),
            "rollout_loop_s": float(timers["rollout_loop_s"]),
            "text_prep_s": text_s,
            "vision_encode_s": float(timers["vision_encode_s"]),
            "vision_decode_s": float(timers["vision_decode_s"]),
            "vision_total_s": vision_s,
            "action_model_s": action_s,
            "env_step_s": float(timers["env_step_s"]),
            "task_check_s": float(timers["task_check_s"]),
            "act_calls": int(timers["act_calls"]),
            "env_steps": int(timers["env_steps"]),
        },
        "major_breakdown": {
            "text_s": text_s,
            "vision_s": vision_s,
            "action_s": action_s,
            "text_pct_of_major": float(text_s / major_total * 100.0) if major_total > 0 else 0.0,
            "vision_pct_of_major": float(vision_s / major_total * 100.0) if major_total > 0 else 0.0,
            "action_pct_of_major": float(action_s / major_total * 100.0) if major_total > 0 else 0.0,
        },
    }

    print(json.dumps(result, indent=2))
    if args.output_json:
        out_path = Path(args.output_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"saved {out_path}")

    eval_mod.cleanup_distributed()


if __name__ == "__main__":
    main()
