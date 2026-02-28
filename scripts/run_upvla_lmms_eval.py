#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_TASKS = "textvqa_val,docvqa_val,chartqa,vizwiz_vqa_val"


def _format_model_args(args_dict):
    parts = []
    for key, value in args_dict.items():
        if value is None:
            continue
        parts.append(f"{key}={value}")
    return ",".join(parts)


def _find_latest_results_file(output_dir: Path):
    candidates = list(output_dir.rglob("*_results.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _extract_numeric_metrics(results_json_path: Path):
    with open(results_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for task_name, metrics in data.get("results", {}).items():
        if not isinstance(metrics, dict):
            continue
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                out[(task_name, metric_name)] = float(value)
    return out


def _run_once(
    repo_root: Path,
    model_config: str,
    tasks: str,
    output_dir: Path,
    model_tag: str,
    device: str,
    dtype: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    limit: int,
    log_samples: bool,
):
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root}:{env.get('PYTHONPATH', '')}".rstrip(":")
    env["LMMS_EVAL_PLUGINS"] = "upvla_lmms_plugin"

    run_out = output_dir / model_tag
    run_out.mkdir(parents=True, exist_ok=True)

    model_args = _format_model_args(
        {
            "model_config": model_config,
            "device": device,
            "dtype": dtype,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
        }
    )

    cmd = [
        sys.executable,
        "-m",
        "lmms_eval",
        "--model",
        "upvla_mmu",
        "--model_args",
        model_args,
        "--tasks",
        tasks,
        "--batch_size",
        "1",
        "--output_path",
        str(run_out),
    ]
    if limit >= 0:
        cmd += ["--limit", str(limit)]
    if log_samples:
        cmd += ["--log_samples"]

    print("Running:")
    print(" ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, check=True, env=env, cwd=str(repo_root))

    latest = _find_latest_results_file(run_out)
    if latest is None:
        raise FileNotFoundError(f"No *_results.json found under {run_out}")
    print(f"Latest results: {latest}")
    return latest


def _print_comparison(base_metrics, cand_metrics, base_name, cand_name):
    common = sorted(set(base_metrics) & set(cand_metrics))
    if not common:
        print("No overlapping numeric metrics found for comparison.")
        return
    print(f"\nComparison ({cand_name} - {base_name}):")
    for task_name, metric_name in common:
        b = base_metrics[(task_name, metric_name)]
        c = cand_metrics[(task_name, metric_name)]
        print(f"{task_name:24s} {metric_name:28s} base={b:.4f} cand={c:.4f} delta={c - b:+.4f}")


def main():
    parser = argparse.ArgumentParser(description="Run UPVLA MMU benchmark with lmms-eval tasks (TextVQA/DocVQA/etc).")
    parser.add_argument("--model-config", required=True, help="Path to UPVLA rollout/train model yaml.")
    parser.add_argument("--baseline-model-config", default=None, help="Optional baseline config path for drop comparison.")
    parser.add_argument("--tasks", default=DEFAULT_TASKS, help="Comma-separated lmms-eval task names.")
    parser.add_argument("--output-dir", default="lmms_outputs/upvla_eval", help="Where lmms-eval outputs are saved.")
    parser.add_argument("--model-tag", default="candidate", help="Output subdir name for candidate run.")
    parser.add_argument("--baseline-tag", default="baseline", help="Output subdir name for baseline run.")
    parser.add_argument("--device", default="cuda", help="Model device passed to upvla_mmu model_args.")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"], help="Model weight dtype.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--limit", type=int, default=-1, help="Set >=0 for quick smoke.")
    parser.add_argument("--log-samples", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir).expanduser()

    try:
        import lmms_eval  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "lmms_eval is not installed. Install it first, e.g. `pip install -e /path/to/lmms-eval`."
        ) from exc

    cand_json = _run_once(
        repo_root=repo_root,
        model_config=args.model_config,
        tasks=args.tasks,
        output_dir=output_dir,
        model_tag=args.model_tag,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        limit=args.limit,
        log_samples=args.log_samples,
    )

    if args.baseline_model_config:
        base_json = _run_once(
            repo_root=repo_root,
            model_config=args.baseline_model_config,
            tasks=args.tasks,
            output_dir=output_dir,
            model_tag=args.baseline_tag,
            device=args.device,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            limit=args.limit,
            log_samples=args.log_samples,
        )
        base_metrics = _extract_numeric_metrics(base_json)
        cand_metrics = _extract_numeric_metrics(cand_json)
        _print_comparison(base_metrics, cand_metrics, args.baseline_tag, args.model_tag)


if __name__ == "__main__":
    main()
