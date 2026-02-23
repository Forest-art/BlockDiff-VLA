#!/usr/bin/env python3
"""Summarize offline action evaluation JSON outputs into a compact table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def _format_float(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    return "nan"


def _load_overall_metrics(json_path: Path) -> Tuple[str, Dict[str, object]]:
    payload = json.loads(json_path.read_text())
    overall = payload.get("overall", {})
    tag = json_path.name.replace("_offline_eval.json", "")
    return tag, overall


def _print_markdown_table(rows: List[Dict[str, str]]) -> None:
    print("| tag | samples | steps | mae_all | rmse_all | mae_pos | mae_rot | mae_gripper | grip_acc |")
    print("|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        print(
            f"| {row['tag']} | {row['samples']} | {row['steps']} | "
            f"{row['mae_all']} | {row['rmse_all']} | {row['mae_pos']} | "
            f"{row['mae_rot']} | {row['mae_gripper']} | {row['grip_acc']} |"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize BlockDiff-VLA offline eval json files.")
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Run root that contains results/<tag>_offline_eval.json",
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        default=["debug_arvla", "debug_mdmvla", "debug_bdvla"],
        help="Tag list used when --run-root is provided.",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=None,
        help="Explicit json paths. If provided, --run-root/--tags are ignored.",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Exit non-zero if any expected file is missing.",
    )
    args = parser.parse_args()

    if args.inputs:
        json_paths = [Path(x).expanduser().resolve() for x in args.inputs]
    else:
        if args.run_root is None:
            parser.error("Either --inputs or --run-root must be provided.")
        run_root = args.run_root.expanduser().resolve()
        json_paths = [run_root / "results" / f"{tag}_offline_eval.json" for tag in args.tags]

    rows: List[Dict[str, str]] = []
    missing: List[str] = []
    for json_path in json_paths:
        if not json_path.exists():
            missing.append(str(json_path))
            continue

        tag, overall = _load_overall_metrics(json_path)
        rows.append(
            {
                "tag": tag,
                "samples": str(overall.get("samples", "nan")),
                "steps": str(overall.get("steps", "nan")),
                "mae_all": _format_float(overall.get("mae_all")),
                "rmse_all": _format_float(overall.get("rmse_all")),
                "mae_pos": _format_float(overall.get("mae_pos")),
                "mae_rot": _format_float(overall.get("mae_rot")),
                "mae_gripper": _format_float(overall.get("mae_gripper")),
                "grip_acc": _format_float(
                    overall.get("grip_acc", overall.get("gripper_sign_acc"))
                ),
            }
        )

    if rows:
        _print_markdown_table(rows)
    else:
        print("No valid offline eval json found.", file=sys.stderr)

    if missing:
        print("Missing files:", file=sys.stderr)
        for path in missing:
            print(f"  - {path}", file=sys.stderr)

    if args.require_all and missing:
        return 1
    if not rows:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
