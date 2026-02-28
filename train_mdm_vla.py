#!/usr/bin/env python3
# coding=utf-8

import sys


_FALSE_LIKE = {"0", "false", "no", "off"}


def _get_cli_value(key: str):
    prefix = f"{key}="
    for arg in sys.argv[1:]:
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return None


def _ensure_cli_value(key: str, expected: str):
    current = _get_cli_value(key)
    if current is None:
        sys.argv.append(f"{key}={expected}")
        return
    if str(current).strip().lower() != str(expected).strip().lower():
        raise ValueError(
            f"{key} must be '{expected}' for train_mdm_vla.py, got '{current}'. "
            "Use train_blockdiff_vla.py for non-MDM objectives."
        )


def _ensure_cli_false(key: str):
    current = _get_cli_value(key)
    if current is None:
        sys.argv.append(f"{key}=false")
        return
    if str(current).strip().lower() not in _FALSE_LIKE:
        raise ValueError(
            f"{key} must be false for train_mdm_vla.py, got '{current}'. "
            "Use train_blockdiff_vla.py when block diffusion is enabled."
        )


def _inject_mdm_constraints():
    _ensure_cli_value("model.framework", "mdmvla")
    _ensure_cli_false("block_diffusion.text_enabled")
    _ensure_cli_false("block_diffusion.action_enabled")


def main():
    _inject_mdm_constraints()
    from train_blockdiff_vla import main as _main

    _main()


if __name__ == "__main__":
    main()
