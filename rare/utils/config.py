from __future__ import annotations

import os
import random
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = REPO_ROOT / "configs" / "base.yaml"


def deep_update(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    with open(BASE_CONFIG) as fh:
        config = yaml.safe_load(fh)
    if path is not None:
        with open(path) as fh:
            config = deep_update(config, yaml.safe_load(fh) or {})
        if not config["output"].get("name"):
            config["output"]["name"] = Path(path).stem
    if overrides:
        config = deep_update(config, overrides)
    return config


def parse_overrides(items: list[str]) -> dict:
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        key, raw = item.split("=", 1)
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError:
            value = raw
        if isinstance(value, str) and re.fullmatch(r"[+-]?\d+(\.\d*)?[eE][+-]?\d+", value):
            value = float(value)
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = True


def resolve_amp_dtype(name: str | None) -> torch.dtype | None:
    if name in (None, "none", "off", False):
        return None
    if name == "bf16":
        if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            return torch.float16
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"unknown amp setting {name!r}")


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p)
