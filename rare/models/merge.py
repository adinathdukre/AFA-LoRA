from __future__ import annotations

import json
from pathlib import Path

import torch

from .delta_soup import mean_raw_delta, merge_lora_souped


def fold_dirs(run_dir: Path) -> list[Path]:
    dirs = sorted(d for d in Path(run_dir).glob("fold_*") if (d / "config.json").exists())
    if not dirs:
        raise FileNotFoundError(f"no fold_*/config.json under {run_dir}")
    return dirs


def load_fold_state(fold_dir: Path) -> dict:
    path = fold_dir / "swa.pt" if (fold_dir / "swa.pt").exists() else fold_dir / "best.pt"
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    return ckpt.get("ema") or ckpt["model"]


def merged_run_states(run_dir: Path, alpha_scale: float, soup_lambda: float):
    dirs = fold_dirs(run_dir)
    configs = [json.loads((d / "config.json").read_text()) for d in dirs]
    rank = int(configs[0]["model"].get("lora_rank") or 0)
    mean_raw = None
    if rank > 0 and soup_lambda != 1.0:
        mean_raw = mean_raw_delta([load_fold_state(d) for d in dirs])
    for d, config in zip(dirs, configs):
        state = load_fold_state(d)
        if rank > 0:
            scale = float(config["model"]["lora_alpha"]) * alpha_scale / rank
            state = merge_lora_souped(state, scale, soup_lambda, mean_raw)
        yield d, config, state
