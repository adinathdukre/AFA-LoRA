from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def load_state(path: str | Path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if isinstance(ckpt.get(key), dict):
                return ckpt[key]
    return ckpt


def average_state_dicts(states: list[dict]) -> dict:
    if not states:
        raise ValueError("no checkpoints to average")
    if len(states) == 1:
        return dict(states[0])
    keys = set(states[0])
    for i, s in enumerate(states[1:], 1):
        if set(s) != keys:
            raise ValueError(f"checkpoint {i} has a different parameter set")
    out: dict[str, torch.Tensor] = {}
    for k, first in states[0].items():
        if not torch.is_floating_point(first):
            out[k] = first.clone()
            continue
        acc = torch.zeros_like(first, dtype=torch.float64)
        for s in states:
            acc += s[k].to(torch.float64)
        out[k] = (acc / len(states)).to(first.dtype)
    return out


def average_checkpoints(paths: list[str | Path]) -> dict:
    return average_state_dicts([load_state(p) for p in paths])


def last_k_checkpoints(run_dir: str | Path, k: int = 3) -> list[Path]:
    return sorted(Path(run_dir).glob("epoch_*.pt"))[-k:]


@torch.no_grad()
def recompute_bn(model: nn.Module, loader, device: torch.device, max_batches: int = 100) -> nn.Module:
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return model
    saved = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    model.train()
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        model(x.to(device, non_blocking=True))
    for m, mom in zip(bns, saved):
        m.momentum = mom
    model.eval()
    return model
