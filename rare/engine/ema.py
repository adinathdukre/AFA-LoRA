from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn


class ModelEMA:

    def __init__(self, model: nn.Module, decay: float = 0.999, warmup_steps: int = 100):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.steps = 0
        self.module = deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.steps += 1
        d = min(self.decay, (1 + self.steps) / (10 + self.steps)) if self.steps <= self.warmup_steps else self.decay
        ema_params = dict(self.module.named_parameters())
        for name, param in model.named_parameters():
            ema_params[name].mul_(d).add_(param.detach(), alpha=1.0 - d)
        ema_buffers = dict(self.module.named_buffers())
        for name, buf in model.named_buffers():
            if buf.dtype.is_floating_point:
                ema_buffers[name].mul_(d).add_(buf.detach(), alpha=1.0 - d)
            else:
                ema_buffers[name].copy_(buf)

    def state_dict(self) -> dict:
        return self.module.state_dict()
