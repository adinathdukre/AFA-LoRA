from __future__ import annotations

import contextlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..metrics.rare_metrics import MetricSummary, evaluate
from .ema import ModelEMA


@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
    best_score: float = -float("inf")
    best_epoch: int = -1
    history: list[dict] = field(default_factory=list)


def build_optimizer(model: nn.Module, lr: float, weight_decay: float,
                    head_lr_mult: float) -> torch.optim.Optimizer:
    decay, no_decay, head = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("head."):
            head.append(param)
        elif param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    groups = [
        {"params": decay, "weight_decay": weight_decay, "lr": lr},
        {"params": no_decay, "weight_decay": 0.0, "lr": lr},
        {"params": head, "weight_decay": weight_decay, "lr": lr * head_lr_mult},
    ]
    return torch.optim.AdamW([g for g in groups if g["params"]], betas=(0.9, 0.999))


def build_scheduler(optimizer, total_steps: int, warmup_frac: float = 0.05,
                    min_lr_frac: float = 0.01):
    warmup_steps = max(1, int(total_steps * warmup_frac))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))
        return min_lr_frac + (1.0 - min_lr_frac) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def autocast(device: torch.device, amp_dtype: torch.dtype | None):
    if amp_dtype is not None and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    return contextlib.nullcontext()


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device,
            amp_dtype: torch.dtype | None = None) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_all, targets_all = [], []
    for images, targets in loader:
        with autocast(device, amp_dtype):
            logits = model(images.to(device, non_blocking=True))
        logits_all.append(logits.float().cpu().numpy().astype(np.float64))
        targets_all.append(targets.numpy())
    return np.concatenate(logits_all), np.concatenate(targets_all)


class Trainer:

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        output_dir: str | Path,
        epochs: int,
        lr: float,
        weight_decay: float,
        head_lr_mult: float,
        grad_clip: float,
        accumulation_steps: int,
        ema_decay: float,
        amp_dtype: torch.dtype | None,
        early_stop_patience: int,
        save_last_k: int,
        log_every: int = 20,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.epochs = epochs
        self.grad_clip = grad_clip
        self.accumulation_steps = max(1, accumulation_steps)
        self.amp_dtype = amp_dtype
        self.early_stop_patience = early_stop_patience
        self.save_last_k = save_last_k
        self.log_every = log_every
        self.scaler = torch.amp.GradScaler(
            device.type, enabled=(amp_dtype is torch.float16 and device.type == "cuda"))
        self.optimizer = build_optimizer(self.model, lr, weight_decay, head_lr_mult)
        steps_per_epoch = max(1, len(train_loader) // self.accumulation_steps)
        self.scheduler = build_scheduler(self.optimizer, steps_per_epoch * epochs)
        self.ema = ModelEMA(self.model, decay=ema_decay)
        self.state = TrainState()

    def train_epoch(self) -> dict:
        self.model.train()
        total, n_batches = 0.0, 0
        start = time.time()
        self.optimizer.zero_grad(set_to_none=True)
        for step, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            with autocast(self.device, self.amp_dtype):
                logits = self.model(images)
                loss = F.binary_cross_entropy_with_logits(logits.reshape(-1), targets.reshape(-1))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {self.state.epoch} step {step}")
            self.scaler.scale(loss / self.accumulation_steps).backward()
            if (step + 1) % self.accumulation_steps == 0:
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.state.global_step += 1
                self.ema.update(self.model)
            total += float(loss.detach())
            n_batches += 1
            if self.log_every and step % self.log_every == 0:
                print(f"  ep{self.state.epoch:02d} step {step:4d}/{len(self.train_loader)} "
                      f"loss={float(loss.detach()):.4f} lr={self.optimizer.param_groups[0]['lr']:.2e}",
                      flush=True)
        return {"loss": total / max(n_batches, 1), "epoch_time_s": time.time() - start}

    def validate(self) -> tuple[MetricSummary, np.ndarray, np.ndarray]:
        logits, targets = predict(self.ema.module, self.val_loader, self.device)
        return evaluate(targets.astype(int), logits, n_bootstrap=0), logits, targets

    def fit(self) -> TrainState:
        ckpt_paths: list[Path] = []
        stale = 0
        for epoch in range(self.epochs):
            self.state.epoch = epoch
            train_stats = self.train_epoch()
            summary, logits, targets = self.validate()
            score = 0.5 * (summary.auroc + summary.auprc)
            self.state.history.append({"epoch": epoch, "selection_score": score,
                                       **{f"train_{k}": v for k, v in train_stats.items()},
                                       **summary.to_dict()})
            print(f"[epoch {epoch:02d}] train_loss={train_stats['loss']:.4f} | {summary} | sel={score:.4f}",
                  flush=True)

            ckpt_path = self.output_dir / f"epoch_{epoch:03d}.pt"
            self.save(ckpt_path, summary)
            ckpt_paths.append(ckpt_path)
            while len(ckpt_paths) > self.save_last_k:
                ckpt_paths.pop(0).unlink(missing_ok=True)

            if score > self.state.best_score:
                self.state.best_score = score
                self.state.best_epoch = epoch
                self.save(self.output_dir / "best.pt", summary)
                np.savez(self.output_dir / "val_predictions.npz", logits=logits, targets=targets,
                         epoch=epoch)
                stale = 0
            else:
                stale += 1
                if stale >= self.early_stop_patience:
                    print(f"early stop at epoch {epoch} (best {self.state.best_epoch})")
                    break
            with open(self.output_dir / "history.json", "w") as fh:
                json.dump(self.state.history, fh, indent=2)
        print(f"best epoch {self.state.best_epoch} with selection score {self.state.best_score:.4f}")
        return self.state

    def save(self, path: str | Path, summary: MetricSummary | None = None) -> None:
        payload = {
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "epoch": self.state.epoch,
            "global_step": self.state.global_step,
        }
        if summary is not None:
            payload["metrics"] = summary.to_dict()
        torch.save(payload, str(path))
