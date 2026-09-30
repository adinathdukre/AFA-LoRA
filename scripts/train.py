#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rare.data.dataset import RareDataset, make_balanced_sampler
from rare.data.transforms import (
    build_eval_transform, build_train_transform, compute_reference_cdfs, seed_worker,
)
from rare.engine.trainer import Trainer, predict
from rare.metrics.rare_metrics import evaluate
from rare.models.backbones import build_model, resolve_img_size
from rare.models.weight_averaging import average_checkpoints, last_k_checkpoints, recompute_bn
from rare.utils.config import (
    load_config, parse_overrides, resolve_amp_dtype, resolve_path, seed_everything,
)


def save_swa(model, out_dir: Path, train_loader, val_loader, val_df, device, config) -> None:
    paths = last_k_checkpoints(out_dir, int(config["train"]["swa_last_k"]))
    if len(paths) < 2:
        return
    model.load_state_dict(average_checkpoints(paths), strict=True)
    bn_loader = DataLoader(train_loader.dataset, batch_size=train_loader.batch_size, shuffle=True,
                           num_workers=max(2, config["data"]["num_workers"] // 2), drop_last=True)
    recompute_bn(model, bn_loader, device, max_batches=60)
    logits, targets = predict(model, val_loader, device)
    torch.save({"model": model.state_dict(), "swa_sources": [p.name for p in paths]},
               out_dir / "swa.pt")
    np.savez(out_dir / "oof_swa.npz", logits=logits, targets=targets,
             sample_id=val_df["sample_id"].to_numpy(), center=val_df["center"].to_numpy())
    print(f"SWA over {len(paths)} epochs: {evaluate(targets.astype(int), logits, n_bootstrap=0)}")


def run_fold(config: dict, fold: int, device: torch.device) -> None:
    seed_everything(config["train"]["seed"] + fold)
    df = pd.read_csv(resolve_path(config["data"]["manifest"]))
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    val_df = df[df["fold"] == fold].reset_index(drop=True)
    train_targets = train_df["target"].to_numpy()
    print(f"fold {fold}: train {len(train_df)} ({int(train_targets.sum())} pos) | "
          f"val {len(val_df)} ({int(val_df['target'].sum())} pos)")

    size = resolve_img_size(config)
    aug = config["augmentation"]
    refs = (compute_reference_cdfs(train_df["image_path"].tolist(), max_refs=64)
            if aug["histogram_prob"] > 0 else None)
    train_tf = build_train_transform(
        size=size,
        processor_strength=aug["processor_strength"],
        processor_prob=aug["processor_prob"],
        specular_prob=aug["specular_prob"],
        fov_prob=aug["fov_prob"],
        geometric_prob=aug["geometric_prob"],
        blur_prob=aug["blur_prob"],
        compression_prob=aug["compression_prob"],
        histogram_prob=aug["histogram_prob"],
        reference_cdfs=refs,
    )
    val_tf = build_eval_transform(size=size, processor_strength=aug["val_processor_strength"])

    workers = config["data"]["num_workers"]
    loader_kwargs = dict(num_workers=workers, pin_memory=True, worker_init_fn=seed_worker,
                         persistent_workers=config["data"]["persistent_workers"] and workers > 0)
    if workers > 0:
        loader_kwargs["prefetch_factor"] = config["data"]["prefetch_factor"]
    train_loader = DataLoader(
        RareDataset(train_df, train_tf),
        batch_size=config["train"]["batch_size"],
        sampler=make_balanced_sampler(train_targets, config["train"]["positive_fraction"]),
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(RareDataset(val_df, val_tf), batch_size=config["train"]["batch_size"] * 2,
                            shuffle=False, **loader_kwargs)

    model = build_model(config)
    print(f"{config['model']['backbone']}: {model.trainable_parameters() / 1e6:.1f}M trainable params")

    out_dir = resolve_path(config["output"]["dir"]) / config["output"]["name"] / f"fold_{fold}"
    t = config["train"]
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=out_dir,
        epochs=t["epochs"],
        lr=t["lr"],
        weight_decay=t["weight_decay"],
        head_lr_mult=t["head_lr_mult"],
        grad_clip=t["grad_clip"],
        accumulation_steps=t["accumulation_steps"],
        ema_decay=t["ema_decay"],
        amp_dtype=resolve_amp_dtype(t["amp"]),
        early_stop_patience=t["early_stop_patience"],
        save_last_k=t["save_last_k"],
    )
    config = {**config, "split": {**config["split"], "fold": fold}}
    with open(out_dir / "config.json", "w") as fh:
        json.dump(config, fh, indent=2, default=str)

    trainer.fit()
    preds = np.load(out_dir / "val_predictions.npz")
    print(f"fold {fold} best epoch: {evaluate(preds['targets'].astype(int), preds['logits'])}")
    np.savez(out_dir / "oof.npz", logits=preds["logits"], targets=preds["targets"],
             sample_id=val_df["sample_id"].to_numpy(), center=val_df["center"].to_numpy())
    save_swa(model, out_dir, train_loader, val_loader, val_df, device, config)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--all-folds", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = parser.parse_args()

    config = load_config(args.config, parse_overrides(args.overrides))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    folds = (list(range(config["split"]["n_folds"])) if args.all_folds
             else [args.fold if args.fold is not None else config["split"]["fold"]])
    for fold in folds:
        run_fold(config, fold, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
