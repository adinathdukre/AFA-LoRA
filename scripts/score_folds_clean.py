#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rare.data.dataset import RareDataset
from rare.data.transforms import build_eval_transform
from rare.engine.trainer import predict
from rare.models.backbones import build_model, resolve_img_size
from rare.models.merge import merged_run_states
from rare.utils.config import resolve_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="runs/clean_oof")
    ap.add_argument("--soup-lambda", type=float, default=0.0)
    ap.add_argument("--lora-alpha-scale", type=float, default=1.5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    manifests: dict[str, pd.DataFrame] = {}
    for run in args.runs:
        run_dir = resolve_path(run)
        for fold_dir, config, state in merged_run_states(run_dir, args.lora_alpha_scale,
                                                         args.soup_lambda):
            fold = int(fold_dir.name.split("_")[-1])
            key = config["data"]["manifest"]
            if key not in manifests:
                manifests[key] = pd.read_csv(resolve_path(key))
            rows = manifests[key]
            rows = rows[rows["fold"] == fold].reset_index(drop=True)

            plain = {**config, "model": {**config["model"], "lora_rank": 0}}
            model = build_model(plain, pretrained=False)
            model.load_state_dict(state, strict=True)
            model = model.to(device)
            loader = DataLoader(
                RareDataset(rows, build_eval_transform(size=resolve_img_size(config))),
                batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                pin_memory=True)
            logits, targets = predict(model, loader, device)
            del model

            dest = resolve_path(args.out) / run_dir.name / fold_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            np.savez(dest / "oof_clean.npz", logits=logits, targets=targets,
                     sample_id=rows["sample_id"].to_numpy(), center=rows["center"].to_numpy())
            print(f"{run_dir.name}/{fold_dir.name}: n={len(logits)} mean {logits.mean():+.3f} "
                  f"sd {logits.std():.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
