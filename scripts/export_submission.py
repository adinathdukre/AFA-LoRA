#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rare.fusion import mean_by_family
from rare.metrics.rare_metrics import evaluate
from rare.models.backbones import resolve_img_size, timm_name
from rare.models.merge import merged_run_states
from rare.utils.config import resolve_path


def round_robin(members: list[dict]) -> list[dict]:
    by_family: dict[str, list[dict]] = {}
    for m in members:
        by_family.setdefault(m["family"], []).append(m)
    ordered, i = [], 0
    while len(ordered) < len(members):
        for fam in sorted(by_family):
            if i < len(by_family[fam]):
                ordered.append(by_family[fam][i])
        i += 1
    return ordered


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--clean-oof", default="runs/clean_oof")
    ap.add_argument("--out", default="submission/model")
    ap.add_argument("--soup-lambda", type=float, default=0.0)
    ap.add_argument("--lora-alpha-scale", type=float, default=1.5)
    args = ap.parse_args()

    out_dir = resolve_path(args.out)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    members, oof = [], {}
    for run in args.runs:
        run_dir = resolve_path(run)
        for fold_dir, config, state in merged_run_states(run_dir, args.lora_alpha_scale,
                                                         args.soup_lambda):
            name = f"{run_dir.name}_{fold_dir.name}"
            family = config["model"].get("family") or run_dir.name
            data = np.load(resolve_path(args.clean_oof) / run_dir.name / fold_dir.name / "oof_clean.npz",
                           allow_pickle=True)
            logits = data["logits"].astype(np.float64)
            mu, sd = float(logits.mean()), float(max(logits.std(), 1e-8))
            oof[name] = dict(zip((str(s) for s in data["sample_id"]),
                                 zip((logits - mu) / sd, data["targets"].astype(int))))

            (out_dir / name).mkdir()
            save_file({k: v.detach().clone().contiguous() for k, v in state.items()},
                      out_dir / name / "model.safetensors")
            (out_dir / name / "meta.json").write_text(json.dumps({
                "backbone": timm_name(config["model"]["backbone"]),
                "img_size": resolve_img_size(config),
                "family": family,
                "weights": "model.safetensors",
            }, indent=2) + "\n")
            members.append({"dir": name, "temperature": 1.0 / sd, "bias": -mu / sd,
                            "weight": 1.0, "family": family})
            print(f"{name}: family {family}, mu {mu:+.3f}, sd {sd:.3f}", flush=True)

    members = round_robin(members)
    (out_dir / "ensemble.json").write_text(
        json.dumps({"fusion": "mean_by_family", "members": members}, indent=2) + "\n")

    per_run: dict[str, dict] = {}
    for m in members:
        run = m["dir"].rsplit("_fold_", 1)[0]
        per_run.setdefault(run, {"family": m["family"], "oof": {}})["oof"].update(oof[m["dir"]])
    ids = sorted(set.intersection(*(set(r["oof"]) for r in per_run.values())))
    matrix = np.stack([[r["oof"][i][0] for i in ids] for r in per_run.values()])
    targets = np.array([next(iter(per_run.values()))["oof"][i][1] for i in ids])
    fused = mean_by_family(matrix, [r["family"] for r in per_run.values()])
    print(f"pooled clean OOF, {len(ids)} images: {evaluate(targets, fused)}")
    print(f"wrote {len(members)} members to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
