#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rare.data.splits import add_grouped_kfold, build_groups, build_manifest, summarize
from rare.utils.config import load_config, resolve_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--root", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--hamming-threshold", type=int, default=24)
    args = parser.parse_args()

    config = load_config(args.config)
    root = Path(args.root or config["data"]["root"])
    out_path = resolve_path(args.out or config["data"]["manifest"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_manifest(root)
    print(f"{len(df)} images, {int((df['target'] == 1).sum())} positive")
    df["group"] = build_groups(df, hamming_threshold=args.hamming_threshold)
    df = add_grouped_kfold(df, n_splits=config["split"]["n_folds"], seed=config["split"]["seed"])
    df.to_csv(out_path, index=False)
    print(summarize(df).to_string(index=False))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
