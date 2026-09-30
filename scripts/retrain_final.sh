#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
RUNS=(vitb_aug_strong vitb_aug_base vitb_aug_light resnet50_dino resnet50_mocov2)
read -r -a GPUS <<< "${GPUS:-0}"
NFOLDS="${NFOLDS:-5}"
mkdir -p runs
i=0
for run in "${RUNS[@]}"; do
  for fold in $(seq 0 $((NFOLDS - 1))); do
    gpu=${GPUS[$((i % ${#GPUS[@]}))]}
    i=$((i + 1))
    python scripts/train.py --config "configs/exp/${run}.yaml" --fold "$fold" --device "cuda:${gpu}" \
      > "runs/${run}_fold_${fold}.log" 2>&1 &
    if (( i % ${#GPUS[@]} == 0 )); then wait; fi
  done
done
wait
