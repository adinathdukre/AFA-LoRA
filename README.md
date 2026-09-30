# AFA-LoRA

Amplified fold-averaged low-rank adaptation with family-balanced logit fusion, for frame-level
detection of neoplasia in Barrett's oesophagus. Submission of Team GenMI to the RARE 2026
challenge (MICCAI 2026).

Model weights: [huggingface.co/adidukre/AFA-LoRA](https://huggingface.co/adidukre/AFA-LoRA)

## Method

The model is an ensemble of 25 binary frame classifiers, each a timm encoder with a linear head
on the pooled feature vector. All encoders are initialised from the public GastroNet-5M
self-supervised checkpoints.

| Run               | Encoder                        | Input   | Adaptation            | Family            |
|-------------------|--------------------------------|---------|-----------------------|-------------------|
| `vitb_aug_strong` | DINOv2 ViT-B/14, 4 registers   | 336x336 | LoRA r16, alpha 16    | `dinov2_vitb`     |
| `vitb_aug_base`   | DINOv2 ViT-B/14, 4 registers   | 336x336 | LoRA r16, alpha 16    | `dinov2_vitb`     |
| `vitb_aug_light`  | DINOv2 ViT-B/14, 4 registers   | 336x336 | LoRA r16, alpha 16    | `dinov2_vitb`     |
| `resnet50_dino`   | ResNet-50 (DINO)               | 384x384 | full fine-tune        | `resnet50_dino`   |
| `resnet50_mocov2` | ResNet-50 (MoCo v2)            | 384x384 | full fine-tune        | `resnet50_mocov2` |

Each run is trained on five grouped, label-stratified cross-validation folds, and every fold is
one ensemble member. The three ViT runs differ only in augmentation strength.

Two transforms are applied to the LoRA adapters when they are merged into the weights, at no
inference cost:

1. **Amplification.** The merged low-rank delta is scaled by 1.5.
2. **Fold averaging.** Each run's low-rank delta is replaced by its mean over the five folds.
   Normalisation layers and the head stay fold-specific.

At inference each member's logit `z` is standardised as `(z - mu) / sd`, with `mu` and `sd`
measured on clean out-of-fold predictions. Standardised logits are averaged within each family,
then across the three families, and mapped to (0, 1) by `0.5 + arctan(x) / pi`. Members are
stored round-robin across families, so if the container stops early to meet the time limit,
every family still contributes.

## Installation

    git clone https://github.com/adinathdukre/AFA-LoRA
    cd AFA-LoRA
    pip install -r requirements.txt

## Inference with the released weights

    pip install huggingface_hub
    huggingface-cli download adidukre/AFA-LoRA --local-dir /path/to/AFA-LoRA-weights

```python
import sys

import numpy as np
import torch

sys.path.insert(0, "submission")
from ensemble import RareEnsemble

ensemble = RareEnsemble.from_directory("/path/to/AFA-LoRA-weights", torch.device("cuda"))
images = np.zeros((4, 512, 640, 3), dtype=np.uint8)
scores = ensemble.predict(images, batch_size=32)
```

`images` is a stack of uint8 RGB frames of shape (N, H, W, 3). Frames are resized to each member's input size without cropping. The output is a ranking
score in (0, 1), not a calibrated probability.

### Grand Challenge container

    cd submission
    MODEL_DIR=/path/to/AFA-LoRA-weights INPUT_DIR=/path/to/test/input ./do_test_run.sh
    MODEL_DIR=/path/to/AFA-LoRA-weights OUT_DIR=/path/to/artifacts ./do_save.sh

`do_save.sh` writes the image and the weights as two tarballs. The weights are mounted at
`/opt/ml/model`. The container reads `RARE_TIME_BUDGET` (seconds, default 1100) and
`RARE_BATCH_SIZE` (default 32).

## Training from scratch

The RARE training release is expected as:

    /path/to/RARE/labeled/<center>/{ndbe,neo}/*.png

Set `data.root` in `configs/base.yaml` to that directory. Download the GastroNet-5M checkpoints
`dinov2.pth`, `RN50_GastroNet-5M_DINOv1.pth` and `RN50_GastroNet-5M_MOCOv2.pth` into one
directory and point `GASTRONET_DIR` at it:

    export GASTRONET_DIR=/path/to/gastronet

**1. Splits.** This clusters near-duplicate frames by perceptual hash and writes
`data/manifest.csv` with five grouped folds:

    python scripts/make_splits.py

**2. Training.** Train one fold of one run:

    python scripts/train.py --config configs/exp/vitb_aug_strong.yaml --fold 0 --device cuda:0

Or train all 25 members, one process per listed GPU:

    GPUS="0 1 2 3" bash scripts/retrain_final.sh

Each fold writes `runs/<run>/fold_<k>/`. The member weights are `swa.pt`, the average of the
last three epochs.

**3. Clean out-of-fold scores.** Score each fold's validation images without augmentation,
using the merged weights (alpha scale 1.5, fold averaging):

    python scripts/score_folds_clean.py \
        --runs runs/vitb_aug_strong runs/vitb_aug_base runs/vitb_aug_light \
               runs/resnet50_dino runs/resnet50_mocov2

**4. Export.** Merge the adapters, write one `model.safetensors` per member and write
`ensemble.json`, which holds the standardisation constants and the member order:

    python scripts/export_submission.py \
        --runs runs/vitb_aug_strong runs/vitb_aug_base runs/vitb_aug_light \
               runs/resnet50_dino runs/resnet50_mocov2 \
        --out submission/model

The order of `--runs` sets the order of members within a family.

## Tests

    python -m pytest tests -q

## Repository layout

    configs/     base configuration and one file per training run
    rare/        data, augmentation, model, training loop, metric, LoRA merge
    scripts/     splits, training, clean scoring, export
    submission/  Grand Challenge inference container
    tests/       metric, container parity, LoRA merge and deadline tests

## Licence

MIT. See `LICENSE`.
