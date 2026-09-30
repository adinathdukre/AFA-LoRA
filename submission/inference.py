import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TORCH_HOME", "/opt/app/resources")

import json
import sys
import time
from glob import glob
from pathlib import Path

import numpy as np
import SimpleITK
import torch

from ensemble import RareEnsemble

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
MODEL_PATH = Path("/opt/ml/model")
IMAGE_SUBDIR = "images/stacked-barretts-esophagus-endoscopy"
OUTPUT_NAME = "stacked-neoplastic-lesion-likelihoods.json"

BATCH_SIZE = int(os.environ.get("RARE_BATCH_SIZE", "32"))
TIME_BUDGET_S = float(os.environ.get("RARE_TIME_BUDGET", "1100"))


def load_input_stack() -> np.ndarray:
    location = INPUT_PATH / IMAGE_SUBDIR
    files = sorted(glob(str(location / "*.tif")) + glob(str(location / "*.tiff"))
                   + glob(str(location / "*.mha")))
    if not files:
        raise FileNotFoundError(f"no stacked image found under {location}")
    array = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(files[0]))
    if array.ndim == 3 and array.shape[-1] in (3, 4):
        array = array[None, ...]
    elif array.ndim == 3:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.ndim != 4:
        raise ValueError(f"unexpected input shape {array.shape}")
    array = array[..., :3]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        torch.zeros(1, device="cuda").sum().item()
        return torch.device("cuda")
    except Exception as exc:
        print(f"CUDA unusable, falling back to CPU: {exc}", flush=True)
        return torch.device("cpu")


def run() -> int:
    start = time.time()
    images = load_input_stack()
    print(f"loaded {len(images)} images of shape {images.shape[1:]}")

    ensemble = RareEnsemble.from_directory(MODEL_PATH, device=resolve_device())
    scores = ensemble.predict(images, batch_size=BATCH_SIZE,
                              deadline_s=TIME_BUDGET_S - (time.time() - start))
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(scores) != len(images) or not np.isfinite(scores).all():
        raise ValueError("invalid scores produced")

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH / OUTPUT_NAME, "w") as fh:
        json.dump([float(s) for s in scores], fh, indent=2)
    print(f"wrote {len(scores)} likelihoods in {time.time() - start:.1f}s, "
          f"range [{scores.min():.4f}, {scores.max():.4f}]")
    return 0


if __name__ == "__main__":
    sys.exit(run())
