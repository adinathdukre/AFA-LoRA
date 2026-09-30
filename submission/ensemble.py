from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
from safetensors.torch import load_file

cv2.setNumThreads(1)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def squash_resize(image: np.ndarray, size: int) -> np.ndarray:
    h, w = image.shape[:2]
    if (h, w) == (size, size):
        return image
    interp = cv2.INTER_AREA if (size < h or size < w) else cv2.INTER_CUBIC
    return cv2.resize(image, (size, size), interpolation=interp)


def preprocess(images: np.ndarray, size: int) -> torch.Tensor:
    out = np.empty((len(images), size, size, 3), dtype=np.float32)
    for i, img in enumerate(images):
        out[i] = squash_resize(img, size).astype(np.float32) / 255.0
    out = (out - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(out.transpose(0, 3, 1, 2)).contiguous()


def fuse(matrix: np.ndarray, families: list[str]) -> np.ndarray:
    if len(set(families)) > 1:
        per_family = np.stack([matrix[[i for i, f in enumerate(families) if f == fam]].mean(axis=0)
                               for fam in sorted(set(families))])
        return 0.5 + np.arctan(per_family.mean(axis=0)) / np.pi
    return 0.5 + np.arctan(matrix.mean(axis=0)) / np.pi


class Member(nn.Module):

    def __init__(self, backbone: str, img_size: int):
        super().__init__()
        kwargs = {"pretrained": False, "num_classes": 0}
        try:
            timm.create_model(backbone, img_size=img_size, **kwargs)
            kwargs["img_size"] = img_size
        except TypeError:
            pass
        self.backbone = timm.create_model(backbone, **kwargs)
        self.head = nn.Linear(self.backbone.num_features, 1)
        self.img_size = img_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x)).reshape(-1)


@dataclass
class MemberSpec:
    path: Path
    backbone: str
    img_size: int
    temperature: float = 1.0
    bias: float = 0.0
    family: str = ""


class RareEnsemble:

    STEP_MARGIN = 1.25
    TAIL_RESERVE_S = 20.0

    def __init__(self, specs: list[MemberSpec], device: torch.device):
        if not specs:
            raise ValueError("no ensemble members provided")
        self.specs = specs
        self.device = device
        self._start_time = time.time()

    def __len__(self) -> int:
        return len(self.specs)

    @classmethod
    def from_directory(cls, root: str | Path, device: torch.device) -> "RareEnsemble":
        root = Path(root)
        manifest = json.loads((root / "ensemble.json").read_text())
        if manifest["fusion"] != "mean_by_family":
            raise ValueError(f"unsupported fusion {manifest['fusion']!r}")
        specs = []
        for entry in manifest["members"]:
            member_dir = root / entry["dir"]
            meta = json.loads((member_dir / "meta.json").read_text())
            specs.append(MemberSpec(
                path=member_dir / meta["weights"],
                backbone=meta["backbone"],
                img_size=int(meta["img_size"]),
                temperature=float(entry["temperature"]),
                bias=float(entry["bias"]),
                family=str(entry["family"]),
            ))
        return cls(specs, device)

    def _load(self, spec: MemberSpec) -> Member:
        model = Member(spec.backbone, spec.img_size)
        missing, unexpected = model.load_state_dict(load_file(str(spec.path)), strict=False)
        if missing or unexpected:
            raise RuntimeError(f"{spec.path}: {len(missing)} missing, {len(unexpected)} unexpected keys")
        return model.to(self.device).eval()

    @torch.inference_mode()
    def _member_logits(self, model: Member, images: np.ndarray, batch_size: int) -> np.ndarray:
        out = np.empty(len(images), dtype=np.float64)
        for start in range(0, len(images), batch_size):
            batch = preprocess(images[start:start + batch_size], model.img_size).to(self.device)
            out[start:start + len(batch)] = model(batch).float().double().cpu().numpy()
        return out

    def predict(self, images: np.ndarray, batch_size: int = 32,
                deadline_s: float | None = None) -> np.ndarray:
        collected, families = [], []
        member_seconds: dict[tuple[str, int], list[float]] = {}
        for i, spec in enumerate(self.specs):
            if i > 0 and deadline_s is not None:
                remaining = deadline_s - (time.time() - self._start_time)
                seen = member_seconds.get((spec.backbone, spec.img_size)) or [
                    t for ts in member_seconds.values() for t in ts]
                if remaining < self.STEP_MARGIN * max(seen) + self.TAIL_RESERVE_S:
                    print(f"stopping at {i}/{len(self.specs)} members, {remaining:.0f}s left", flush=True)
                    break
            started = time.time()
            model = self._load(spec)
            logits = self._member_logits(model, images, batch_size)
            collected.append(spec.temperature * logits + spec.bias)
            families.append(spec.family)
            took = time.time() - started
            member_seconds.setdefault((spec.backbone, spec.img_size), []).append(took)
            print(f"member {i + 1}/{len(self.specs)} ({spec.backbone} @ {spec.img_size}) "
                  f"done in {took:.1f}s", flush=True)
            del model
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        return fuse(np.stack(collected), families)
