from __future__ import annotations

import math
import os
from pathlib import Path

import timm
import torch
import torch.nn as nn

GASTRONET_DIR = Path(os.environ.get("GASTRONET_DIR", "weights"))

BACKBONE_PRESETS: dict[str, dict] = {
    "gastronet_rn50_dino": {
        "timm_name": "resnet50", "size": 384,
        "weights": "RN50_GastroNet-5M_DINOv1.pth"},
    "gastronet_rn50_mocov2": {
        "timm_name": "resnet50", "size": 384,
        "weights": "RN50_GastroNet-5M_MOCOv2.pth"},
    "gastronet_dinov2_vitb": {
        "timm_name": "vit_base_patch14_reg4_dinov2.lvd142m", "size": 336,
        "weights": "dinov2.pth"},
}


def timm_name(preset: str) -> str:
    return BACKBONE_PRESETS[preset]["timm_name"]


def resolve_img_size(config: dict) -> int:
    model_cfg = config["model"]
    return int(model_cfg.get("img_size") or BACKBONE_PRESETS[model_cfg["backbone"]]["size"])


def _adapt_dinov2(state: dict) -> dict:
    out = {k: v for k, v in state.items() if k != "mask_token"}
    if "register_tokens" in out:
        out["reg_token"] = out.pop("register_tokens")
    pos = out.get("pos_embed")
    if pos is not None and pos.shape[1] == 577:
        out["pos_embed"] = pos[:, 1:, :]
    return out


def load_pretrained_state(path: str | Path) -> dict:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    is_dinov2 = isinstance(ckpt, dict) and "teacher" in ckpt
    for key in ("teacher", "student", "model_state_dict", "state_dict", "model"):
        if isinstance(ckpt, dict) and isinstance(ckpt.get(key), dict):
            ckpt = ckpt[key]
            break
    state = {}
    for k, v in ckpt.items():
        for p in ("module.", "backbone.", "encoder."):
            if k.startswith(p):
                k = k[len(p):]
        state[k] = v
    return _adapt_dinov2(state) if is_dinov2 else state


class LoRALinear(nn.Module):

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.scaling = alpha / rank
        self.lora_a = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * (x @ self.lora_a.t() @ self.lora_b.t())


def apply_lora(model: nn.Module, rank: int, alpha: float,
               target_keys: tuple[str, ...] = ("attn.qkv", "attn.proj")) -> int:
    replaced = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and any(k in full for k in target_keys):
                setattr(module, child_name, LoRALinear(child, rank, alpha))
                replaced += 1
    return replaced


class RareClassifier(nn.Module):

    def __init__(
        self,
        backbone_name: str = "gastronet_dinov2_vitb",
        pretrained: bool = True,
        img_size: int | None = None,
        drop_path_rate: float = 0.1,
        lora_rank: int = 0,
        lora_alpha: float = 16.0,
        prior_pos: float | None = None,
    ):
        super().__init__()
        preset = BACKBONE_PRESETS[backbone_name]
        name = preset["timm_name"]
        self.img_size = img_size or preset["size"]

        kwargs: dict = {"pretrained": False, "num_classes": 0, "drop_rate": 0.0}
        for key, value in (("drop_path_rate", drop_path_rate), ("img_size", self.img_size)):
            try:
                timm.create_model(name, pretrained=False, num_classes=0, **{key: value})
                kwargs[key] = value
            except TypeError:
                pass
        self.backbone = timm.create_model(name, **kwargs)

        if pretrained:
            path = GASTRONET_DIR / preset["weights"]
            state = load_pretrained_state(path)
            missing, unexpected = self.backbone.load_state_dict(state, strict=False)
            loaded = len(state) - len(unexpected)
            if loaded == 0:
                raise RuntimeError(f"no weights from {path} matched {name}")
            print(f"[{backbone_name}] loaded {loaded}/{len(state)} tensors from {path.name} "
                  f"({len(missing)} missing, {len(unexpected)} unexpected)")

        self.head = nn.Linear(self.backbone.num_features, 1)
        nn.init.zeros_(self.head.weight)
        if prior_pos is not None and 0 < prior_pos < 1:
            nn.init.constant_(self.head.bias, -math.log((1 - prior_pos) / prior_pos))
        else:
            nn.init.zeros_(self.head.bias)

        if lora_rank > 0:
            n = apply_lora(self.backbone, lora_rank, lora_alpha)
            if n == 0:
                raise RuntimeError(f"lora_rank={lora_rank} but {name} has no attention layers")
            for pname, param in self.backbone.named_parameters():
                param.requires_grad_(any(k in pname for k in ("lora_", "norm", "bn", "head")))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x)).reshape(-1)

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(config: dict, pretrained: bool = True) -> RareClassifier:
    m = config["model"]
    return RareClassifier(
        backbone_name=m["backbone"],
        pretrained=pretrained,
        img_size=m.get("img_size"),
        drop_path_rate=m.get("drop_path_rate", 0.1),
        lora_rank=int(m.get("lora_rank") or 0),
        lora_alpha=float(m.get("lora_alpha", 16.0)),
        prior_pos=m.get("prior_pos"),
    )
