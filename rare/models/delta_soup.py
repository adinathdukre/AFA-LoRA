from __future__ import annotations

import torch


def lora_prefixes(state: dict) -> set[str]:
    return {k[: -len(".lora_a")] for k in state if k.endswith(".lora_a")}


def raw_delta(state: dict, prefix: str) -> torch.Tensor:
    a = state[f"{prefix}.lora_a"].to(torch.float32)
    b = state[f"{prefix}.lora_b"].to(torch.float32)
    return b @ a


def mean_raw_delta(states: list[dict]) -> dict[str, torch.Tensor]:
    if not states:
        raise ValueError("no states to average")
    prefixes = lora_prefixes(states[0])
    if not prefixes:
        raise ValueError("no LoRA adapters found in the first state")
    for n, st in enumerate(states[1:], start=1):
        if lora_prefixes(st) != prefixes:
            raise ValueError(f"state {n} adapts a different module set")
        for p in prefixes:
            k = f"{p}.base.weight"
            if not torch.equal(states[0][k], st[k]):
                raise ValueError(f"state {n} has a different base weight at {k}")
    return {p: torch.stack([raw_delta(st, p) for st in states]).mean(0) for p in prefixes}


def merge_lora_souped(state: dict, alpha_scale: float, lam: float,
                      mean_raw: dict[str, torch.Tensor] | None) -> dict:
    prefixes = lora_prefixes(state)
    if not prefixes:
        return dict(state)
    if lam != 1.0 and mean_raw is None:
        raise ValueError("lam < 1 requires the run mean delta")
    out: dict = {}
    for k, v in state.items():
        if k.endswith(".lora_a") or k.endswith(".lora_b"):
            continue
        if ".base." in k:
            prefix, suffix = k.split(".base.", 1)
            if suffix == "weight" and prefix in prefixes:
                r = raw_delta(state, prefix)
                if lam != 1.0:
                    r = lam * r + (1.0 - lam) * mean_raw[prefix]
                v = v.to(torch.float32) + alpha_scale * r
            out[f"{prefix}.{suffix}"] = v
        else:
            out[k] = v
    return out
