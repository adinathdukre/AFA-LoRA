from __future__ import annotations

import numpy as np


def mean_by_family(logits: np.ndarray, families: list[str]) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    if len(set(families)) > 1:
        per_family = np.stack([logits[[i for i, f in enumerate(families) if f == fam]].mean(axis=0)
                               for fam in sorted(set(families))])
        return 0.5 + np.arctan(per_family.mean(axis=0)) / np.pi
    return 0.5 + np.arctan(logits.mean(axis=0)) / np.pi
