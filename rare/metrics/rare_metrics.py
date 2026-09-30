from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

IMBALANCE_RATIO = 100
N_BOOTSTRAP = 1000
RECALL_LEVEL = 0.9


def ppv_at_recall(y_true: np.ndarray, y_score: np.ndarray, recall: float = RECALL_LEVEL) -> float:
    precisions, recalls, _ = precision_recall_curve(y_true, y_score)
    return float(np.interp(recall, recalls[::-1], precisions[::-1]))


def fpr_at_tpr(y_true: np.ndarray, y_score: np.ndarray, tpr_level: float = RECALL_LEVEL) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    reached = tpr >= tpr_level
    if reached.any():
        return float(fpr[reached].min())
    return float(np.interp(tpr_level, tpr, fpr))


def bootstrap_metrics(y_true, y_pred, n_iterations: int = N_BOOTSTRAP,
                      imbalance_ratio: int = IMBALANCE_RATIO, seed: int | None = 0) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    if not np.isfinite(y_pred).all():
        raise ValueError("y_pred contains non-finite values")
    ndbe_idx = np.where(y_true == 0)[0]
    neo_idx = np.where(y_true == 1)[0]
    if len(ndbe_idx) == 0 or len(neo_idx) == 0:
        raise ValueError("both classes must be present")

    rng = np.random.default_rng(seed)
    n_sample = max(1, int(len(ndbe_idx) / imbalance_ratio))
    boot = np.empty((n_iterations, 3), dtype=np.float64)
    for i in range(n_iterations):
        idx = np.concatenate([ndbe_idx, rng.choice(neo_idx, size=n_sample, replace=True)])
        yt, yp = y_true[idx], y_pred[idx]
        precisions, recalls, _ = precision_recall_curve(yt, yp)
        boot[i] = (
            roc_auc_score(yt, yp),
            average_precision_score(yt, yp),
            np.interp(RECALL_LEVEL, recalls[::-1], precisions[::-1]),
        )
    return {
        "score": float(np.median(boot[:, 2])),
        "ppv90_ci": (float(np.percentile(boot[:, 2], 2.5)), float(np.percentile(boot[:, 2], 97.5))),
        "auroc": float(np.median(boot[:, 0])),
        "auprc": float(np.median(boot[:, 1])),
    }


@dataclass
class MetricSummary:
    n: int = 0
    n_pos: int = 0
    auroc: float = float("nan")
    auprc: float = float("nan")
    ppv90: float = float("nan")
    fpr_at_90tpr: float = float("nan")
    score: float = float("nan")
    ppv90_ci: tuple[float, float] = (float("nan"), float("nan"))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ppv90_ci_lower"], d["ppv90_ci_upper"] = d.pop("ppv90_ci")
        return d

    def __str__(self) -> str:
        return (f"score={self.score:.4f} FPR@90TPR={self.fpr_at_90tpr:.4f} "
                f"AUROC={self.auroc:.4f} AUPRC={self.auprc:.4f} n={self.n} (pos={self.n_pos})")


def evaluate(y_true, y_pred, n_bootstrap: int = N_BOOTSTRAP, seed: int | None = 0) -> MetricSummary:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    summary = MetricSummary(
        n=len(y_true),
        n_pos=int((y_true == 1).sum()),
        auroc=float(roc_auc_score(y_true, y_pred)),
        auprc=float(average_precision_score(y_true, y_pred)),
        ppv90=ppv_at_recall(y_true, y_pred),
        fpr_at_90tpr=fpr_at_tpr(y_true, y_pred),
    )
    if n_bootstrap > 0:
        boot = bootstrap_metrics(y_true, y_pred, n_bootstrap, seed=seed)
        summary.score = boot["score"]
        summary.ppv90_ci = boot["ppv90_ci"]
    return summary
