from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rare.metrics.rare_metrics import bootstrap_metrics, evaluate, ppv_at_recall
from rare.utils.config import load_config, parse_overrides


def reference_bootstrap(y_true, y_pred, n_iterations=1000, imbalance_ratio=100, seed=0):
    from sklearn.metrics import precision_recall_curve

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ndbe = np.where(y_true == 0)[0]
    neo = np.where(y_true == 1)[0]
    rng = np.random.default_rng(seed)
    n_sample = max(1, int(len(ndbe) / imbalance_ratio))
    out = []
    for _ in range(n_iterations):
        idx = np.concatenate([ndbe, rng.choice(neo, size=n_sample, replace=True)])
        precisions, recalls, _ = precision_recall_curve(y_true[idx], y_pred[idx])
        out.append(np.interp(0.9, recalls[::-1], precisions[::-1]))
    return float(np.median(out))


@pytest.fixture
def scores():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(1500), np.ones(120)].astype(int)
    s = np.r_[rng.normal(0, 1, 1500), rng.normal(2.2, 1, 120)]
    return y, s


def test_matches_organizer_reference(scores):
    y, s = scores
    assert bootstrap_metrics(y, s, seed=0)["score"] == pytest.approx(reference_bootstrap(y, s), abs=1e-12)


def test_monotone_transforms_are_no_ops(scores):
    y, s = scores
    base = ppv_at_recall(y, s)
    for transform in (lambda x: 3.0 * x + 7.0, lambda x: 1 / (1 + np.exp(-x)), lambda x: np.exp(x / 5)):
        assert ppv_at_recall(y, transform(s)) == pytest.approx(base, abs=1e-9)


def test_rejects_non_finite(scores):
    y, s = scores
    s = s.copy()
    s[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        bootstrap_metrics(y, s)


def test_evaluate_panel(scores):
    y, s = scores
    summary = evaluate(y, s, n_bootstrap=100)
    assert 0 < summary.auroc < 1
    assert 0 <= summary.fpr_at_90tpr <= 1
    assert summary.n_pos == 120


def test_scientific_notation_override_is_float():
    for text in ("3e-4", "3.0e-4", "1e-5", "0.0003"):
        assert isinstance(parse_overrides([f"train.lr={text}"])["train"]["lr"], float)


def test_every_run_config_loads():
    configs = sorted((Path(__file__).resolve().parents[1] / "configs" / "exp").glob("*.yaml"))
    assert len(configs) == 5
    for path in configs:
        config = load_config(path)
        assert config["output"]["name"] == path.stem
        assert config["model"]["family"]
