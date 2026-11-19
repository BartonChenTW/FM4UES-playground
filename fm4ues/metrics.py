"""Energy-domain evaluation metrics for forecasting, building stocks and land use."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import f1_score


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    yt, yp = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(yp - yt)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Mean absolute percentage error (%)."""
    yt, yp = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(yt) > eps, np.abs(yt), np.nan)
    return float(100.0 * np.nanmean(np.abs(yp - yt) / denom))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    yt, yp = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((yp - yt) ** 2)))


def peak_hour_mae(y_true: np.ndarray, y_pred: np.ndarray, top_pct: float = 5.0) -> float:
    """MAE restricted to the highest `top_pct`% actual values (peak-demand risk)."""
    yt, yp = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    threshold = np.percentile(yt, 100.0 - top_pct)
    mask = yt >= threshold
    return float(np.mean(np.abs(yp[mask] - yt[mask])))


def ramp_rate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAE of step-to-step changes (ramps) along the last axis."""
    yt, yp = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(np.diff(yp, axis=-1) - np.diff(yt, axis=-1))))


def interval_coverage(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Empirical prediction-interval coverage (%)."""
    yt = np.asarray(y_true, dtype=float)
    lo_a, hi_a = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    return float(100.0 * np.mean((yt >= lo_a) & (yt <= hi_a)))


def pinball_loss(y_true: np.ndarray, y_q: np.ndarray, q: float) -> float:
    """Mean pinball (quantile) loss for quantile level `q` in (0, 1)."""
    yt, yq = np.asarray(y_true, dtype=float), np.asarray(y_q, dtype=float)
    diff = yt - yq
    return float(np.mean(np.maximum(q * diff, (q - 1.0) * diff)))


def winkler_score(
    y_true: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    alpha: float = 0.2,
) -> float:
    """Mean Winkler interval score (width plus overshoot penalty; lower is better)."""
    yt = np.asarray(y_true, dtype=float)
    lo_a, hi_a = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    width = hi_a - lo_a
    under = np.maximum(lo_a - yt, 0.0)
    over = np.maximum(yt - hi_a, 0.0)
    return float(np.mean(width + (2.0 / alpha) * (under + over)))


def portfolio_bias_pct(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """Signed percentage bias of weighted total across all items (e.g., total stock energy)."""
    yt, yp = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    w = np.ones_like(yt) if weights is None else np.asarray(weights, dtype=float)
    total_true = np.sum(w * yt)
    total_pred = np.sum(w * yp)
    return float(100.0 * (total_pred - total_true) / total_true)


def class_f1_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
) -> dict[str, float]:
    """Per-class F1 scores plus macro F1 for classification tasks."""
    labels = list(range(len(class_names)))
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    out = {name: float(score) for name, score in zip(class_names, per_class)}
    out["macro_f1"] = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    return out


def crps_from_quantiles(
    y_true: np.ndarray,
    quantile_preds: np.ndarray,
    quantiles: Sequence[float],
) -> float:
    """Discrete CRPS approximation: 2 * mean pinball loss across `quantiles`."""
    yt = np.asarray(y_true, dtype=float)
    qp = np.asarray(quantile_preds, dtype=float)
    if qp.shape[-1] != len(quantiles):
        raise ValueError(f"Expected last axis {len(quantiles)}, got {qp.shape[-1]}")
    losses = [pinball_loss(yt, qp[..., i], float(q)) for i, q in enumerate(quantiles)]
    return float(2.0 * np.mean(losses))


def stylized_imbalance_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    up_price: float = 80.0,
    down_price: float = 40.0,
) -> float:
    """Asymmetric hourly balancing cost (EUR) for under- and over-forecasting MW."""
    yt, yp = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    shortfall = np.maximum(yt - yp, 0.0)
    surplus = np.maximum(yp - yt, 0.0)
    return float(np.sum(up_price * shortfall + down_price * surplus))
