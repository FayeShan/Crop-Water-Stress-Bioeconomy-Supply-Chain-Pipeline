"""
Cross-validation, evaluation metrics, and calibration utilities.
"""

import logging
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, median_absolute_error, r2_score,
)


def temporal_cv_splits(
    dates: pd.Series,
    n_splits: Optional[int] = None,
    test_months: int = 24,
    min_train_months: int = 12,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate expanding-window temporal CV splits."""
    unique_dates = sorted(dates.unique())
    total = len(unique_dates)

    if n_splits is None:
        n_splits = max(1, (total - min_train_months) // test_months)

    splits = []
    for i in range(n_splits):
        test_end = total - i * test_months
        test_start = test_end - test_months
        if test_start < min_train_months:
            break

        test_dates = set(unique_dates[test_start:test_end])
        train_dates = set(unique_dates[:test_start])

        train_idx = np.where(dates.isin(train_dates))[0]
        test_idx = np.where(dates.isin(test_dates))[0]

        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))

    return splits[::-1]  # chronological order


def get_spatial_cv_splits(
    groups: pd.Series,
    n_folds: int = 5,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Spatial CV via GroupKFold on grid_id."""
    gkf = GroupKFold(n_splits=n_folds)
    return list(gkf.split(np.zeros(len(groups)), groups=groups))


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_lower: Optional[np.ndarray] = None,
    y_pred_upper: Optional[np.ndarray] = None,
    prefix: str = "",
    extreme_percentiles: List[float] = [90, 95],
) -> Dict[str, float]:
    """
    Compute regression, extreme-value, and uncertainty metrics.
    
    Matches the old HARM_unified evaluate_predictions output format.
    """
    y_range = float(np.ptp(y_true))
    if y_range == 0:
        y_range = 1.0

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    metrics = {
        f"{prefix}rmse": rmse,
        f"{prefix}nrmse": rmse / y_range,
        f"{prefix}mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}medae": float(median_absolute_error(y_true, y_pred)),
        f"{prefix}r2": float(r2_score(y_true, y_pred)),
        f"{prefix}n_samples": len(y_true),
        f"{prefix}bias": float(np.mean(y_pred - y_true)),
    }

    # ── Extreme-value metrics (P90, P95) ──
    for pct in extreme_percentiles:
        threshold = np.percentile(y_true, pct)
        mask = y_true >= threshold
        n_extreme = int(np.sum(mask))

        metrics[f"{prefix}n_extreme_p{int(pct)}"] = n_extreme

        if n_extreme > 10:
            metrics[f"{prefix}rmse_extreme_p{int(pct)}"] = float(
                np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
            )
            metrics[f"{prefix}nrmse_extreme_p{int(pct)}"] = (
                metrics[f"{prefix}rmse_extreme_p{int(pct)}"] / y_range
            )
            metrics[f"{prefix}bias_extreme_p{int(pct)}"] = float(
                np.mean(y_pred[mask] - y_true[mask])
            )

            # Hit rate: fraction of true extremes predicted as extreme
            pred_threshold = np.percentile(y_pred, pct)
            pred_extreme = y_pred >= pred_threshold
            metrics[f"{prefix}hit_rate_p{int(pct)}"] = float(
                (pred_extreme & mask).sum() / mask.sum()
            )

            if pred_extreme.sum() > 0:
                metrics[f"{prefix}false_alarm_rate_p{int(pct)}"] = float(
                    (pred_extreme & ~mask).sum() / pred_extreme.sum()
                )

    # ── Uncertainty metrics (require q10 and q90) ──
    if y_pred_lower is not None and y_pred_upper is not None:
        in_interval = (y_true >= y_pred_lower) & (y_true <= y_pred_upper)
        metrics[f"{prefix}picp_80"] = float(np.mean(in_interval))
        metrics[f"{prefix}piw_80"] = float(np.mean(y_pred_upper - y_pred_lower))

        # Calibration: fraction below q10, above q90
        metrics[f"{prefix}calibration_below_q10"] = float(np.mean(y_true < y_pred_lower))
        metrics[f"{prefix}calibration_above_q90"] = float(np.mean(y_true > y_pred_upper))
        metrics[f"{prefix}calibration_in_interval"] = float(np.mean(in_interval))

        # CRPS approximation
        metrics[f"{prefix}crps"] = float(
            _approx_crps(y_true, y_pred, y_pred_lower, y_pred_upper)
        )

    return metrics


def _approx_crps(y_true, y_pred_q50, y_pred_q10, y_pred_q90):
    """Approximate CRPS from 3 quantile predictions."""
    # Pinball losses for each quantile
    loss_10 = np.mean(pinball_loss(y_true, y_pred_q10, 0.1))
    loss_50 = np.mean(pinball_loss(y_true, y_pred_q50, 0.5))
    loss_90 = np.mean(pinball_loss(y_true, y_pred_q90, 0.9))
    return (loss_10 + loss_50 + loss_90) / 3.0


def compute_crps_from_quantiles(
    y_true: np.ndarray,
    q_preds: Dict[float, np.ndarray],
) -> float:
    """Approximate CRPS from quantile predictions."""
    total = 0.0
    for q, pred in sorted(q_preds.items()):
        total += np.mean(pinball_loss(y_true, pred, q))
    return total / len(q_preds)


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> np.ndarray:
    diff = y_true - y_pred
    return np.where(diff >= 0, quantile * diff, (quantile - 1) * diff)


def compute_within_between_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> Dict[str, float]:
    """Decompose error into within-group and between-group components."""
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "group": groups})
    group_means = df.groupby("group").agg(
        true_mean=("y_true", "mean"), pred_mean=("y_pred", "mean")
    )
    between_mse = mean_squared_error(group_means["true_mean"], group_means["pred_mean"])
    return {"between_rmse": np.sqrt(between_mse)}


def compute_yearly_metrics(
    df: pd.DataFrame,
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    year_col: str = "year",
) -> pd.DataFrame:
    """Compute metrics per year."""
    records = []
    for yr, grp in df.groupby(year_col):
        m = evaluate_predictions(grp[y_true_col].values, grp[y_pred_col].values)
        m["year"] = yr
        records.append(m)
    return pd.DataFrame(records)


def compute_improvement_metrics(
    base_metrics: Dict, improved_metrics: Dict, metric_key: str = "rmse",
) -> Dict[str, float]:
    """Compute improvement (%) of improved over base."""
    base_val = base_metrics.get(metric_key, 0)
    imp_val = improved_metrics.get(metric_key, 0)
    if base_val > 0:
        pct = (base_val - imp_val) / base_val * 100
    else:
        pct = 0.0
    return {"improvement_pct": pct, "base": base_val, "improved": imp_val}
