"""Core training utilities: quantile model training, grid search, Optuna tuning."""

import gc
import json
import logging
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

from .metrics import (
    evaluate_predictions, temporal_cv_splits, pinball_loss,
)
from .utils import clear_memory, clear_gpu_memory


def train_quantile_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Dict,
    device: str = "cpu",
    quantiles: List[float] = [0.1, 0.5, 0.9],
    random_state: int = 42,
    sample_weight: Optional[np.ndarray] = None,
) -> Dict[float, XGBRegressor]:
    """Train one XGBRegressor per quantile.
    
    Parameters
    ----------
    device : 'cuda' or 'cpu'
    sample_weight : optional array of per-sample weights
    """
    models = {}
    for q in quantiles:
        p = deepcopy(params)
        p.update({
            "objective": "reg:quantileerror",
            "quantile_alpha": q,
            "tree_method": "hist",
            "device": device,
            "random_state": random_state,
            "verbosity": 0,
            "n_jobs": -1,
        })

        model = XGBRegressor(**p)
        model.fit(X_train, y_train, sample_weight=sample_weight)
        models[q] = model
    return models


def predict_with_quantiles(
    models: Dict[float, XGBRegressor],
    X: np.ndarray,
) -> Dict[float, np.ndarray]:
    """Generate predictions for each quantile model."""
    return {q: model.predict(X) for q, model in models.items()}


def grid_search_cv(
    X: np.ndarray,
    y: np.ndarray,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    param_grid: Dict[str, list],
    fixed_params: Dict,
    device: str = "cpu",
    quantiles: List[float] = [0.1, 0.5, 0.9],
    random_state: int = 42,
) -> Tuple[Dict, List[Dict]]:
    """
    Exhaustive grid search over param_grid with temporal/spatial CV.

    Returns
    -------
    best_params : dict — best parameter combination
    all_results : list of dicts with scores for each combo
    """
    keys = sorted(param_grid.keys())
    combos = [dict(zip(keys, vals)) for vals in product(*[param_grid[k] for k in keys])]

    best_score = np.inf
    best_params = {}
    all_results = []

    for combo in combos:
        params = {**fixed_params, **combo}
        fold_scores = []

        for train_idx, val_idx in cv_splits:
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            scaler = MinMaxScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)

            models = train_quantile_models(
                X_tr_s, y_tr, params, device,
                quantiles, random_state,
            )
            y_pred = models[0.5].predict(X_val_s)
            y_pred = np.maximum(y_pred, 0)

            metrics = evaluate_predictions(y_val, y_pred)
            fold_scores.append(metrics["rmse"])

            clear_memory()

        mean_rmse = np.mean(fold_scores)
        all_results.append({"params": combo, "mean_rmse": mean_rmse, "fold_scores": fold_scores})

        if mean_rmse < best_score:
            best_score = mean_rmse
            best_params = combo

    logging.info(f"[GRID] Best RMSE={best_score:.4f}, params={best_params}")
    return best_params, all_results


def compute_weighted_params(results: List[Dict], top_k: int = 3) -> Dict:
    """Average top-k parameter sets, weighted by inverse RMSE."""
    sorted_results = sorted(results, key=lambda x: x["mean_rmse"])[:top_k]
    weights = [1.0 / r["mean_rmse"] for r in sorted_results]
    total_w = sum(weights)

    all_keys = set()
    for r in sorted_results:
        all_keys.update(r["params"].keys())

    weighted = {}
    for key in all_keys:
        vals = [r["params"].get(key, 0) for r in sorted_results]
        if all(isinstance(v, (int, float)) for v in vals):
            weighted[key] = sum(w * v for w, v in zip(weights, vals)) / total_w
            if all(isinstance(v, int) for v in vals):
                weighted[key] = int(round(weighted[key]))
        else:
            weighted[key] = sorted_results[0]["params"][key]

    return weighted


def create_optuna_objective(
    X_tr_scaled: np.ndarray,
    y_tr: np.ndarray,
    cv_splits: List,
    device: str = "cpu",
    random_state: int = 42,
):
    """Return an Optuna objective function for hyperparameter tuning."""
    import optuna

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }

        fold_scores = []
        for train_idx, val_idx in cv_splits:
            xgb_params = {
                **params,
                "objective": "reg:quantileerror",
                "quantile_alpha": 0.5,
                "tree_method": "hist",
                "device": device,
                "random_state": random_state,
                "verbosity": 0,
                "n_jobs": -1,
            }

            model = XGBRegressor(**xgb_params)
            model.fit(X_tr_scaled[train_idx], y_tr[train_idx])

            y_pred = np.maximum(model.predict(X_tr_scaled[val_idx]), 0)
            fold_scores.append(np.sqrt(mean_squared_error_safe(y_tr[val_idx], y_pred)))

            del model
            clear_memory()

        return np.mean(fold_scores)

    return objective


def mean_squared_error_safe(y_true, y_pred):
    """MSE that handles edge cases."""
    from sklearn.metrics import mean_squared_error
    return mean_squared_error(y_true, y_pred)
