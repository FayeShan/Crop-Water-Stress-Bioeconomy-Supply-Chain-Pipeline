"""Global Layer (M0) — XGBoost quantile regression on all data."""

import gc
import json
import logging
from datetime import datetime
from multiprocessing import Manager
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

from .utils import (
    setup_gpu_environment, clear_gpu_memory, clear_memory,
    check_memory_threshold, clean_for_json, compute_sample_weights,
)
from .metrics import evaluate_predictions, temporal_cv_splits
from .training import (
    train_quantile_models, predict_with_quantiles,
    compute_weighted_params,
)


def _create_optuna_objective(
    X_tr_scaled, y_tr, cv_splits, device, random_state,
    gpu_queue, gpu_lock,
):
    """Optuna objective with GPU queue for multi-GPU environments."""
    import optuna

    use_cuda = (device == "cuda")

    def objective(trial):
        if use_cuda:
            with gpu_lock:
                gpu_id = gpu_queue.pop(0) if gpu_queue else 0
        else:
            gpu_id = None

        try:
            param = {
                "n_estimators": trial.suggest_int("n_estimators", 80, 200),
                "max_depth": trial.suggest_int("max_depth", 6, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            }

            xgb_kwargs = {
                "objective": "reg:squarederror",
                "tree_method": "hist",
                "device": device,
                "random_state": random_state,
                "verbosity": 0,
                "max_bin": 512,
                **param,
            }

            cv_scores = []
            for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
                model = XGBRegressor(**xgb_kwargs)
                model.fit(
                    X_tr_scaled[train_idx], y_tr[train_idx],
                    eval_set=[(X_tr_scaled[val_idx], y_tr[val_idx])],
                    verbose=False,
                )
                y_pred = model.predict(X_tr_scaled[val_idx])
                rmse = np.sqrt(mean_squared_error(y_tr[val_idx], y_pred))
                cv_scores.append(rmse)

                trial.report(rmse, fold_idx)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

                del model
                if use_cuda:
                    clear_gpu_memory()

            return np.mean(cv_scores)
        finally:
            if use_cuda:
                with gpu_lock:
                    gpu_queue.append(gpu_id)

    return objective


def _train_global_optuna(
    X_tr, y_tr, groups_tr,
    n_inner_folds, device, n_trials,
    random_state, n_jobs,
):
    """Run Optuna study and return (scaler, best_params, best_value)."""
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner

    use_cuda = (device == "cuda")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    scaler = MinMaxScaler().fit(X_tr)
    X_tr_scaled = scaler.transform(X_tr)

    inner_cv = GroupKFold(n_splits=n_inner_folds)
    cv_splits = list(inner_cv.split(X_tr_scaled, y_tr, groups_tr))

    manager = Manager()
    gpu_lock = manager.Lock()
    gpu_queue = manager.list(range(n_jobs)) if use_cuda else manager.list([0])

    objective = _create_optuna_objective(
        X_tr_scaled, y_tr.values, cv_splits, device,
        random_state, gpu_queue, gpu_lock,
    )

    study = optuna.create_study(
        sampler=TPESampler(seed=random_state),
        pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=5),
        direction="minimize",
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=max(1, n_jobs))

    best_params = study.best_params
    best_value = study.best_value

    del X_tr_scaled, cv_splits
    if use_cuda:
        clear_gpu_memory()
    gc.collect()

    return scaler, best_params, best_value


def train_global_layer(
    cfg,
    crop: str,
    df: pd.DataFrame,
    feature_cols: List[str],
    model_dir: Path,
) -> Dict:
    """Train the Global layer (M0) of HARM; returns a summary dict."""
    logging.info(f"[GLOBAL] Starting Global layer training for {crop}")

    random_state = cfg.common.random_state
    quantiles = cfg.common.quantiles
    gl_cfg = cfg.global_layer

    # Detect actual GPU availability (falls back to CPU if not found)
    gpu_available, n_gpus, device = setup_gpu_environment(cfg.common.use_gpu)
    n_jobs = max(1, n_gpus if gpu_available else 1)
    model_dir.mkdir(parents=True, exist_ok=True)

    X = df[feature_cols]
    y = df[crop]
    dates = pd.to_datetime(df["time"])
    grid_ids = df["grid50_id"]

    logging.info(f"[GLOBAL] Data shape: X={X.shape}, y={y.shape}")

    # Sample weighting (for right-skewed targets)
    sw_cfg = gl_cfg.get("sample_weight", {})
    sample_weights = compute_sample_weights(
        y.values,
        enabled=sw_cfg.get("enabled", False),
        alpha=sw_cfg.get("alpha", 1.0),
        reference_quantile=sw_cfg.get("reference_quantile", 0.9),
        max_weight=sw_cfg.get("max_weight", 10.0),
    )

    splits = temporal_cv_splits(
        dates,
        test_months=gl_cfg["test_months"],
        min_train_months=gl_cfg["min_train_months"],
    )
    logging.info(f"[GLOBAL] Temporal CV: {len(splits)} splits")

    rng = np.random.RandomState(random_state)
    metrics_list, fi_records = [], []
    all_rmse = []
    best_params_per_fold = []
    all_val_predictions = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
        logging.info(f"[GLOBAL] Fold {fold_idx}/{len(splits)}: "
                     f"train={len(train_idx)}, val={len(val_idx)}")
        check_memory_threshold()

        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        groups_tr = grid_ids.iloc[train_idx]

        if gl_cfg["tuning_method"] == "optuna":
            scaler, best_params, best_value = _train_global_optuna(
                X_tr, y_tr, groups_tr,
                gl_cfg["n_inner_folds"], device, gl_cfg["n_trials"],
                random_state, n_jobs,
            )
        else:
            scaler = MinMaxScaler().fit(X_tr)
            best_params = gl_cfg["fixed_params"]
            best_value = None

        logging.info(f"[GLOBAL] Fold {fold_idx}: best RMSE={best_value}")
        best_params_per_fold.append(best_params)

        X_tr_s = scaler.transform(X_tr)
        X_val_s = scaler.transform(X_val)

        # Slice weights for this fold's training set
        fold_weights = sample_weights[train_idx] if sample_weights is not None else None

        q_models = train_quantile_models(
            X_tr_s, y_tr.values, best_params,
            device, quantiles, random_state, fold_weights,
        )

        q_preds = predict_with_quantiles(q_models, X_val_s)
        y_pred = np.maximum(q_preds[0.5], 0)
        y_pred_q10 = np.maximum(q_preds.get(0.1, y_pred), 0)
        y_pred_q90 = np.maximum(q_preds.get(0.9, y_pred), 0)

        # Evaluate (with quantile predictions for uncertainty metrics)
        fold_metrics = evaluate_predictions(
            y_val.values, y_pred, y_pred_q10, y_pred_q90, prefix="",
        )
        metrics_list.append({"fold": fold_idx, **fold_metrics})
        all_rmse.append(fold_metrics["rmse"])

        importances = q_models[0.5].feature_importances_
        for feat, imp in zip(feature_cols, importances):
            fi_records.append({"fold": fold_idx, "feature": feat, "importance": float(imp)})

        all_val_predictions.append(pd.DataFrame({
            "fold": fold_idx,
            "time": dates.iloc[val_idx].values,
            "grid50_id": grid_ids.iloc[val_idx].values,
            "y_true": y_val.values,
            "y_pred": y_pred,
            "y_pred_q10": y_pred_q10,
            "y_pred_q90": y_pred_q90,
            "residual": y_val.values - y_pred,
        }))

        del X_tr, X_val, X_tr_s, X_val_s, scaler, q_models
        clear_memory()
        if gpu_available:
            clear_gpu_memory()

    if gl_cfg["tuning_method"] == "optuna":
        weighted_params = compute_weighted_params(
            [{"params": p, "mean_rmse": r} for p, r in zip(best_params_per_fold, all_rmse)]
        )
    else:
        weighted_params = gl_cfg["fixed_params"]

    logging.info(f"[GLOBAL] Weighted params: {weighted_params}")

    # ── Train final models on ALL data ──
    logging.info("[GLOBAL] Training final models on all data")
    final_scaler = MinMaxScaler().fit(X)
    X_scaled = final_scaler.transform(X)

    final_models = train_quantile_models(
        X_scaled, y.values, weighted_params,
        device, quantiles, random_state, sample_weights,
    )

    for q, model in final_models.items():
        model.save_model(str(model_dir / f"model_q{int(q * 100)}.json"))

    joblib.dump(final_scaler, model_dir / "scaler.joblib")

    with open(model_dir / "best_params.json", "w") as f:
        json.dump(weighted_params, f, indent=2)

    with open(model_dir / "feature_names.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    val_df = pd.concat(all_val_predictions, ignore_index=True)
    val_df.to_parquet(model_dir / "val_predictions.parquet", index=False)

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(model_dir / "cv_metrics.csv", index=False)

    fi_df = pd.DataFrame(fi_records)
    fi_df.to_csv(model_dir / "feature_importance.csv", index=False)

    # Bootstrap CI
    boot = rng.choice(all_rmse, size=(1000, len(all_rmse))).mean(axis=1)
    ci_rmse = np.percentile(boot, [2.5, 97.5])

    mean_metrics = metrics_df.mean(numeric_only=True).to_dict()

    summary = {
        "layer": "global",
        "crop": crop,
        "n_features": len(feature_cols),
        "n_samples": len(X),
        "n_folds": len(splits),
        "metrics": {
            "mean": clean_for_json(mean_metrics),
            "ci_rmse_95": [float(ci_rmse[0]), float(ci_rmse[1])],
        },
        "best_params": clean_for_json(weighted_params),
        "model_dir": str(model_dir),
        "timestamp": datetime.now().isoformat(),
    }

    with open(model_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logging.info(
        f"[GLOBAL] Done. RMSE={mean_metrics.get('rmse', 'N/A'):.4f}, "
        f"R²={mean_metrics.get('r2', 'N/A'):.4f}"
    )
    return summary
