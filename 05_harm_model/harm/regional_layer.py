"""Regional Layer (M1) — per-region residual correction over the 9 agro-climatic regions (THZ x MST)."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from itertools import product

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

from harm.metrics import evaluate_predictions, temporal_cv_splits
from harm.training import train_quantile_models, predict_with_quantiles
from harm.utils import (
    setup_gpu_environment, clear_gpu_memory, clear_memory,
    clean_for_json,
)


def train_regional_layer(
    cfg,
    crop: str,
    df: pd.DataFrame,
    feature_cols: List[str],
    global_dir: Path,
    regional_dir: Path,
) -> Dict:
    """
    Train the Regional layer (M1) of HARM.

    Uses out-of-fold residuals from Global layer, grouped by regiontype.
    """
    reg_cfg = cfg.regional_layer
    gpu_available, n_gpus, device = setup_gpu_environment(cfg.common.use_gpu)
    quantiles = cfg.common.quantiles
    random_state = cfg.common.random_state

    regional_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"[REGIONAL] Starting Regional layer training for {crop}")

    # ── Load Global artifacts ──
    global_scaler = joblib.load(global_dir / "scaler.joblib")
    global_model = XGBRegressor()
    global_model.load_model(str(global_dir / "model_q50.json"))
    with open(global_dir / "feature_names.json") as f:
        global_features = json.load(f)

    # ── Load out-of-fold predictions for residuals ──
    val_pred_path = global_dir / "val_predictions.parquet"
    if not val_pred_path.exists():
        raise FileNotFoundError(
            f"No val_predictions.parquet at {global_dir}. Run Global first."
        )
    val_preds = pd.read_parquet(val_pred_path)
    logging.info(f"[REGIONAL] Loaded {len(val_preds)} out-of-fold predictions")

    # ── Merge regiontype from training data ──
    if "regiontype" not in df.columns:
        raise ValueError("No 'regiontype' column in training data")

    df["time"] = pd.to_datetime(df["time"])
    val_preds["time"] = pd.to_datetime(val_preds["time"])
    val_preds = val_preds.merge(
        df[["time", "grid50_id", "regiontype"]].drop_duplicates(),
        on=["time", "grid50_id"],
        how="left",
    )

    n_missing = val_preds["regiontype"].isna().sum()
    if n_missing > 0:
        logging.warning(f"[REGIONAL] {n_missing} samples missing regiontype, dropping")
        val_preds = val_preds.dropna(subset=["regiontype"])

    region_types = sorted(val_preds["regiontype"].unique())
    logging.info(f"[REGIONAL] Found {len(region_types)} regions: {region_types}")

    # ── Train per region ──
    region_results = {}
    trained_regions = []
    skipped_regions = []

    for region_type in region_types:
        region_str = str(int(region_type))
        region_dir_i = regional_dir / region_str
        region_dir_i.mkdir(parents=True, exist_ok=True)

        mask = val_preds["regiontype"] == region_type
        df_region = val_preds[mask].copy()
        n_samples = len(df_region)

        if n_samples < reg_cfg.get("min_samples", 5000):
            logging.warning(
                f"[REGIONAL-{region_str}] Only {n_samples} samples, skipping"
            )
            _save_skip_marker(region_dir_i, region_str, n_samples, "insufficient_samples")
            skipped_regions.append(region_str)
            continue

        # ── Zero-stress region check ──
        # If >90% of samples have near-zero stress, flag this region
        # The model can't learn much here — better to predict zero/mean
        y_region = df_region["y_true"].values
        zero_frac = float(np.mean(y_region == 0))
        near_zero_frac = float(np.mean(y_region < np.percentile(y_region[y_region > 0], 10) if np.any(y_region > 0) else 1))
        median_stress = float(np.median(y_region))

        if zero_frac > 0.9:
            logging.info(
                f"[REGIONAL-{region_str}] Low-stress region: "
                f"{zero_frac*100:.0f}% zeros, median={median_stress:.0f}. "
                f"Flagged as zero-stress."
            )
            _save_skip_marker(
                region_dir_i, region_str, n_samples,
                f"zero_stress_region (zeros={zero_frac*100:.0f}%, median={median_stress:.0f})",
            )
            region_results[region_str] = {
                "use_regional": False,
                "reason": "zero_stress_region",
                "n_samples": n_samples,
                "zero_frac": zero_frac,
                "median_stress": median_stress,
                "flag_predict_zero": True,
            }
            skipped_regions.append(region_str)
            continue

        logging.info(f"[REGIONAL-{region_str}] Training (n={n_samples})")

        result = _train_single_region(
            cfg=cfg,
            crop=crop,
            df_region=df_region,
            df_full=df,
            global_scaler=global_scaler,
            global_model=global_model,
            global_features=global_features,
            global_dir=global_dir,
            region_str=region_str,
            region_dir=region_dir_i,
            device=device,
            quantiles=quantiles,
            random_state=random_state,
        )

        region_results[region_str] = result

        if result["use_regional"]:
            trained_regions.append(region_str)
            logging.info(
                f"[REGIONAL-{region_str}] ADOPTED — "
                f"NRMSE improvement: {result.get('nrmse_improvement_pct', 0):.2f}%"
            )
        else:
            skipped_regions.append(region_str)
            logging.info(
                f"[REGIONAL-{region_str}] REJECTED — "
                f"improvement {result.get('nrmse_improvement_pct', 0):.2f}% "
                f"< threshold {reg_cfg.get('improvement_threshold', 2.0)}%"
            )

        clear_memory()
        if gpu_available:
            clear_gpu_memory()

    # ── Summary ──
    summary = {
        "layer": "regional",
        "crop": crop,
        "n_regions": len(region_types),
        "n_trained": len(trained_regions),
        "n_skipped": len(skipped_regions),
        "trained_regions": trained_regions,
        "skipped_regions": skipped_regions,
        "region_results": {k: clean_for_json(v) for k, v in region_results.items()},
        "model_dir": str(regional_dir),
        "timestamp": datetime.now().isoformat(),
    }

    with open(regional_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logging.info(
        f"[REGIONAL] Done. {len(trained_regions)}/{len(region_types)} regions adopted"
    )
    return summary


def _train_single_region(
    cfg,
    crop: str,
    df_region: pd.DataFrame,
    df_full: pd.DataFrame,
    global_scaler,
    global_model,
    global_features: List[str],
    global_dir: Path,
    region_str: str,
    region_dir: Path,
    device: str,
    quantiles: List[float],
    random_state: int,
) -> Dict:
    """Train regional model for a single region on Global residuals."""
    reg_cfg = cfg.regional_layer
    tag = f"[REGIONAL-{region_str}]"

    # ── Match OOF predictions back to original features ──
    region_keys = df_region[["time", "grid50_id"]].drop_duplicates()
    df_matched = df_full.merge(region_keys, on=["time", "grid50_id"], how="inner")

    X = df_matched[global_features].values
    y_true = df_matched[crop].values
    dates = pd.to_datetime(df_matched["time"])

    # Global predictions → residuals
    X_scaled = global_scaler.transform(X)
    global_pred = np.maximum(global_model.predict(X_scaled), 0)
    residuals = y_true - global_pred

    y_range = float(np.ptp(y_true))
    if y_range == 0:
        y_range = 1.0

    logging.info(
        f"{tag} Residual: mean={np.mean(residuals):.0f}, "
        f"std={np.std(residuals):.0f}"
    )

    # ── Temporal CV ──
    cv_splits = temporal_cv_splits(
        dates,
        test_months=reg_cfg.get("test_months", 24),
        min_train_months=reg_cfg.get("min_train_months", 12),
    )

    if not cv_splits:
        logging.warning(f"{tag} No valid CV splits")
        _save_skip_marker(region_dir, region_str, len(X), "no_cv_splits")
        return {"use_regional": False, "reason": "no_cv_splits"}

    # ── Grid search ──
    best_params = _grid_search_regional(
        X_scaled, residuals, cv_splits,
        reg_cfg.get("param_grid", {}),
        reg_cfg.get("fixed_params", {}),
        device, random_state, tag,
    )

    # ── Evaluate: Global-only vs Global+Regional ──
    metrics_global = []
    metrics_combined = []
    fi_records = []
    all_val_predictions = []

    # Load global q10/q90 models for uncertainty propagation
    global_q10, global_q90 = None, None
    if (global_dir / "model_q10.json").exists():
        global_q10 = XGBRegressor()
        global_q10.load_model(str(global_dir / "model_q10.json"))
    if (global_dir / "model_q90.json").exists():
        global_q90 = XGBRegressor()
        global_q90.load_model(str(global_dir / "model_q90.json"))

    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits, start=1):
        X_tr = X_scaled[train_idx]
        X_val = X_scaled[val_idx]
        r_tr = residuals[train_idx]
        y_val = y_true[val_idx]
        g_val = global_pred[val_idx]

        fold_scaler = MinMaxScaler().fit(X_tr)
        X_tr_s = fold_scaler.transform(X_tr)
        X_val_s = fold_scaler.transform(X_val)

        r_models = train_quantile_models(
            X_tr_s, r_tr, best_params, device, quantiles, random_state,
        )

        q_preds = predict_with_quantiles(r_models, X_val_s)
        r_pred_q50 = q_preds[0.5]
        r_pred_q10 = q_preds.get(0.1, r_pred_q50)
        r_pred_q90 = q_preds.get(0.9, r_pred_q50)

        # Combined predictions (Global + Regional residual)
        y_combined = np.maximum(g_val + r_pred_q50, 0)

        # Combined q10/q90: global quantile + regional residual quantile
        if global_q10 is not None and global_q90 is not None:
            g_q10_val = np.maximum(global_q10.predict(X_val), 0)
            g_q90_val = np.maximum(global_q90.predict(X_val), 0)
            y_combined_q10 = np.maximum(g_q10_val + r_pred_q10, 0)
            y_combined_q90 = np.maximum(g_q90_val + r_pred_q90, 0)
        else:
            y_combined_q10 = np.maximum(g_val + r_pred_q10, 0)
            y_combined_q90 = np.maximum(g_val + r_pred_q90, 0)

        # Global-only metrics (no q10/q90 from global at fold level)
        m_g = evaluate_predictions(y_val, np.maximum(g_val, 0), prefix="global_")
        metrics_global.append(m_g)

        # Combined metrics WITH uncertainty
        m_c = evaluate_predictions(
            y_val, y_combined, y_combined_q10, y_combined_q90,
            prefix="combined_",
        )
        metrics_combined.append(m_c)

        for feat, imp in zip(global_features, r_models[0.5].feature_importances_):
            fi_records.append({
                "fold": fold_idx, "feature": feat, "importance": float(imp),
            })

        all_val_predictions.append(pd.DataFrame({
            "fold": fold_idx,
            "time": dates.iloc[val_idx].values,
            "grid50_id": df_matched["grid50_id"].iloc[val_idx].values,
            "y_true": y_val,
            "y_pred_global": np.maximum(g_val, 0),
            "y_pred_combined": y_combined,
            "y_pred_combined_q10": y_combined_q10,
            "y_pred_combined_q90": y_combined_q90,
            "residual": y_val - y_combined,
        }))

        del r_models, fold_scaler
        clear_memory()

    mean_g_nrmse = np.mean([m["global_nrmse"] for m in metrics_global])
    mean_c_nrmse = np.mean([m["combined_nrmse"] for m in metrics_combined])
    mean_g_r2 = np.mean([m["global_r2"] for m in metrics_global])
    mean_c_r2 = np.mean([m["combined_r2"] for m in metrics_combined])

    improvement = (
        (mean_g_nrmse - mean_c_nrmse) / mean_g_nrmse * 100
        if mean_g_nrmse > 0 else 0
    )

    # Adaptive threshold: lower bar when Global is very poor
    base_threshold = reg_cfg.get("improvement_threshold", 2.0)
    if mean_g_r2 < 0:
        # Global worse than mean — any consistent improvement is worth taking
        threshold = max(0.5, base_threshold * 0.25)
        logging.info(f"{tag} Global R²={mean_g_r2:.2f} < 0, using relaxed threshold={threshold:.1f}%")
    elif mean_g_r2 < 0.2:
        threshold = max(1.0, base_threshold * 0.5)
        logging.info(f"{tag} Global R²={mean_g_r2:.2f} < 0.2, using reduced threshold={threshold:.1f}%")
    else:
        threshold = base_threshold

    use_regional = improvement >= threshold

    logging.info(
        f"{tag} Global: NRMSE={mean_g_nrmse:.6f} R²={mean_g_r2:.4f} | "
        f"Combined: NRMSE={mean_c_nrmse:.6f} R²={mean_c_r2:.4f} | "
        f"Improvement={improvement:.2f}%"
    )

    # ── Train final models if adopted ──
    if use_regional:
        final_scaler = MinMaxScaler().fit(X_scaled)
        X_final = final_scaler.transform(X_scaled)

        final_models = train_quantile_models(
            X_final, residuals, best_params, device, quantiles, random_state,
        )

        for q, model in final_models.items():
            model.save_model(str(region_dir / f"model_q{int(q * 100)}.json"))
        joblib.dump(final_scaler, region_dir / "scaler.joblib")

        with open(region_dir / "best_params.json", "w") as f:
            json.dump(best_params, f, indent=2)

    pd.DataFrame(fi_records).to_csv(region_dir / "feature_importance.csv", index=False)

    # Save full CV metrics (all columns from evaluate_predictions)
    cv_records = []
    for i, (mg, mc) in enumerate(zip(metrics_global, metrics_combined)):
        row = {"fold": i + 1}
        row.update(mg)   # global_rmse, global_nrmse, global_r2, global_hit_rate_p90, ...
        row.update(mc)   # combined_rmse, combined_nrmse, combined_r2, combined_picp_80, ...
        cv_records.append(row)
    cv_df = pd.DataFrame(cv_records)
    cv_df.to_csv(region_dir / "cv_metrics.csv", index=False)

    if all_val_predictions:
        val_df = pd.concat(all_val_predictions, ignore_index=True)
        val_df.to_parquet(region_dir / "val_predictions.parquet", index=False)

    result = {
        "use_regional": use_regional,
        "n_samples": len(X),
        "n_folds": len(cv_splits),
        "global_nrmse": float(mean_g_nrmse),
        "global_r2": float(mean_g_r2),
        "combined_nrmse": float(mean_c_nrmse),
        "combined_r2": float(mean_c_r2),
        "nrmse_improvement_pct": float(improvement),
        "best_params": best_params,
    }

    with open(region_dir / "region_summary.json", "w") as f:
        json.dump(clean_for_json(result), f, indent=2)

    return result


def _grid_search_regional(
    X_scaled, residuals, cv_splits, param_grid, fixed_params,
    device, random_state, tag,
):
    """Grid search over param_grid using temporal CV on residuals."""
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    if not keys:
        return fixed_params.copy()

    combos = list(product(*values))
    logging.info(f"{tag} Grid search: {len(combos)} param combinations")

    best_rmse = float("inf")
    best_params = fixed_params.copy()

    for combo in combos:
        params = {**fixed_params, **dict(zip(keys, combo))}
        fold_rmses = []

        for train_idx, val_idx in cv_splits:
            X_tr = X_scaled[train_idx]
            X_val = X_scaled[val_idx]
            r_tr = residuals[train_idx]
            r_val = residuals[val_idx]

            scaler = MinMaxScaler().fit(X_tr)
            model = XGBRegressor(
                objective="reg:squarederror",
                tree_method="hist",
                device=device,
                random_state=random_state,
                **params,
            )
            model.fit(scaler.transform(X_tr), r_tr)
            pred = model.predict(scaler.transform(X_val))
            fold_rmses.append(float(np.sqrt(np.mean((r_val - pred) ** 2))))

        mean_rmse = np.mean(fold_rmses)
        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_params = params.copy()

    logging.info(f"{tag} Best params: {best_params} (RMSE={best_rmse:.0f})")
    return best_params


def _save_skip_marker(region_dir, region_str, n_samples, reason):
    """Save marker for skipped regions."""
    with open(region_dir / "region_summary.json", "w") as f:
        json.dump({
            "region": region_str,
            "n_samples": n_samples,
            "reason": reason,
            "use_regional": False,
        }, f, indent=2)
