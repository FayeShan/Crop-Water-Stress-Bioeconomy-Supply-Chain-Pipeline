"""Extreme Layer (M2) — detect and correct systematic underestimation of high values."""

import gc
import json
import logging
from copy import deepcopy
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import (
    r2_score, mean_squared_error, roc_auc_score,
)
from xgboost import XGBRegressor, XGBClassifier

from harm.metrics import evaluate_predictions, temporal_cv_splits
from harm.utils import (
    setup_gpu_environment, clear_gpu_memory, clear_memory, clean_for_json,
)


def _identify_extreme_underestimates(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ytrue_percentile: float = 85,
    underestimate_threshold: float = 0.15,
) -> Tuple[np.ndarray, Dict]:
    """
    Identify samples that are both high-value AND systematically underestimated.

    Returns
    -------
    under_mask : bool array — True for underestimated extreme samples
    stats : dict with diagnostic info
    """
    threshold = np.percentile(y_true, ytrue_percentile)
    high_mask = y_true >= threshold
    under_mask = high_mask & (y_pred < (1 - underestimate_threshold) * y_true)
    over_mask = high_mask & (y_pred > (1 + underestimate_threshold) * y_true)

    stats = {
        "ytrue_threshold": float(threshold),
        "n_high_value": int(np.sum(high_mask)),
        "n_underestimate": int(np.sum(under_mask)),
        "n_overestimate": int(np.sum(over_mask)),
        "underestimate_pct": float(np.mean(under_mask[high_mask]) * 100) if high_mask.any() else 0,
    }
    return under_mask, stats


def _apply_bounded_correction(
    y_base: np.ndarray,
    detector_probs: np.ndarray,
    corrector_pred: np.ndarray,
    prob_threshold: float = 0.6,
    max_correction_ratio: float = 0.05,
    max_relative_correction: float = 0.5,
    conservation_tolerance: float = 0.05,
) -> Tuple[np.ndarray, Dict]:
    """
    Apply corrections with safety bounds.

    1. Select top samples by detector probability (capped at max_correction_ratio)
    2. Clip corrections to ±max_relative_correction of base value
    3. Conservation constraint: total sum change ≤ tolerance
    4. Clip negatives to zero
    """
    n = len(y_base)
    y_out = y_base.copy()

    eligible = detector_probs >= prob_threshold
    n_eligible = int(np.sum(eligible))
    max_n = int(n * max_correction_ratio)

    if n_eligible > max_n:
        idx_eligible = np.where(eligible)[0]
        top_idx = idx_eligible[np.argsort(detector_probs[idx_eligible])[-max_n:]]
        final_mask = np.zeros(n, dtype=bool)
        final_mask[top_idx] = True
    else:
        final_mask = eligible

    n_corrected = int(np.sum(final_mask))
    if n_corrected == 0:
        return np.maximum(y_out, 0), {
            "n_eligible": n_eligible, "n_corrected": 0,
            "mean_correction": 0, "conservation_ratio": 1.0,
        }

    base_vals = y_base[final_mask]
    raw_corr = corrector_pred[final_mask]
    max_abs = np.abs(base_vals) * max_relative_correction
    bounded_corr = np.clip(raw_corr, -max_abs, max_abs)

    y_out[final_mask] = base_vals + bounded_corr

    # Conservation constraint
    sum_base = np.sum(y_base)
    if sum_base > 0:
        total_change = np.sum(y_out) - sum_base
        allowed = np.abs(sum_base) * conservation_tolerance
        if np.abs(total_change) > allowed:
            scale = allowed / np.abs(total_change)
            y_out[final_mask] = base_vals + bounded_corr * scale

    y_out = np.maximum(y_out, 0)

    stats = {
        "n_eligible": n_eligible,
        "n_corrected": n_corrected,
        "correction_pct": n_corrected / n * 100,
        "mean_correction": float(np.mean(np.abs(y_out[final_mask] - base_vals))),
        "conservation_ratio": float(np.sum(y_out) / sum_base) if sum_base > 0 else 1.0,
    }
    return y_out, stats


def _grid_search_detector(
    X, y_label, cv_splits, param_grid, fixed_params,
    device, random_state, tag="",
):
    """Grid search for detector (classifier) — optimize AUC."""
    keys = sorted(param_grid.keys())
    combos = [dict(zip(keys, vals)) for vals in product(*[param_grid[k] for k in keys])]

    n_pos = int(np.sum(y_label))
    n_neg = len(y_label) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    best_auc = -1
    best_params = combos[0] if combos else {}

    for combo in combos:
        params = {**fixed_params, **combo, "scale_pos_weight": scale_pos_weight}
        fold_aucs = []

        for train_idx, val_idx in cv_splits:
            y_tr, y_val = y_label[train_idx], y_label[val_idx]
            if np.sum(y_val) == 0 or np.sum(y_tr) == 0:
                continue

            scaler = MinMaxScaler()
            X_tr_s = scaler.fit_transform(X[train_idx])
            X_val_s = scaler.transform(X[val_idx])

            model = XGBClassifier(
                objective="binary:logistic", eval_metric="auc",
                tree_method="hist", device=device,
                random_state=random_state, verbosity=0, **params,
            )
            model.fit(X_tr_s, y_tr)
            probs = model.predict_proba(X_val_s)[:, 1]
            fold_aucs.append(roc_auc_score(y_val, probs))
            del model

        if fold_aucs:
            mean_auc = np.mean(fold_aucs)
            if mean_auc > best_auc:
                best_auc = mean_auc
                best_params = combo

    logging.info(f"{tag} Detector best: {best_params} (AUC={best_auc:.4f})")
    clear_memory()
    return best_params, best_auc


def _grid_search_corrector(
    X, y_resid, cv_splits, param_grid, fixed_params,
    device, random_state, tag="",
):
    """Grid search for corrector (regressor) — optimize RMSE on extreme residuals."""
    keys = sorted(param_grid.keys())
    combos = [dict(zip(keys, vals)) for vals in product(*[param_grid[k] for k in keys])]

    best_rmse = np.inf
    best_params = combos[0] if combos else {}

    for combo in combos:
        params = {**fixed_params, **combo}
        fold_rmses = []

        for train_idx, val_idx in cv_splits:
            scaler = MinMaxScaler()
            X_tr_s = scaler.fit_transform(X[train_idx])
            X_val_s = scaler.transform(X[val_idx])

            model = XGBRegressor(
                objective="reg:squarederror", tree_method="hist",
                device=device, random_state=random_state, verbosity=0,
                **params,
            )
            model.fit(X_tr_s, y_resid[train_idx])
            pred = model.predict(X_val_s)
            fold_rmses.append(np.sqrt(mean_squared_error(y_resid[val_idx], pred)))
            del model

        if fold_rmses:
            mean_rmse = np.mean(fold_rmses)
            if mean_rmse < best_rmse:
                best_rmse = mean_rmse
                best_params = combo

    logging.info(f"{tag} Corrector best: {best_params} (RMSE={best_rmse:.0f})")
    clear_memory()
    return best_params, best_rmse


def _decide_adoption(cv_metrics: List[Dict], ext_cfg: Dict, tag: str = "") -> Tuple[bool, str, Dict]:
    """
    Decision based on:
      1. Hard veto: overall RMSE worsens > threshold
      2. Hard veto: conservation ratio out of bounds
      3. Score: extreme improvement, RMSE change, R² change, stability
    """
    if not cv_metrics:
        return False, "no_cv_metrics", {}

    max_rmse_deg = ext_cfg.get("max_rmse_degradation_pct", 5.0)
    max_conservation_dev = ext_cfg.get("max_conservation_deviation", 0.10)

    rmse_changes = [m["rmse_change_pct"] for m in cv_metrics]
    extreme_improvs = [m["extreme_improvement_pct"] for m in cv_metrics]
    conservations = [m["conservation_ratio"] for m in cv_metrics]
    r2_bases = [m["r2_base"] for m in cv_metrics]
    r2_finals = [m["r2_final"] for m in cv_metrics]

    mean_rmse_change = np.mean(rmse_changes)
    mean_extreme_improv = np.mean(extreme_improvs)
    std_extreme_improv = np.std(extreme_improvs)
    mean_conservation = np.mean(conservations)
    mean_r2_base = np.mean(r2_bases)
    mean_r2_final = np.mean(r2_finals)

    # Hard vetoes
    if mean_rmse_change > max_rmse_deg:
        return False, f"RMSE_degradation_{mean_rmse_change:.1f}%", {}
    if abs(mean_conservation - 1.0) > max_conservation_dev:
        return False, f"conservation_out_of_bounds_{mean_conservation:.3f}", {}

    # Scoring
    score = 0.0
    components = {}

    # Extreme improvement (max 5 pts)
    base_r2 = mean_r2_base
    if base_r2 >= 0.9:
        dyn_threshold = 0.3
    elif base_r2 >= 0.7:
        dyn_threshold = 1.0
    else:
        dyn_threshold = 2.0

    ratio = mean_extreme_improv / dyn_threshold if dyn_threshold > 0 else 0
    components["improvement"] = min(5.0, ratio * 2.5)
    score += components["improvement"]

    # RMSE (max 2 pts)
    if mean_rmse_change <= 0:
        components["rmse"] = min(2.0, -mean_rmse_change * 0.5)
    else:
        components["rmse"] = max(-3.0, -mean_rmse_change * 0.3)
    score += components["rmse"]

    # Conservation (max 1 pt)
    dev = abs(mean_conservation - 1.0)
    components["conservation"] = 1.0 if dev <= 0.02 else (0.5 if dev <= 0.05 else 0.0)
    score += components["conservation"]

    # R² change (max 1 pt)
    r2_change = mean_r2_final - mean_r2_base
    components["r2_change"] = min(1.0, r2_change * 50) if r2_change >= 0 else max(-2.0, r2_change * 100)
    score += components["r2_change"]

    # Stability (max 1 pt)
    components["stability"] = max(0, 1 - min(1, std_extreme_improv / max(0.1, mean_extreme_improv)))
    score += components["stability"]

    # Decision (threshold at 1.0 — no marginal accepts)
    if score >= 3.0:
        decision = "STRONG_ACCEPT"
    elif score >= 1.0:
        decision = "ACCEPT"
    else:
        decision = "REJECT"

    use = score >= 1.0

    breakdown = {
        "score": float(score), "components": {k: float(v) for k, v in components.items()},
        "mean_rmse_change": float(mean_rmse_change),
        "mean_extreme_improv": float(mean_extreme_improv),
        "mean_conservation": float(mean_conservation),
        "mean_r2_base": float(mean_r2_base),
        "mean_r2_final": float(mean_r2_final),
    }

    logging.info(f"{tag} {decision}: score={score:.2f}, "
                 f"ext_improv={mean_extreme_improv:.1f}%, "
                 f"rmse_Δ={mean_rmse_change:+.2f}%")
    return use, decision, breakdown


def train_extreme_layer(
    cfg,
    crop: str,
    df: pd.DataFrame,
    feature_cols: List[str],
    global_dir: Path,
    regional_dir: Path,
    extreme_dir: Path,
) -> Dict:
    """
    Train the Extreme layer (M2) of HARM.

    Reads G+R val_predictions, trains Detector+Corrector per region,
    applies adoption gate per region.
    """
    ext_cfg = cfg.extreme_layer
    gpu_available, n_gpus, device = setup_gpu_environment(cfg.common.use_gpu)
    random_state = cfg.common.random_state

    extreme_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"[EXTREME] Starting Extreme layer for {crop}")

    # ── Load G+R artifacts ──
    global_scaler = joblib.load(global_dir / "scaler.joblib")
    with open(global_dir / "feature_names.json") as f:
        global_features = json.load(f)

    global_model = XGBRegressor()
    global_model.load_model(str(global_dir / "model_q50.json"))

    # ── Load val_predictions ──
    # Regional saves per-region: {crop}/regional/{region_str}/val_predictions.parquet
    # We need to collect all of them and merge into one DataFrame.
    # Fallback: Global val_predictions at {crop}/global/val_predictions.parquet
    global_vp = global_dir / "val_predictions.parquet"

    regional_vp_parts = []
    if regional_dir.exists():
        for region_subdir in sorted(regional_dir.iterdir()):
            vp_path = region_subdir / "val_predictions.parquet"
            if region_subdir.is_dir() and vp_path.exists():
                part = pd.read_parquet(vp_path)
                part["_region_str"] = region_subdir.name
                regional_vp_parts.append(part)

    if regional_vp_parts:
        val_preds = pd.concat(regional_vp_parts, ignore_index=True)
        base_col = "y_pred_combined"
        logging.info(f"[EXTREME] Loaded Regional val_predictions from "
                     f"{len(regional_vp_parts)} regions ({len(val_preds)} rows)")

        # Supplement with Global val_predictions for non-adopted regions
        # Regional only covers adopted regions; remaining samples need Global baseline
        if global_vp.exists():
            global_vp_df = pd.read_parquet(global_vp)
            global_vp_df["time"] = pd.to_datetime(global_vp_df["time"])
            val_preds["time"] = pd.to_datetime(val_preds["time"])

            # Use int64 timestamps for reliable matching (avoids datetime string precision issues)
            regional_keys = set(zip(
                val_preds["time"].astype(np.int64), val_preds["grid50_id"]
            ))
            global_keys = list(zip(
                global_vp_df["time"].astype(np.int64), global_vp_df["grid50_id"]
            ))
            global_mask = np.array([k not in regional_keys for k in global_keys])
            missing = global_vp_df[global_mask].copy()

            if len(missing) > 0:
                # Rename Global columns to match Regional schema
                missing = missing.rename(columns={"y_pred": "y_pred_combined"})
                if "y_pred_q10" in missing.columns:
                    missing = missing.rename(columns={
                        "y_pred_q10": "y_pred_combined_q10",
                        "y_pred_q90": "y_pred_combined_q90",
                    })
                missing["y_pred_global"] = missing["y_pred_combined"]
                val_preds = pd.concat([val_preds, missing], ignore_index=True)
                logging.info(f"[EXTREME] Supplemented {len(missing)} samples from Global "
                             f"(non-adopted regions). Total: {len(val_preds)}")
            del global_vp_df

        del regional_vp_parts
    elif global_vp.exists():
        val_preds = pd.read_parquet(global_vp)
        base_col = "y_pred"
        logging.info(f"[EXTREME] Using Global val_predictions — no Regional found "
                     f"({len(val_preds)} rows)")
    else:
        raise FileNotFoundError(
            f"No val_predictions found in {regional_dir}/*/val_predictions.parquet "
            f"or {global_vp}"
        )

    # ── Merge regiontype from training data ──
    df["time"] = pd.to_datetime(df["time"])
    val_preds["time"] = pd.to_datetime(val_preds["time"])

    # Ensure unique (time, grid50_id) → regiontype lookup
    regiontype_lookup = df[["time", "grid50_id", "regiontype"]].drop_duplicates(
        subset=["time", "grid50_id"], keep="first"
    )
    val_preds = val_preds.merge(regiontype_lookup, on=["time", "grid50_id"], how="left")
    val_preds = val_preds.dropna(subset=["regiontype"])
    logging.info(f"[EXTREME] After regiontype merge: {len(val_preds)} rows")

    # ── Merge features from training data ──
    features_df = df[["time", "grid50_id"] + global_features].drop_duplicates(
        subset=["time", "grid50_id"], keep="first"
    )
    val_preds = val_preds.merge(features_df, on=["time", "grid50_id"], how="inner")
    logging.info(f"[EXTREME] After features merge: {len(val_preds)} rows")

    # ── Average OOF predictions across CV folds ──
    # val_predictions contain one row per (sample × fold). The same (time, grid50_id)
    # appears in multiple folds with different y_pred values. We average them to get
    # a single, more robust OOF estimate per sample.
    n_before = len(val_preds)
    pred_cols = [c for c in val_preds.columns
                 if c.startswith("y_pred") or c == "residual"]
    agg_dict = {c: "mean" for c in pred_cols}
    agg_dict["y_true"] = "first"
    agg_dict["regiontype"] = "first"
    for feat in global_features:
        if feat in val_preds.columns:
            agg_dict[feat] = "first"
    val_preds = val_preds.groupby(["time", "grid50_id"], as_index=False).agg(agg_dict)
    logging.info(f"[EXTREME] Fold averaging: {n_before} → {len(val_preds)} "
                 f"(~{n_before / max(len(val_preds), 1):.1f} folds/sample)")

    logging.info(f"[EXTREME] Final dataset: {len(val_preds)} samples with features")

    # ── Train per region ──
    region_types = sorted(val_preds["regiontype"].unique())
    logging.info(f"[EXTREME] Found {len(region_types)} regions")

    trained_regions = []
    skipped_regions = []
    region_results = {}

    for region_type in region_types:
        region_str = str(int(region_type))
        tag = f"[EXTREME-{region_str}]"
        region_dir_i = extreme_dir / region_str
        region_dir_i.mkdir(parents=True, exist_ok=True)

        mask = val_preds["regiontype"] == region_type
        df_region = val_preds[mask].copy()
        n_samples = len(df_region)

        # ── Sample size check ──
        if n_samples < ext_cfg.get("min_samples", 5000):
            logging.info(f"{tag} SKIP: only {n_samples} samples")
            _save_extreme_skip(region_dir_i, region_str, n_samples, "insufficient_samples")
            skipped_regions.append(region_str)
            continue

        y_true = df_region["y_true"].values
        y_base = np.maximum(df_region[base_col].values, 0)
        X_raw = df_region[global_features].values

        base_r2 = r2_score(y_true, y_base)
        if base_r2 < ext_cfg.get("min_base_r2", 0.0):
            logging.info(f"{tag} SKIP: base R²={base_r2:.4f} too low")
            _save_extreme_skip(region_dir_i, region_str, n_samples, f"low_base_r2_{base_r2:.4f}")
            skipped_regions.append(region_str)
            continue

        # ── Identify extreme underestimates ──
        under_mask, ext_stats = _identify_extreme_underestimates(
            y_true, y_base,
            ext_cfg.get("ytrue_percentile", 85),
            ext_cfg.get("underestimate_threshold", 0.15),
        )

        if ext_stats["n_underestimate"] < ext_cfg.get("min_extreme_samples", 100):
            logging.info(f"{tag} SKIP: only {ext_stats['n_underestimate']} extreme underestimates")
            _save_extreme_skip(region_dir_i, region_str, n_samples,
                             f"insufficient_extremes_{ext_stats['n_underestimate']}")
            skipped_regions.append(region_str)
            continue

        logging.info(f"{tag} n={n_samples}, base_R²={base_r2:.4f}, "
                     f"underest={ext_stats['n_underestimate']} ({ext_stats['underestimate_pct']:.1f}%)")

        # ── Temporal CV splits ──
        dates = pd.to_datetime(df_region["time"])
        cv_splits = temporal_cv_splits(
            dates,
            test_months=ext_cfg.get("test_months", 24),
            min_train_months=ext_cfg.get("min_train_months", 12),
        )
        if not cv_splits:
            logging.info(f"{tag} SKIP: no valid CV splits")
            _save_extreme_skip(region_dir_i, region_str, n_samples, "no_cv_splits")
            skipped_regions.append(region_str)
            continue

        # ── Grid search (on first CV fold's train set) ──
        train_idx_0 = cv_splits[0][0]
        X_gs = X_raw[train_idx_0]
        y_gs_true = y_true[train_idx_0]
        y_gs_base = y_base[train_idx_0]

        under_gs, _ = _identify_extreme_underestimates(
            y_gs_true, y_gs_base,
            ext_cfg.get("ytrue_percentile", 85),
            ext_cfg.get("underestimate_threshold", 0.15),
        )
        residuals_gs = y_gs_true - y_gs_base

        # Inner CV for grid search (on the training subset)
        # NOTE: temporal_cv_splits may return empty if subset is too short
        # (e.g., 12 months can't split into 24-month test + 12-month train).
        # Fallback to KFold on the training subset — indices must be 0-based
        # relative to X_gs, NOT the full region.
        inner_splits = temporal_cv_splits(
            dates.iloc[train_idx_0],
            test_months=ext_cfg.get("test_months", 24),
            min_train_months=ext_cfg.get("min_train_months", 12),
        )
        if not inner_splits:
            n_gs = len(X_gs)
            n_inner_folds = min(3, n_gs // 1000)
            if n_inner_folds >= 2:
                inner_splits = list(KFold(
                    n_splits=n_inner_folds, shuffle=True, random_state=random_state,
                ).split(np.arange(n_gs)))
                logging.info(f"{tag} Inner temporal CV empty, using {n_inner_folds}-fold KFold "
                             f"on {n_gs} training samples")
            else:
                logging.info(f"{tag} SKIP: training subset too small for grid search ({n_gs})")
                _save_extreme_skip(region_dir_i, region_str, n_samples,
                                   f"train_subset_too_small_{n_gs}")
                skipped_regions.append(region_str)
                continue

        det_best, det_auc = _grid_search_detector(
            X_gs, under_gs, inner_splits,
            ext_cfg.get("detector_param_grid", {}),
            ext_cfg.get("detector_fixed_params", {}),
            device, random_state, tag,
        )

        # Corrector grid search (on extreme subset)
        ext_idx = np.where(under_gs)[0]
        if len(ext_idx) > 50:
            n_corr_folds = min(3, len(ext_idx) // 20)
            if n_corr_folds >= 2:
                inner_corr_splits = list(KFold(
                    n_splits=n_corr_folds, shuffle=True, random_state=random_state,
                ).split(ext_idx))
                inner_corr_splits = [
                    (ext_idx[tr], ext_idx[va]) for tr, va in inner_corr_splits
                ]
            else:
                inner_corr_splits = inner_splits
        else:
            inner_corr_splits = inner_splits

        corr_best, corr_rmse = _grid_search_corrector(
            X_gs, residuals_gs, inner_corr_splits,
            ext_cfg.get("corrector_param_grid", {}),
            ext_cfg.get("corrector_fixed_params", {}),
            device, random_state, tag,
        )

        # ── CV evaluation ──
        all_cv_metrics = []
        prob_threshold = ext_cfg.get("detector_prob_threshold", 0.6)
        max_corr_ratio = ext_cfg.get("max_correction_sample_ratio", 0.05)
        max_rel_corr = ext_cfg.get("max_relative_correction", 0.5)
        conservation_tol = ext_cfg.get("conservation_tolerance", 0.05)
        ytrue_pct = ext_cfg.get("ytrue_percentile", 85)

        det_full_params = {**ext_cfg.get("detector_fixed_params", {}), **det_best}
        corr_full_params = {**ext_cfg.get("corrector_fixed_params", {}), **corr_best}

        n_pos_all = int(np.sum(under_mask))
        n_neg_all = len(under_mask) - n_pos_all
        spw = n_neg_all / n_pos_all if n_pos_all > 0 else 1.0

        for fold_idx, (tr_idx, va_idx) in enumerate(cv_splits):
            X_tr, X_va = X_raw[tr_idx], X_raw[va_idx]
            y_tr_true, y_va_true = y_true[tr_idx], y_true[va_idx]
            y_tr_base, y_va_base = y_base[tr_idx], y_base[va_idx]

            # Train detector
            under_tr, _ = _identify_extreme_underestimates(
                y_tr_true, y_tr_base, ytrue_pct,
                ext_cfg.get("underestimate_threshold", 0.15),
            )
            if np.sum(under_tr) < 10:
                continue

            scaler_d = MinMaxScaler()
            X_tr_s = scaler_d.fit_transform(X_tr)
            X_va_s = scaler_d.transform(X_va)

            n_pos_tr = int(np.sum(under_tr))
            n_neg_tr = len(under_tr) - n_pos_tr
            spw_tr = n_neg_tr / n_pos_tr if n_pos_tr > 0 else 1.0

            detector = XGBClassifier(
                objective="binary:logistic", eval_metric="auc",
                tree_method="hist", device=device,
                random_state=random_state, verbosity=0,
                scale_pos_weight=spw_tr, **det_full_params,
            )
            detector.fit(X_tr_s, under_tr)

            # Train corrector (on extreme subset)
            resid_tr = y_tr_true - y_tr_base
            X_ext = X_tr_s[under_tr]
            r_ext = resid_tr[under_tr]

            corrector = XGBRegressor(
                objective="reg:squarederror", tree_method="hist",
                device=device, random_state=random_state, verbosity=0,
                **corr_full_params,
            )
            corrector.fit(X_ext, r_ext)

            # Evaluate on val fold
            det_probs = detector.predict_proba(X_va_s)[:, 1]
            corr_pred = corrector.predict(X_va_s)

            y_corrected, corr_stats = _apply_bounded_correction(
                y_va_base, det_probs, corr_pred,
                prob_threshold, max_corr_ratio, max_rel_corr, conservation_tol,
            )

            rmse_base = np.sqrt(mean_squared_error(y_va_true, y_va_base))
            rmse_final = np.sqrt(mean_squared_error(y_va_true, y_corrected))
            rmse_change_pct = (rmse_final - rmse_base) / rmse_base * 100 if rmse_base > 0 else 0

            r2_base_fold = r2_score(y_va_true, y_va_base)
            r2_final_fold = r2_score(y_va_true, y_corrected)

            # Extreme-specific improvement
            ext_threshold = np.percentile(y_va_true, ytrue_pct)
            ext_mask_va = y_va_true >= ext_threshold
            ext_improv = 0.0
            if np.sum(ext_mask_va) > 10:
                rmse_ext_base = np.sqrt(mean_squared_error(
                    y_va_true[ext_mask_va], y_va_base[ext_mask_va]))
                rmse_ext_final = np.sqrt(mean_squared_error(
                    y_va_true[ext_mask_va], y_corrected[ext_mask_va]))
                ext_improv = (rmse_ext_base - rmse_ext_final) / rmse_ext_base * 100 if rmse_ext_base > 0 else 0

            fold_metrics = {
                "rmse_base": float(rmse_base),
                "rmse_final": float(rmse_final),
                "rmse_change_pct": float(rmse_change_pct),
                "r2_base": float(r2_base_fold),
                "r2_final": float(r2_final_fold),
                "extreme_improvement_pct": float(ext_improv),
                "conservation_ratio": corr_stats["conservation_ratio"],
                "n_corrected": corr_stats["n_corrected"],
            }
            all_cv_metrics.append(fold_metrics)

            logging.info(f"{tag} Fold {fold_idx}: RMSE_Δ={rmse_change_pct:+.2f}%, "
                        f"ext_improv={ext_improv:.1f}%, "
                        f"conserv={corr_stats['conservation_ratio']:.3f}")

            del detector, corrector, scaler_d
            clear_memory()

        if not all_cv_metrics:
            logging.info(f"{tag} SKIP: all CV folds failed")
            _save_extreme_skip(region_dir_i, region_str, n_samples, "cv_failed")
            skipped_regions.append(region_str)
            continue

        # ── Adoption decision ──
        use_extreme, decision, breakdown = _decide_adoption(all_cv_metrics, ext_cfg, tag)

        # ── Train final models (if adopted) ──
        if use_extreme:
            final_scaler = MinMaxScaler().fit(X_raw)
            X_final = final_scaler.transform(X_raw)

            # Final detector
            final_detector = XGBClassifier(
                objective="binary:logistic", eval_metric="auc",
                tree_method="hist", device=device,
                random_state=random_state, verbosity=0,
                scale_pos_weight=spw, **det_full_params,
            )
            final_detector.fit(X_final, under_mask)

            # Final corrector
            residuals_all = y_true - y_base
            X_ext_all = X_final[under_mask]
            r_ext_all = residuals_all[under_mask]

            final_corrector = XGBRegressor(
                objective="reg:squarederror", tree_method="hist",
                device=device, random_state=random_state, verbosity=0,
                **corr_full_params,
            )
            final_corrector.fit(X_ext_all, r_ext_all)

            # Save models
            final_detector.save_model(str(region_dir_i / "detector.json"))
            final_corrector.save_model(str(region_dir_i / "corrector.json"))
            joblib.dump(final_scaler, region_dir_i / "scaler.joblib")
            with open(region_dir_i / "feature_names.json", "w") as f:
                json.dump(global_features, f)
            with open(region_dir_i / "detector_params.json", "w") as f:
                json.dump(clean_for_json(det_full_params), f, indent=2)
            with open(region_dir_i / "corrector_params.json", "w") as f:
                json.dump(clean_for_json(corr_full_params), f, indent=2)

            trained_regions.append(region_str)
        else:
            skipped_regions.append(region_str)

        # Save region summary
        result = clean_for_json({
            "region": region_str,
            "n_samples": n_samples,
            "base_r2": float(base_r2),
            "extreme_stats": ext_stats,
            "cv_metrics": all_cv_metrics,
            "use_extreme": use_extreme,
            "decision": decision,
            "score_breakdown": breakdown,
        })
        with open(region_dir_i / "region_summary.json", "w") as f:
            json.dump(result, f, indent=2)

        region_results[region_str] = result
        clear_memory()
        if gpu_available:
            clear_gpu_memory()

    # ── Crop summary ──
    summary = clean_for_json({
        "layer": "extreme",
        "crop": crop,
        "n_regions": len(region_types),
        "n_trained": len(trained_regions),
        "n_skipped": len(skipped_regions),
        "trained_regions": trained_regions,
        "skipped_regions": skipped_regions,
        "region_results": region_results,
        "timestamp": datetime.now().isoformat(),
    })

    with open(extreme_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logging.info(f"[EXTREME] Done. {len(trained_regions)}/{len(region_types)} regions adopted")
    return summary


def _save_extreme_skip(region_dir, region_str, n_samples, reason):
    """Save a skip marker for regions that didn't qualify."""
    summary = {
        "region": region_str,
        "use_extreme": False,
        "reason": reason,
        "n_samples": n_samples,
    }
    with open(region_dir / "region_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


# Weighted-base variants (GW+RW base instead of G+R).
# The two functions below mirror the unweighted implementation above
# but consume base predictions from the GW (Global Weighted) and RW
# (Regional Weighted) pipelines. They are invoked by src/train/run_extreme.py
# when --weighted is passed.


def load_gw_rw_base(crop, df, result_root):
    """
    Load GW + RW models and predict on df.
    Returns (y_base, feature_names).
    """
    gw_dir = result_root / crop / "global_weighted"
    rw_dir = result_root / crop / "regional_weighted"

    with open(gw_dir / "feature_names.json") as f:
        features = json.load(f)

    gw_scaler = joblib.load(gw_dir / "scaler.joblib")
    gw_model = XGBRegressor()
    gw_model.load_model(str(gw_dir / "model_q50.json"))

    X = df[features].values
    X_s = gw_scaler.transform(X)
    y_gw = np.maximum(gw_model.predict(X_s), 0)

    # RW
    y_base = y_gw.copy()
    rw_summary_path = rw_dir / "summary.json"
    if rw_summary_path.exists() and "regiontype" in df.columns:
        with open(rw_summary_path) as f:
            adopted = json.load(f).get("trained_regions", [])
        for region_str in adopted:
            rdir = rw_dir / region_str
            if not (rdir / "model_q50.json").exists():
                continue
            r_scaler = joblib.load(rdir / "scaler.joblib")
            r_model = XGBRegressor()
            r_model.load_model(str(rdir / "model_q50.json"))
            region_val = int(region_str) if region_str.isdigit() else region_str
            mask = df["regiontype"].values == region_val
            if mask.sum() == 0:
                continue
            r_pred = r_model.predict(r_scaler.transform(X_s[mask]))
            y_base[mask] = np.maximum(y_gw[mask] + r_pred, 0)
            del r_model, r_scaler

    base_r2 = r2_score(df[crop].values, y_base)
    logging.info(f"[BASE] GW+RW: mean={y_base.mean():.0f}, R²={base_r2:.4f}")
    del gw_model, gw_scaler
    gc.collect()

    return y_base, features


def train_extreme_weighted(cfg, crop, df, result_root, extreme_dir):
    """Train Extreme layer using GW+RW as base. Same logic as extreme_layer.py."""
    ext_cfg = cfg.extreme_layer
    gpu_available, _, device = setup_gpu_environment(cfg.common.use_gpu)
    random_state = cfg.common.random_state

    extreme_dir.mkdir(parents=True, exist_ok=True)

    # ── Get GW+RW base predictions ──
    y_base_all, global_features = load_gw_rw_base(crop, df, result_root)
    y_true_all = df[crop].values

    if "regiontype" not in df.columns:
        return {"status": "no_regiontype"}

    region_types = sorted(df["regiontype"].unique())
    logging.info(f"[EXTREME-W] {len(region_types)} regions")

    trained_regions, skipped_regions, region_results = [], [], {}

    ytrue_pct = ext_cfg.get("ytrue_percentile", 85)
    under_thresh = ext_cfg.get("underestimate_threshold", 0.15)
    prob_threshold = ext_cfg.get("detector_prob_threshold", 0.6)
    max_corr_ratio = ext_cfg.get("max_correction_sample_ratio", 0.05)
    max_rel_corr = ext_cfg.get("max_relative_correction", 0.5)
    conservation_tol = ext_cfg.get("conservation_tolerance", 0.05)

    for region_type in region_types:
        region_str = str(int(region_type))
        tag = f"[EXT-W-{region_str}]"
        region_dir_i = extreme_dir / region_str
        region_dir_i.mkdir(parents=True, exist_ok=True)

        mask = df["regiontype"].values == region_type
        n_samples = int(mask.sum())

        # ── Skip checks ──
        if n_samples < ext_cfg.get("min_samples", 5000):
            _save_extreme_skip(region_dir_i, region_str, n_samples, "insufficient_samples")
            skipped_regions.append(region_str)
            continue

        y_true = y_true_all[mask]
        y_base = y_base_all[mask]
        X_raw = df[global_features].values[mask]

        base_r2 = r2_score(y_true, y_base)
        if base_r2 < ext_cfg.get("min_base_r2", 0.0):
            _save_extreme_skip(region_dir_i, region_str, n_samples, f"low_base_r2_{base_r2:.4f}")
            skipped_regions.append(region_str)
            continue

        under_mask, ext_stats = _identify_extreme_underestimates(
            y_true, y_base, ytrue_pct, under_thresh)

        if ext_stats["n_underestimate"] < ext_cfg.get("min_extreme_samples", 100):
            _save_extreme_skip(region_dir_i, region_str, n_samples,
                             f"insufficient_extremes_{ext_stats['n_underestimate']}")
            skipped_regions.append(region_str)
            continue

        logging.info(f"{tag} n={n_samples}, base_R²={base_r2:.4f}, "
                     f"underest={ext_stats['n_underestimate']} ({ext_stats['underestimate_pct']:.1f}%)")

        # ── Temporal CV ──
        dates = pd.to_datetime(df["time"].values[mask])
        dates = pd.Series(dates)
        cv_splits = temporal_cv_splits(
            dates,
            test_months=ext_cfg.get("test_months", 24),
            min_train_months=ext_cfg.get("min_train_months", 12),
        )
        if not cv_splits:
            _save_extreme_skip(region_dir_i, region_str, n_samples, "no_cv_splits")
            skipped_regions.append(region_str)
            continue

        # ── Grid search on first fold's training set ──
        tr0 = cv_splits[0][0]
        X_gs, y_gs_true, y_gs_base = X_raw[tr0], y_true[tr0], y_base[tr0]
        under_gs, _ = _identify_extreme_underestimates(y_gs_true, y_gs_base, ytrue_pct, under_thresh)
        residuals_gs = y_gs_true - y_gs_base

        # Inner CV
        inner_splits = temporal_cv_splits(
            dates.iloc[tr0],
            test_months=ext_cfg.get("test_months", 24),
            min_train_months=ext_cfg.get("min_train_months", 12),
        )
        if not inner_splits:
            n_gs = len(X_gs)
            n_folds = min(3, n_gs // 1000)
            if n_folds >= 2:
                inner_splits = list(KFold(n_splits=n_folds, shuffle=True,
                                          random_state=random_state).split(np.arange(n_gs)))
            else:
                _save_extreme_skip(region_dir_i, region_str, n_samples, f"train_too_small_{n_gs}")
                skipped_regions.append(region_str)
                continue

        det_best, _ = _grid_search_detector(
            X_gs, under_gs, inner_splits,
            ext_cfg.get("detector_param_grid", {}),
            ext_cfg.get("detector_fixed_params", {}),
            device, random_state, tag)

        # Corrector grid search
        ext_idx = np.where(under_gs)[0]
        if len(ext_idx) > 50:
            nc = min(3, len(ext_idx) // 20)
            if nc >= 2:
                inner_corr = [(ext_idx[t], ext_idx[v])
                              for t, v in KFold(n_splits=nc, shuffle=True,
                                                random_state=random_state).split(ext_idx)]
            else:
                inner_corr = inner_splits
        else:
            inner_corr = inner_splits

        corr_best, _ = _grid_search_corrector(
            X_gs, residuals_gs, inner_corr,
            ext_cfg.get("corrector_param_grid", {}),
            ext_cfg.get("corrector_fixed_params", {}),
            device, random_state, tag)

        # ── CV evaluation ──
        det_full = {**ext_cfg.get("detector_fixed_params", {}), **det_best}
        corr_full = {**ext_cfg.get("corrector_fixed_params", {}), **corr_best}
        spw = (len(under_mask) - np.sum(under_mask)) / max(np.sum(under_mask), 1)

        all_cv_metrics = []
        for fi, (tr_idx, va_idx) in enumerate(cv_splits):
            X_tr, X_va = X_raw[tr_idx], X_raw[va_idx]
            y_tr_t, y_va_t = y_true[tr_idx], y_true[va_idx]
            y_tr_b, y_va_b = y_base[tr_idx], y_base[va_idx]

            u_tr, _ = _identify_extreme_underestimates(y_tr_t, y_tr_b, ytrue_pct, under_thresh)
            if np.sum(u_tr) < 10:
                continue

            sc = MinMaxScaler()
            Xts = sc.fit_transform(X_tr)
            Xvs = sc.transform(X_va)

            spw_tr = (len(u_tr) - np.sum(u_tr)) / max(np.sum(u_tr), 1)
            det = XGBClassifier(objective="binary:logistic", eval_metric="auc",
                                tree_method="hist", device=device,
                                random_state=random_state, verbosity=0,
                                scale_pos_weight=spw_tr, **det_full)
            det.fit(Xts, u_tr)

            cor = XGBRegressor(objective="reg:squarederror", tree_method="hist",
                               device=device, random_state=random_state, verbosity=0,
                               **corr_full)
            cor.fit(Xts[u_tr], (y_tr_t - y_tr_b)[u_tr])

            dp = det.predict_proba(Xvs)[:, 1]
            cp = cor.predict(Xvs)
            y_corr, cs = _apply_bounded_correction(
                y_va_b, dp, cp, prob_threshold, max_corr_ratio, max_rel_corr, conservation_tol)

            rmse_b = np.sqrt(mean_squared_error(y_va_t, y_va_b))
            rmse_f = np.sqrt(mean_squared_error(y_va_t, y_corr))

            ext_thr = np.percentile(y_va_t, ytrue_pct)
            em = y_va_t >= ext_thr
            ei = 0.0
            if em.sum() > 10:
                rb = np.sqrt(mean_squared_error(y_va_t[em], y_va_b[em]))
                rf = np.sqrt(mean_squared_error(y_va_t[em], y_corr[em]))
                ei = (rb - rf) / rb * 100 if rb > 0 else 0

            all_cv_metrics.append({
                "rmse_base": float(rmse_b), "rmse_final": float(rmse_f),
                "rmse_change_pct": float((rmse_f - rmse_b) / rmse_b * 100) if rmse_b > 0 else 0,
                "r2_base": float(r2_score(y_va_t, y_va_b)),
                "r2_final": float(r2_score(y_va_t, y_corr)),
                "extreme_improvement_pct": float(ei),
                "conservation_ratio": cs["conservation_ratio"],
                "n_corrected": cs["n_corrected"],
            })
            logging.info(f"{tag} Fold {fi}: RMSE_Δ={(rmse_f-rmse_b)/rmse_b*100:+.2f}%, ext={ei:.1f}%")
            del det, cor, sc; clear_memory()

        if not all_cv_metrics:
            _save_extreme_skip(region_dir_i, region_str, n_samples, "cv_failed")
            skipped_regions.append(region_str)
            continue

        # ── Adoption ──
        use_ext, decision, breakdown = _decide_adoption(all_cv_metrics, ext_cfg, tag)

        if use_ext:
            fsc = MinMaxScaler().fit(X_raw)
            Xf = fsc.transform(X_raw)
            fd = XGBClassifier(objective="binary:logistic", eval_metric="auc",
                               tree_method="hist", device=device,
                               random_state=random_state, verbosity=0,
                               scale_pos_weight=spw, **det_full)
            fd.fit(Xf, under_mask)
            fc = XGBRegressor(objective="reg:squarederror", tree_method="hist",
                              device=device, random_state=random_state, verbosity=0,
                              **corr_full)
            fc.fit(Xf[under_mask], (y_true - y_base)[under_mask])

            fd.save_model(str(region_dir_i / "detector.json"))
            fc.save_model(str(region_dir_i / "corrector.json"))
            joblib.dump(fsc, region_dir_i / "scaler.joblib")
            with open(region_dir_i / "feature_names.json", "w") as f:
                json.dump(global_features, f)
            with open(region_dir_i / "detector_params.json", "w") as f:
                json.dump(clean_for_json(det_full), f, indent=2)
            with open(region_dir_i / "corrector_params.json", "w") as f:
                json.dump(clean_for_json(corr_full), f, indent=2)
            trained_regions.append(region_str)
        else:
            skipped_regions.append(region_str)

        with open(region_dir_i / "region_summary.json", "w") as f:
            json.dump(clean_for_json({
                "region": region_str, "base_model": "GW+RW",
                "n_samples": n_samples, "base_r2": float(base_r2),
                "extreme_stats": ext_stats, "cv_metrics": all_cv_metrics,
                "use_extreme": use_ext, "decision": decision,
                "score_breakdown": breakdown,
            }), f, indent=2)
        region_results[region_str] = {"use_extreme": use_ext, "decision": decision}
        clear_memory()

    summary = clean_for_json({
        "layer": "extreme", "base_model": "GW+RW", "crop": crop,
        "n_regions": len(region_types),
        "n_trained": len(trained_regions), "n_skipped": len(skipped_regions),
        "trained_regions": trained_regions, "skipped_regions": skipped_regions,
        "timestamp": datetime.now().isoformat(),
    })
    with open(extreme_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"[EXTREME-W] {len(trained_regions)}/{len(region_types)} adopted")
    return summary
