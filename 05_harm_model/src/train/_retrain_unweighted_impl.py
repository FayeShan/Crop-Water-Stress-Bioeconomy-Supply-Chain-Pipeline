"""HARM final retrain (unweighted): train G/R/Extreme on 2000-2018 using eval-phase decisions and params."""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor, XGBClassifier

from harm.config import load_config
from harm.utils import (
    load_crop_data, get_feature_columns, setup_logging,
    setup_gpu_environment, clear_memory, clear_gpu_memory, clean_for_json,
)
from harm.training import train_quantile_models, predict_with_quantiles
from harm.extreme_layer import _identify_extreme_underestimates, _apply_bounded_correction

from harm.constants import ALL_CROPS

TRAIN_YEARS = list(range(2000, 2019))  # 2000-2018


def retrain_global(crop, df, feature_cols, best_params, output_dir,
                   quantiles, device, random_state):
    """Train final Global model on full 2000-2018 data."""
    output_dir.mkdir(parents=True, exist_ok=True)

    X = df[feature_cols]
    y = df[crop]

    scaler = MinMaxScaler()
    X_s = scaler.fit_transform(X)

    models = train_quantile_models(X_s, y.values, best_params, device, quantiles, random_state)

    for q, model in models.items():
        model.save_model(str(output_dir / f"model_q{int(q*100)}.json"))
    joblib.dump(scaler, output_dir / "scaler.joblib")

    with open(output_dir / "feature_names.json", "w") as f:
        json.dump(feature_cols, f)
    with open(output_dir / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    y_pred = np.maximum(models[0.5].predict(X_s), 0)
    train_r2 = float(r2_score(y.values, y_pred))
    train_rmse = float(np.sqrt(mean_squared_error(y.values, y_pred)))

    summary = {
        "crop": crop,
        "layer": "global",
        "mode": "retrain_00-18",
        "n_samples": len(df),
        "n_features": len(feature_cols),
        "train_years": f"{TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]}",
        "train_r2": train_r2,
        "train_rmse": train_rmse,
        "best_params": best_params,
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def generate_global_val_predictions(global_dir, df, crop, feature_cols, quantiles):
    """Generate val_predictions.parquet for downstream layers.

    Retrain uses full 00-18 (no CV), so the model predicts its own training
    data: slight information leakage, but acceptable for the final production
    model since eval-phase decisions were made with proper OOF.
    """
    scaler = joblib.load(global_dir / "scaler.joblib")
    X_s = scaler.transform(df[feature_cols])
    y_true = df[crop].values

    models = {}
    for q in quantiles:
        m = XGBRegressor()
        m.load_model(str(global_dir / f"model_q{int(q*100)}.json"))
        models[q] = m

    y_pred = np.maximum(models[0.5].predict(X_s), 0)
    y_q10 = np.maximum(models[0.1].predict(X_s), 0)
    y_q90 = np.maximum(models[0.9].predict(X_s), 0)

    val_df = pd.DataFrame({
        "fold": 0,
        "time": pd.to_datetime(df["time"]).values,
        "grid50_id": df["grid50_id"].values,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_pred_q10": y_q10,
        "y_pred_q90": y_q90,
        "residual": y_true - y_pred,
    })
    val_df.to_parquet(global_dir / "val_predictions.parquet", index=False)
    logging.info(f"  Global val_predictions: {len(val_df):,} rows, "
                 f"mean_residual={np.mean(y_true - y_pred):.0f}, "
                 f"R²={r2_score(y_true, y_pred):.4f}")


def retrain_regional(crop, df, feature_cols, eval_root, output_root,
                     quantiles, device, random_state):
    """Retrain Regional layer for adopted regions only, reusing eval adoption decisions and best_params."""
    eval_regional_dir = eval_root / crop / "regional"
    output_global_dir = output_root / crop / "global"
    output_regional_dir = output_root / crop / "regional"
    output_regional_dir.mkdir(parents=True, exist_ok=True)

    eval_summary_path = eval_regional_dir / "summary.json"
    if not eval_summary_path.exists():
        logging.warning(f"  [REGIONAL] No eval summary at {eval_summary_path}, skipping")
        return {"status": "no_eval_summary"}

    with open(eval_summary_path) as f:
        eval_summary = json.load(f)

    adopted_regions = eval_summary.get("trained_regions", [])
    if not adopted_regions:
        logging.info(f"  [REGIONAL] No regions adopted in eval, skipping")
        _save_regional_summary(output_regional_dir, crop, [], eval_summary.get("skipped_regions", []))
        return {"status": "no_adopted_regions"}

    logging.info(f"  [REGIONAL] Eval adopted {len(adopted_regions)} regions: {adopted_regions}")

    global_scaler = joblib.load(output_global_dir / "scaler.joblib")
    global_model = XGBRegressor()
    global_model.load_model(str(output_global_dir / "model_q50.json"))
    with open(output_global_dir / "feature_names.json") as f:
        global_features = json.load(f)

    # Load quantile models for combined predictions
    global_q10 = XGBRegressor()
    global_q10.load_model(str(output_global_dir / "model_q10.json"))
    global_q90 = XGBRegressor()
    global_q90.load_model(str(output_global_dir / "model_q90.json"))

    df["time"] = pd.to_datetime(df["time"])
    X_all = df[global_features].values
    X_all_scaled = global_scaler.transform(X_all)
    y_all = df[crop].values
    global_pred = np.maximum(global_model.predict(X_all_scaled), 0)
    global_q10_pred = np.maximum(global_q10.predict(X_all_scaled), 0)
    global_q90_pred = np.maximum(global_q90.predict(X_all_scaled), 0)
    residuals_all = y_all - global_pred

    if "regiontype" not in df.columns:
        raise ValueError("No 'regiontype' column in training data")

    trained_regions = []
    skipped_regions = []

    for region_str in adopted_regions:
        region_type = int(region_str) if region_str.isdigit() else region_str
        mask = df["regiontype"] == region_type
        n_region = int(mask.sum())

        if n_region == 0:
            logging.warning(f"  [REGIONAL-{region_str}] No samples in retrain data, skipping")
            skipped_regions.append(region_str)
            continue

        eval_params_path = eval_regional_dir / region_str / "best_params.json"
        if not eval_params_path.exists():
            # Try region_summary.json for params
            eval_rs_path = eval_regional_dir / region_str / "region_summary.json"
            if eval_rs_path.exists():
                with open(eval_rs_path) as f:
                    rs = json.load(f)
                best_params = rs.get("best_params", {})
            else:
                logging.warning(f"  [REGIONAL-{region_str}] No params found in eval, skipping")
                skipped_regions.append(region_str)
                continue
        else:
            with open(eval_params_path) as f:
                best_params = json.load(f)

        if not best_params:
            logging.warning(f"  [REGIONAL-{region_str}] Empty best_params, skipping")
            skipped_regions.append(region_str)
            continue

        logging.info(f"  [REGIONAL-{region_str}] Training (n={n_region}, params={best_params})")

        X_region = X_all_scaled[mask]
        residuals_region = residuals_all[mask]

        # Regional has its own scaler on top of Global's scaled features
        region_scaler = MinMaxScaler()
        X_region_s = region_scaler.fit_transform(X_region)

        r_models = train_quantile_models(
            X_region_s, residuals_region, best_params,
            device, quantiles, random_state,
        )

        region_dir = output_regional_dir / region_str
        region_dir.mkdir(parents=True, exist_ok=True)

        for q, model in r_models.items():
            model.save_model(str(region_dir / f"model_q{int(q*100)}.json"))
        joblib.dump(region_scaler, region_dir / "scaler.joblib")

        with open(region_dir / "best_params.json", "w") as f:
            json.dump(best_params, f, indent=2)

        # Generate val_predictions for this region (for Extreme layer)
        r_preds = predict_with_quantiles(r_models, X_region_s)
        y_combined = np.maximum(global_pred[mask] + r_preds[0.5], 0)
        y_combined_q10 = np.maximum(global_q10_pred[mask] + r_preds.get(0.1, r_preds[0.5]), 0)
        y_combined_q90 = np.maximum(global_q90_pred[mask] + r_preds.get(0.9, r_preds[0.5]), 0)

        val_df = pd.DataFrame({
            "fold": 0,
            "time": df.loc[mask, "time"].values,
            "grid50_id": df.loc[mask, "grid50_id"].values,
            "y_true": y_all[mask],
            "y_pred_global": global_pred[mask],
            "y_pred_combined": y_combined,
            "y_pred_combined_q10": y_combined_q10,
            "y_pred_combined_q90": y_combined_q90,
            "residual": y_all[mask] - y_combined,
        })
        val_df.to_parquet(region_dir / "val_predictions.parquet", index=False)

        combined_r2 = float(r2_score(y_all[mask], y_combined))
        global_r2 = float(r2_score(y_all[mask], global_pred[mask]))

        with open(region_dir / "region_summary.json", "w") as f:
            json.dump({
                "region": region_str,
                "use_regional": True,
                "mode": "retrain_00-18",
                "n_samples": n_region,
                "global_r2_train": global_r2,
                "combined_r2_train": combined_r2,
                "best_params": best_params,
            }, f, indent=2)

        trained_regions.append(region_str)
        logging.info(f"  [REGIONAL-{region_str}] ✓ Global R²={global_r2:.4f} → G+R R²={combined_r2:.4f}")

        del r_models, region_scaler
        clear_memory()

    # Also note regions that were skipped/rejected in eval
    all_skipped = list(set(
        eval_summary.get("skipped_regions", []) + skipped_regions
    ))

    # Write skip markers for non-adopted regions
    for region_str in all_skipped:
        region_dir = output_regional_dir / region_str
        region_dir.mkdir(parents=True, exist_ok=True)
        with open(region_dir / "region_summary.json", "w") as f:
            json.dump({
                "region": region_str,
                "use_regional": False,
                "reason": "not_adopted_in_eval",
            }, f, indent=2)

    summary = _save_regional_summary(output_regional_dir, crop, trained_regions, all_skipped)
    return summary


def _save_regional_summary(regional_dir, crop, trained_regions, skipped_regions):
    regional_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "layer": "regional",
        "crop": crop,
        "mode": "retrain_00-18",
        "n_trained": len(trained_regions),
        "n_skipped": len(skipped_regions),
        "trained_regions": trained_regions,
        "skipped_regions": skipped_regions,
        "timestamp": datetime.now().isoformat(),
    }
    with open(regional_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def retrain_extreme(crop, df, feature_cols, eval_root, output_root,
                    ext_cfg, device, random_state):
    """Retrain Extreme layer for adopted regions only; y_base comes from retrained Global+Regional predictions."""
    eval_extreme_dir = eval_root / crop / "extreme"
    output_global_dir = output_root / crop / "global"
    output_regional_dir = output_root / crop / "regional"
    output_extreme_dir = output_root / crop / "extreme"
    output_extreme_dir.mkdir(parents=True, exist_ok=True)

    eval_summary_path = eval_extreme_dir / "summary.json"
    if not eval_summary_path.exists():
        logging.warning(f"  [EXTREME] No eval summary at {eval_summary_path}, skipping")
        return {"status": "no_eval_summary"}

    with open(eval_summary_path) as f:
        eval_summary = json.load(f)

    adopted_regions = eval_summary.get("trained_regions", [])
    if not adopted_regions:
        logging.info(f"  [EXTREME] No regions adopted in eval, skipping")
        _save_extreme_summary(output_extreme_dir, crop, [], eval_summary.get("skipped_regions", []))
        return {"status": "no_adopted_regions"}

    logging.info(f"  [EXTREME] Eval adopted {len(adopted_regions)} regions: {adopted_regions}")

    global_scaler = joblib.load(output_global_dir / "scaler.joblib")
    global_model = XGBRegressor()
    global_model.load_model(str(output_global_dir / "model_q50.json"))
    with open(output_global_dir / "feature_names.json") as f:
        global_features = json.load(f)

    # Load retrained Regional models (for adopted regions)
    regional_models = {}
    regional_scalers = {}
    regional_summary_path = output_regional_dir / "summary.json"

    if regional_summary_path.exists():
        with open(regional_summary_path) as f:
            reg_summary = json.load(f)
        for r_str in reg_summary.get("trained_regions", []):
            r_model_path = output_regional_dir / r_str / "model_q50.json"
            r_scaler_path = output_regional_dir / r_str / "scaler.joblib"
            if r_model_path.exists() and r_scaler_path.exists():
                m = XGBRegressor()
                m.load_model(str(r_model_path))
                regional_models[r_str] = m
                regional_scalers[r_str] = joblib.load(r_scaler_path)

    logging.info(f"  [EXTREME] Loaded {len(regional_models)} retrained Regional models")

    df["time"] = pd.to_datetime(df["time"])
    X_all = df[global_features].values
    X_all_gs = global_scaler.transform(X_all)  # Global-scaled
    y_all = df[crop].values
    global_pred = np.maximum(global_model.predict(X_all_gs), 0)

    if "regiontype" not in df.columns:
        raise ValueError("No 'regiontype' column in training data")

    # Compute G+R predictions per sample
    y_base_all = global_pred.copy()
    for r_str, r_model in regional_models.items():
        region_type = int(r_str) if r_str.isdigit() else r_str
        r_mask = (df["regiontype"] == region_type).values
        if r_mask.sum() > 0:
            X_r_gs = X_all_gs[r_mask]
            X_r_s = regional_scalers[r_str].transform(X_r_gs)
            r_pred = r_model.predict(X_r_s)
            y_base_all[r_mask] = np.maximum(global_pred[r_mask] + r_pred, 0)

    ytrue_pct = ext_cfg.get("ytrue_percentile", 85)
    under_thresh = ext_cfg.get("underestimate_threshold", 0.15)
    prob_threshold = ext_cfg.get("detector_prob_threshold", 0.6)
    max_corr_ratio = ext_cfg.get("max_correction_sample_ratio", 0.05)
    max_rel_corr = ext_cfg.get("max_relative_correction", 0.5)
    conservation_tol = ext_cfg.get("conservation_tolerance", 0.05)

    trained_regions = []
    skipped_regions = []

    for region_str in adopted_regions:
        region_type = int(region_str) if region_str.isdigit() else region_str
        tag = f"[EXTREME-{region_str}]"
        mask = (df["regiontype"] == region_type).values
        n_region = int(mask.sum())

        if n_region == 0:
            logging.warning(f"  {tag} No samples in retrain data, skipping")
            skipped_regions.append(region_str)
            continue

        eval_region_dir = eval_extreme_dir / region_str
        det_params_path = eval_region_dir / "detector_params.json"
        corr_params_path = eval_region_dir / "corrector_params.json"

        if not det_params_path.exists() or not corr_params_path.exists():
            logging.warning(f"  {tag} No eval params found, skipping")
            skipped_regions.append(region_str)
            continue

        with open(det_params_path) as f:
            det_params = json.load(f)
        with open(corr_params_path) as f:
            corr_params = json.load(f)

        y_true = y_all[mask]
        y_base = y_base_all[mask]
        X_raw = X_all[mask]

        base_r2 = float(r2_score(y_true, y_base))
        logging.info(f"  {tag} n={n_region}, base_R²={base_r2:.4f}")

        under_mask, ext_stats = _identify_extreme_underestimates(
            y_true, y_base, ytrue_pct, under_thresh,
        )

        n_extreme = ext_stats["n_underestimate"]
        if n_extreme < 10:
            logging.warning(f"  {tag} Too few extremes ({n_extreme}), cannot train")
            skipped_regions.append(region_str)
            _save_extreme_region_summary(output_extreme_dir / region_str,
                                        region_str, n_region, False,
                                        f"too_few_extremes_retrain_{n_extreme}")
            continue

        if n_extreme < ext_cfg.get("min_extreme_samples", 100):
            logging.warning(f"  {tag} Only {n_extreme} extreme samples (below threshold), "
                          f"but training anyway — eval adopted this region")

        logging.info(f"  {tag} Underestimates: {n_extreme} ({ext_stats['underestimate_pct']:.1f}%)")

        extreme_scaler = MinMaxScaler()
        X_scaled = extreme_scaler.fit_transform(X_raw)

        n_pos = int(np.sum(under_mask))
        n_neg = len(under_mask) - n_pos
        spw = n_neg / n_pos if n_pos > 0 else 1.0

        detector = XGBClassifier(
            objective="binary:logistic", eval_metric="auc",
            tree_method="hist", device=device,
            random_state=random_state, verbosity=0,
            scale_pos_weight=spw, **det_params,
        )
        detector.fit(X_scaled, under_mask)

        # Train corrector on the extreme subset only
        residuals = y_true - y_base
        X_ext = X_scaled[under_mask]
        r_ext = residuals[under_mask]

        corrector = XGBRegressor(
            objective="reg:squarederror", tree_method="hist",
            device=device, random_state=random_state, verbosity=0,
            **corr_params,
        )
        corrector.fit(X_ext, r_ext)

        # Quick train-set evaluation (diagnostics only)
        det_probs = detector.predict_proba(X_scaled)[:, 1]
        corr_pred = corrector.predict(X_scaled)
        y_corrected, corr_stats = _apply_bounded_correction(
            y_base, det_probs, corr_pred,
            prob_threshold, max_corr_ratio, max_rel_corr, conservation_tol,
        )
        final_r2 = float(r2_score(y_true, y_corrected))
        final_rmse = float(np.sqrt(mean_squared_error(y_true, y_corrected)))
        base_rmse = float(np.sqrt(mean_squared_error(y_true, y_base)))

        logging.info(f"  {tag} ✓ Train: R² {base_r2:.4f}→{final_r2:.4f}, "
                     f"RMSE {base_rmse:.0f}→{final_rmse:.0f}, "
                     f"corrected={corr_stats['n_corrected']}, "
                     f"conservation={corr_stats['conservation_ratio']:.3f}")

        region_dir = output_extreme_dir / region_str
        region_dir.mkdir(parents=True, exist_ok=True)

        detector.save_model(str(region_dir / "detector.json"))
        corrector.save_model(str(region_dir / "corrector.json"))
        joblib.dump(extreme_scaler, region_dir / "scaler.joblib")
        with open(region_dir / "feature_names.json", "w") as f:
            json.dump(global_features, f)
        with open(region_dir / "detector_params.json", "w") as f:
            json.dump(clean_for_json(det_params), f, indent=2)
        with open(region_dir / "corrector_params.json", "w") as f:
            json.dump(clean_for_json(corr_params), f, indent=2)

        _save_extreme_region_summary(
            region_dir, region_str, n_region, True,
            "retrained_with_eval_params",
            extra={
                "base_r2_train": base_r2,
                "final_r2_train": final_r2,
                "base_rmse_train": base_rmse,
                "final_rmse_train": final_rmse,
                "n_corrected": corr_stats["n_corrected"],
                "conservation_ratio": corr_stats["conservation_ratio"],
                "extreme_stats": clean_for_json(ext_stats),
            },
        )

        trained_regions.append(region_str)
        del detector, corrector, extreme_scaler
        clear_memory()

    # Write skip markers for non-adopted regions
    all_skipped = list(set(
        eval_summary.get("skipped_regions", []) + skipped_regions
    ))
    for r_str in all_skipped:
        _save_extreme_region_summary(
            output_extreme_dir / r_str, r_str, 0, False, "not_adopted_in_eval",
        )

    summary = _save_extreme_summary(output_extreme_dir, crop, trained_regions, all_skipped)
    return summary


def _save_extreme_region_summary(region_dir, region_str, n_samples, use_extreme, reason, extra=None):
    region_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "region": region_str,
        "use_extreme": use_extreme,
        "reason": reason,
        "n_samples": n_samples,
        "mode": "retrain_00-18",
    }
    if extra:
        result.update(extra)
    with open(region_dir / "region_summary.json", "w") as f:
        json.dump(clean_for_json(result), f, indent=2)


def _save_extreme_summary(extreme_dir, crop, trained_regions, skipped_regions):
    extreme_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "layer": "extreme",
        "crop": crop,
        "mode": "retrain_00-18",
        "n_trained": len(trained_regions),
        "n_skipped": len(skipped_regions),
        "trained_regions": trained_regions,
        "skipped_regions": skipped_regions,
        "timestamp": datetime.now().isoformat(),
    }
    with open(extreme_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description="HARM Final Retrain (00-18)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--phase", choices=["eval", "final"], default="eval", help="Which config phase block to use (default: eval)")
    parser.add_argument("--crops", nargs="+", default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--eval-root", type=str, default=None,
                        help="Where to read eval decisions/params from (default: HARM_v2)")
    parser.add_argument("--output-root", type=str, default=None,
                        help="Where to save retrained models (default: HARM_v2_final)")
    parser.add_argument("--layers", nargs="+", default=["global"],
                        choices=["global", "regional", "extreme", "all"],
                        help="Which layers to retrain")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config, phase=args.phase if hasattr(args, 'phase') else 'eval')
    crops = args.crops or cfg.crops

    data_root = Path(args.data_root) if args.data_root else cfg.paths.data_root

    # Eval root: where 00-16 results live (decisions + params)
    eval_root = Path(args.eval_root) if args.eval_root else Path(
        str(cfg.paths.result_root).replace("/final", "/eval")
    )
    # Output root: where 00-18 retrained models go
    output_root = Path(args.output_root) if args.output_root else cfg.paths.result_root

    layers = args.layers
    if "all" in layers:
        layers = ["global", "regional", "extreme"]
    layer_order = ["global", "regional", "extreme"]
    layers = [l for l in layer_order if l in layers]

    gpu_available, n_gpus, device = setup_gpu_environment(cfg.common.use_gpu)
    quantiles = cfg.common.quantiles
    random_state = cfg.common.random_state

    print("=" * 60)
    print("HARM Final Retrain (2000-2018)")
    print(f"  Layers:      {' → '.join(layers)}")
    print(f"  Crops:       {crops}")
    print(f"  Data:        {data_root}")
    print(f"  Eval from:   {eval_root}")
    print(f"  Output:      {output_root}")
    print(f"  GPU:         {device}")
    print("=" * 60)

    results = {}

    for crop in crops:
        print(f"\n{'━' * 50}")
        print(f"  {crop}")
        print(f"{'━' * 50}")

        crop_root = output_root / crop
        global_dir = crop_root / "global"
        regional_dir = crop_root / "regional"
        extreme_dir = crop_root / "extreme"

        setup_logging(crop_root / "logs", crop)

        try:
            df = load_crop_data(data_root, crop, years=TRAIN_YEARS)
            feature_cols = get_feature_columns(df, crop, cfg.features)
            print(f"  Data: {len(df):,} rows, {len(feature_cols)} features")

            if "global" in layers:
                if not args.force and (global_dir / "summary.json").exists():
                    print(f"  [SKIP] Global already done. Use --force.")
                else:
                    params_path = eval_root / crop / "global" / "best_params.json"
                    if not params_path.exists():
                        print(f"  [SKIP] No eval best_params at {params_path}")
                        results[crop] = "no_eval_params"
                        continue

                    with open(params_path) as f:
                        best_params = json.load(f)

                    summary = retrain_global(
                        crop, df, feature_cols, best_params, global_dir,
                        quantiles, device, random_state,
                    )
                    print(f"  ✓ Global: train R²={summary['train_r2']:.4f}")

                # Always generate val_predictions for downstream layers
                if "regional" in layers or "extreme" in layers:
                    generate_global_val_predictions(
                        global_dir, df, crop, feature_cols, quantiles,
                    )

            if "regional" in layers:
                if not args.force and (regional_dir / "summary.json").exists():
                    print(f"  [SKIP] Regional already done. Use --force.")
                else:
                    reg_result = retrain_regional(
                        crop, df, feature_cols, eval_root, output_root,
                        quantiles, device, random_state,
                    )
                    n_trained = reg_result.get("n_trained", 0) if isinstance(reg_result, dict) else 0
                    print(f"  ✓ Regional: {n_trained} regions retrained")

            if "extreme" in layers:
                if not args.force and (extreme_dir / "summary.json").exists():
                    print(f"  [SKIP] Extreme already done. Use --force.")
                else:
                    ext_result = retrain_extreme(
                        crop, df, feature_cols, eval_root, output_root,
                        cfg.extreme_layer, device, random_state,
                    )
                    n_trained = ext_result.get("n_trained", 0) if isinstance(ext_result, dict) else 0
                    print(f"  ✓ Extreme: {n_trained} regions retrained")

            results[crop] = "done"

            del df
            clear_memory()
            if gpu_available:
                clear_gpu_memory()

        except Exception as e:
            print(f"  [FAIL] {e}")
            import traceback
            traceback.print_exc()
            results[crop] = f"failed: {e}"

    print(f"\n{'=' * 60}")
    print("Retrain Summary:")
    for crop, status in results.items():
        print(f"  {crop}: {status}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
