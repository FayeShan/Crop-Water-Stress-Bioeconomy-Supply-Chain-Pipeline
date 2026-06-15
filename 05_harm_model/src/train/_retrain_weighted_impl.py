"""HARM final retrain (weighted): GW -> GW+RW -> Extreme on 2000-2018 using eval-phase decisions and params."""

import argparse
import json
import logging
import gc
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
from harm.utils import compute_sample_weights

from harm.constants import UNWEIGHTED_CROPS, WEIGHTED_CROPS

TRAIN_YEARS = list(range(2000, 2019))


def retrain_gw(crop, df, feature_cols, best_params, output_dir,
               quantiles, device, random_state):
    """Train final Global Weighted model on full 2000-2018 data."""
    output_dir.mkdir(parents=True, exist_ok=True)

    X = df[feature_cols]
    y = df[crop]

    scaler = MinMaxScaler()
    X_s = scaler.fit_transform(X)

    sample_weight = compute_sample_weights(
        y.values, enabled=True, alpha=1.0, reference_quantile=0.9, max_weight=10.0)
    
    models = train_quantile_models(
        X_s, y.values, best_params, device, quantiles, random_state,
        sample_weight=sample_weight)

    for q, model in models.items():
        model.save_model(str(output_dir / f"model_q{int(q*100)}.json"))
    joblib.dump(scaler, output_dir / "scaler.joblib")

    with open(output_dir / "feature_names.json", "w") as f:
        json.dump(feature_cols, f)
    with open(output_dir / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    y_pred = np.maximum(models[0.5].predict(X_s), 0)
    train_r2 = float(r2_score(y.values, y_pred))

    summary = {
        "crop": crop, "layer": "global_weighted", "mode": "retrain_00-18",
        "n_samples": len(df), "n_features": len(feature_cols),
        "train_r2": train_r2,
        "best_params": best_params,
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def retrain_rw(crop, df, feature_cols, eval_root, output_root,
               quantiles, device, random_state):
    """Retrain Regional Weighted for adopted regions only."""
    eval_rw_dir = eval_root / crop / "regional_weighted"
    output_gw_dir = output_root / crop / "global_weighted"
    output_rw_dir = output_root / crop / "regional_weighted"
    output_rw_dir.mkdir(parents=True, exist_ok=True)

    eval_summary_path = eval_rw_dir / "summary.json"
    if not eval_summary_path.exists():
        logging.warning(f"  [RW] No eval summary, skipping")
        return _save_rw_summary(output_rw_dir, crop, [], [])

    with open(eval_summary_path) as f:
        eval_summary = json.load(f)

    adopted = eval_summary.get("trained_regions", [])
    if not adopted:
        logging.info(f"  [RW] No regions adopted in eval")
        return _save_rw_summary(output_rw_dir, crop, [], eval_summary.get("skipped_regions", []))

    logging.info(f"  [RW] Eval adopted {len(adopted)} regions: {adopted}")

    # Load retrained GW model
    gw_scaler = joblib.load(output_gw_dir / "scaler.joblib")
    with open(output_gw_dir / "feature_names.json") as f:
        gw_features = json.load(f)

    gw_models = {}
    for q in quantiles:
        m = XGBRegressor()
        m.load_model(str(output_gw_dir / f"model_q{int(q*100)}.json"))
        gw_models[q] = m

    df["time"] = pd.to_datetime(df["time"])
    X_all = df[gw_features].values
    X_all_s = gw_scaler.transform(X_all)
    y_all = df[crop].values

    gw_pred = {q: np.maximum(gw_models[q].predict(X_all_s), 0) for q in quantiles}
    residuals_all = y_all - gw_pred[0.5]

    if "regiontype" not in df.columns:
        return _save_rw_summary(output_rw_dir, crop, [], adopted)

    trained_regions = []
    skipped_regions = []

    for region_str in adopted:
        region_val = int(region_str) if region_str.isdigit() else region_str
        mask = df["regiontype"] == region_val
        n_region = int(mask.sum())

        if n_region == 0:
            skipped_regions.append(region_str)
            continue

        bp = _load_region_params(eval_rw_dir / region_str)
        if not bp:
            skipped_regions.append(region_str)
            continue

        logging.info(f"  [RW-{region_str}] Training (n={n_region:,})")

        r_scaler = MinMaxScaler()
        X_r_s = r_scaler.fit_transform(X_all_s[mask])

        r_models = train_quantile_models(
            X_r_s, residuals_all[mask], bp, device, quantiles, random_state)

        r_dir = output_rw_dir / region_str
        r_dir.mkdir(parents=True, exist_ok=True)
        for q, m in r_models.items():
            m.save_model(str(r_dir / f"model_q{int(q*100)}.json"))
        joblib.dump(r_scaler, r_dir / "scaler.joblib")
        with open(r_dir / "best_params.json", "w") as f:
            json.dump(bp, f, indent=2)

        # Combined predictions for diagnostics + downstream Extreme
        r_preds = predict_with_quantiles(r_models, X_r_s)
        y_combined = np.maximum(gw_pred[0.5][mask] + r_preds[0.5], 0)

        val_df = pd.DataFrame({
            "fold": 0,
            "time": df.loc[mask, "time"].values,
            "grid50_id": df.loc[mask, "grid50_id"].values,
            "y_true": y_all[mask],
            "y_pred_global": gw_pred[0.5][mask],
            "y_pred_combined": y_combined,
            "residual": y_all[mask] - y_combined,
        })
        val_df.to_parquet(r_dir / "val_predictions.parquet", index=False)

        gw_r2 = float(r2_score(y_all[mask], gw_pred[0.5][mask]))
        combined_r2 = float(r2_score(y_all[mask], y_combined))

        with open(r_dir / "region_summary.json", "w") as f:
            json.dump({"region": region_str, "use_regional": True,
                       "mode": "retrain_00-18", "n_samples": n_region,
                       "gw_r2_train": gw_r2, "combined_r2_train": combined_r2,
                       "best_params": bp}, f, indent=2)

        trained_regions.append(region_str)
        logging.info(f"  [RW-{region_str}] ✓ GW R²={gw_r2:.4f} → GW+RW R²={combined_r2:.4f}")

        del r_models, r_scaler
        clear_memory()

    # Skip markers for non-adopted
    all_skipped = list(set(eval_summary.get("skipped_regions", []) + skipped_regions))
    for r_str in all_skipped:
        r_dir = output_rw_dir / r_str
        r_dir.mkdir(parents=True, exist_ok=True)
        with open(r_dir / "region_summary.json", "w") as f:
            json.dump({"region": r_str, "use_regional": False, "reason": "not_adopted_in_eval"}, f, indent=2)

    return _save_rw_summary(output_rw_dir, crop, trained_regions, all_skipped)


def _save_rw_summary(rw_dir, crop, trained, skipped):
    rw_dir.mkdir(parents=True, exist_ok=True)
    s = {"layer": "regional_weighted", "crop": crop, "mode": "retrain_00-18",
         "n_trained": len(trained), "n_skipped": len(skipped),
         "trained_regions": trained, "skipped_regions": skipped,
         "timestamp": datetime.now().isoformat()}
    with open(rw_dir / "summary.json", "w") as f:
        json.dump(s, f, indent=2)
    return s


def _load_region_params(region_dir):
    """Load best_params from eval region directory."""
    bp_path = region_dir / "best_params.json"
    if bp_path.exists():
        with open(bp_path) as f:
            return json.load(f)
    rs_path = region_dir / "region_summary.json"
    if rs_path.exists():
        with open(rs_path) as f:
            return json.load(f).get("best_params", {})
    return {}


def retrain_extreme_weighted(crop, df, feature_cols, eval_root, output_root,
                             ext_cfg, device, random_state):
    """Retrain Extreme layer using retrained GW+RW as y_base, reusing eval adoption decisions."""
    eval_ext_dir = eval_root / crop / "extreme_weighted"
    output_gw_dir = output_root / crop / "global_weighted"
    output_rw_dir = output_root / crop / "regional_weighted"
    output_ext_dir = output_root / crop / "extreme_weighted"
    output_ext_dir.mkdir(parents=True, exist_ok=True)

    eval_summary_path = eval_ext_dir / "summary.json"
    if not eval_summary_path.exists():
        logging.warning(f"  [EXT-W] No eval summary, skipping")
        return _save_ext_summary(output_ext_dir, crop, [], [])

    with open(eval_summary_path) as f:
        eval_summary = json.load(f)

    adopted = eval_summary.get("trained_regions", [])
    if not adopted:
        logging.info(f"  [EXT-W] No regions adopted in eval")
        return _save_ext_summary(output_ext_dir, crop, [], eval_summary.get("skipped_regions", []))

    logging.info(f"  [EXT-W] Eval adopted {len(adopted)} regions: {adopted}")

    # Load retrained GW model
    gw_scaler = joblib.load(output_gw_dir / "scaler.joblib")
    gw_model = XGBRegressor()
    gw_model.load_model(str(output_gw_dir / "model_q50.json"))
    with open(output_gw_dir / "feature_names.json") as f:
        gw_features = json.load(f)

    # Load retrained RW models
    rw_models = {}
    rw_scalers = {}
    rw_summary_path = output_rw_dir / "summary.json"
    if rw_summary_path.exists():
        with open(rw_summary_path) as f:
            rw_summary = json.load(f)
        for r_str in rw_summary.get("trained_regions", []):
            r_path = output_rw_dir / r_str
            if (r_path / "model_q50.json").exists() and (r_path / "scaler.joblib").exists():
                m = XGBRegressor()
                m.load_model(str(r_path / "model_q50.json"))
                rw_models[r_str] = m
                rw_scalers[r_str] = joblib.load(r_path / "scaler.joblib")

    logging.info(f"  [EXT-W] Loaded {len(rw_models)} retrained RW models")

    # Compute GW+RW base predictions
    df["time"] = pd.to_datetime(df["time"])
    X_all = df[gw_features].values
    X_all_gs = gw_scaler.transform(X_all)
    y_all = df[crop].values
    gw_pred = np.maximum(gw_model.predict(X_all_gs), 0)

    y_base_all = gw_pred.copy()
    for r_str, r_model in rw_models.items():
        region_val = int(r_str) if r_str.isdigit() else r_str
        r_mask = (df["regiontype"] == region_val).values
        if r_mask.sum() > 0:
            X_r_s = rw_scalers[r_str].transform(X_all_gs[r_mask])
            r_pred = r_model.predict(X_r_s)
            y_base_all[r_mask] = np.maximum(gw_pred[r_mask] + r_pred, 0)

    base_r2 = r2_score(y_all, y_base_all)
    logging.info(f"  [EXT-W] GW+RW base: mean={y_base_all.mean():.0f}, R²={base_r2:.4f}")

    if "regiontype" not in df.columns:
        return _save_ext_summary(output_ext_dir, crop, [], adopted)

    ytrue_pct = ext_cfg.get("ytrue_percentile", 85)
    under_thresh = ext_cfg.get("underestimate_threshold", 0.15)
    prob_threshold = ext_cfg.get("detector_prob_threshold", 0.6)
    max_corr_ratio = ext_cfg.get("max_correction_sample_ratio", 0.05)
    max_rel_corr = ext_cfg.get("max_relative_correction", 0.5)
    conservation_tol = ext_cfg.get("conservation_tolerance", 0.05)

    trained_regions = []
    skipped_regions = []

    for region_str in adopted:
        region_val = int(region_str) if region_str.isdigit() else region_str
        tag = f"[EXT-W-{region_str}]"
        mask = (df["regiontype"] == region_val).values
        n_region = int(mask.sum())

        if n_region == 0:
            skipped_regions.append(region_str)
            continue

        eval_region_dir = eval_ext_dir / region_str
        det_path = eval_region_dir / "detector_params.json"
        corr_path = eval_region_dir / "corrector_params.json"
        if not (det_path.exists() and corr_path.exists()):
            logging.warning(f"  {tag} No eval params, skipping")
            skipped_regions.append(region_str)
            continue

        with open(det_path) as f:
            det_params = json.load(f)
        with open(corr_path) as f:
            corr_params = json.load(f)

        y_true = y_all[mask]
        y_base = y_base_all[mask]
        X_raw = X_all[mask]

        region_base_r2 = float(r2_score(y_true, y_base))
        logging.info(f"  {tag} n={n_region:,}, base_R²={region_base_r2:.4f}")

        under_mask, ext_stats = _identify_extreme_underestimates(
            y_true, y_base, ytrue_pct, under_thresh)

        n_extreme = ext_stats["n_underestimate"]
        if n_extreme < 10:
            logging.warning(f"  {tag} Too few extremes ({n_extreme})")
            skipped_regions.append(region_str)
            _save_ext_region(output_ext_dir / region_str, region_str, n_region, False,
                           f"too_few_extremes_{n_extreme}")
            continue

        if n_extreme < ext_cfg.get("min_extreme_samples", 100):
            logging.warning(f"  {tag} Only {n_extreme} extremes, training anyway (eval adopted)")

        logging.info(f"  {tag} Underestimates: {n_extreme} ({ext_stats['underestimate_pct']:.1f}%)")

        e_scaler = MinMaxScaler()
        X_scaled = e_scaler.fit_transform(X_raw)

        n_pos = int(np.sum(under_mask))
        n_neg = len(under_mask) - n_pos
        spw = n_neg / n_pos if n_pos > 0 else 1.0

        detector = XGBClassifier(
            objective="binary:logistic", eval_metric="auc",
            tree_method="hist", device=device,
            random_state=random_state, verbosity=0,
            scale_pos_weight=spw, **det_params)
        detector.fit(X_scaled, under_mask)

        # Train corrector on the extreme subset only
        residuals = y_true - y_base
        corrector = XGBRegressor(
            objective="reg:squarederror", tree_method="hist",
            device=device, random_state=random_state, verbosity=0,
            **corr_params)
        corrector.fit(X_scaled[under_mask], residuals[under_mask])

        # Quick diagnostics
        det_probs = detector.predict_proba(X_scaled)[:, 1]
        corr_pred = corrector.predict(X_scaled)
        y_corrected, corr_stats = _apply_bounded_correction(
            y_base, det_probs, corr_pred,
            prob_threshold, max_corr_ratio, max_rel_corr, conservation_tol)

        final_r2 = float(r2_score(y_true, y_corrected))
        base_rmse = float(np.sqrt(mean_squared_error(y_true, y_base)))
        final_rmse = float(np.sqrt(mean_squared_error(y_true, y_corrected)))

        logging.info(f"  {tag} ✓ R² {region_base_r2:.4f}→{final_r2:.4f}, "
                     f"RMSE {base_rmse:.0f}→{final_rmse:.0f}, "
                     f"corrected={corr_stats['n_corrected']}")

        region_dir = output_ext_dir / region_str
        region_dir.mkdir(parents=True, exist_ok=True)
        detector.save_model(str(region_dir / "detector.json"))
        corrector.save_model(str(region_dir / "corrector.json"))
        joblib.dump(e_scaler, region_dir / "scaler.joblib")
        with open(region_dir / "feature_names.json", "w") as f:
            json.dump(gw_features, f)
        with open(region_dir / "detector_params.json", "w") as f:
            json.dump(clean_for_json(det_params), f, indent=2)
        with open(region_dir / "corrector_params.json", "w") as f:
            json.dump(clean_for_json(corr_params), f, indent=2)

        _save_ext_region(region_dir, region_str, n_region, True,
                        "retrained_with_eval_params",
                        extra={"base_r2_train": region_base_r2,
                               "final_r2_train": final_r2,
                               "n_corrected": corr_stats["n_corrected"],
                               "conservation_ratio": corr_stats["conservation_ratio"],
                               "extreme_stats": clean_for_json(ext_stats)})

        trained_regions.append(region_str)
        del detector, corrector, e_scaler
        clear_memory()

    # Skip markers
    all_skipped = list(set(eval_summary.get("skipped_regions", []) + skipped_regions))
    for r_str in all_skipped:
        _save_ext_region(output_ext_dir / r_str, r_str, 0, False, "not_adopted_in_eval")

    return _save_ext_summary(output_ext_dir, crop, trained_regions, all_skipped)


def _save_ext_region(region_dir, region_str, n_samples, use_extreme, reason, extra=None):
    region_dir.mkdir(parents=True, exist_ok=True)
    result = {"region": region_str, "use_extreme": use_extreme, "base_model": "GW+RW",
              "reason": reason, "n_samples": n_samples, "mode": "retrain_00-18"}
    if extra:
        result.update(extra)
    with open(region_dir / "region_summary.json", "w") as f:
        json.dump(clean_for_json(result), f, indent=2)


def _save_ext_summary(ext_dir, crop, trained, skipped):
    ext_dir.mkdir(parents=True, exist_ok=True)
    s = {"layer": "extreme", "base_model": "GW+RW", "crop": crop,
         "mode": "retrain_00-18",
         "n_trained": len(trained), "n_skipped": len(skipped),
         "trained_regions": trained, "skipped_regions": skipped,
         "timestamp": datetime.now().isoformat()}
    with open(ext_dir / "summary.json", "w") as f:
        json.dump(s, f, indent=2)
    return s


def main():
    parser = argparse.ArgumentParser(description="HARM Weighted Retrain (GW+RW+E, 00-18)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--phase", choices=["eval", "final"], default="eval", help="Which config phase block to use (default: eval)")
    parser.add_argument("--crops", nargs="+", default=WEIGHTED_CROPS)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--eval-root", type=str, default=None,
                        help="Where to read eval decisions (default: HARM_v2)")
    parser.add_argument("--output-root", type=str, default=None,
                        help="Where to save retrained models (default: HARM_v2_final)")
    parser.add_argument("--layers", nargs="+", default=["global", "regional", "extreme"],
                        choices=["global", "regional", "extreme", "all"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config, phase=args.phase if hasattr(args, 'phase') else 'eval')
    crops = args.crops

    data_root = Path(args.data_root) if args.data_root else cfg.paths.data_root
    eval_root = Path(args.eval_root) if args.eval_root else Path(
        str(cfg.paths.result_root).replace("/final", "/eval"))
    output_root = Path(args.output_root) if args.output_root else cfg.paths.result_root

    layers = args.layers
    if "all" in layers:
        layers = ["global", "regional", "extreme"]
    layer_order = ["global", "regional", "extreme"]
    layers = [l for l in layer_order if l in layers]

    gpu_available, _, device = setup_gpu_environment(cfg.common.use_gpu)
    quantiles = cfg.common.quantiles
    random_state = cfg.common.random_state

    print("=" * 60)
    print("HARM Weighted Retrain (GW → GW+RW → Extreme)")
    print(f"  Layers:    {' → '.join(layers)}")
    print(f"  Crops:     {crops}")
    print(f"  Data:      {data_root}")
    print(f"  Eval from: {eval_root}")
    print(f"  Output:    {output_root}")
    print(f"  GPU:       {device}")
    print("=" * 60, flush=True)

    results = {}

    for crop in crops:
        print(f"\n{'━' * 50}\n  {crop}\n{'━' * 50}", flush=True)

        if crop in UNWEIGHTED_CROPS:
            print(f"  [SKIP] {crop} uses unweighted — use run_retrain.py")
            continue

        gw_dir = output_root / crop / "global_weighted"
        rw_dir = output_root / crop / "regional_weighted"
        ext_dir = output_root / crop / "extreme_weighted"

        setup_logging(output_root / crop / "logs", crop)

        try:
            df = load_crop_data(data_root, crop, years=TRAIN_YEARS)
            feature_cols = get_feature_columns(df, crop, cfg.features)
            print(f"  Data: {len(df):,} rows, {len(feature_cols)} features", flush=True)

            if "global" in layers:
                if not args.force and (gw_dir / "summary.json").exists():
                    print(f"  [SKIP] GW already done. Use --force.")
                else:
                    params_path = eval_root / crop / "global_weighted" / "best_params.json"
                    if not params_path.exists():
                        print(f"  [SKIP] No eval GW params at {params_path}")
                        results[crop] = "no_eval_params"
                        continue

                    with open(params_path) as f:
                        best_params = json.load(f)

                    with open(eval_root / crop / "global_weighted" / "feature_names.json") as f:
                        gw_features = json.load(f)

                    summary = retrain_gw(
                        crop, df, gw_features, best_params, gw_dir,
                        quantiles, device, random_state)
                    print(f"  ✓ GW: train R²={summary['train_r2']:.4f}", flush=True)

            if "regional" in layers:
                if not args.force and (rw_dir / "summary.json").exists():
                    print(f"  [SKIP] RW already done. Use --force.")
                else:
                    rw_result = retrain_rw(
                        crop, df, feature_cols, eval_root, output_root,
                        quantiles, device, random_state)
                    n_t = rw_result.get("n_trained", 0) if isinstance(rw_result, dict) else 0
                    print(f"  ✓ RW: {n_t} regions retrained", flush=True)

            if "extreme" in layers:
                if not args.force and (ext_dir / "summary.json").exists():
                    print(f"  [SKIP] Extreme already done. Use --force.")
                else:
                    ext_result = retrain_extreme_weighted(
                        crop, df, feature_cols, eval_root, output_root,
                        cfg.extreme_layer, device, random_state)
                    n_t = ext_result.get("n_trained", 0) if isinstance(ext_result, dict) else 0
                    print(f"  ✓ Extreme (weighted): {n_t} regions retrained", flush=True)

            results[crop] = "done"
            del df; clear_memory()
            if gpu_available:
                clear_gpu_memory()

        except Exception as e:
            print(f"  [FAIL] {e}", flush=True)
            import traceback; traceback.print_exc()
            results[crop] = f"failed: {e}"

    print(f"\n{'=' * 60}")
    print("Retrain Weighted Summary:")
    for crop, status in results.items():
        print(f"  {crop}: {status}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
