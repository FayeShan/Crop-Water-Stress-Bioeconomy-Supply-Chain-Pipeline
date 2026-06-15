
"""HARM prediction pipeline: filter then apply retrained G -> G+R -> G+R+E to 2019-2024."""

import argparse
import gc
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor, XGBClassifier

warnings.filterwarnings("ignore")

from harm.config import load_config
from harm.utils import (
    get_feature_columns, setup_logging,
    setup_gpu_environment, clear_memory, clear_gpu_memory, clean_for_json,
)
from harm.extreme_layer import _apply_bounded_correction

PRED_YEARS = list(range(2019, 2025))  # 2019-2024

# ── Paths are loaded from config; see config.yaml / config.yaml ──
# These module-level variables are set in main() after config is loaded.
HIST_DATA_ROOT = None  # set from cfg.paths.hist_data_root
HARVEST_DIR = None     # set from cfg.paths.harvest_area_dir
SPAM_2020_DIR = None   # set from cfg.paths.spam_2020_dir

from harm.constants import (
    ALL_CROPS, KOPPEN_CROPS, UNWEIGHTED_CROPS,
    CROPS_NO_SPAM, ACEA_TO_SPAM,
)

# Monthly mask reference years
MASK_YEARS = [2013, 2014, 2015, 2016, 2017, 2018]


def build_spam_spatial_mask(crop: str) -> Optional[set]:
    """Build (lon, lat) mask where SPAM 2020 irrigated area > 0; None if crop has no SPAM mapping."""
    if crop in CROPS_NO_SPAM or crop not in ACEA_TO_SPAM:
        logging.info(f"[FILTER-SPAM] {crop}: no SPAM mapping, skipping spatial filter")
        return None

    try:
        import xarray as xr
    except ImportError:
        logging.warning("[FILTER-SPAM] xarray not available, skipping")
        return None

    spam_codes = ACEA_TO_SPAM[crop]
    combined = None

    for spam_code in spam_codes:
        ha_ir = None

        nc_path = HARVEST_DIR / f"{spam_code}_2000_2025.nc"
        if nc_path.exists():
            ds = xr.open_dataset(nc_path)
            ha_ir = ds["harvested_area_ir"].sel(time="2020").squeeze()
            ds.close()
            logging.info(f"  {spam_code}: loaded from NC")
        else:
            # Fallback: SPAM 2020 TIF
            try:
                import rioxarray
                tif_path = SPAM_2020_DIR / f"2020_{spam_code}_I.tif"
                if tif_path.exists():
                    ha_ir = rioxarray.open_rasterio(tif_path).squeeze("band", drop=True)
                    if not ha_ir.rio.crs:
                        ha_ir = ha_ir.rio.write_crs("EPSG:4326")
                    rename_map = {}
                    if "y" in ha_ir.dims:
                        rename_map["y"] = "lat"
                    if "x" in ha_ir.dims:
                        rename_map["x"] = "lon"
                    if rename_map:
                        ha_ir = ha_ir.rename(rename_map)
                    logging.info(f"  {spam_code}: loaded from TIF")
                else:
                    logging.warning(f"  {spam_code}: no file found, skipping")
                    continue
            except ImportError:
                logging.warning(f"  {spam_code}: rioxarray not available")
                continue

        if ha_ir is None:
            continue
        if combined is None:
            combined = ha_ir
        else:
            if combined.shape != ha_ir.shape:
                ha_ir = ha_ir.interp_like(combined, method="nearest")
            combined = combined + ha_ir

    return combined


def apply_spam_filter(df: pd.DataFrame, crop: str) -> pd.DataFrame:
    """Filter df to only keep (lon, lat) where SPAM irrigated area > 0."""
    import xarray as xr

    ha_ir = build_spam_spatial_mask(crop)
    if ha_ir is None:
        return df

    n_before = len(df)
    points = df[["lon", "lat"]].drop_duplicates().reset_index(drop=True)

    vals = ha_ir.sel(
        lon=xr.DataArray(points["lon"].values, dims="points"),
        lat=xr.DataArray(points["lat"].values, dims="points"),
        method="nearest",
    ).values

    valid_points = points[vals > 0][["lon", "lat"]]
    df = df.merge(valid_points, on=["lon", "lat"], how="inner")

    n_after = len(df)
    logging.info(f"[FILTER-SPAM] {crop}: {len(valid_points)}/{len(points)} points, "
                 f"{n_after:,}/{n_before:,} rows kept "
                 f"({(1 - n_after / n_before) * 100:.1f}% removed)")
    return df


def build_monthly_mask(crop: str, hist_root: Path = HIST_DATA_ROOT) -> pd.DataFrame:
    """Build (lon, lat, month) mask from MASK_YEARS historical data; only seen tuples are kept."""
    hist_dir = hist_root / crop
    fpath = hist_dir / f"{crop}.parquet"
    if not fpath.exists():
        # Try directory-level parquet
        if hist_dir.exists():
            fpath = hist_dir
        else:
            raise FileNotFoundError(f"Historical data not found: {hist_dir}")

    df_hist = pd.read_parquet(fpath)
    df_hist["time"] = pd.to_datetime(df_hist["time"])
    df_hist["year"] = df_hist["time"].dt.year
    df_hist["month"] = df_hist["time"].dt.month

    df_recent = df_hist[df_hist["year"].isin(MASK_YEARS)]
    mask = df_recent[["lon", "lat", "month"]].drop_duplicates().reset_index(drop=True)

    n_points = df_recent[["lon", "lat"]].drop_duplicates().shape[0]
    logging.info(f"[FILTER-MONTHLY] {crop}: {len(mask):,} (lon,lat,month) tuples "
                 f"from {n_points:,} points ({MASK_YEARS})")

    del df_hist, df_recent
    gc.collect()
    return mask


def apply_monthly_mask(df: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows matching (lon, lat, month) in the mask."""
    n_before = len(df)
    df["time"] = pd.to_datetime(df["time"])
    if "month" not in df.columns:
        df["month"] = df["time"].dt.month

    df = df.merge(mask, on=["lon", "lat", "month"], how="inner")
    n_after = len(df)

    logging.info(f"[FILTER-MONTHLY] {n_after:,}/{n_before:,} rows kept "
                 f"({(1 - n_after / n_before) * 100:.1f}% removed)")
    return df


def apply_all_filters(df: pd.DataFrame, crop: str) -> pd.DataFrame:
    """Apply SPAM spatial filter then historical (lon, lat, month) mask."""
    logging.info(f"[FILTER] Applying pre-prediction filters for {crop.upper()}")
    n_orig = len(df)

    if crop not in CROPS_NO_SPAM:
        try:
            df = apply_spam_filter(df, crop)
        except Exception as e:
            logging.warning(f"[FILTER-SPAM] Failed for {crop}: {e}, continuing without")

    try:
        monthly_mask = build_monthly_mask(crop)
        df = apply_monthly_mask(df, monthly_mask)
        del monthly_mask
    except Exception as e:
        logging.warning(f"[FILTER-MONTHLY] Failed for {crop}: {e}, continuing without")

    reduction = (1 - len(df) / n_orig) * 100 if n_orig > 0 else 0
    logging.info(f"[FILTER] Final: {len(df):,}/{n_orig:,} rows "
                 f"({reduction:.1f}% removed total)")

    gc.collect()
    return df


def get_pred_data_root(crop: str, cfg) -> Path:
    """Return the correct prediction data path for a crop."""
    return cfg.get_pred_data_root(crop)


def load_prediction_data(pred_data_root: Path, crop: str, years: list) -> pd.DataFrame:
    """Load prediction dataset for a crop, filter by years."""
    crop_dir = pred_data_root / crop
    fpath = crop_dir / f"{crop}.parquet"
    if not fpath.exists():
        raise FileNotFoundError(f"Prediction data not found: {fpath}")

    df = pd.read_parquet(fpath)
    logging.info(f"[PRED] Loaded {crop}: {len(df):,} rows, {len(df.columns)} columns")

    if years is not None and "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df["year"] = df["time"].dt.year
        df = df[df["year"].isin(years)].copy()
        logging.info(f"[PRED] Filtered to years {years[0]}-{years[-1]}: {len(df):,} rows")

    return df


def predict_global(df, global_dir, feature_cols, quantiles):
    """Apply Global model; returns (preds dict keyed by quantile, X_scaled for Regional layer, model_features)."""
    scaler = joblib.load(global_dir / "scaler.joblib")
    with open(global_dir / "feature_names.json") as f:
        model_features = json.load(f)

    missing = set(model_features) - set(df.columns)
    if missing:
        logging.warning(f"[GLOBAL] Missing features in prediction data: {missing}")
        for col in missing:
            df[col] = 0

    X = df[model_features].values
    X_scaled = scaler.transform(X)

    predictions = {}
    for q in quantiles:
        model = XGBRegressor()
        model.load_model(str(global_dir / f"model_q{int(q*100)}.json"))
        predictions[q] = np.maximum(model.predict(X_scaled), 0)
        del model

    logging.info(f"[GLOBAL] Predicted {len(df):,} samples, "
                 f"mean={predictions[0.5].mean():.0f}, "
                 f"median={np.median(predictions[0.5]):.0f}")

    return predictions, X_scaled, model_features


def predict_regional(df, X_global_scaled, y_global, regional_dir):
    """Apply Regional corrections to adopted regions."""
    summary_path = regional_dir / "summary.json"
    if not summary_path.exists():
        logging.info("[REGIONAL] No regional summary, returning Global predictions")
        return y_global.copy(), {}

    with open(summary_path) as f:
        summary = json.load(f)

    adopted = summary.get("trained_regions", [])
    if not adopted:
        logging.info("[REGIONAL] No regions adopted")
        return y_global.copy(), {}

    y_out = y_global.copy()
    regional_models = {}

    if "regiontype" not in df.columns:
        logging.warning("[REGIONAL] No regiontype column, skipping")
        return y_out, {}

    n_total_corrected = 0
    for region_str in adopted:
        rdir = regional_dir / region_str
        if not (rdir / "model_q50.json").exists():
            continue

        r_scaler = joblib.load(rdir / "scaler.joblib")
        r_model = XGBRegressor()
        r_model.load_model(str(rdir / "model_q50.json"))

        region_val = int(region_str) if region_str.isdigit() else region_str
        mask = df["regiontype"].values == region_val
        n_match = int(mask.sum())

        if n_match == 0:
            continue

        X_region = r_scaler.transform(X_global_scaled[mask])
        r_pred = r_model.predict(X_region)
        y_out[mask] = np.maximum(y_global[mask] + r_pred, 0)
        n_total_corrected += n_match

        regional_models[region_str] = (r_scaler, r_model)
        del r_model

    logging.info(f"[REGIONAL] Applied {len(regional_models)} regions, "
                 f"corrected {n_total_corrected:,}/{len(df):,} samples")

    return y_out, regional_models


def predict_regional_quantiles(df, X_global_scaled, global_preds, regional_dir, quantiles):
    """Predict q10/q90 through Regional layer."""
    summary_path = regional_dir / "summary.json"
    if not summary_path.exists():
        return {q: global_preds[q].copy() for q in quantiles if q != 0.5}

    with open(summary_path) as f:
        summary = json.load(f)

    adopted = summary.get("trained_regions", [])
    result = {q: global_preds[q].copy() for q in quantiles if q != 0.5}

    if not adopted or "regiontype" not in df.columns:
        return result

    for region_str in adopted:
        rdir = regional_dir / region_str
        region_val = int(region_str) if region_str.isdigit() else region_str
        mask = df["regiontype"].values == region_val
        if mask.sum() == 0:
            continue

        r_scaler_path = rdir / "scaler.joblib"
        if not r_scaler_path.exists():
            continue
        r_scaler = joblib.load(r_scaler_path)
        X_region = r_scaler.transform(X_global_scaled[mask])

        for q in result:
            q_model_path = rdir / f"model_q{int(q*100)}.json"
            if q_model_path.exists():
                r_model = XGBRegressor()
                r_model.load_model(str(q_model_path))
                r_pred = r_model.predict(X_region)
                result[q][mask] = np.maximum(global_preds[q][mask] + r_pred, 0)
                del r_model

    return result


def predict_extreme(df, y_gr, extreme_dir, ext_cfg):
    """
    Apply Extreme layer corrections to adopted regions.

    IMPORTANT: Extreme layer has its own MinMaxScaler fitted on RAW features
    (not Global-scaled). We must apply it to raw features from df.
    """
    summary_path = extreme_dir / "summary.json"
    if not summary_path.exists():
        logging.info("[EXTREME] No extreme summary, returning G+R predictions")
        return y_gr.copy()

    with open(summary_path) as f:
        summary = json.load(f)

    adopted = summary.get("trained_regions", [])
    if not adopted:
        logging.info("[EXTREME] No regions adopted")
        return y_gr.copy()

    y_out = y_gr.copy()

    if "regiontype" not in df.columns:
        logging.warning("[EXTREME] No regiontype column, skipping")
        return y_out

    prob_threshold = ext_cfg.get("detector_prob_threshold", 0.6)
    max_corr_ratio = ext_cfg.get("max_correction_sample_ratio", 0.05)
    max_rel_corr = ext_cfg.get("max_relative_correction", 0.5)
    conservation_tol = ext_cfg.get("conservation_tolerance", 0.05)

    regiontype_vals = df["regiontype"].values

    n_total_corrected = 0
    for region_str in adopted:
        rdir = extreme_dir / region_str
        det_path = rdir / "detector.json"
        corr_path = rdir / "corrector.json"
        scaler_path = rdir / "scaler.joblib"

        if not (det_path.exists() and corr_path.exists() and scaler_path.exists()):
            logging.warning(f"[EXTREME-{region_str}] Missing model files, skipping")
            continue

        feat_path = rdir / "feature_names.json"
        if not feat_path.exists():
            logging.warning(f"[EXTREME-{region_str}] No feature_names.json, skipping")
            continue
        with open(feat_path) as f:
            ext_features = json.load(f)

        e_scaler = joblib.load(scaler_path)
        detector = XGBClassifier()
        detector.load_model(str(det_path))
        corrector = XGBRegressor()
        corrector.load_model(str(corr_path))

        region_val = int(region_str) if region_str.isdigit() else region_str
        mask = regiontype_vals == region_val
        n_match = int(mask.sum())

        if n_match == 0:
            continue

        # Extreme scaler is fitted on RAW features, not Global-scaled
        X_region_raw = df[ext_features].values[mask]
        X_region_scaled = e_scaler.transform(X_region_raw)

        det_probs = detector.predict_proba(X_region_scaled)[:, 1]
        corr_pred = corrector.predict(X_region_scaled)

        y_region_base = y_gr[mask]
        y_corrected, corr_stats = _apply_bounded_correction(
            y_region_base, det_probs, corr_pred,
            prob_threshold, max_corr_ratio, max_rel_corr, conservation_tol,
        )

        y_out[mask] = y_corrected
        n_total_corrected += corr_stats["n_corrected"]

        logging.info(f"[EXTREME-{region_str}] n={n_match:,}, "
                     f"corrected={corr_stats['n_corrected']}, "
                     f"conservation={corr_stats['conservation_ratio']:.3f}")

        del detector, corrector, e_scaler
        clear_memory()

    logging.info(f"[EXTREME] Total corrected: {n_total_corrected:,}/{len(df):,}")
    return y_out


def predict_crop(crop, model_root, output_dir, cfg, pred_years,
                 apply_filters=True):
    """Full prediction pipeline: filter → G → G+R → G+R+E → save."""

    use_weighted = crop not in UNWEIGHTED_CROPS
    if use_weighted:
        global_dir = model_root / crop / "global_weighted"
        regional_dir = model_root / crop / "regional_weighted"
        extreme_dir = model_root / crop / "extreme_weighted"
        pipeline_label = "GW+RW+EW"
    else:
        global_dir = model_root / crop / "global"
        regional_dir = model_root / crop / "regional"
        extreme_dir = model_root / crop / "extreme"
        pipeline_label = "G+R+E"

    logging.info(f"[{crop}] Pipeline: {pipeline_label}")

    if not (global_dir / "model_q50.json").exists():
        # Fallback: try the other weighted/unweighted variant
        alt_global = model_root / crop / ("global" if use_weighted else "global_weighted")
        if (alt_global / "model_q50.json").exists():
            logging.warning(f"[{crop}] {pipeline_label} not found, falling back to other variant")
            global_dir = alt_global
            regional_dir = model_root / crop / ("regional" if use_weighted else "regional_weighted")
            extreme_dir = model_root / crop / ("extreme" if use_weighted else "extreme_weighted")
            pipeline_label += " (fallback)"
        else:
            logging.warning(f"[{crop}] No retrained model at {global_dir}, skipping")
            return None

    pred_data_root = get_pred_data_root(crop, cfg)
    df = load_prediction_data(pred_data_root, crop, pred_years)
    if len(df) == 0:
        logging.warning(f"[{crop}] No prediction data")
        return None

    n_raw = len(df)

    if apply_filters:
        df = apply_all_filters(df, crop)
        df = df.reset_index(drop=True)
        if len(df) == 0:
            logging.warning(f"[{crop}] No data after filtering")
            return None

    with open(global_dir / "feature_names.json") as f:
        feature_cols = json.load(f)

    quantiles = cfg.common.quantiles

    global_preds, X_scaled, model_features = predict_global(
        df, global_dir, feature_cols, quantiles,
    )
    y_global = global_preds[0.5]

    y_gr, _ = predict_regional(df, X_scaled, y_global, regional_dir)

    gr_quantiles = predict_regional_quantiles(
        df, X_scaled, global_preds, regional_dir, quantiles,
    )

    y_gre = predict_extreme(df, y_gr, extreme_dir, cfg.extreme_layer)

    out = pd.DataFrame({
        "time": pd.to_datetime(df["time"]).values,
        "grid50_id": df["grid50_id"].values,
        "y_pred_global": y_global,
        "y_pred_gr": y_gr,
        "y_pred_gre": y_gre,
        "y_pred_q10": gr_quantiles.get(0.1, global_preds.get(0.1, y_global)),
        "y_pred_q90": gr_quantiles.get(0.9, global_preds.get(0.9, y_global)),
    })

    for col in ["regiontype", "lat", "lon"]:
        if col in df.columns:
            out[col] = df[col].values

    out["year"] = out["time"].dt.year

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "predictions.parquet"
    out.to_parquet(out_path, index=False)

    summary = {
        "crop": crop,
        "pipeline": pipeline_label,
        "n_raw": n_raw,
        "n_after_filter": len(out),
        "filter_reduction_pct": float((1 - len(out) / n_raw) * 100) if n_raw > 0 else 0,
        "n_samples": len(out),
        "years": sorted(out["year"].unique().tolist()),
        "n_grid_cells": int(out["grid50_id"].nunique()),
        "pred_data_root": str(get_pred_data_root(crop, cfg)),
        "global_mean": float(y_global.mean()),
        "gr_mean": float(y_gr.mean()),
        "gre_mean": float(y_gre.mean()),
        "gr_vs_global_change_pct": float(
            (y_gr.sum() - y_global.sum()) / y_global.sum() * 100
        ) if y_global.sum() > 0 else 0,
        "gre_vs_gr_change_pct": float(
            (y_gre.sum() - y_gr.sum()) / y_gr.sum() * 100
        ) if y_gr.sum() > 0 else 0,
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_dir / "prediction_summary.json", "w") as f:
        json.dump(clean_for_json(summary), f, indent=2)

    logging.info(f"[{crop}] Saved {len(out):,} predictions to {out_path}")
    logging.info(f"[{crop}] Means: G={y_global.mean():.0f}, "
                 f"G+R={y_gr.mean():.0f}, G+R+E={y_gre.mean():.0f}")

    del df, out
    clear_memory()
    return summary


def main():
    parser = argparse.ArgumentParser(description="HARM Prediction Pipeline (2019-2024)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--phase", choices=["eval", "final"], default="eval", help="Which config phase block to use (default: eval)")
    parser.add_argument("--crops", nargs="+", default=None)
    parser.add_argument("--model-root", type=str, default=None,
                        help="Where retrained models live (default: HARM_v2_final)")
    parser.add_argument("--output-root", type=str, default=None,
                        help="Where to save predictions (default: {model_root}/predictions)")
    parser.add_argument("--years", nargs="+", type=int, default=None,
                        help="Override prediction years")
    parser.add_argument("--no-filter", action="store_true",
                        help="Skip SPAM + monthly mask filtering")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config, phase=args.phase if hasattr(args, 'phase') else 'eval')
    crops = args.crops or cfg.crops

    model_root = Path(args.model_root) if args.model_root else cfg.paths.result_root
    output_root = Path(args.output_root) if args.output_root else model_root / "predictions"
    pred_years = args.years or PRED_YEARS
    apply_filters = not args.no_filter

    # Set module-level paths from config (used by filter functions)
    global HIST_DATA_ROOT, HARVEST_DIR, SPAM_2020_DIR
    HIST_DATA_ROOT = cfg.paths.hist_data_root
    HARVEST_DIR = cfg.paths.harvest_area_dir
    SPAM_2020_DIR = cfg.paths.spam_2020_dir

    print("=" * 60)
    print("HARM Prediction Pipeline")
    print(f"  Crops:       {crops}")
    print(f"  Years:       {pred_years[0]}-{pred_years[-1]}")
    print(f"  Models from: {model_root}")
    print(f"  Output:      {output_root}")
    print(f"  Filtering:   {'ON (SPAM + monthly mask)' if apply_filters else 'OFF'}")
    print(f"  Unweighted:  {[c for c in crops if c in UNWEIGHTED_CROPS] or 'none'}")
    print(f"  Weighted:    {[c for c in crops if c not in UNWEIGHTED_CROPS] or 'none'}")
    print(f"  Köppen:      {[c for c in crops if c in KOPPEN_CROPS] or 'none'}")
    print("=" * 60)

    results = {}

    for crop in crops:
        print(f"\n{'━' * 50}")
        print(f"  {crop}")
        print(f"{'━' * 50}")

        crop_output = output_root / crop
        setup_logging(crop_output / "logs", crop)

        if not args.force and (crop_output / "predictions.parquet").exists():
            print(f"  [SKIP] Already done. Use --force.")
            results[crop] = "skipped"
            continue

        try:
            summary = predict_crop(
                crop, model_root, crop_output, cfg, pred_years,
                apply_filters=apply_filters,
            )

            if summary:
                print(f"  ✓ [{summary['pipeline']}] {summary['n_samples']:,} predictions "
                      f"(filtered {summary['filter_reduction_pct']:.1f}% from {summary['n_raw']:,})")
                print(f"    G={summary['global_mean']:.0f}, "
                      f"G+R={summary['gr_mean']:.0f} ({summary['gr_vs_global_change_pct']:+.1f}%), "
                      f"G+R+E={summary['gre_mean']:.0f} ({summary['gre_vs_gr_change_pct']:+.1f}%)")
                results[crop] = "done"
            else:
                results[crop] = "no_model"

        except Exception as e:
            print(f"  [FAIL] {e}")
            import traceback
            traceback.print_exc()
            results[crop] = f"failed: {e}"

    print(f"\n{'=' * 60}")
    print("Prediction Summary:")
    for crop, status in results.items():
        print(f"  {crop}: {status}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
