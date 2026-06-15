"""HARM eval predictions: apply trained G -> G+R -> G+R+E on held-out 2017-2018, saving y_true alongside."""

import argparse
import gc
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor, XGBClassifier

warnings.filterwarnings("ignore")

from harm.extreme_layer import _apply_bounded_correction

from harm.config import load_config
from harm.constants import ALL_CROPS, UNWEIGHTED_CROPS, KOPPEN_CROPS

EVAL_YEARS = [2017, 2018]

# Extreme layer config (same as run_predict.py defaults)
EXTREME_CFG = {
    "detector_prob_threshold": 0.6,
    "max_correction_sample_ratio": 0.05,
    "max_relative_correction": 0.5,
    "conservation_tolerance": 0.05,
}

QUANTILES = [0.1, 0.5, 0.9]


# Prediction functions copied verbatim from run_predict.py
def predict_global(df, global_dir, feature_cols, quantiles):
    scaler = joblib.load(global_dir / "scaler.joblib")
    with open(global_dir / "feature_names.json") as f:
        model_features = json.load(f)

    missing = set(model_features) - set(df.columns)
    if missing:
        print(f"    ⚠ Missing features: {missing}")
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

    return predictions, X_scaled, model_features


def predict_regional(df, X_global_scaled, y_global, regional_dir):
    summary_path = regional_dir / "summary.json"
    if not summary_path.exists():
        return y_global.copy()

    with open(summary_path) as f:
        summary = json.load(f)

    adopted = summary.get("trained_regions", [])
    if not adopted or "regiontype" not in df.columns:
        return y_global.copy()

    y_out = y_global.copy()
    n_corrected = 0

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
        n_corrected += n_match
        del r_model

    print(f"    Regional: corrected {n_corrected:,}/{len(df):,} samples")
    return y_out


def predict_regional_quantiles(df, X_global_scaled, global_preds, regional_dir, quantiles):
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
    summary_path = extreme_dir / "summary.json"
    if not summary_path.exists():
        return y_gr.copy()

    with open(summary_path) as f:
        summary = json.load(f)

    adopted = summary.get("trained_regions", [])
    if not adopted or "regiontype" not in df.columns:
        return y_gr.copy()

    y_out = y_gr.copy()
    prob_threshold = ext_cfg.get("detector_prob_threshold", 0.6)
    max_corr_ratio = ext_cfg.get("max_correction_sample_ratio", 0.05)
    max_rel_corr = ext_cfg.get("max_relative_correction", 0.5)
    conservation_tol = ext_cfg.get("conservation_tolerance", 0.05)

    n_total_corrected = 0
    for region_str in adopted:
        rdir = extreme_dir / region_str
        det_path = rdir / "detector.json"
        corr_path = rdir / "corrector.json"
        scaler_path = rdir / "scaler.joblib"
        feat_path = rdir / "feature_names.json"

        if not all(p.exists() for p in [det_path, corr_path, scaler_path, feat_path]):
            continue

        with open(feat_path) as f:
            ext_features = json.load(f)

        e_scaler = joblib.load(scaler_path)
        detector = XGBClassifier()
        detector.load_model(str(det_path))
        corrector = XGBRegressor()
        corrector.load_model(str(corr_path))

        region_val = int(region_str) if region_str.isdigit() else region_str
        mask = df["regiontype"].values == region_val
        n_match = int(mask.sum())

        if n_match == 0:
            del detector, corrector, e_scaler
            continue

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

        del detector, corrector, e_scaler
        gc.collect()

    print(f"    Extreme: corrected {n_total_corrected:,}/{len(df):,} samples")
    return y_out


def eval_crop(crop, model_root, data_root_9, data_root_koppen):
    """Load 2017-2018 data, predict through all layers, save with y_true."""

    use_weighted = crop not in UNWEIGHTED_CROPS
    if use_weighted:
        global_dir = model_root / crop / "global_weighted"
        regional_dir = model_root / crop / "regional_weighted"
        extreme_dir = model_root / crop / "extreme_weighted"
        pipeline = "GW+RW+EW"
    else:
        global_dir = model_root / crop / "global"
        regional_dir = model_root / crop / "regional"
        extreme_dir = model_root / crop / "extreme"
        pipeline = "G+R+E"

    if not (global_dir / "model_q50.json").exists():
        print(f"    ⚠ No model at {global_dir}")
        return None

    data_root = data_root_koppen if crop in KOPPEN_CROPS else data_root_9

    fpath = data_root / crop / f"{crop}.parquet"
    if not fpath.exists():
        print(f"    ⚠ No data at {fpath}")
        return None

    print(f"    Loading {fpath.name}...")
    df = pd.read_parquet(fpath)
    df["time"] = pd.to_datetime(df["time"])
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month

    df = df[df["year"].isin(EVAL_YEARS)].copy().reset_index(drop=True)
    print(f"    Eval data: {len(df):,} rows ({df['year'].unique().tolist()})")

    if len(df) == 0:
        return None

    # The crop column holds the target value
    y_true = df[crop].values.astype(np.float64)

    print(f"    Global layer...")
    global_preds, X_scaled, model_features = predict_global(
        df, global_dir, None, QUANTILES
    )

    print(f"    Regional layer...")
    y_gr = predict_regional(df, X_scaled, global_preds[0.5], regional_dir)
    gr_quantiles = predict_regional_quantiles(
        df, X_scaled, global_preds, regional_dir, QUANTILES
    )

    print(f"    Extreme layer...")
    y_gre = predict_extreme(df, y_gr, extreme_dir, EXTREME_CFG)

    out = pd.DataFrame({
        "time": df["time"].values,
        "lat": df["lat"].values,
        "lon": df["lon"].values,
        "grid50_id": df["grid50_id"].values,
        "regiontype": df["regiontype"].values if "regiontype" in df.columns else np.nan,
        "year": df["year"].values,
        "month": df["month"].values,
        "y_true": y_true,
        "y_pred_global_q10": global_preds.get(0.1, np.zeros(len(df))),
        "y_pred_global_q50": global_preds[0.5],
        "y_pred_global_q90": global_preds.get(0.9, np.zeros(len(df))),
        "y_pred_gr_q10": gr_quantiles.get(0.1, global_preds.get(0.1, np.zeros(len(df)))),
        "y_pred_gr_q50": y_gr,
        "y_pred_gr_q90": gr_quantiles.get(0.9, global_preds.get(0.9, np.zeros(len(df)))),
        # Extreme layer produces a Q50 point estimate only
        "y_pred_gre_q50": y_gre,
    })

    mask_pos = y_true > 0
    if mask_pos.sum() > 0:
        corr = np.corrcoef(y_true[mask_pos], out["y_pred_gre_q50"].values[mask_pos])[0, 1]
    else:
        corr = np.nan
    total_ratio = out["y_pred_gre_q50"].sum() / y_true.sum() if y_true.sum() > 0 else np.nan

    print(f"    Sanity: n={len(out):,}, n_pos={mask_pos.sum():,}, "
          f"corr(pos)={corr:.4f}, total_ratio={total_ratio:.4f}")

    return out


def main():
    parser = argparse.ArgumentParser(description="HARM Eval Predictions (2017-2018)")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--phase", choices=["eval", "final"], default="eval", help="Which config phase block to use (default: eval)")
    parser.add_argument("--crops", nargs="+", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config, phase=args.phase)
    crops = args.crops or ALL_CROPS
    model_root = cfg.paths.result_root
    data_root_9 = cfg.paths.data_root
    data_root_koppen = cfg.paths.data_root_koppen
    output_root = model_root / "eval_predictions"
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("HARM Eval Predictions (2017-2018)")
    print(f"  Crops:   {len(crops)} crops")
    print(f"  Models:  {model_root}")
    print(f"  Data:    {data_root_9}")
    print(f"  Output:  {output_root}")
    print("=" * 60)

    results = {}

    for crop in crops:
        out_path = output_root / f"{crop}_eval_predictions.parquet"

        if not args.force and out_path.exists():
            print(f"\n  {crop}: [SKIP] exists. Use --force.")
            results[crop] = "skipped"
            continue

        print(f"\n{'━' * 50}")
        print(f"  {crop.upper()} ({'G+R+E' if crop in UNWEIGHTED_CROPS else 'GW+RW+EW'})")
        print(f"{'━' * 50}")

        try:
            out = eval_crop(crop, model_root, data_root_9, data_root_koppen)
            if out is not None:
                out.to_parquet(out_path, index=False)
                print(f"    ✅ Saved: {out_path} ({len(out):,} rows)")
                results[crop] = "done"
            else:
                results[crop] = "no_data"
        except Exception as e:
            print(f"    ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results[crop] = f"failed: {e}"

        gc.collect()

    print(f"\n{'=' * 60}")
    print("Summary:")
    done = sum(1 for v in results.values() if v == "done")
    print(f"  ✅ Done: {done}/{len(crops)}")
    for crop, status in results.items():
        if status != "done":
            print(f"  {crop}: {status}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()