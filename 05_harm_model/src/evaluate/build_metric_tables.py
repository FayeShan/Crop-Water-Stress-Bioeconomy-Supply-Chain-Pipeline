"""Assemble paper-ready per-crop metric tables from eval-phase model outputs."""

import json, joblib, gc, sys
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import r2_score, mean_squared_error

from harm.extreme_layer import _apply_bounded_correction

from harm.config import load_config
from harm.constants import ALL_CROPS, UNWEIGHTED_CROPS as UNWEIGHTED, KOPPEN_CROPS as KOPPEN

_cfg = load_config("config.yaml")
EVAL = _cfg.paths.result_root
DATA = _cfg.paths.data_root
DATA_KOP = _cfg.paths.data_root_koppen
OUT = _cfg.paths.result_root / "tables"
OUT.mkdir(parents=True, exist_ok=True)

EVAL_YEARS = [2017, 2018]

def compute_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nrmse = rmse / (y_true.max() - y_true.min()) if (y_true.max() - y_true.min()) > 0 else 0
    p90 = np.percentile(y_true, 90)
    extreme_mask = y_true >= p90
    hit_rate = (y_pred[extreme_mask] >= p90).mean() if extreme_mask.sum() > 0 else 0
    bias = float((y_pred - y_true).mean())
    return {"r2": r2, "rmse": rmse, "nrmse": nrmse, "hit_p90": hit_rate, "bias": bias}


def predict_layers(df, crop, use_weighted):
    """Predict all 3 layers, return y_g, y_gr, y_gre."""
    g_sub = "global_weighted" if use_weighted else "global"
    r_sub = "regional_weighted" if use_weighted else "regional"
    e_sub = "extreme_weighted" if use_weighted else "extreme"

    g_dir = EVAL / crop / g_sub
    r_dir = EVAL / crop / r_sub
    e_dir = EVAL / crop / e_sub

    with open(g_dir / "feature_names.json") as f:
        feats = json.load(f)
    scaler = joblib.load(g_dir / "scaler.joblib")
    model = XGBRegressor()
    model.load_model(str(g_dir / "model_q50.json"))

    X_s = scaler.transform(df[feats].values)
    y_g = np.maximum(model.predict(X_s), 0)

    y_gr = y_g.copy()
    if (r_dir / "summary.json").exists() and "regiontype" in df.columns:
        with open(r_dir / "summary.json") as f:
            adopted_r = json.load(f).get("trained_regions", [])
        for r_str in adopted_r:
            rdir = r_dir / r_str
            if not (rdir / "model_q50.json").exists():
                continue
            r_scaler = joblib.load(rdir / "scaler.joblib")
            r_model = XGBRegressor()
            r_model.load_model(str(rdir / "model_q50.json"))
            region_val = int(r_str) if r_str.isdigit() else r_str
            mask = df["regiontype"].values == region_val
            if mask.sum() > 0:
                y_gr[mask] = np.maximum(y_g[mask] + r_model.predict(r_scaler.transform(X_s[mask])), 0)

    y_gre = y_gr.copy()
    if (e_dir / "summary.json").exists() and "regiontype" in df.columns:
        with open(e_dir / "summary.json") as f:
            adopted_e = json.load(f).get("trained_regions", [])
        for r_str in adopted_e:
            edir = e_dir / r_str
            if not (edir / "detector.json").exists():
                continue
            e_scaler = joblib.load(edir / "scaler.joblib")
            with open(edir / "feature_names.json") as f:
                e_feats = json.load(f)
            det = XGBClassifier()
            det.load_model(str(edir / "detector.json"))
            corr = XGBRegressor()
            corr.load_model(str(edir / "corrector.json"))
            region_val = int(r_str) if r_str.isdigit() else r_str
            mask = df["regiontype"].values == region_val
            if mask.sum() == 0:
                continue
            X_es = e_scaler.transform(df[e_feats].values[mask])
            det_probs = det.predict_proba(X_es)[:, 1]
            corr_pred = corr.predict(X_es)
            y_corrected, _ = _apply_bounded_correction(
                y_gr[mask], det_probs, corr_pred, 0.6, 0.05, 0.5, 0.05)
            y_gre[mask] = y_corrected

    return y_g, y_gr, y_gre, feats


all_rows = []

for crop in ALL_CROPS:
    print(f"\n{'─'*40}\n  {crop.upper()}\n{'─'*40}", flush=True)

    data_root = DATA_KOP if crop in KOPPEN else DATA
    df = pd.read_parquet(data_root / crop / f"{crop}.parquet")
    df["year"] = pd.to_datetime(df["time"]).dt.year
    df = df[df["year"].isin(EVAL_YEARS)]

    if len(df) == 0:
        print("  [SKIP] No eval data")
        continue

    y_true = df[crop].values
    use_w = crop not in UNWEIGHTED

    try:
        y_g, y_gr, y_gre, feats = predict_layers(df, crop, use_w)
    except Exception as e:
        print(f"  [FAIL] {e}")
        continue

    prefix = "GW" if use_w else "G"
    layers = [
        (f"{prefix}", y_g),
        (f"{prefix}+R{'W' if use_w else ''}", y_gr),
        (f"{prefix}+R{'W' if use_w else ''}+E", y_gre),
    ]

    for label, y_pred in layers:
        m = compute_metrics(y_true, y_pred)
        row = {"crop": crop, "layer": label, **m}
        all_rows.append(row)
        print(f"  {label:>10}: R²={m['r2']:.4f}, NRMSE={m['nrmse']:.6f}, "
              f"hit_p90={m['hit_p90']:.3f}, bias={m['bias']:.0f}")

    del df; gc.collect()

df_all = pd.DataFrame(all_rows)
df_all.to_csv(OUT / "table_s1b_all_layers_metrics.csv", index=False)

pivot = df_all.pivot(index="crop", columns="layer", values=["r2", "nrmse", "hit_p90"])
pivot.columns = [f"{metric}_{layer}" for metric, layer in pivot.columns]
pivot = pivot.reset_index()
pivot.to_csv(OUT / "table_s1b_pivot.csv", index=False)

print(f"\n{'='*70}")
print("Summary: Mean metrics by layer")
print("=" * 70)
summary = df_all.groupby("layer")[["r2", "nrmse", "hit_p90"]].mean().round(4)
print(summary.to_string())

print(f"\nSaved to {OUT}/")