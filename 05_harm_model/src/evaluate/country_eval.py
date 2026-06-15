"""Aggregate grid-level HARM predictions to country level and compute metrics on eval data (2017-2018)."""

import argparse
import json
import gc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import joblib
from shapely.geometry import Point
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

from harm.config import load_config
from harm.constants import ALL_CROPS as CROPS_ALL

_cfg = load_config("config.yaml")
DEFAULT_RESULT_ROOT = _cfg.paths.result_root
DEFAULT_DATA_ROOT   = _cfg.paths.data_root
SHAPEFILE_PATH      = _cfg.paths.country_shapefile

EVAL_YEARS = [2017, 2018]


def load_model_and_predict(model_dir, X, features):
    """Load model, scaler, predict."""
    scaler = joblib.load(model_dir / "scaler.joblib")
    model = XGBRegressor()
    model.load_model(str(model_dir / "model_q50.json"))
    X_s = scaler.transform(X[features])
    y_pred = np.maximum(model.predict(X_s), 0)
    return y_pred, X_s, scaler


def load_regional_models(regional_dir):
    """Load adopted regional models."""
    summary_path = regional_dir / "summary.json"
    if not summary_path.exists():
        return {}
    with open(summary_path) as f:
        summary = json.load(f)
    models = {}
    for region_str in summary.get("trained_regions", []):
        rdir = regional_dir / region_str
        if not (rdir / "model_q50.json").exists():
            continue
        r_scaler = joblib.load(rdir / "scaler.joblib")
        r_model = XGBRegressor()
        r_model.load_model(str(rdir / "model_q50.json"))
        models[region_str] = (r_scaler, r_model)
    return models


def apply_regional(df, y_pred, X_scaled, regional_models):
    """Apply regional corrections in-place."""
    n_corrected = 0
    if "regiontype" not in df.columns or not regional_models:
        return y_pred, 0

    y_pred = y_pred.copy()
    for region_str, (r_scaler, r_model) in regional_models.items():
        mask = df["regiontype"].astype(int).astype(str).values == region_str
        if mask.sum() == 0:
            continue
        X_region = r_scaler.transform(X_scaled[mask])
        r_pred = r_model.predict(X_region)
        y_pred[mask] = np.maximum(y_pred[mask] + r_pred, 0)
        n_corrected += int(mask.sum())
    return y_pred, n_corrected


def spatial_join_countries(df, shapefile_path):
    """Join grid points to countries. Returns df with 'country' column."""
    print("  Loading shapefile...")
    countries = gpd.read_file(shapefile_path)
    if countries.crs is None:
        countries = countries.set_crs("EPSG:4326")
    elif str(countries.crs) != "EPSG:4326":
        countries = countries.to_crs("EPSG:4326")

    coords = df[["lat", "lon"]].drop_duplicates()
    print(f"  Joining {len(coords):,} unique grid points to countries...")

    geometry = [Point(xy) for xy in zip(coords["lon"], coords["lat"])]
    gdf = gpd.GeoDataFrame(coords, geometry=geometry, crs="EPSG:4326")
    joined = gpd.sjoin(gdf, countries[["ADMIN", "geometry"]], how="left", predicate="within")

    lookup = joined.groupby(["lat", "lon"])["ADMIN"].first().reset_index()
    lookup = lookup.rename(columns={"ADMIN": "country"})

    df = df.merge(lookup, on=["lat", "lon"], how="left")

    unmatched = df["country"].isna().sum()
    if unmatched > 0:
        pct = unmatched / len(df) * 100
        print(f"  ⚠ {unmatched:,} rows ({pct:.1f}%) unmatched → dropping")
        df = df.dropna(subset=["country"])

    print(f"  ✓ {df['country'].nunique()} countries matched")
    return df


def compute_country_metrics(df_agg):
    """Compute R², RMSE, correlation at country-year level."""
    y_true = df_agg["y_true_sum"].values
    y_pred = df_agg["y_pred_sum"].values

    n = len(y_true)
    if n < 3:
        return {}

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    corr = np.corrcoef(y_true, y_pred)[0, 1] if n >= 2 else 0

    mean_true = np.mean(y_true)
    nrmse = rmse / mean_true if mean_true > 0 else 0

    return {
        "n_country_years": n,
        "n_countries": df_agg["country"].nunique(),
        "r2": float(r2),
        "rmse": float(rmse),
        "mae": float(mae),
        "nrmse": float(nrmse),
        "correlation": float(corr),
        "mean_true": float(mean_true),
        "mean_pred": float(np.mean(y_pred)),
    }


def evaluate_crop(crop, result_root, data_root, shapefile_path,
                  model_subdir, regional_subdir, use_regional):
    """Full country-level evaluation for one crop."""
    print(f"\n{'─'*50}")
    print(f"  {crop.upper()}: Country-Level Evaluation")
    print(f"{'─'*50}")

    data_path = data_root / crop / f"{crop}.parquet"
    if not data_path.exists():
        print(f"  [SKIP] No data at {data_path}")
        return None

    df = pd.read_parquet(data_path)
    df["time"] = pd.to_datetime(df["time"])
    df = df[df["time"].dt.year.isin(EVAL_YEARS)].copy()
    print(f"  Eval data: {len(df):,} rows ({EVAL_YEARS})")

    if len(df) == 0:
        return None

    y_true = df[crop].values

    model_dir = result_root / crop / model_subdir
    if not (model_dir / "model_q50.json").exists():
        print(f"  [SKIP] No model at {model_dir}")
        return None

    with open(model_dir / "feature_names.json") as f:
        features = json.load(f)

    y_pred, X_scaled, scaler = load_model_and_predict(model_dir, df, features)
    print(f"  Global predictions done")

    n_corrected = 0
    if use_regional:
        regional_dir = result_root / crop / regional_subdir
        regional_models = load_regional_models(regional_dir)
        if regional_models:
            y_pred, n_corrected = apply_regional(df, y_pred, X_scaled, regional_models)
            print(f"  Regional corrections: {n_corrected:,} samples, {len(regional_models)} regions")

    df["y_true"] = y_true
    df["y_pred"] = y_pred
    df["year"] = df["time"].dt.year

    # Grid-level R² for reference
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    grid_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    df = spatial_join_countries(df, shapefile_path)

    # ── Country-Year aggregation (sum) ──
    agg_year = df.groupby(["country", "year"]).agg(
        y_true_sum=("y_true", "sum"),
        y_pred_sum=("y_pred", "sum"),
        n_grids=("y_true", "count"),
    ).reset_index()

    metrics_year = compute_country_metrics(agg_year)
    metrics_year["aggregation"] = "country_year"
    metrics_year["grid_r2"] = grid_r2

    print(f"\n  Results:")
    print(f"    Grid-level R²:         {grid_r2:.4f}")
    print(f"    Country-Year R²:       {metrics_year['r2']:.4f}")
    print(f"    Country-Year corr:     {metrics_year['correlation']:.4f}")
    print(f"    Country-Year NRMSE:    {metrics_year['nrmse']:.4f}")
    print(f"    N country-years:       {metrics_year['n_country_years']}")
    print(f"    N countries:           {metrics_year['n_countries']}")

    # ── Country-only aggregation (sum across all eval years) ──
    agg_country = df.groupby("country").agg(
        y_true_sum=("y_true", "sum"),
        y_pred_sum=("y_pred", "sum"),
        n_grids=("y_true", "count"),
    ).reset_index()

    metrics_country = compute_country_metrics(agg_country)
    print(f"    Country-Total R²:      {metrics_country['r2']:.4f}  (summed across years)")
    print(f"    Country-Total corr:    {metrics_country['correlation']:.4f}")

    output_dir = result_root / crop / "country_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{model_subdir}"
    if use_regional:
        suffix += f"_{regional_subdir}"

    agg_year.to_csv(output_dir / f"country_year{suffix}.csv", index=False)
    agg_country.to_csv(output_dir / f"country_total{suffix}.csv", index=False)

    all_metrics = {
        "crop": crop,
        "model": model_subdir,
        "regional": regional_subdir if use_regional else None,
        "grid_r2": grid_r2,
        "country_year": metrics_year,
        "country_total": metrics_country,
        "n_corrected": n_corrected,
    }
    with open(output_dir / f"metrics{suffix}.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    del df
    gc.collect()

    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="Country-Level Evaluation")
    parser.add_argument("--crops", nargs="+", default=None)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--shapefile", type=Path, default=SHAPEFILE_PATH)
    parser.add_argument("--model-subdir", default="global")
    parser.add_argument("--regional", action="store_true")
    parser.add_argument("--regional-subdir", default=None)
    args = parser.parse_args()

    crops = args.crops or CROPS_ALL

    if args.regional and args.regional_subdir is None:
        args.regional_subdir = (
            "regional_weighted" if "weighted" in args.model_subdir else "regional"
        )

    mode_str = args.model_subdir
    if args.regional:
        mode_str += f" + {args.regional_subdir}"

    print("=" * 60)
    print("HARM Country-Level Evaluation")
    print(f"  Crops:    {crops}")
    print(f"  Model:    {mode_str}")
    print(f"  Eval:     {EVAL_YEARS}")
    print("=" * 60)

    all_results = []

    for crop in crops:
        try:
            result = evaluate_crop(
                crop, args.result_root, args.data_root, args.shapefile,
                args.model_subdir, args.regional_subdir, args.regional,
            )
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"  [FAIL] {crop}: {e}")
            import traceback
            traceback.print_exc()

    # ── Summary table ──
    if all_results:
        print(f"\n{'='*80}")
        print(f"SUMMARY: Country-Level R² ({mode_str})")
        print(f"{'='*80}")
        print(f"{'Crop':<6} {'Grid R²':>9} {'Ctry-Yr R²':>12} {'Ctry-Tot R²':>13} {'Corr':>8} {'N_ctry':>8}")
        print("─" * 60)

        for r in sorted(all_results, key=lambda x: x["country_year"]["r2"], reverse=True):
            cy = r["country_year"]
            ct = r["country_total"]
            print(f"{r['crop']:<6} {r['grid_r2']:>9.4f} {cy['r2']:>12.4f} "
                  f"{ct['r2']:>13.4f} {cy['correlation']:>8.4f} {cy['n_countries']:>8}")

        summary_df = pd.DataFrame([
            {
                "crop": r["crop"],
                "grid_r2": r["grid_r2"],
                "country_year_r2": r["country_year"]["r2"],
                "country_total_r2": r["country_total"]["r2"],
                "correlation": r["country_year"]["correlation"],
                "n_countries": r["country_year"]["n_countries"],
                "n_country_years": r["country_year"]["n_country_years"],
            }
            for r in all_results
        ])
        out_path = args.result_root / f"country_eval_summary_{args.model_subdir}.csv"
        summary_df.to_csv(out_path, index=False)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
