"""Compute metrics at all aggregation levels (grid/country/global) from eval predictions."""

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats
from pathlib import Path
from tqdm import tqdm


from harm.config import load_config
from harm.constants import ALL_CROPS

_cfg = load_config("config.yaml")
EVAL_DIR = _cfg.paths.result_root / "eval_predictions"
SHAPEFILE = _cfg.paths.country_shapefile
OUT_DIR = _cfg.paths.result_root / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Layers to evaluate (columns in eval_predictions parquet)
# Format: (layer_name, q50_col, q10_col, q90_col)
LAYERS = [
    ("global",   "y_pred_global_q50", "y_pred_global_q10", "y_pred_global_q90"),
    ("regional", "y_pred_gr_q50",     "y_pred_gr_q10",     "y_pred_gr_q90"),
    ("extreme",  "y_pred_gre_q50",    "y_pred_gr_q10",     "y_pred_gr_q90"),  # extreme has no own Q10/Q90, reuse G+R
]

def compute_metrics(y_true, y_pred, y_q10=None, y_q90=None):
    """Compute a comprehensive set of metrics."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = y_true[mask]
    yp = y_pred[mask]
    n = len(yt)

    if n < 3:
        return {"n": n, "r2": np.nan, "rmse": np.nan, "mae": np.nan,
                "nrmse": np.nan, "bias": np.nan, "corr": np.nan,
                "kge": np.nan, "kge_r": np.nan, "kge_alpha": np.nan,
                "kge_beta": np.nan, "pbias": np.nan,
                "picp_80": np.nan, "hit_p90": np.nan}

    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = np.sqrt(np.mean((yt - yp) ** 2))
    mae = np.mean(np.abs(yt - yp))
    nrmse = rmse / np.mean(yt) if np.mean(yt) > 0 else np.nan
    bias = np.mean(yp - yt)
    corr = np.corrcoef(yt, yp)[0, 1] if np.std(yt) > 0 and np.std(yp) > 0 else np.nan

    result = {
        "n": n, "r2": r2, "rmse": rmse, "mae": mae,
        "nrmse": nrmse, "bias": bias, "corr": corr,
    }

    # KGE (Kling-Gupta Efficiency) — standard in hydrology
    if np.std(yt) > 0 and np.std(yp) > 0 and np.mean(yt) > 0:
        r = np.corrcoef(yt, yp)[0, 1]
        alpha = np.std(yp) / np.std(yt)       # variability ratio
        beta = np.mean(yp) / np.mean(yt)      # bias ratio
        kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        result["kge"] = kge
        result["kge_r"] = r
        result["kge_alpha"] = alpha
        result["kge_beta"] = beta
    else:
        result["kge"] = np.nan
        result["kge_r"] = np.nan
        result["kge_alpha"] = np.nan
        result["kge_beta"] = np.nan

    # Percent bias (PBIAS) — positive = overprediction
    result["pbias"] = (np.sum(yp - yt) / np.sum(yt) * 100) if np.sum(yt) > 0 else np.nan

    # Prediction interval coverage (PICP)
    if y_q10 is not None and y_q90 is not None:
        q10 = y_q10[mask]
        q90 = y_q90[mask]
        covered = ((yt >= q10) & (yt <= q90)).sum()
        result["picp_80"] = covered / n if n > 0 else np.nan
    else:
        result["picp_80"] = np.nan

    # Hit rate for P90 (capturing high-stress events)
    if n > 10:
        p90_threshold = np.percentile(yt[yt > 0], 90) if (yt > 0).sum() > 10 else np.nan
        if np.isfinite(p90_threshold) and p90_threshold > 0:
            actual_high = yt >= p90_threshold
            pred_high = yp >= p90_threshold * 0.5  # Within 50% of threshold
            if actual_high.sum() > 0:
                result["hit_p90"] = (actual_high & pred_high).sum() / actual_high.sum()
            else:
                result["hit_p90"] = np.nan
        else:
            result["hit_p90"] = np.nan
    else:
        result["hit_p90"] = np.nan

    # Ratio of totals (important for yearly comparison)
    total_true = yt.sum()
    total_pred = yp.sum()
    result["total_ratio"] = total_pred / total_true if total_true > 0 else np.nan

    return result


def build_country_lookup(df, shapefile_path):
    """Assign country to each (lat, lon) pair."""
    from shapely.geometry import Point

    coords = df[["lat", "lon"]].drop_duplicates()
    gdf = gpd.GeoDataFrame(
        coords,
        geometry=[Point(xy) for xy in zip(coords["lon"], coords["lat"])],
        crs="EPSG:4326"
    )
    countries = gpd.read_file(shapefile_path)
    if countries.crs is None:
        countries = countries.set_crs("EPSG:4326")

    joined = gpd.sjoin(gdf, countries[["ADMIN", "geometry"]], how="left", predicate="within")
    lookup = joined.groupby(["lat", "lon"]).first()[["ADMIN"]].reset_index()
    lookup.rename(columns={"ADMIN": "country"}, inplace=True)
    return lookup


all_metrics = []

# ── Build country lookup from ALL crops' coordinates (not just the first!) ──
print("Collecting all unique (lat, lon) across crops...")
all_coords = []
valid_crops = []
for crop in ALL_CROPS:
    eval_path = EVAL_DIR / f"{crop}_eval_predictions.parquet"
    if not eval_path.exists():
        continue
    valid_crops.append(crop)
    df_tmp = pd.read_parquet(eval_path, columns=["lat", "lon"])
    all_coords.append(df_tmp[["lat", "lon"]].drop_duplicates())
    del df_tmp

if all_coords:
    all_coords_df = pd.concat(all_coords, ignore_index=True).drop_duplicates()
    print(f"  Total unique points: {len(all_coords_df):,}")
    print("Building country lookup (one-time, all crops)...")
    country_lookup = build_country_lookup(all_coords_df, SHAPEFILE)
    n_matched = country_lookup["country"].notna().sum()
    print(f"  Country lookup: {len(country_lookup):,} points, {n_matched:,} matched")
    del all_coords, all_coords_df
else:
    country_lookup = None
    valid_crops = ALL_CROPS

for crop in tqdm(ALL_CROPS, desc="Computing metrics"):
    eval_path = EVAL_DIR / f"{crop}_eval_predictions.parquet"
    if not eval_path.exists():
        print(f"  ⚠️  {crop}: no eval predictions found, skipping")
        continue

    df = pd.read_parquet(eval_path)
    print(f"\n  {crop}: {len(df):,} rows")

    if country_lookup is not None and "lat" in df.columns:
        df = df.merge(country_lookup, on=["lat", "lon"], how="left")
        n_with_country = df["country"].notna().sum()
        print(f"    Country matched: {n_with_country:,}/{len(df):,} "
              f"({n_with_country/len(df)*100:.1f}%)")

    for layer_name, q50_col, q10_col, q90_col in LAYERS:
        if q50_col not in df.columns:
            continue

        # ── Level 1: Monthly grid ──
        m = compute_metrics(
            df["y_true"].values, df[q50_col].values,
            df[q10_col].values if q10_col in df.columns else None,
            df[q90_col].values if q90_col in df.columns else None,
        )
        m.update({"crop": crop, "layer": layer_name, "level": "monthly_grid"})
        all_metrics.append(m)

        # ── Level 2: Yearly grid ──
        yg = df.groupby(["lat", "lon", "year"]).agg(
            y_true=("y_true", "sum"),
            y_pred=(q50_col, "sum"),
            y_q10=(q10_col, "sum") if q10_col in df.columns else ("y_true", "count"),
            y_q90=(q90_col, "sum") if q90_col in df.columns else ("y_true", "count"),
        ).reset_index()
        m = compute_metrics(
            yg["y_true"].values, yg["y_pred"].values,
            yg["y_q10"].values if q10_col in df.columns else None,
            yg["y_q90"].values if q90_col in df.columns else None,
        )
        m.update({"crop": crop, "layer": layer_name, "level": "yearly_grid"})
        all_metrics.append(m)

        # ── Level 3: Monthly country ──
        if "country" in df.columns:
            mc = df.groupby(["country", "year", "month"]).agg(
                y_true=("y_true", "sum"),
                y_pred=(q50_col, "sum"),
                y_q10=(q10_col, "sum") if q10_col in df.columns else ("y_true", "count"),
                y_q90=(q90_col, "sum") if q90_col in df.columns else ("y_true", "count"),
            ).reset_index()
            mc = mc[mc["y_true"].notna()]  # Drop NaN only (match old code behavior)
            m = compute_metrics(
                mc["y_true"].values, mc["y_pred"].values,
                mc["y_q10"].values if q10_col in df.columns else None,
                mc["y_q90"].values if q90_col in df.columns else None,
            )
            m.update({"crop": crop, "layer": layer_name, "level": "monthly_country"})
            all_metrics.append(m)

        # ── Level 4: Yearly country ──
        if "country" in df.columns:
            yc = df.groupby(["country", "year"]).agg(
                y_true=("y_true", "sum"),
                y_pred=(q50_col, "sum"),
            ).reset_index()
            yc = yc[yc["y_true"].notna()]  # Keep all country-years (match old code)
            m = compute_metrics(yc["y_true"].values, yc["y_pred"].values)
            m.update({"crop": crop, "layer": layer_name, "level": "yearly_country"})
            all_metrics.append(m)

        # ── Level 5: Yearly global ──
        yglob = df.groupby("year").agg(
            y_true=("y_true", "sum"),
            y_pred=(q50_col, "sum"),
        ).reset_index()
        m = compute_metrics(yglob["y_true"].values, yglob["y_pred"].values)
        m["total_ratio_2017"] = (
            yglob.loc[yglob["year"] == 2017, "y_pred"].values[0] /
            yglob.loc[yglob["year"] == 2017, "y_true"].values[0]
            if 2017 in yglob["year"].values and yglob.loc[yglob["year"] == 2017, "y_true"].values[0] > 0
            else np.nan
        )
        m["total_ratio_2018"] = (
            yglob.loc[yglob["year"] == 2018, "y_pred"].values[0] /
            yglob.loc[yglob["year"] == 2018, "y_true"].values[0]
            if 2018 in yglob["year"].values and yglob.loc[yglob["year"] == 2018, "y_true"].values[0] > 0
            else np.nan
        )
        m.update({"crop": crop, "layer": layer_name, "level": "yearly_global"})
        all_metrics.append(m)

    del df


print(f"\n{'━' * 60}")
print("  ALL CROPS (summed across crops)")
print(f"{'━' * 60}")

all_dfs = []
for crop in ALL_CROPS:
    eval_path = EVAL_DIR / f"{crop}_eval_predictions.parquet"
    if not eval_path.exists():
        continue
    cdf = pd.read_parquet(eval_path)
    cdf["crop"] = crop
    if country_lookup is not None and "lat" in cdf.columns:
        cdf = cdf.merge(country_lookup, on=["lat", "lon"], how="left")
    all_dfs.append(cdf)

if all_dfs:
    df_all = pd.concat(all_dfs, ignore_index=True)
    del all_dfs
    print(f"  Stacked: {len(df_all):,} rows from {df_all['crop'].nunique()} crops")

    for layer_name, q50_col, q10_col, q90_col in LAYERS:
        if q50_col not in df_all.columns:
            continue

        # ── Level 1: Monthly grid (sum all crops per cell×month) ──
        mg = df_all.groupby(["lat", "lon", "year", "month"]).agg(
            y_true=("y_true", "sum"),
            y_pred=(q50_col, "sum"),
            y_q10=(q10_col, "sum") if q10_col in df_all.columns else ("y_true", "count"),
            y_q90=(q90_col, "sum") if q90_col in df_all.columns else ("y_true", "count"),
        ).reset_index()
        m = compute_metrics(
            mg["y_true"].values, mg["y_pred"].values,
            mg["y_q10"].values if q10_col in df_all.columns else None,
            mg["y_q90"].values if q90_col in df_all.columns else None,
        )
        m.update({"crop": "ALL", "layer": layer_name, "level": "monthly_grid"})
        all_metrics.append(m)

        # ── Level 2: Yearly grid (sum all crops per cell×year) ──
        yg = df_all.groupby(["lat", "lon", "year"]).agg(
            y_true=("y_true", "sum"),
            y_pred=(q50_col, "sum"),
            y_q10=(q10_col, "sum") if q10_col in df_all.columns else ("y_true", "count"),
            y_q90=(q90_col, "sum") if q90_col in df_all.columns else ("y_true", "count"),
        ).reset_index()
        m = compute_metrics(
            yg["y_true"].values, yg["y_pred"].values,
            yg["y_q10"].values if q10_col in df_all.columns else None,
            yg["y_q90"].values if q90_col in df_all.columns else None,
        )
        m.update({"crop": "ALL", "layer": layer_name, "level": "yearly_grid"})
        all_metrics.append(m)

        # ── Level 3: Monthly country (sum all crops per country×month) ──
        if "country" in df_all.columns:
            mc = df_all.groupby(["country", "year", "month"]).agg(
                y_true=("y_true", "sum"),
                y_pred=(q50_col, "sum"),
                y_q10=(q10_col, "sum") if q10_col in df_all.columns else ("y_true", "count"),
                y_q90=(q90_col, "sum") if q90_col in df_all.columns else ("y_true", "count"),
            ).reset_index()
            mc = mc[mc["y_true"].notna()]  # Keep all (match old code)
            m = compute_metrics(
                mc["y_true"].values, mc["y_pred"].values,
                mc["y_q10"].values if q10_col in df_all.columns else None,
                mc["y_q90"].values if q90_col in df_all.columns else None,
            )
            m.update({"crop": "ALL", "layer": layer_name, "level": "monthly_country"})
            all_metrics.append(m)

        # ── Level 4: Yearly country (sum all crops per country×year) ──
        if "country" in df_all.columns:
            yc = df_all.groupby(["country", "year"]).agg(
                y_true=("y_true", "sum"),
                y_pred=(q50_col, "sum"),
            ).reset_index()
            yc = yc[yc["y_true"].notna()]  # Keep all country-years (match old code)
            m = compute_metrics(yc["y_true"].values, yc["y_pred"].values)
            m.update({"crop": "ALL", "layer": layer_name, "level": "yearly_country"})
            all_metrics.append(m)

        # ── Level 5: Yearly global ──
        yglob = df_all.groupby("year").agg(
            y_true=("y_true", "sum"),
            y_pred=(q50_col, "sum"),
        ).reset_index()
        m = compute_metrics(yglob["y_true"].values, yglob["y_pred"].values)
        m["total_ratio_2017"] = (
            yglob.loc[yglob["year"] == 2017, "y_pred"].values[0] /
            yglob.loc[yglob["year"] == 2017, "y_true"].values[0]
            if 2017 in yglob["year"].values and yglob.loc[yglob["year"] == 2017, "y_true"].values[0] > 0
            else np.nan
        )
        m["total_ratio_2018"] = (
            yglob.loc[yglob["year"] == 2018, "y_pred"].values[0] /
            yglob.loc[yglob["year"] == 2018, "y_true"].values[0]
            if 2018 in yglob["year"].values and yglob.loc[yglob["year"] == 2018, "y_true"].values[0] > 0
            else np.nan
        )
        m.update({"crop": "ALL", "layer": layer_name, "level": "yearly_global"})
        all_metrics.append(m)

    del df_all
else:
    print("  ⚠️  No crop files found for ALL-crop aggregation")


df_metrics = pd.DataFrame(all_metrics)
out_path = OUT_DIR / "eval_metrics_multilevel.csv"
df_metrics.to_csv(out_path, index=False)
print(f"\n\nSaved: {out_path}")

# ── Summary table: R² by level ──
print(f"\n{'='*80}")
print("R² SUMMARY BY AGGREGATION LEVEL (extreme layer, Q50)")
print(f"{'='*80}")

summary = df_metrics[df_metrics["layer"] == "extreme"].pivot_table(
    index="crop", columns="level", values="r2"
)
col_order = ["monthly_grid", "yearly_grid", "monthly_country", "yearly_country"]
col_order = [c for c in col_order if c in summary.columns]
summary = summary[col_order]

print(f"\n{'crop':>4}", end="")
for col in col_order:
    short = col.replace("monthly_", "m_").replace("yearly_", "y_").replace("country", "ctry")
    print(f"  {short:>10}", end="")
print()
print("─" * (6 + 12 * len(col_order)))

for crop in summary.index:
    marker = " ◀" if crop == "ALL" else ""
    print(f"{crop:>4}", end="")
    for col in col_order:
        v = summary.loc[crop, col]
        print(f"  {v:>10.3f}" if pd.notna(v) else f"  {'N/A':>10}", end="")
    print(marker)

# ── Mean R² across crops (excluding ALL) ──
print("─" * (6 + 12 * len(col_order)))
per_crop = summary.drop("ALL", errors="ignore")
print(f"{'MEAN':>4}", end="")
for col in col_order:
    print(f"  {per_crop[col].mean():>10.3f}", end="")
print("  (per-crop mean)")
if "ALL" in summary.index:
    print(f"{'ALL':>4}", end="")
    for col in col_order:
        v = summary.loc["ALL", col]
        print(f"  {v:>10.3f}" if pd.notna(v) else f"  {'N/A':>10}", end="")
    print("  ◀ all crops summed")

# ── KGE summary ──
print(f"\n\n{'='*80}")
print("KGE SUMMARY BY AGGREGATION LEVEL (extreme layer)")
print(f"{'='*80}")
ext = df_metrics[df_metrics["layer"] == "extreme"]
for metric_name, metric_col, fmt_str in [
    ("KGE", "kge", ".3f"),
    ("NRMSE", "nrmse", ".3f"),
    ("PBIAS%", "pbias", ".1f"),
    ("PICP_80", "picp_80", ".3f"),
]:
    if metric_col not in ext.columns:
        continue
    tbl = ext.pivot_table(index="crop", columns="level", values=metric_col)
    tbl_cols = [c for c in col_order if c in tbl.columns]
    if not tbl_cols:
        continue
    tbl = tbl[tbl_cols]
    
    print(f"\n  {metric_name}:")
    print(f"  {'crop':>4}", end="")
    for col in tbl_cols:
        short = col.replace("monthly_", "m_").replace("yearly_", "y_").replace("country", "ctry")
        print(f"  {short:>10}", end="")
    print()
    print("  " + "─" * (4 + 12 * len(tbl_cols)))
    
    per = tbl.drop("ALL", errors="ignore")
    print(f"  {'MEAN':>4}", end="")
    for col in tbl_cols:
        print(f"  {per[col].mean():>10{fmt_str}}", end="")
    print()
    if "ALL" in tbl.index:
        print(f"  {'ALL':>4}", end="")
        for col in tbl_cols:
            v = tbl.loc["ALL", col]
            print(f"  {v:>10{fmt_str}}" if pd.notna(v) else f"  {'N/A':>10}", end="")
        print(" ◀")

# ── Key message for paper ──
print(f"\n\n{'='*80}")
print("KEY METRICS FOR PAPER (extreme layer)")
print(f"{'='*80}")

ext_all = df_metrics[(df_metrics["layer"] == "extreme") & (df_metrics["crop"] == "ALL")]
ext_per = df_metrics[(df_metrics["layer"] == "extreme") & (df_metrics["crop"] != "ALL")]

for level in col_order:
    r2_vals = per_crop[level].dropna() if level in per_crop.columns else pd.Series()
    all_row = ext_all[ext_all["level"] == level]
    
    print(f"\n  {level}:")
    if not all_row.empty:
        ar = all_row.iloc[0]
        print(f"    ALL crops: R²={ar.get('r2', np.nan):.3f}, "
              f"KGE={ar.get('kge', np.nan):.3f}, "
              f"NRMSE={ar.get('nrmse', np.nan):.3f}, "
              f"PBIAS={ar.get('pbias', np.nan):.1f}%")
    if len(r2_vals) > 0:
        above_07 = (r2_vals > 0.7).sum()
        print(f"    Per-crop:  mean R²={r2_vals.mean():.3f}, "
              f"median={r2_vals.median():.3f}, "
              f"{above_07}/{len(r2_vals)} crops R²>0.7")

# ── Total ratio (global level) ──
print(f"\n\n{'='*80}")
print("YEARLY GLOBAL TOTAL RATIO (pred/true)")
print(f"{'='*80}")
glob = df_metrics[(df_metrics["layer"] == "extreme") & (df_metrics["level"] == "yearly_global")]
for _, r in glob.iterrows():
    r17 = r.get("total_ratio_2017", np.nan)
    r18 = r.get("total_ratio_2018", np.nan)
    print(f"  {r['crop']:>4}: 2017={r17:.3f}, 2018={r18:.3f}" if pd.notna(r17) else f"  {r['crop']:>4}: N/A")