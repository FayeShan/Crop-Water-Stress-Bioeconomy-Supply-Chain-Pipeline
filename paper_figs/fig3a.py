"""
Fig 3a — Country x yearly observed vs predicted scatter.

Metrics box: computed on UNFILTERED country-crop-year totals
             (matches paper text "R2 = 0.96" and Table S6.4).
Plot: restricted to y_true > THRESHOLD = 1e8 m^3 for visual clarity
      (focus on policy-relevant high-volume cases).
Legend per-year PICP still describes plotted points only.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
from sklearn.metrics import r2_score
from scipy import stats
from config import EVAL_PRED_DIR, SHAPEFILE, OUTPUT_DIR

# Config
EVAL_DIR = EVAL_PRED_DIR
OUTPUT = OUTPUT_DIR / "fig3"
OUTPUT.mkdir(parents=True, exist_ok=True)

ALL_CROPS = [
    'aff','bar','bea','cas','ckp','coc','cof','cot','cwp',
    'mai','mil','nut','oac','pdc','pec','plm','pot','rap',
    'ri1','ri2','sgb','sgc','sor','soy','sun','vgt','wh',
]

THRESHOLD = 1e8  # raw m^3 — plot-only filter (= 100 million m^3)

COLOR_2017 = "#1F4E79"
COLOR_2018 = "#C1272D"

CROP_FULL_NAMES = {
    'wh':'Wheat','ri1':'Rice','ri2':'Rice','cot':'Cotton','soy':'Soybean',
    'mai':'Maize','sgc':'Sugarcane','sgb':'Sugarbeet','pot':'Potato',
    'vgt':'Vegetables','oac':'Other annual','pec':'Perennial evergreen',
    'pdc':'Perennial deciduous','rap':'Rapeseed','sun':'Sunflower',
    'bar':'Barley','sor':'Sorghum','mil':'Millet','cas':'Cassava',
    'bea':'Beans','ckp':'Chickpea','cwp':'Cowpea','plm':'Oil palm',
    'cof':'Coffee','coc':'Cocoa','nut':'Nuts','aff':'Alfalfa',
}

LABELS = [
    ("India",    "wh",  2018,  12, -18),
    ("India",    "ri1", 2018,  12,  10),
    ("China",    "wh",  2018, -60,  12),
    ("Pakistan", "wh",  2018, -65, -10),
    ("Pakistan", "cot", 2018,  15,  -5),
    ("Iran",     "wh",  2018,  15,  10),
    ("Egypt",    "wh",  2018,  12, -15),
]


# 1. Spatial join — point → country
print("Building country lookup...")
all_coords = []
for crop in ALL_CROPS:
    p = EVAL_DIR / f"{crop}_eval_predictions.parquet"
    if p.exists():
        all_coords.append(
            pd.read_parquet(p, columns=["lat", "lon"]).drop_duplicates()
        )

all_coords_df = pd.concat(all_coords).drop_duplicates()
gdf = gpd.GeoDataFrame(
    all_coords_df,
    geometry=[Point(xy) for xy in zip(all_coords_df["lon"], all_coords_df["lat"])],
    crs="EPSG:4326",
)
countries_gdf = gpd.read_file(SHAPEFILE)
if countries_gdf.crs is None:
    countries_gdf = countries_gdf.set_crs("EPSG:4326")
joined = gpd.sjoin(
    gdf, countries_gdf[["ADMIN", "geometry"]],
    how="left", predicate="within"
)
lookup = (
    joined.groupby(["lat", "lon"]).first()[["ADMIN"]]
    .reset_index()
    .rename(columns={"ADMIN": "country"})
)
del all_coords, all_coords_df, gdf, joined


# 2. Load all crops with required columns
print("Loading crops with q10/q50/q90 from gr + gre.q50...")
NEED_COLS = [
    "lat", "lon", "year", "month",
    "y_true",
    "y_pred_gr_q10",
    "y_pred_gr_q50",
    "y_pred_gr_q90",
    "y_pred_gre_q50",
]

all_dfs = []
for crop in ALL_CROPS:
    p = EVAL_DIR / f"{crop}_eval_predictions.parquet"
    if not p.exists():
        continue
    df = pd.read_parquet(p)
    keep_cols = [c for c in NEED_COLS if c in df.columns]
    df = df[keep_cols].copy()
    df["crop"] = crop
    df = df.merge(lookup, on=["lat", "lon"], how="left").dropna(subset=["country"])
    all_dfs.append(df)
df_all = pd.concat(all_dfs, ignore_index=True)
del all_dfs


# 3. Aggregate to country x year x crop
yc_full = (
    df_all
    .groupby(["country", "year", "crop"])
    .agg(
        y_true     = ("y_true",          "sum"),
        y_pred_q50 = ("y_pred_gre_q50",  "sum"),
        y_pred_q10 = ("y_pred_gr_q10",   "sum"),
        y_pred_q90 = ("y_pred_gr_q90",   "sum"),
    )
    .reset_index()
)

# Holdout years only
yc_full = yc_full[yc_full["year"].isin([2017, 2018])].copy()

# Unfiltered PI coverage flag (for unfiltered PICP below)
yc_full["in_pi80"] = (
    (yc_full["y_true"] >= yc_full["y_pred_q10"]) &
    (yc_full["y_true"] <= yc_full["y_pred_q90"])
)

# Plot-only filter on y_true (avoid circular filter on prediction)
yc = yc_full[yc_full["y_true"] > THRESHOLD].copy()

# Scale to million m^3 for plotting
for col in ["y_true", "y_pred_q50", "y_pred_q10", "y_pred_q90"]:
    yc[col.replace("y_true", "yt").replace("y_pred_", "yp_")] = yc[col] / 1e6

print(f"Unfiltered: {len(yc_full):,} points  |  Plotted (>1e8): {len(yc):,} "
      f"(2017: {(yc['year']==2017).sum()}, 2018: {(yc['year']==2018).sum()})")


# 4. Metrics on UNFILTERED data (matches paper text & Table S6.4)
y_true_f = yc_full["y_true"].values
y_pred_f = yc_full["y_pred_q50"].values

r2_linear = r2_score(y_true_f, y_pred_f)

mask_pos = (y_true_f > 0) & (y_pred_f > 0)
r2_log = r2_score(np.log10(y_true_f[mask_pos]), np.log10(y_pred_f[mask_pos]))

corr  = np.corrcoef(y_true_f, y_pred_f)[0, 1]
alpha = np.std(y_pred_f) / np.std(y_true_f)
beta  = np.mean(y_pred_f) / np.mean(y_true_f)
kge   = 1 - np.sqrt((corr - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

rmse_log = np.sqrt(
    np.mean((np.log10(y_true_f[mask_pos]) - np.log10(y_pred_f[mask_pos]))**2)
)

picp = yc_full["in_pi80"].mean()


# 5. Outlier inspection on plotted (filtered) subset
yc_outliers = yc.copy()
yc_outliers["log_residual"] = (
    np.log10(yc_outliers["y_pred_q50"]) - np.log10(yc_outliers["y_true"])
)
print("\n=== Top 10 over-predictions (model > truth) ===")
print(yc_outliers.sort_values("log_residual", ascending=False)
      .head(10)[["country","crop","year","y_true","y_pred_q50","log_residual"]]
      .to_string(index=False))
print("\n=== Top 10 under-predictions (model < truth) ===")
print(yc_outliers.sort_values("log_residual", ascending=True)
      .head(10)[["country","crop","year","y_true","y_pred_q50","log_residual"]]
      .to_string(index=False))


# 6. Per-year log regression helper
def log_regression(x, y, confidence=0.95):
    mask = (x > 0) & (y > 0)
    x = x[mask]; y = y[mask]
    lx = np.log10(x); ly = np.log10(y)
    slope, intercept, r_val, p_val, se = stats.linregress(lx, ly)
    x_fit = np.linspace(lx.min() - 0.2, lx.max() + 0.2, 200)
    y_fit = intercept + slope * x_fit
    n = len(lx)
    t_val = stats.t.ppf((1 + confidence) / 2, n - 2)
    residuals = ly - (intercept + slope * lx)
    s_err = np.sqrt(np.sum(residuals ** 2) / (n - 2))
    lx_mean = np.mean(lx)
    se_fit = s_err * np.sqrt(
        1 / n + (x_fit - lx_mean) ** 2 / np.sum((lx - lx_mean) ** 2)
    )
    ci = t_val * se_fit
    return (10 ** x_fit, 10 ** y_fit,
            10 ** (y_fit - ci), 10 ** (y_fit + ci),
            slope, r_val)


# 7. PLOT
fig, ax = plt.subplots(figsize=(8, 8))

lo = min(yc["yt"].min(), yc["yp_q50"].min()) * 0.3
hi = max(yc["yt"].max(), yc["yp_q50"].max()) * 3

# Layer 1: 1:1 line
ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.5, zorder=1)

# Layer 2: q10-q90 PI error bars for out-of-PI points only
# In-PI points already fall within their interval; adding bars there is noise.
yc["in_pi80_plot"] = (
    (yc["y_true"] >= yc["y_pred_q10"]) &
    (yc["y_true"] <= yc["y_pred_q90"])
)
for yr, color in [(2017, COLOR_2017), (2018, COLOR_2018)]:
    sub = yc[(yc["year"] == yr) & (~yc["in_pi80_plot"])]
    if len(sub) == 0:
        continue
    yerr_lo = (sub["yp_q50"] - sub["yp_q10"]).clip(lower=0)
    yerr_hi = (sub["yp_q90"] - sub["yp_q50"]).clip(lower=0)
    ax.errorbar(
        sub["yt"], sub["yp_q50"],
        yerr=[yerr_lo, yerr_hi],
        fmt="none",
        ecolor=color, alpha=0.35,
        capsize=2, linewidth=0.6,
        zorder=2,
    )

# Layer 3: scatter with in-PI / out-PI distinction (on plotted subset)
for yr, color in [(2017, COLOR_2017), (2018, COLOR_2018)]:
    sub = yc[yc["year"] == yr]
    in_pi = sub["in_pi80_plot"]
    picp_yr = in_pi.mean()
    # In-interval — filled
    ax.scatter(
        sub.loc[in_pi, "yt"], sub.loc[in_pi, "yp_q50"],
        c=color, s=22, alpha=0.70,
        edgecolors="white", linewidth=0.3,
        label=f"{yr} in 80% PI (n={in_pi.sum()}, PICP={picp_yr:.2f})",
        zorder=3,
    )
    # Out-of-interval — open marker
    ax.scatter(
        sub.loc[~in_pi, "yt"], sub.loc[~in_pi, "yp_q50"],
        facecolors="none", edgecolors=color,
        s=24, linewidth=0.9,
        label=f"{yr} out 80% PI (n={(~in_pi).sum()})",
        zorder=3,
    )

# Layer 4: per-year log-log regression + 95% CI
for yr, color in [(2017, COLOR_2017), (2018, COLOR_2018)]:
    sub = yc[yc["year"] == yr]
    if len(sub) < 6:
        continue
    x_fit, y_fit, ci_lo, ci_hi, slope, r_val = log_regression(
        sub["yt"].values, sub["yp_q50"].values
    )
    ax.plot(x_fit, y_fit, color=color, linewidth=1.8, zorder=4)
    ax.fill_between(x_fit, ci_lo, ci_hi, color=color, alpha=0.12, zorder=2)

# Axes
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_aspect("equal")
ax.set_xlabel(r"Observed CWS (million m$^3$ world-equivalent yr$^{-1}$)", fontsize=12)
ax.set_ylabel(r"Predicted CWS (million m$^3$ world-equivalent yr$^{-1}$)", fontsize=12)
ax.tick_params(labelsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Country labels
for country, crop, year, dx, dy in LABELS:
    row = yc[
        (yc["country"] == country) &
        (yc["crop"] == crop) &
        (yc["year"] == year)
    ]
    if len(row) == 0:
        print(f"  (label skipped: {country} x {crop} x {year} not in filtered data)")
        continue
    r = row.iloc[0]
    crop_name = CROP_FULL_NAMES.get(crop, crop)
    ax.annotate(
        f"{country} ({crop_name})",
        xy=(r["yt"], r["yp_q50"]),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=8, color="#333333", fontweight="bold",
        arrowprops=dict(arrowstyle="-", color="#999999", lw=0.6),
        zorder=8,
    )

# Metrics box — unfiltered values matching paper text & Table S6.4
metrics_lines = [
    ("R²",         f"{r2_linear:.3f}"),
    ("KGE",        f"{kge:.3f}"),
    ("RMSE(log)",  f"{rmse_log:.3f}"),
    ("PICP(80%)",  f"{picp:.3f}"),
    ("n",          f"{len(yc_full)}"),
]

metrics_x = 0.04
metrics_y = 0.96
line_height = 0.032

n_lines = len(metrics_lines)
box_w = 0.24
box_h = line_height * n_lines + 0.02
box_patch = plt.Rectangle(
    (metrics_x - 0.012, metrics_y - box_h + 0.005),
    box_w, box_h,
    transform=ax.transAxes,
    facecolor="white", edgecolor="#CCCCCC", alpha=0.85,
    zorder=9,
)
ax.add_patch(box_patch)

for i, (key, val) in enumerate(metrics_lines):
    y = metrics_y - i * line_height - 0.005
    ax.text(
        metrics_x, y,
        f"{key:<10}= {val}",
        transform=ax.transAxes,
        fontsize=10, fontweight="normal",
        verticalalignment="top",
        family="monospace", color="#444444",
        zorder=10,
    )

# Legend
ax.legend(
    loc="lower right", fontsize=8.5, framealpha=0.9,
    edgecolor="#CCCCCC", markerscale=1.3, handletextpad=0.4,
)

plt.subplots_adjust(left=0.10, right=0.95, top=0.95, bottom=0.10)

outfile = OUTPUT / "fig3a_country_yearly_scatter.png"
fig.savefig(outfile, dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
plt.show()

print(f"\n  Saved: {outfile}")
print(f"\n  Metrics (unfiltered, n={len(yc_full)}):")
print(f"    R²(linear) = {r2_linear:.3f}")
print(f"    R²(log)    = {r2_log:.3f}")
print(f"    KGE        = {kge:.3f}")
print(f"    RMSE(log)  = {rmse_log:.3f}")
print(f"    PICP(80%)  = {picp:.3f}")
print(f"  Plotted: n={len(yc)} (y_true > {THRESHOLD:.0e} m^3)")

