"""
Fig 3b — Density scatter: observed vs predicted for 3 crops + pooled all.

Crops: Wheat (wh), Rice (ri1+ri2 summed), Cotton (cot), Vegetables (vgt)
Plus one pooled-all figure combining 27 crops.

Style: plasma cmap, OLS regression + identity line, standard colorbar.
Metrics box (R², KGE, n): computed on UNFILTERED data (only NaN/inf removed).
Plotting: restricted to MIN_VALUE_RAW = 1000 m³ for visual clarity.
Max plot points: 300,000.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score

from config import EVAL_PRED_DIR, OUTPUT_DIR


EVAL_DIR = EVAL_PRED_DIR
OUTPUT = OUTPUT_DIR / "fig3c"
OUTPUT.mkdir(parents=True, exist_ok=True)

Y_TRUE_COL = "y_true"
Y_PRED_COL = "y_pred_gre_q50"

MAX_PLOT_POINTS = 300_000
MIN_VALUE_RAW = 1000.0  # raw m³ — plot-only filter for visual clarity

# Custom blue→red colormap (matches identity / regression line colors)
RED, BLUE = "#E74C3C", "#4A90D9"
BLUE_RED = mcolors.LinearSegmentedColormap.from_list("blue_red", [BLUE, RED])


PANEL_SPECS = [
    {"key": "wheat",      "title": "Wheat",      "crops": ["wh"]},
    {"key": "rice",       "title": "Rice",       "crops": ["ri1", "ri2"]},
    {"key": "cotton",     "title": "Cotton",     "crops": ["cot"]},
    {"key": "vegetables", "title": "Vegetables", "crops": ["vgt"]},
]

ALL_CROPS_FOR_POOL = [
    'aff','bar','bea','cas','ckp','coc','cof','cot','cwp',
    'mai','mil','nut','oac','pdc','pec','plm','pot','rap',
    'ri1','ri2','sgb','sgc','sor','soy','sun','vgt','wh',
]


def calculate_metrics(y_true, y_pred):
    """R², KGE, n on raw data (only NaN/inf removed; no value filter)."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]

    if len(y_true) < 5:
        return {"r2": np.nan, "kge": np.nan, "n": len(y_true)}

    r2 = r2_score(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    alpha = np.std(y_pred) / np.std(y_true) if np.std(y_true) > 0 else np.nan
    beta = np.mean(y_pred) / np.mean(y_true) if np.mean(y_true) > 0 else np.nan
    kge = 1 - np.sqrt((corr - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    return {"r2": r2, "kge": kge, "n": len(y_true)}


def load_crop_data(crop_codes):
    """Load and optionally sum multiple crops at grid×month. No value filter."""
    dfs = []
    for c in crop_codes:
        p = EVAL_DIR / f"{c}_eval_predictions.parquet"
        if not p.exists():
            print(f"  Warning: {p.name} not found")
            continue
        df = pd.read_parquet(
            p, columns=["lat", "lon", "year", "month", Y_TRUE_COL, Y_PRED_COL]
        )
        df = df.replace([np.inf, -np.inf], np.nan).dropna(
            subset=[Y_TRUE_COL, Y_PRED_COL]
        )
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    df_combined = pd.concat(dfs, ignore_index=True)

    # Sum at grid×month if multiple crop codes (e.g. ri1+ri2)
    if len(crop_codes) > 1:
        df_combined = (
            df_combined
            .groupby(["lat", "lon", "year", "month"], as_index=False)
            .agg({Y_TRUE_COL: "sum", Y_PRED_COL: "sum"})
        )

    return df_combined


def density_scatter_plot(df, outfile, y_true_col=Y_TRUE_COL, y_pred_col=Y_PRED_COL):
    """Log-log density scatter. Metrics box reports unfiltered values."""
    df = df[[y_true_col, y_pred_col]].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    if len(df) == 0:
        print("  Skip: no valid points.")
        return None

    metrics = calculate_metrics(df[y_true_col].values, df[y_pred_col].values)

    plot_df = df[
        (df[y_true_col] > MIN_VALUE_RAW) & (df[y_pred_col] > MIN_VALUE_RAW)
    ].copy()
    if len(plot_df) == 0:
        print("  Skip: no points pass plot filter.")
        return metrics

    # Convert to million m³ for plot axes
    plot_df["yt_mil"] = plot_df[y_true_col] / 1e6
    plot_df["yp_mil"] = plot_df[y_pred_col] / 1e6

    # OLS regression in log10 space (on filtered plot data, full points)
    full_log_x = np.log10(plot_df["yt_mil"].values)
    full_log_y = np.log10(plot_df["yp_mil"].values)
    slope, intercept = np.polyfit(full_log_x, full_log_y, 1)

    # Subsample for rendering speed
    if len(plot_df) > MAX_PLOT_POINTS:
        plot_df = plot_df.sample(MAX_PLOT_POINTS, random_state=42)

    x = plot_df["yt_mil"].values
    y = plot_df["yp_mil"].values

    # KDE density in log space
    log_x, log_y = np.log10(x), np.log10(y)
    xy = np.vstack([log_x, log_y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    # Axis limits from plotted data
    lo = 10 ** np.floor(np.log10(min(x.min(), y.min())))
    hi = 10 ** np.ceil(np.log10(max(x.max(), y.max())))

    fig, ax = plt.subplots(figsize=(7.2, 6.6))

    sc = ax.scatter(
        x, y, c=z, s=5, alpha=0.75,
        cmap=BLUE_RED, edgecolors="none",
    )

    # Identity line
    ax.plot([lo, hi], [lo, hi], "-", color="#4A90D9", linewidth=1.2,
            alpha=0.8, zorder=3, label="Identity line")

    # OLS regression in log space
    reg_x = np.array([np.log10(lo), np.log10(hi)])
    reg_y = slope * reg_x + intercept
    ax.plot(10 ** reg_x, 10 ** reg_y, ":", color="#E74C3C", linewidth=1.5,
            alpha=0.9, zorder=3,
            label=f"$y = {intercept:+.2f} {slope:+.2f}x$ (log₁₀)")

    ax.legend(loc="lower right", fontsize=9, framealpha=0.9,
              edgecolor="#cccccc")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(r"Observed CWS (million m$^3$ month$^{-1}$)", fontsize=12)
    ax.set_ylabel(r"Predicted CWS (million m$^3$ month$^{-1}$)", fontsize=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Metrics box — unfiltered values
    n_str = f"{metrics['n']:,}" if metrics['n'] < 1e6 else f"{metrics['n']/1e6:.1f}M"
    metrics_text = (
        f"R² = {metrics['r2']:.3f}\n"
        f"KGE = {metrics['kge']:.3f}\n"
        f"n = {n_str}"
    )
    ax.text(
        0.05, 0.95, metrics_text,
        transform=ax.transAxes, va="top", ha="left",
        fontsize=10, family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white", edgecolor="#cccccc", alpha=0.9,
        ),
    )

    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Density of points", fontsize=10)

    plt.tight_layout()

    fig.savefig(outfile, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {outfile.name}")
    print(f"  OLS (log10): log(y) = {intercept:.3f} + {slope:.3f} * log(x)")
    print(f"  Metrics on n={metrics['n']:,} (unfiltered); plotted on filtered subset")

    return metrics


print("=== Generating per-crop figures ===")
all_metrics = {}

for spec in PANEL_SPECS:
    print(f"\n--- {spec['title']} ({spec['crops']}) ---")
    df = load_crop_data(spec["crops"])
    if len(df) == 0:
        print(f"  No data for {spec['title']}, skipping")
        continue

    outfile = OUTPUT / f"fig3c_{spec['key']}.png"
    m = density_scatter_plot(df, outfile)
    if m:
        all_metrics[spec["title"]] = m


print("\n=== Generating pooled-all figure (27 crops) ===")

pooled_dfs = []
for c in ALL_CROPS_FOR_POOL:
    p = EVAL_DIR / f"{c}_eval_predictions.parquet"
    if not p.exists():
        continue
    df = pd.read_parquet(
        p, columns=["lat", "lon", "year", "month", Y_TRUE_COL, Y_PRED_COL]
    )
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[Y_TRUE_COL, Y_PRED_COL]
    )
    pooled_dfs.append(df)

df_pooled = pd.concat(pooled_dfs, ignore_index=True)
outfile = OUTPUT / "fig3b_pooled_27crops.png"
m = density_scatter_plot(df_pooled, outfile)
all_metrics["Pooled All (27 crops)"] = m


print("\n=== Summary (unfiltered metrics) ===")
print(f"{'Crop':<24} {'R²':>8} {'KGE':>8} {'n':>14}")
print("-" * 58)
for title, m in all_metrics.items():
    n_str = f"{m['n']:,}" if m['n'] < 1e6 else f"{m['n']/1e6:.1f}M"
    print(f"{title:<24} {m['r2']:>8.3f} {m['kge']:>8.3f} {n_str:>14}")

print(f"\nDone. Files saved to {OUTPUT}/")