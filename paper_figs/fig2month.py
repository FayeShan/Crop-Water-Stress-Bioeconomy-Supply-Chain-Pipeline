import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from config import COUNTRY_CSV_DIR, OUTPUT_DIR

CSV_DIR = COUNTRY_CSV_DIR
OUTPUT = OUTPUT_DIR / "monthly_barcharts"
OUTPUT.mkdir(parents=True, exist_ok=True)

YEAR = 2024

TOP6 = {
    "wh":   {"name": "Wheat",               "crops": ["wh"],         "colors": ["#FFFFFF","#FFF8E1","#FFECB3","#FFD54F","#FFB300","#E65100","#4E342E"]},
    "rice": {"name": "Rice",                "crops": ["ri1", "ri2"], "colors": ["#FFFFFF","#FDE0DD","#FA9FB5","#F768A1","#C51B8A","#7A0177"]},
    "cot":  {"name": "Cotton",              "crops": ["cot"],        "colors": ["#FFFFFF","#FEF0D9","#FDCC8A","#FC8D59","#E34A33","#B30000"]},
    "vgt":  {"name": "Vegetables",          "crops": ["vgt"],        "colors": ["#FFFFFF","#E0F7FA","#80DEEA","#26C6DA","#00897B","#004D40"]},
    "oac":  {"name": "Other Annual",        "crops": ["oac"],        "colors": ["#FFFFFF","#FFF0F0","#FFBCBC","#FF7878","#D93A3A","#8B0000"]},
    "pec":  {"name": "Perennial Evergreen", "crops": ["pec"],        "colors": ["#FFFFFF","#F0F9E8","#BAE4BC","#7BCCC4","#43A2CA","#0868AC"]},
}

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def read_monthly_csv_total(crop_list, year):
    """
    Read crop-level country csv files and aggregate to global monthly total.

    Expected columns:
    year, month, waterstress

    For rice, crop_list = ["ri1", "ri2"], so the two files are summed.
    """
    monthly_total = pd.Series(0.0, index=np.arange(1, 13), name="waterstress")

    for crop in crop_list:
        csv_path = CSV_DIR / f"{crop}_country.csv"

        if not csv_path.exists():
            print(f"  Warning: {csv_path} not found, skipping")
            continue

        df = pd.read_csv(csv_path)

        df_year = df[df["year"] == year].copy()

        monthly = (
            df_year
            .groupby("month")["waterstress"]
            .sum()
            .reindex(np.arange(1, 13), fill_value=0)
        )

        monthly_total = monthly_total + monthly

    out = monthly_total.reset_index()
    out.columns = ["month", "waterstress"]

    return out


for key, cfg in TOP6.items():
    print(f"\nPlotting monthly barchart for {key} ({cfg['name']})...")

    monthly_ws = read_monthly_csv_total(cfg["crops"], YEAR)

    if monthly_ws["waterstress"].sum() <= 0:
        print(f"  {key}: no positive data, skipping")
        continue

    # ── Colormap: same palette as the global map ──
    cmap = mcolors.LinearSegmentedColormap.from_list(
        f"ws_{key}",
        cfg["colors"],
        N=256
    )

    # For monthly barchart, use monthly values to map top colors.
    # This preserves visible month-to-month differences.
    norm = mcolors.Normalize(
        vmin=monthly_ws["waterstress"].min(),
        vmax=monthly_ws["waterstress"].max()
    )

    months = monthly_ws["month"].values
    values = monthly_ws["waterstress"].values

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(4.2, 2.4))

    bars = ax.bar(
        months,
        values,
        width=0.72,
        color="none",
        edgecolor="black",
        linewidth=0.45,
        zorder=3
    )

    # Bottom color of each bar
    base_color = cfg["colors"][1]

    # Vertical gradient template
    grad = np.linspace(0, 1, 256).reshape(256, 1)

    for i, (bar, val) in enumerate(zip(bars, values)):
        if not np.isfinite(val) or val <= 0:
            continue

        x = bar.get_x()
        w = bar.get_width()
        h = bar.get_height()

        # Top color comes from the same crop colormap
        top_color = cmap(norm(val))

        # Each bar: light base color -> value-specific top color
        bar_cmap = mcolors.LinearSegmentedColormap.from_list(
            f"{key}_bar_{i}",
            [base_color, top_color],
            N=256
        )

        ax.imshow(
            grad,
            extent=[x, x + w, 0, h],
            origin="lower",
            aspect="auto",
            cmap=bar_cmap,
            clip_path=bar,
            clip_on=True,
            zorder=2
        )

    ax.set_xlim(0.4, 12.6)
    ax.set_ylim(0, values.max() * 1.08)

    ax.set_xticks(months)
    ax.set_xticklabels(MONTH_LABELS, fontsize=7)

    ax.tick_params(axis="y", labelsize=7, length=3, width=0.5)
    ax.tick_params(axis="x", length=2.5, width=0.5)

    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    ax.set_xlabel("")
    ax.set_ylabel("Crop Water Stress [m$^3$ World-eq]", fontsize=6)
    ax.set_title(f"{cfg['name']} Monthly Attribution ({YEAR})", fontsize=8, pad=4)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)

    plt.tight_layout()

    outfile = OUTPUT / f"cwss_{key}_{YEAR}_monthly_bar.png"
    fig.savefig(outfile, dpi=300, bbox_inches="tight", transparent=True)
    fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight", transparent=True)

    plt.show()
    plt.close(fig)

    print(f"  Saved: {outfile}")
    print(f"  Annual total = {values.sum():.6e}")
    print(f"  Monthly min = {values.min():.6e}")
    print(f"  Monthly max = {values.max():.6e}")

print("\nDone!")