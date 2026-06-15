"""
Compares GLORIA's native Agriculture water-stress satellite (Pinero et al. 2021)
with our reconstructed crop water stress (CWS), stacked by country, 1995-2024.

Produces ONE figure:
   country_absolute_stackbar.{png,pdf}
   Two panels (GLORIA, Ours), stacked bars of ABSOLUTE country-level CWS,
   top 10 countries by mean volume plus a "Remaining" bucket, shared y-axis.

Inputs
------
data/gloria_agri_water_stress_by_country_yearly.csv   (from step 01)
data/ours_icws_by_country_yearly.csv                  (from step 02)

Outputs
-------
data/figs/country_absolute_stackbar.png + .pdf
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from config import COMPARISON_DATA_DIR

CODE_DIR = Path(__file__).resolve().parent
DATA_DIR = COMPARISON_DATA_DIR
FIG_DIR = DATA_DIR / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

N_TOP = 10
# Boundary between the physically grounded reference period (1995-2018)
# and the HARM-predicted years (2019-2024).
HARM_BOUNDARY = 2018.5

PAL_COUNTRY = [
    "#E85D75",  # coral red
    "#4AADE8",  # sky blue
    "#F5B946",  # gold orange
    "#7E57C2",  # iris purple
    "#26A69A",  # peacock teal
    "#FF8A65",  # warm persimmon
    "#42A5F5",  # clear sky blue
    "#EC407A",  # peach pink
    "#66BB6A",  # grass green
    "#AB47BC",  # orchid violet
    "#BDC3C7",  # light grey for Remaining
]

# Plot styling (mirrors theme_ws in vis_R.py)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "grey",
    "axes.linewidth": 0.6,
    "grid.color": "grey",
    "grid.alpha": 0.3,
    "grid.linewidth": 0.3,
})


def load_records() -> tuple[pd.DataFrame, pd.DataFrame]:
    gloria = pd.read_csv(DATA_DIR / "gloria_agri_water_stress_by_country_yearly.csv")
    ours = pd.read_csv(DATA_DIR / "ours_icws_by_country_yearly.csv")
    gloria = gloria.rename(columns={"value_Mm3_H2Oeq": "ws"})
    ours = ours.rename(columns={"value_Mm3_H2Oeq": "ws"})
    gloria["source"] = "GLORIA native"
    ours["source"] = "Our record"
    # Standardise on column 'country'
    gloria["country"] = gloria["region_name"]
    ours["country"] = ours["region_name"]
    gloria = gloria[["year", "country", "ws", "source"]]
    ours = ours[["year", "country", "ws", "source"]]
    return gloria, ours


def pick_top_countries(df: pd.DataFrame, n: int = N_TOP) -> list[str]:
    """Top n countries by mean ws over the record."""
    means = df.groupby("country")["ws"].mean().sort_values(ascending=False)
    return means.head(n).index.tolist()


def collapse_to_top(df: pd.DataFrame, top: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["group"] = df["country"].where(df["country"].isin(top), "Remaining")
    df = df.groupby(["year", "group", "source"], as_index=False)["ws"].sum()
    return df


def stacked_bar(ax, df: pd.DataFrame, group_order: list[str],
                value_col: str, source_label: str,
                pal: list[str], y_label: str, title: str,
                show_legend: bool = False) -> None:
    years = sorted(df["year"].unique())
    pos_bottom = np.zeros(len(years))
    neg_bottom = np.zeros(len(years))
    handles = []
    cmap = {g: pal[i % len(pal)] for i, g in enumerate(group_order)}

    for g in group_order:
        sub = df[df["group"] == g].set_index("year").reindex(years).fillna(0.0)
        vals = sub[value_col].values
        pos = np.maximum(vals, 0)
        neg = np.minimum(vals, 0)
        ax.bar(years, pos, bottom=pos_bottom, color=cmap[g],
               edgecolor="white", linewidth=0.2, width=0.8, label=g)
        ax.bar(years, neg, bottom=neg_bottom, color=cmap[g],
               edgecolor="white", linewidth=0.2, width=0.8)
        pos_bottom += pos
        neg_bottom += neg
        handles.append(Patch(color=cmap[g], label=g))

    ax.axhline(0, color="grey", linewidth=0.6)
    # Reference (1995-2018) vs HARM-predicted (2019-2024) boundary
    ax.axvline(HARM_BOUNDARY, color="grey", linewidth=0.6, linestyle="--", alpha=0.6)

    ax.set_xticks([y for y in years if y % 5 == 0 or y == years[-1]])
    ax.set_xlabel("")
    ax.set_ylabel(y_label)
    ax.set_title(f"{source_label} — {title}")
    ax.grid(axis="y")
    if show_legend:
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
                  frameon=False, handlelength=1.2)


def make_figure_absolute(gloria: pd.DataFrame, ours: pd.DataFrame) -> None:
    """Absolute country-level water stress (million m3 world-eq), two panels."""
    # Choose top countries on the combined record so both panels share the same set
    combined = pd.concat([gloria, ours], ignore_index=True)
    top = pick_top_countries(combined, N_TOP)
    print(f"Top {N_TOP} countries by combined mean: {top}")

    g_top = collapse_to_top(gloria, top)
    o_top = collapse_to_top(ours, top)

    # Order: top countries first (by combined mean), Remaining last
    group_order = top + ["Remaining"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharey=True)
    # Shared y-axis limit
    ymax = max(
        g_top.groupby("year")["ws"].sum().max(),
        o_top.groupby("year")["ws"].sum().max(),
    ) * 1.05
    axes[0].set_ylim(0, ymax)
    axes[1].set_ylim(0, ymax)

    stacked_bar(axes[0], g_top, group_order, "ws",
                "GLORIA native (Pinero et al. 2021)",
                PAL_COUNTRY,
                y_label="Crop water stress (million m³ world-eq)",
                title="absolute country-level record",
                show_legend=False)
    stacked_bar(axes[1], o_top, group_order, "ws",
                "Our record (1995–2024)",
                PAL_COUNTRY,
                y_label="",
                title="absolute country-level record",
                show_legend=True)

    fig.suptitle("Country-level crop water stress, 1995–2024",
                 fontsize=12, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 0.92, 0.96])
    fig.savefig(FIG_DIR / "country_absolute_stackbar.png", dpi=600,
                transparent=True, bbox_inches="tight")
    fig.savefig(FIG_DIR / "country_absolute_stackbar.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {FIG_DIR / 'country_absolute_stackbar.png'}")


def main() -> None:
    gloria, ours = load_records()
    print(f"GLORIA rows: {len(gloria):,} | Ours rows: {len(ours):,}")
    print(f"GLORIA years: {sorted(gloria['year'].unique())[:3]} ... {sorted(gloria['year'].unique())[-3:]}")
    print(f"Ours years:   {sorted(ours['year'].unique())[:3]} ... {sorted(ours['year'].unique())[-3:]}")
    make_figure_absolute(gloria, ours)


if __name__ == "__main__":
    main()