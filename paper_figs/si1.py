"""
Supplementary Fig. S1 - Concentration of crop water stress (CWS)

Saves TWO separate single-panel figures (to be stitched later):
  figS1_stock_2024.{png,pdf}           - stock concentration in 2024
  figS1_increment_1995_2018.{png,pdf}  - concentration of the 1995->2018 increment

Choices
-------
* STOCK uses a single year (2024) to match the main-text statement
  "in 2024 ... top 10% ... 78% ... top 1% ... 28%".
* The INCREMENT uses reference-period-only 3-year-mean endpoints
  (1995-1997 -> 2016-2018): NO predicted years, fully physically grounded,
  which avoids the >100% overshoot seen with a 2024 endpoint.
* Increment shares are reported relative to the NET increase (sum of per-cell
  changes); a gross-positive variant is also printed so you can pick the
  framing you cite.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
from config import YEARLY_TOTAL_NC, OUTPUT_DIR

# ------------------------------- Config -------------------------------
NC_FILE = YEARLY_TOTAL_NC
OUTPUT = OUTPUT_DIR
OUTPUT.mkdir(parents=True, exist_ok=True)
VAR_NAME = "total_water_stress"

STOCK_YEAR = 2024                   # stock: single year

BASE_YEARS = (1995, 1997)           # increment baseline (mean)
END_YEARS  = (2016, 2018)           # increment endpoint (mean) - reference period only

TOP_MARKS  = [0.01, 0.05, 0.10]
FIGSIZE    = (7, 5.8)               # same size for both, so they stitch cleanly

# ------------------------------- Load -------------------------------
ds = xr.open_dataset(NC_FILE)
da = ds[VAR_NAME]
stock_field = da.sel(year=STOCK_YEAR).values.ravel()
base_field  = da.sel(year=slice(*BASE_YEARS)).mean("year").values.ravel()
end_field   = da.sel(year=slice(*END_YEARS)).mean("year").values.ravel()
ds.close()

# ------------------------------- Helper -------------------------------
def share_at(frac, cell_share, value_share):
    idx = min(np.searchsorted(cell_share, frac), len(value_share) - 1)
    return value_share[idx]

def save_pareto(cell, value_share, title, ylabel, color, n_cells, outstem):
    """Draw one Pareto/Lorenz panel and save it as its own PNG + PDF."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x, y = cell * 100, value_share * 100
    ax.fill_between(x, y, 0, alpha=0.18, color=color)
    ax.plot(x, y, lw=2.5, color=color)
    ax.plot([0, 100], [0, 100], "--", lw=1.1, color="0.5", alpha=0.8)   # equality line
    ax.axhline(100, ls=":", lw=1.0, color="0.6")
    for f in TOP_MARKS:
        yv = share_at(f, cell, value_share) * 100
        ax.axvline(f * 100, ls="--", lw=1.0, color=color, alpha=0.5)
        ax.scatter([f * 100], [yv], s=36, color=color, zorder=5)
    y10 = share_at(0.10, cell, value_share) * 100
    ax.annotate(f"top 10% \u2192 {y10:.0f}%",
                xy=(10, y10), xytext=(24, max(18, y10 - 24)), fontsize=11, color="0.15",
                arrowprops=dict(arrowstyle="->", color="0.5", lw=1.0),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.95))
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_ylim(0, max(105, y.max() * 1.02))
    ax.set_xlabel("Cumulative share of active cropland cells (%)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(True, ls=":", lw=0.7, alpha=0.5)
    sec = ax.secondary_xaxis(
        "top", functions=(lambda v: v / 100 * n_cells, lambda c: c / n_cells * 100))
    sec.set_xlabel("Cumulative number of grid cells", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{outstem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT / f"{outstem}.pdf", bbox_inches="tight", facecolor="white")
    return fig

# ------------------------- STOCK (2024) -------------------------
m_stock  = np.isfinite(stock_field) & (stock_field > 0)
s_sorted = np.sort(stock_field[m_stock])[::-1]          # largest first
s_cum    = np.cumsum(s_sorted)
s_total  = s_cum[-1]
s_cell   = np.arange(1, s_sorted.size + 1) / s_sorted.size
s_share  = s_cum / s_total

save_pareto(s_cell, s_share,
            f"a  Stock concentration ({STOCK_YEAR})",
            "Cumulative share of crop water stress (%)",
            "#1f77b4", s_sorted.size, "figS1_stock_2024")

# ------------------- INCREMENT (reference period 1995->2018) -------------------
m_inc    = np.isfinite(base_field) & np.isfinite(end_field) & ((base_field > 0) | (end_field > 0))
delta    = end_field[m_inc] - base_field[m_inc]
d_sorted = np.sort(delta)[::-1]                          # biggest increases first
d_cum    = np.cumsum(d_sorted)
d_net    = d_cum[-1]                                     # net increase
d_cell   = np.arange(1, d_sorted.size + 1) / d_sorted.size
d_share  = d_cum / d_net                                 # NET share (printed below; overshoots >100%)
gross_pos = d_sorted[d_sorted > 0].sum()                 # gross positive increase
# Plot the GROSS-positive share: cumulative positive increase / total positive increase.
# Declining cells add nothing, so the curve is monotonic and plateaus at 100% (no overshoot).
d_share_plot = np.cumsum(np.clip(d_sorted, 0, None)) / gross_pos

save_pareto(d_cell, d_share_plot,
            f"b  Increment concentration ({BASE_YEARS[0]}\u2013{BASE_YEARS[1]} "
            f"\u2192 {END_YEARS[0]}\u2013{END_YEARS[1]})",
            "Cumulative share of the increase in CWS (%)",
            "#d62728", d_sorted.size, "figS1_increment_1995_2018")

plt.show()

# ------------------------------- Summary -------------------------------
print("=" * 64)
print(f"STOCK concentration - {STOCK_YEAR}  (active cropland cells: {s_sorted.size:,})")
for f in TOP_MARKS:
    print(f"   top {f*100:>4.0f}%  ->  {share_at(f, s_cell, s_share)*100:5.1f}%  of total CWS")
print()
print(f"INCREMENT concentration - {BASE_YEARS[0]}-{BASE_YEARS[1]} -> {END_YEARS[0]}-{END_YEARS[1]} (reference period)")
print(f"   active cells: {d_sorted.size:,}")
print(f"   net increase   = {d_net:,.0f}   (same units as the input field)")
print(f"   gross positive = {gross_pos:,.0f}")
print("   share of NET increase:")
for f in TOP_MARKS:
    print(f"      top {f*100:>4.0f}%  ->  {share_at(f, d_cell, d_share)*100:6.1f}%")
print("   share of GROSS positive increase:")
for f in TOP_MARKS:
    idx = min(int(np.ceil(f * d_sorted.size)) - 1, d_sorted.size - 1)
    print(f"      top {f*100:>4.0f}%  ->  {d_cum[idx] / gross_pos * 100:6.1f}%")
print("=" * 64)
print(f"Saved: {OUTPUT / 'figS1_stock_2024.png'}  (+ .pdf)")
print(f"Saved: {OUTPUT / 'figS1_increment_1995_2018.png'}  (+ .pdf)")