import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from config import YEARLY_TOTAL_NC, OUTPUT_DIR

NC_YEARLY = YEARLY_TOTAL_NC
OUTPUT = OUTPUT_DIR

YEAR_END = 2024
YEAR_START = 1995
AVG_YEARS = 3 

ZOOM_REGIONS = [
    (66, 89, 23, 33, "Indo-Gangetic",        "#E85D75"),
    (24, 36, 23, 33, "Nile Delta",            "#FF8A65"),
    (-10, 30, 35, 46, "Mediterranean Europe",  "#66BB6A"),
    (44, 64, 25, 40, "Iran",                  "#42A5F5"),
]

# multiple year average
def compute_change(ds, year_end=2024, year_start=1995, avg_years=3):
    """
    Compute absolute change between two periods.

    Parameters
    ----------
    ds : xr.Dataset
        Must contain 'total_water_stress' with a 'year' dimension.
    year_end : int
        End of the later period (e.g. 2024 → uses 2022-2024 for avg_years=3).
    year_start : int
        Start of the earlier period (e.g. 1995 → uses 1995-1997 for avg_years=3).
    avg_years : int
        Number of years to average. 1 = single year (original behavior).

    Returns
    -------
    abs_change : np.ndarray
    label_start, label_end : str
    """
    end_years = list(range(year_end - avg_years + 1, year_end + 1))
    start_years = list(range(year_start, year_start + avg_years))

    data_end = ds["total_water_stress"].sel(year=end_years).mean(dim="year").values
    data_start = ds["total_water_stress"].sel(year=start_years).mean(dim="year").values

    abs_change = (data_end - data_start) / 1e6
    both_nan = np.isnan(data_start) & np.isnan(data_end)
    both_zero = (np.nan_to_num(data_start, 0) == 0) & (np.nan_to_num(data_end, 0) == 0)
    abs_change[both_nan | both_zero] = np.nan

    if avg_years == 1:
        label_start, label_end = str(year_start), str(year_end)
    else:
        label_start = f"{start_years[0]}–{start_years[-1]}"
        label_end = f"{end_years[0]}–{end_years[-1]}"

    return abs_change, label_start, label_end


# ── Load data ──
ds = xr.open_dataset(NC_YEARLY)
lat, lon = ds["lat"].values, ds["lon"].values
abs_change, label_start, label_end = compute_change(ds, YEAR_END, YEAR_START, AVG_YEARS)
ds.close()

outfile = OUTPUT / f"cwss_change_abs_{label_end}_vs_{label_start}_with_boxes.png"

div_colors = [
    "#0E4D4D", "#1A7A70", "#3FA898", "#8ECFC2",
    "#F5F0EB",
    "#E8B870", "#CC7A28", "#A84820", "#6E1A0A",
]
cmap_div = mcolors.LinearSegmentedColormap.from_list("div_bwr", div_colors, N=256)
cmap_div.set_bad(color="white", alpha=0)

proj = ccrs.Robinson()
data_crs = ccrs.PlateCarree()

fig, ax = plt.subplots(
    figsize=(16, 8),
    subplot_kw={"projection": proj},
)

x_left,  _ = proj.transform_point(-125, 0,  data_crs)
x_right, _ = proj.transform_point( 166, 0,  data_crs)
_, y_bot    = proj.transform_point(0,  -60,  data_crs)
_, y_top    = proj.transform_point(0,   80,  data_crs)

rect = mpath.Path([
    (x_left,  y_bot),
    (x_right, y_bot),
    (x_right, y_top),
    (x_left,  y_top),
    (x_left,  y_bot),
])
ax.set_boundary(rect, transform=ax.transData)
ax.set_global()

# ── Map features ──
ax.add_feature(cfeature.LAND, facecolor="white", edgecolor="none", zorder=0)
ax.add_feature(cfeature.OCEAN, facecolor="white", edgecolor="none", zorder=0)
ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color="#666666", zorder=3)
ax.add_feature(cfeature.BORDERS, linewidth=0.4, color="#666666", zorder=4)
ax.set_facecolor("white")

valid = abs_change[~np.isnan(abs_change)]
abs_max = np.percentile(np.abs(valid), 98)
vmin, vmax = -abs_max, abs_max
norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

plot_data = np.ma.masked_where(np.isnan(abs_change), abs_change)
im = ax.pcolormesh(
    lon, lat, plot_data,
    transform=data_crs, cmap=cmap_div, norm=norm,
    shading="auto", zorder=1, rasterized=True,
)

# ── Draw zoom-in boxes ──
for lon_min, lon_max, lat_min, lat_max, label, color in ZOOM_REGIONS:
    width = lon_max - lon_min
    height = lat_max - lat_min
    rect_patch = mpatches.Rectangle(
        (lon_min, lat_min), width, height,
        linewidth=2, edgecolor=color, facecolor="none",
        transform=data_crs, zorder=5,
    )
    ax.add_patch(rect_patch)

# ── Spine ──
for spine in ax.spines.values():
    spine.set_linewidth(0.5)

cbar_ax = fig.add_axes([0.47, 0.205, 0.30, 0.022])
cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal", extend="both")
cbar.ax.tick_params(labelsize=7, length=3, width=0.5)
cbar.outline.set_linewidth(0.5)
cbar.ax.set_title(
    f"Δ Crop Water Stress ({label_end} − {label_start}) [million m³ World - eq]",
    fontsize=11, pad=4,
)

plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.02)

OUTPUT.mkdir(parents=True, exist_ok=True)
fig.savefig(outfile, dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
plt.show()
print(f"Saved: {outfile}")
print(f"  vmin={vmin:.2e}, vmax={vmax:.2e}")