import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from config import YEARLY_TOTAL_NC, OUTPUT_DIR

NC_YEARLY = YEARLY_TOTAL_NC
OUTPUT = OUTPUT_DIR
OUTPUT.mkdir(parents=True, exist_ok=True)

ZOOM_REGIONS = [
    (66, 89, 23, 33, "indo_gangetic",  "#E85D75"),
    (24, 36, 22, 33, "nile_delta",     "#FF8A65"),
    (-10, 30, 35, 46, "med_europe",    "#66BB6A"),
    (44, 64, 25, 40, "iran",           "#42A5F5"),
]

FIG_HEIGHT = 6  # inches — same for all 4 panels

ds = xr.open_dataset(NC_YEARLY)
lat, lon = ds["lat"].values, ds["lon"].values
data_1995 = ds["total_water_stress"].sel(year=1995).values.copy()
data_2024 = ds["total_water_stress"].sel(year=2024).values.copy()
ds.close()

abs_change = (data_2024 - data_1995) / 1e6 
both_nan = np.isnan(data_1995) & np.isnan(data_2024)
both_zero = (np.nan_to_num(data_1995, 0) == 0) & (np.nan_to_num(data_2024, 0) == 0)
abs_change[both_nan | both_zero] = np.nan

valid = abs_change[~np.isnan(abs_change)]
abs_max = np.percentile(np.abs(valid), 98)

div_colors = [
    "#0E4D4D", "#1A7A70", "#3FA898", "#8ECFC2",
    "#F5F0EB",
    "#E8B870", "#CC7A28", "#A84820", "#6E1A0A",
]
cmap_div = mcolors.LinearSegmentedColormap.from_list("div_bwr", div_colors, N=256)
cmap_div.set_bad(color="white", alpha=0)

# ── Plot each region ──
for lon_min, lon_max, lat_min, lat_max, name, color in ZOOM_REGIONS:

    # Compute figure width to keep height constant
    # Account for latitude compression (cos correction)
    mid_lat = np.radians((lat_min + lat_max) / 2)
    lon_span = (lon_max - lon_min) * np.cos(mid_lat)
    lat_span = lat_max - lat_min
    aspect = lon_span / lat_span
    fig_width = FIG_HEIGHT * aspect
    # Clamp to reasonable range
    fig_width = max(4, min(16, fig_width))

    fig, ax = plt.subplots(
        figsize=(fig_width, FIG_HEIGHT),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="white", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, color="#555555", zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, color="#888888", zorder=3)
    ax.add_feature(cfeature.RIVERS, linewidth=0.3, color="#AACCEE", zorder=2)
    ax.set_facecolor("white")
    import matplotlib.ticker as mticker
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

    ax.set_xticks(np.arange(np.ceil(lon_min / 10) * 10, lon_max, 10), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(np.ceil(lat_min / 5) * 5, lat_max, 5), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(labelsize=13, length=3, width=0.5, direction="out")

    # Use region-specific or global norm
    # Option A: shared global scale (better for comparison across panels)
    norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
    # Option B: region-specific scale (uncomment to use)
    # region_mask = (
    #     (lon[None, :] >= lon_min) & (lon[None, :] <= lon_max) &
    #     (lat[:, None] >= lat_min) & (lat[:, None] <= lat_max)
    # )
    # region_valid = abs_change[region_mask & ~np.isnan(abs_change)]
    # if len(region_valid) > 10:
    #     r_max = np.percentile(np.abs(region_valid), 98)
    #     norm = mcolors.TwoSlopeNorm(vmin=-r_max, vcenter=0, vmax=r_max)

    plot_data = np.ma.masked_where(np.isnan(abs_change), abs_change)
    im = ax.pcolormesh(
        lon, lat, plot_data,
        transform=ccrs.PlateCarree(), cmap=cmap_div, norm=norm,
        shading="auto", zorder=1, rasterized=True,
    )

    # Border box in the region's color
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_edgecolor(color)

    # Colorbar
    # cbar_ax = fig.add_axes([0.15, 0.08, 0.70, 0.020])
    # cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal", extend="both")
    # cbar.ax.tick_params(labelsize=8, length=3, width=0.5)
    # cbar.outline.set_linewidth(0.5)
    # cbar.ax.set_title("Δ Crop Water Stress (2024 − 1995)", fontsize=9, pad=4)

    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.12)

    outfile = OUTPUT / f"zoom_{name}_change_2024_vs_1995.png"
    fig.savefig(outfile, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.show()
    print(f"Saved: {outfile}  (size: {fig_width:.1f} x {FIG_HEIGHT})")

    plt.close(fig)