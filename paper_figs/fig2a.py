import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from config import YEARLY_BY_CROP_NC, OUTPUT_DIR

NC_YEARLY = YEARLY_BY_CROP_NC
OUTPUT = OUTPUT_DIR / "fig2"
OUTPUT.mkdir(parents=True, exist_ok=True)

YEAR = 2024

# Top 6 crops by global water stress
TOP6 = {
    "wh": {
        "name": "Wheat",
        "crops": ["wh"],
        "colors": ["#FFFFFF", "#FFF8E1", "#FFECB3", "#FFD54F", "#FFB300", "#E65100", "#4E342E"]
    },
    "rice": {
        "name": "Rice",
        "crops": ["ri1", "ri2"],
        "colors": ["#FFFFFF", "#FDE0DD", "#FA9FB5", "#F768A1", "#C51B8A", "#7A0177"]
    },
    "cot": {
        "name": "Cotton",
        "crops": ["cot"],
        "colors": ["#FFFFFF", "#FEF0D9", "#FDCC8A", "#FC8D59", "#E34A33", "#B30000"]
    },
    "vgt": {
        "name": "Vegetables",
        "crops": ["vgt"],
        "colors": ["#FFFFFF", "#E0F7FA", "#80DEEA", "#26C6DA", "#00897B", "#004D40"]
    },
    "oac": {
        "name": "Other Annual",
        "crops": ["oac"],
        "colors": ["#FFFFFF", "#FFF0F0", "#FFBCBC", "#FF7878", "#D93A3A", "#8B0000"]
    },
    "pec": {
        "name": "Perennial Evergreen",
        "crops": ["pec"],
        "colors": ["#FFFFFF", "#F0F9E8", "#BAE4BC", "#7BCCC4", "#43A2CA", "#0868AC"]
    },
}

ds = xr.open_dataset(NC_YEARLY)
lat = ds["lat"].values
lon = ds["lon"].values

for key, cfg in TOP6.items():
    print(f"\nPlotting {key} ({cfg['name']})...", flush=True)

    # Sum multiple crops if needed, e.g. ri1 + ri2
    data = None

    for crop in cfg["crops"]:
        if crop not in ds:
            print(f"  Warning: {crop} not found in dataset, skipping.")
            continue

        layer = ds[crop].sel(year=YEAR).values.copy()

        if data is None:
            data = np.where(np.isnan(layer), 0, layer)
        else:
            data += np.where(np.isnan(layer), 0, layer)

    if data is None:
        print(f"  {key}: no data, skipping")
        continue

    # Mask zero / negative values
    data[data <= 0] = np.nan

    if np.all(np.isnan(data)):
        print(f"  {key}: all NaN, skipping")
        continue

    cmap = mcolors.LinearSegmentedColormap.from_list(
        f"ws_{key}",
        cfg["colors"],
        N=256
    )
    cmap.set_bad(color="white", alpha=0)

    # Original unit: m³ World-eq
    vmin = 1e3
    vmax = np.nanpercentile(data, 99.5)

    if vmax <= vmin:
        vmax = np.nanmax(data)

    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(
        figsize=(14, 9),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    ax.set_global()
    ax.set_extent([-180, 180, -60, 80], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor="white", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color="#666666", zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, color="#999999", zorder=3)
    ax.set_facecolor("white")

    plot_data = np.ma.masked_where(np.isnan(data), data)

    im = ax.pcolormesh(
        lon,
        lat,
        plot_data,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        shading="auto",
        zorder=1,
        rasterized=True,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    cbar_ax = fig.add_axes([0.47, 0.25, 0.30, 0.022])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal", extend="max")

    cbar.ax.tick_params(labelsize=7, length=3, width=0.5)
    cbar.outline.set_linewidth(0.5)

    cbar.ax.set_title(
        f"{cfg['name']} — Crop Water Stress ({YEAR}) [m³ World-eq]",
        fontsize=12,
        pad=4
    )

    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.02)
    outfile = OUTPUT / f"cwss_{key}_{YEAR}.png"
    fig.savefig(outfile, dpi=300, bbox_inches="tight", transparent=True)
    fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")

    plt.show()
    plt.close(fig)

    print(f"  Saved: {outfile}")

ds.close()
print("\nDone!")