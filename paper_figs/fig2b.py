import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import uniform_filter1d
from pathlib import Path
from config import YEARLY_BY_CROP_NC, OUTPUT_DIR

NC_FILE = YEARLY_BY_CROP_NC
OUTPUT = OUTPUT_DIR / "fig2"
OUTPUT.mkdir(parents=True, exist_ok=True)

CROPS = {
    "wh":  {"name": "Wheat",      "vars": ["wh"],       "colors": ["#FFFFFF","#FFF8E1","#FFECB3","#FFD54F","#FFB300","#E65100","#4E342E"]},
    "rice":{"name": "Rice",       "vars": ["ri1","ri2"], "colors": ["#FFFFFF","#FDE0DD","#FA9FB5","#F768A1","#C51B8A","#7A0177"]},
    "cot": {"name": "Cotton",     "vars": ["cot"],       "colors": ["#FFFFFF","#FEF0D9","#FDCC8A","#FC8D59","#E34A33","#B30000"]},
    "vgt": {"name": "Vegetables", "vars": ["vgt"],       "colors": ["#FFFFFF","#E0F7FA","#80DEEA","#26C6DA","#00897B","#004D40"]},
}

FIG_HEIGHT = 9  

ds = xr.open_dataset(NC_FILE)
lat = ds["lat"].values

for key, info in CROPS.items():
    # Sum crop variables for 2024
    data_2024 = None
    for var in info["vars"]:
        d = ds[var].sel(year=2024).values.copy()
        d[d <= 0] = 0
        d = np.nan_to_num(d, 0)
        data_2024 = d if data_2024 is None else data_2024 + d

    # Zonal mean
    zonal = np.nanmean(data_2024, axis=1)
    zonal_smooth = uniform_filter1d(zonal, size=5)

    # Build colormap from crop colors
    cmap = mcolors.LinearSegmentedColormap.from_list(
        f"cmap_{key}", info["colors"], N=256
    )
    # Dark color for the line (second-to-last in palette)
    line_color = info["colors"][-2]

    fig, ax = plt.subplots(figsize=(2.5, FIG_HEIGHT))

    # Gradient fill: split into horizontal slices colored by value
    z_norm = zonal_smooth / np.max(zonal_smooth) if np.max(zonal_smooth) > 0 else zonal_smooth
    for i in range(len(lat) - 1):
        color = cmap(z_norm[i] * 0.9 + 0.1)  # shift away from pure white
        ax.fill_betweenx(
            [lat[i], lat[i+1]], 0, [zonal_smooth[i], zonal_smooth[i+1]],
            color=color, alpha=0.8, linewidth=0,
        )

    ax.plot(zonal_smooth, lat, color=line_color, linewidth=1.3)

    ax.set_ylim(-60, 80)
    ax.set_xlim(0, np.max(zonal_smooth) * 1.1)
    ax.set_ylabel("Latitude (°)", fontsize=10)
    # ax.set_xlabel("Mean Crop Water Stress", fontsize=9)
    # ax.set_title(info["name"], fontsize=11, fontweight="bold", pad=8)

    ax.tick_params(labelsize=8)
    ax.ticklabel_format(axis="x", style="scientific", scilimits=(0, 0))
    ax.xaxis.offsetText.set_fontsize(7)

    for lat_val in [-30, 0, 30, 60]:
        ax.axhline(lat_val, color="#DDDDDD", linewidth=0.5, linestyle=":")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)

    plt.tight_layout()

    outfile = OUTPUT / f"latitude_profile_{key}_2024.png"
    fig.savefig(outfile, dpi=300, bbox_inches="tight", transparent=True)
    fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight", transparent=True)
    plt.show()
    print(f"Saved: {outfile}")
    plt.close(fig)

ds.close()