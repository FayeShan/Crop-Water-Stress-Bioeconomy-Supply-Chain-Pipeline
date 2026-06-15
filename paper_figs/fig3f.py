from pathlib import Path
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.cluster.hierarchy import linkage, leaves_list
from config import SHAP_ROOT

FIG_DIR   = SHAP_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── Custom blue→red colormap (colors from the density-scatter script) ──
RED, BLUE = "#E74C3C", "#4A90D9"
BLUE_RED = mcolors.LinearSegmentedColormap.from_list("blue_red", [BLUE, RED])  # blue=low, red=high

def plot_heatmap(
    top_n=20,
    cluster_crops=True,
    scale='sqrt',                # 'linear' | 'sqrt' | 'log'
    annotate_threshold=None,     # 例如 0.10,只标 >= 0.10 的 cell
    grid=False,
    cmap=BLUE_RED,
    save=True,
):
    crops = sorted([d.name for d in SHAP_ROOT.iterdir()
                    if d.is_dir() and (d / "shap_values.npy").exists()
                    and d.name not in {'figures', 'all_crops_combined'}])

    rows = {}
    for c in crops:
        sv = np.load(SHAP_ROOT / c / "shap_values.npy")
        with open(SHAP_ROOT / c / "meta.json") as f:
            meta = json.load(f)
        mean_abs = np.nanmean(np.abs(sv), axis=0)
        rows[c] = pd.Series(mean_abs, index=meta['feature_names'])

    df = pd.DataFrame(rows)               # rows=features, cols=crops, NaN for missing

    # --- 2. Within-crop normalize (each column sums to 1) ---
    df = df.div(df.sum(axis=0), axis=1)

    # --- 3. Select top features (by mean importance across crops) ---
    top_feats = df.mean(axis=1).sort_values(ascending=False).head(top_n).index
    df = df.loc[top_feats]

    # --- 4. Cluster crops by importance pattern similarity ---
    if cluster_crops and df.shape[1] >= 3:
        X = df.fillna(0).T.values         # crops × features
        link = linkage(X, method='average', metric='euclidean')
        df = df.iloc[:, leaves_list(link)]

    vmax = float(np.nanmax(df.values))
    if scale == 'linear':
        norm = mcolors.Normalize(vmin=0, vmax=vmax)
    elif scale == 'sqrt':
        norm = mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=vmax)
    elif scale == 'log':
        vmin = float(np.nanmin(df.values[df.values > 0]))
        norm = mcolors.LogNorm(vmin=max(vmin, 1e-4), vmax=vmax)

    cmap_use = cmap.copy()
    cmap_use.set_bad('#eeeeee')           # NaN cells → light gray

    # --- 6. 画 ---
    n_feat, n_crop = df.shape
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * n_crop + 3),
                                     max(6, 0.32 * n_feat + 2)))
    im = ax.imshow(df.values, cmap=cmap_use, aspect='auto', norm=norm)

    ax.set_xticks(range(n_crop))
    ax.set_xticklabels(df.columns, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(df.index, fontsize=10)

    if grid:
        ax.set_xticks(np.arange(-0.5, n_crop, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_feat, 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=0.3)
        ax.tick_params(which='minor', length=0)

    if annotate_threshold is not None:
        for i in range(n_feat):
            for j in range(n_crop):
                v = df.values[i, j]
                if np.isnan(v) or v < annotate_threshold:
                    continue
                # 按该 cell 实际颜色的亮度自动选黑/白字（红蓝双色亮度非单调，不能只看 norm 值）
                # r, g, b, _ = cmap_use(norm(v))
                # lum = 0.299 * r + 0.587 * g + 0.114 * b
                # color = 'black' if lum > 0.6 else 'white'
                ax.text(j, i, f"{v:.2f}", ha='center', va='center',
                        color='black', fontsize=8, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar_label = "Relative importance (column-normalized)"
    if scale != 'linear':
        cbar_label += f"  ({scale} scale)"
    cbar.set_label(cbar_label, fontsize=9)

    # ax.set_title(f"SHAP feature importance — top {top_n} features × {n_crop} crops",
    #              fontsize=11, loc='left', pad=10)
    ax.set_xlabel("Crop", fontsize=10)
    ax.set_ylabel("Feature", fontsize=10)
    plt.tight_layout()

    if save:
        suffix = f"_top{top_n}_within_crop"
        if cluster_crops: suffix += "_clustered"
        if scale != 'linear': suffix += f"_{scale}"
        suffix += f"_{getattr(cmap, 'name', 'custom')}"   # 把 colormap 名写进文件名,避免覆盖其它配色的输出
        fp = FIG_DIR / f"heatmap{suffix}.png"
        fig.savefig(fp, dpi=600, bbox_inches='tight')
        print(f"Saved → {fp}")

    plt.show()
    return df


plot_heatmap(top_n=18, cluster_crops=True, scale='sqrt',
                annotate_threshold=0.10, grid=True)