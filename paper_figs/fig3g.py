from pathlib import Path
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import shap

from config import SHAP_ROOT

FIG_DIR   = SHAP_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── Custom blue→red colormap (colors from the density-scatter script) ──
# beeswarm 里颜色编码的是“特征值高低”：蓝=低特征值，红=高特征值（与 SHAP 默认习惯一致）
RED, BLUE = "#E74C3C", "#4A90D9"
BLUE_RED = mcolors.LinearSegmentedColormap.from_list("blue_red", [BLUE, RED])
# 想换方向（红=低特征值、蓝=高）就把列表顺序颠倒成 [RED, BLUE]


def plot_beeswarm(crop, top_n=20, cmap=BLUE_RED, save=True):
    """画一个 crop 的 beeswarm,直接在 Jupyter 里 inline 显示。"""
    cd = SHAP_ROOT / crop
    shap_vals = np.load(cd / "shap_values.npy")
    X_raw     = pd.read_parquet(cd / "X_raw.parquet")
    with open(cd / "meta.json") as f:
        meta = json.load(f)

    plt.figure(figsize=(9, 8))
    shap.summary_plot(
        shap_vals, X_raw,
        feature_names=meta['feature_names'],
        max_display=top_n,
        plot_type="dot",
        show=False,
        cmap=cmap,
    )
    # plt.title(f"SHAP beeswarm — {crop} ({meta['variant']}, n={meta['n_samples']})",
    #           fontsize=11, loc='left')
    fig = plt.gcf()
    for ax in fig.axes:
        for coll in ax.collections:
            try: coll.set_cmap(cmap)
            except: pass

    plt.tight_layout()
    if save:
        fp = FIG_DIR / f"beeswarm_{crop}_{getattr(cmap, 'name', 'custom')}.png"
        plt.savefig(fp, dpi=600, bbox_inches='tight')
        print(f"Saved → {fp}")
    plt.show()
    

plot_beeswarm("wh")