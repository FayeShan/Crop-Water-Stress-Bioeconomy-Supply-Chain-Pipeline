"""Preflight checks run by main.py to surface missing pre-processing outputs early."""

from pathlib import Path
from typing import List

from harm.config import load_config
from harm.constants import KOPPEN_CROPS


def check_koppen_data_available(
    crops: List[str],
    config_path: str = "config.yaml",
    phase: str = "eval",
    mode: str = "train",
) -> None:
    """Ensure Köppen-region parquets exist for wheat / maize when needed.

    The 30-class Köppen–Geiger classification (used for ``wh`` and
    ``mai``) lives in a separate parquet tree derived from the default
    9-region parquets via ``main.py koppen-regions``. Without those
    parquets the training code would silently fall back to the 9-region
    files and train on the wrong features; this check makes that
    situation impossible.
    """
    needed = [c for c in crops if c in KOPPEN_CROPS]
    if not needed:
        return

    cfg = load_config(config_path, phase=phase)

    if mode == "train":
        data_dir = cfg.paths.data_root_koppen
    elif mode == "predict":
        data_dir = cfg.paths.pred_data_root_koppen
    else:
        raise ValueError(f"mode must be 'train' or 'predict', got {mode!r}")

    if data_dir is None:
        raise SystemExit(
            f"\nERROR: crops {needed} require Köppen-region parquets, but\n"
            f"config.yaml does not define paths.{'data_root_koppen' if mode=='train' else 'pred_data_root_koppen'}.\n"
        )

    missing = []
    for c in needed:
        parquet = Path(data_dir) / c / f"{c}.parquet"
        if not parquet.exists():
            missing.append((c, parquet))

    if not missing:
        return

    listing = "\n".join(f"    {c} : {p}" for c, p in missing)
    raise SystemExit(
        f"\nERROR: The following {mode} parquets for the 30-region "
        f"Köppen pipeline are missing:\n"
        f"{listing}\n\n"
        f"Run the preprocessing step once before training or predicting "
        f"for these crops:\n\n"
        f"    python main.py koppen-regions\n"
    )
