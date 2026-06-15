"""
Shared utilities: GPU setup, memory management, logging, data loading, feature selection.
"""

import gc
import os
import json
import logging
import subprocess
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def setup_gpu_environment(use_gpu: bool = True) -> Tuple[bool, int, str]:
    """Detect GPU availability and configure XGBoost.
    
    Returns (gpu_available, n_gpus, device_str) where device_str is
    'cuda' or 'cpu'. XGBoost 2.0+ uses tree_method='hist' for both
    CPU and GPU; GPU is selected via the 'device' parameter instead.
    """
    if not use_gpu:
        return False, 0, "cpu"
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            gpus = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
            n_gpus = len(gpus)
            logging.info(f"[GPU] Found {n_gpus} GPU(s): {gpus}")
            return True, n_gpus, "cuda"
    except Exception as e:
        logging.warning(f"[GPU] Detection failed: {e}")
    logging.info("[GPU] No GPU available, using CPU")
    return False, 0, "cpu"


def clear_gpu_memory():
    """Release GPU memory via XGBoost / PyTorch caches."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def clear_memory():
    """Force garbage collection."""
    gc.collect()


def check_memory_threshold(threshold_mb: float = 8000) -> bool:
    """Return True if available memory is below *threshold_mb*."""
    try:
        import psutil
        avail = psutil.virtual_memory().available / 1e6
        return avail < threshold_mb
    except ImportError:
        return False


def setup_logging(log_path: Path, crop: str) -> logging.Logger:
    """Configure root logger with file + console output.
    
    All logging.info() / logging.warning() calls throughout the pipeline
    will print to both console and log file.
    """
    log_path.mkdir(parents=True, exist_ok=True)

    # Configure root logger (used by logging.info() everywhere)
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Avoid duplicate handlers on repeated calls
    root.handlers.clear()

    fh = logging.FileHandler(log_path / f"{crop}.log", mode="a")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    root.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    root.addHandler(ch)

    return root


def get_logger(crop: str) -> logging.Logger:
    return logging.getLogger(crop)


def load_crop_data(
    data_root: Path,
    crop: str,
    years: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Load a single crop's parquet file, optionally filtering by year."""
    crop_dir = data_root / crop
    fpath = crop_dir / f"{crop}.parquet"
    if not fpath.exists():
        raise FileNotFoundError(f"Data file not found: {fpath}")

    df = pd.read_parquet(fpath)
    logging.info(f"[DATA] Loaded {crop}: {len(df)} rows, {len(df.columns)} columns")

    if years is not None:
        if "time" in df.columns:
            df["year"] = pd.to_datetime(df["time"]).dt.year
        if "year" in df.columns:
            df = df[df["year"].isin(years)].copy()
            logging.info(f"[DATA] Filtered to years {years[0]}-{years[-1]}: {len(df)} rows")

    return df


def get_feature_columns(
    df: pd.DataFrame,
    crop: str,
    features_cfg: dict,
) -> List[str]:
    """Determine feature columns by excluding non-feature and time columns per config."""
    time_cfg = features_cfg["time_config"]
    time_mode = time_cfg["overrides"].get(crop, time_cfg["default"])
    time_drops = time_cfg["drops"].get(time_mode, [])

    exclude = set(features_cfg.get("exclude_columns", []))
    exclude.add(crop)  # target column
    exclude.update(time_drops)

    # Select features (only columns actually present in df)
    feature_cols = [c for c in df.columns if c not in exclude]

    logging.info(f"[FEATURES] Time config: {time_mode}")
    logging.info(f"[FEATURES] Dropped time features: {[t for t in time_drops if t in df.columns]}")
    logging.info(f"[FEATURES] Total features: {len(feature_cols)}")

    return feature_cols


def compute_sample_weights(
    y: np.ndarray,
    enabled: bool = False,
    alpha: float = 1.0,
    reference_quantile: float = 0.9,
    max_weight: float = 10.0,
) -> Optional[np.ndarray]:
    """
    Compute sample weights for right-skewed targets.

    weight = 1 + alpha * (y / q_ref),  capped at max_weight

    where q_ref = quantile(y, reference_quantile). This gives:
      - baseline samples (y << q_ref): weight ≈ 1
      - high-value samples (y ≈ q_ref): weight ≈ 1 + alpha
      - extreme samples (y >> q_ref):   weight = max_weight (capped)

    The cap prevents extreme outliers from dominating training and
    causing memory issues with very large datasets.

    Returns None if disabled (XGBoost treats None as uniform weights).
    """
    if not enabled:
        return None

    q_ref = np.quantile(y[y > 0], reference_quantile) if np.any(y > 0) else 1.0
    if q_ref <= 0:
        q_ref = 1.0

    weights = 1.0 + alpha * np.clip(y, 0, None) / q_ref
    weights = np.clip(weights, 1.0, max_weight)

    n_capped = int(np.sum(weights >= max_weight))
    logging.info(f"[WEIGHTS] alpha={alpha}, q{int(reference_quantile*100)}={q_ref:.1f}, "
                 f"max_cap={max_weight}, "
                 f"weight range=[{weights.min():.2f}, {weights.max():.2f}], "
                 f"mean={weights.mean():.2f}, capped={n_capped}")
    return weights


def clean_for_json(obj: Any) -> Any:
    """Recursively convert numpy/Path types to JSON-serializable equivalents."""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [clean_for_json(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
