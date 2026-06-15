"""
Compute SHAP values for all 27 crops in the HARM_v2 pipeline.

Jupyter usage:
    from compute_shap import run, ALL_CROPS
    print(len(ALL_CROPS), ALL_CROPS)        # verify crop list
    results = run(['mil'])                  # test one crop
    results = run(['mil', 'soy', 'wh'])     # test a few
    results = run()                         # run all crops

CLI usage:
    python compute_shap.py                  # run all
    python compute_shap.py mil soy wh       # run specific crops
"""

from pathlib import Path
import numpy as np
import pandas as pd
import json
import joblib
import xgboost as xgb
import time
import sys

from config import HARM_EVAL_DIR, TRAIN_DATA_ROOT, TRAIN_DATA_ROOT_KOPPEN

# =============================================================================
# CONFIG
# =============================================================================
HARM_V2          = HARM_EVAL_DIR
DATA_ROOT        = TRAIN_DATA_ROOT
DATA_ROOT_KOPPEN = TRAIN_DATA_ROOT_KOPPEN
OUT_ROOT         = HARM_V2 / "shap_outputs"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

N_SAMPLES   = 5000        # samples per crop for SHAP
RANDOM_SEED = 42

# Crop-variant mapping (from config_retrain.yaml)
WEIGHTED_CROPS   = {'aff', 'bar', 'bea', 'cas', 'ckp', 'coc', 'cof', 'cot', 'cwp',
                    'mai', 'mil', 'nut', 'plm', 'pot', 'ri1', 'ri2', 'sun', 'vgt', 'wh'}
UNWEIGHTED_CROPS = {'soy', 'sor', 'sgb', 'rap', 'sgc'}
KOPPEN_CROPS     = {'wh', 'mai'}
# Edge case: oac/pec/pdc — auto-detect variant below

# Skip non-crop directories at the top level of HARM_v2
NON_CROP_DIRS = {'shap_outputs', 'predictions', 'weighted_a1.0'}

# Build crop list: only directories that have at least one of global/global_weighted
# Note: parentheses around `or` are critical here — fixed from earlier version
ALL_CROPS = sorted([
    d.name for d in HARM_V2.iterdir()
    if d.is_dir()
    and d.name not in NON_CROP_DIRS
    and ((d / 'global').exists() or (d / 'global_weighted').exists())
])

# =============================================================================
# HELPERS
# =============================================================================
def resolve_variant(crop: str) -> str:
    """Return 'global_weighted' or 'global' based on crop classification."""
    if crop in WEIGHTED_CROPS:
        return 'global_weighted'
    if crop in UNWEIGHTED_CROPS:
        return 'global'
    # Edge cases (oac, pec, pdc): prefer global_weighted if exists, else global
    if (HARM_V2 / crop / 'global_weighted').exists():
        return 'global_weighted'
    return 'global'

def resolve_data_path(crop: str) -> Path:
    """Return the parquet path for a crop (koppen variant for wh/mai)."""
    root = DATA_ROOT_KOPPEN if crop in KOPPEN_CROPS else DATA_ROOT
    return root / crop / f"{crop}.parquet"

# =============================================================================
# CORE: per-crop SHAP computation
# =============================================================================
def compute_shap_for_crop(crop: str, skip_existing: bool = True) -> dict:
    """Compute and save SHAP for a single crop. Returns status dict."""
    t0 = time.time()
    
    # ★ 新加：如果已经算过就跳过 ★
    out_dir = OUT_ROOT / crop
    if skip_existing and (out_dir / "shap_values.npy").exists() \
                    and (out_dir / "meta.json").exists():
        return {'crop': crop, 'status': 'skip', 'reason': 'already exists'}
    
    variant     = resolve_variant(crop)
    variant_dir = HARM_V2 / crop / variant
    data_path   = resolve_data_path(crop)

    if not variant_dir.exists():
        return {'crop': crop, 'status': 'skip', 'reason': f'no {variant}/ dir'}
    if not data_path.exists():
        return {'crop': crop, 'status': 'skip', 'reason': f'no data at {data_path}'}

    # --- Load feature list (training order) ---
    with open(variant_dir / "feature_names.json") as f:
        feature_names = json.load(f)
    n_feat = len(feature_names)

    # --- Load crop parquet, subsample ---
    df      = pd.read_parquet(data_path)
    n_total = len(df)
    n_take  = min(N_SAMPLES, n_total)
    df_sub  = df.sample(n=n_take, random_state=RANDOM_SEED).reset_index(drop=True)

    # --- Build X in training feature order ---
    missing = [c for c in feature_names if c not in df_sub.columns]
    if missing:
        return {'crop': crop, 'status': 'error',
                'reason': f'missing features: {missing[:3]}...'}
    X_raw = df_sub[feature_names].copy()

    # --- Scale (model was trained on MinMax-scaled features) ---
    scaler   = joblib.load(variant_dir / "scaler.joblib")
    X_scaled = scaler.transform(X_raw.values)

    # --- Load q50 booster ---
    booster = xgb.Booster()
    booster.load_model(str(variant_dir / "model_q50.json"))

    import os
    booster.set_param({'nthread': os.cpu_count()})

    # --- Compute SHAP via XGBoost native (fast, supports GPU if available) ---
    dmat           = xgb.DMatrix(X_scaled, feature_names=feature_names)
    contribs       = booster.predict(dmat, pred_contribs=True)   # (n, n_feat + 1)
    shap_values    = contribs[:, :-1].astype(np.float32)         # drop bias column
    expected_value = float(contribs[:, -1].mean())               # mean bias = expected value

    # --- Sanity check ---
    if shap_values.shape != (n_take, n_feat):
        return {'crop': crop, 'status': 'error',
                'reason': f'shape mismatch {shap_values.shape} vs ({n_take},{n_feat})'}

    # --- Save outputs ---
    out_dir = OUT_ROOT / crop
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "shap_values.npy", shap_values)
    X_raw.to_parquet(out_dir / "X_raw.parquet", index=False)

    # Also save the join keys so we can trace back samples
    keys_cols = [c for c in ['grid50_id', 'time', 'date'] if c in df_sub.columns]
    if keys_cols:
        df_sub[keys_cols].to_parquet(out_dir / "sample_keys.parquet", index=False)

    meta = {
        'crop': crop,
        'variant': variant,
        'data_path': str(data_path),
        'n_samples': int(n_take),
        'n_features': int(n_feat),
        'feature_names': feature_names,
        'expected_value': expected_value,
        'random_seed': RANDOM_SEED,
        'elapsed_sec': round(time.time() - t0, 2),
    }
    with open(out_dir / "meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    return {'crop': crop, 'status': 'ok', 'variant': variant,
            'n_samples': n_take, 'n_features': n_feat,
            'elapsed': meta['elapsed_sec']}

# =============================================================================
# BATCH RUNNER (Jupyter-safe)
# =============================================================================
def run(crops=None):
    """
    Run SHAP computation.
      crops=None       → run all crops in ALL_CROPS
      crops=['mil',..] → run only these crops
    Returns list of result dicts.
    """
    crops_to_run = crops if crops else ALL_CROPS
    print(f"Running SHAP on {len(crops_to_run)} crops → {OUT_ROOT}\n")

    results = []
    for i, crop in enumerate(crops_to_run, 1):
        print(f"[{i:2d}/{len(crops_to_run)}] {crop:5s}", end=' ', flush=True)
        try:
            r = compute_shap_for_crop(crop)
            results.append(r)
            if r['status'] == 'ok':
                print(f"✅ {r['variant']:15s} n={r['n_samples']:5d}  feat={r['n_features']:2d}  ({r['elapsed']}s)")
            else:
                print(f"⚠️  {r['status']}: {r.get('reason','')}")
        except Exception as e:
            print(f"❌ ERROR: {e}")
            """
Compute SHAP values for all 27 crops in the HARM_v2 pipeline.

Jupyter usage:
    from compute_shap import run, ALL_CROPS
    print(len(ALL_CROPS), ALL_CROPS)        # verify crop list
    results = run(['mil'])                  # test one crop
    results = run(['mil', 'soy', 'wh'])     # test a few
    results = run()                         # run all crops

CLI usage:
    python compute_shap.py                  # run all
    python compute_shap.py mil soy wh       # run specific crops
"""

from pathlib import Path
import numpy as np
import pandas as pd
import json
import joblib
import xgboost as xgb
import time
import sys

from config import HARM_EVAL_DIR, TRAIN_DATA_ROOT, TRAIN_DATA_ROOT_KOPPEN

# =============================================================================
# CONFIG
# =============================================================================
HARM_V2          = HARM_EVAL_DIR
DATA_ROOT        = TRAIN_DATA_ROOT
DATA_ROOT_KOPPEN = TRAIN_DATA_ROOT_KOPPEN
OUT_ROOT         = HARM_V2 / "shap_outputs"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

N_SAMPLES   = 5000        # samples per crop for SHAP
RANDOM_SEED = 42

# Crop-variant mapping (from config_retrain.yaml)
WEIGHTED_CROPS   = {'aff', 'bar', 'bea', 'cas', 'ckp', 'coc', 'cof', 'cot', 'cwp',
                    'mai', 'mil', 'nut', 'plm', 'pot', 'ri1', 'ri2', 'sun', 'vgt', 'wh'}
UNWEIGHTED_CROPS = {'soy', 'sor', 'sgb', 'rap', 'sgc'}
KOPPEN_CROPS     = {'wh', 'mai'}
# Edge case: oac/pec/pdc — auto-detect variant below

# Skip non-crop directories at the top level of HARM_v2
NON_CROP_DIRS = {'shap_outputs', 'predictions', 'weighted_a1.0'}

# Build crop list: only directories that have at least one of global/global_weighted
# Note: parentheses around `or` are critical here — fixed from earlier version
ALL_CROPS = sorted([
    d.name for d in HARM_V2.iterdir()
    if d.is_dir()
    and d.name not in NON_CROP_DIRS
    and ((d / 'global').exists() or (d / 'global_weighted').exists())
])

# =============================================================================
# HELPERS
# =============================================================================
def resolve_variant(crop: str) -> str:
    """Return 'global_weighted' or 'global' based on crop classification."""
    if crop in WEIGHTED_CROPS:
        return 'global_weighted'
    if crop in UNWEIGHTED_CROPS:
        return 'global'
    # Edge cases (oac, pec, pdc): prefer global_weighted if exists, else global
    if (HARM_V2 / crop / 'global_weighted').exists():
        return 'global_weighted'
    return 'global'

def resolve_data_path(crop: str) -> Path:
    """Return the parquet path for a crop (koppen variant for wh/mai)."""
    root = DATA_ROOT_KOPPEN if crop in KOPPEN_CROPS else DATA_ROOT
    return root / crop / f"{crop}.parquet"

# =============================================================================
# CORE: per-crop SHAP computation
# =============================================================================
def compute_shap_for_crop(crop: str, skip_existing: bool = True) -> dict:
    """Compute and save SHAP for a single crop. Returns status dict."""
    t0 = time.time()
    
    # ★ 新加：如果已经算过就跳过 ★
    out_dir = OUT_ROOT / crop
    if skip_existing and (out_dir / "shap_values.npy").exists() \
                    and (out_dir / "meta.json").exists():
        return {'crop': crop, 'status': 'skip', 'reason': 'already exists'}
    
    variant     = resolve_variant(crop)
    variant_dir = HARM_V2 / crop / variant
    data_path   = resolve_data_path(crop)

    if not variant_dir.exists():
        return {'crop': crop, 'status': 'skip', 'reason': f'no {variant}/ dir'}
    if not data_path.exists():
        return {'crop': crop, 'status': 'skip', 'reason': f'no data at {data_path}'}

    # --- Load feature list (training order) ---
    with open(variant_dir / "feature_names.json") as f:
        feature_names = json.load(f)
    n_feat = len(feature_names)

    # --- Load crop parquet, subsample ---
    df      = pd.read_parquet(data_path)
    n_total = len(df)
    n_take  = min(N_SAMPLES, n_total)
    df_sub  = df.sample(n=n_take, random_state=RANDOM_SEED).reset_index(drop=True)

    # --- Build X in training feature order ---
    missing = [c for c in feature_names if c not in df_sub.columns]
    if missing:
        return {'crop': crop, 'status': 'error',
                'reason': f'missing features: {missing[:3]}...'}
    X_raw = df_sub[feature_names].copy()

    # --- Scale (model was trained on MinMax-scaled features) ---
    scaler   = joblib.load(variant_dir / "scaler.joblib")
    X_scaled = scaler.transform(X_raw.values)

    # --- Load q50 booster ---
    booster = xgb.Booster()
    booster.load_model(str(variant_dir / "model_q50.json"))

    import os
    booster.set_param({'nthread': os.cpu_count()})

    # --- Compute SHAP via XGBoost native (fast, supports GPU if available) ---
    dmat           = xgb.DMatrix(X_scaled, feature_names=feature_names)
    contribs       = booster.predict(dmat, pred_contribs=True)   # (n, n_feat + 1)
    shap_values    = contribs[:, :-1].astype(np.float32)         # drop bias column
    expected_value = float(contribs[:, -1].mean())               # mean bias = expected value

    # --- Sanity check ---
    if shap_values.shape != (n_take, n_feat):
        return {'crop': crop, 'status': 'error',
                'reason': f'shape mismatch {shap_values.shape} vs ({n_take},{n_feat})'}

    # --- Save outputs ---
    out_dir = OUT_ROOT / crop
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "shap_values.npy", shap_values)
    X_raw.to_parquet(out_dir / "X_raw.parquet", index=False)

    # Also save the join keys so we can trace back samples
    keys_cols = [c for c in ['grid50_id', 'time', 'date'] if c in df_sub.columns]
    if keys_cols:
        df_sub[keys_cols].to_parquet(out_dir / "sample_keys.parquet", index=False)

    meta = {
        'crop': crop,
        'variant': variant,
        'data_path': str(data_path),
        'n_samples': int(n_take),
        'n_features': int(n_feat),
        'feature_names': feature_names,
        'expected_value': expected_value,
        'random_seed': RANDOM_SEED,
        'elapsed_sec': round(time.time() - t0, 2),
    }
    with open(out_dir / "meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    return {'crop': crop, 'status': 'ok', 'variant': variant,
            'n_samples': n_take, 'n_features': n_feat,
            'elapsed': meta['elapsed_sec']}

# =============================================================================
# BATCH RUNNER (Jupyter-safe)
# =============================================================================
def run(crops=None):
    """
    Run SHAP computation.
      crops=None       → run all crops in ALL_CROPS
      crops=['mil',..] → run only these crops
    Returns list of result dicts.
    """
    crops_to_run = crops if crops else ALL_CROPS
    print(f"Running SHAP on {len(crops_to_run)} crops → {OUT_ROOT}\n")

    results = []
    for i, crop in enumerate(crops_to_run, 1):
        print(f"[{i:2d}/{len(crops_to_run)}] {crop:5s}", end=' ', flush=True)
        try:
            r = compute_shap_for_crop(crop)
            results.append(r)
            if r['status'] == 'ok':
                print(f"✅ {r['variant']:15s} n={r['n_samples']:5d}  feat={r['n_features']:2d}  ({r['elapsed']}s)")
            else:
                print(f"⚠️  {r['status']}: {r.get('reason','')}")
        except Exception as e:
            print(f"❌ ERROR: {e}")