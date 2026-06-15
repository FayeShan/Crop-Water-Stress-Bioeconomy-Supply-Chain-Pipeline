"""Shared input/output paths for the Python figure scripts.

All paths are resolved relative to this file, so the scripts run from any
working directory. Each entry points at the pipeline stage that produces
the data (large inputs are gitignored — regenerate them by running the
corresponding stage, or fetch the published dataset from Zenodo).
"""

from pathlib import Path

_FIGS = Path(__file__).resolve().parent      # paper_figs/
_REPO = _FIGS.parent                          # repository root

# ── Stage 08 — yearly gridded crop water stress ──
YEARLY_TOTAL_NC = _REPO / "08_final_dataset/final_dataset/netcdf/yearly/yearly_total_1990_2024.nc"
YEARLY_BY_CROP_NC = _REPO / "08_final_dataset/final_dataset/netcdf/yearly/yearly_by_crop_1990_2024.nc"

# ── Stage 07 — MRIO supply-chain aggregation (consumed by the R scripts) ──
AGG_CSV = _REPO / "07_gloria_mrio/dashboard_data/aggregated_all_years.csv"

# ── Stage 06 — per-crop country-month predictions (fig2month) ──
COUNTRY_CSV_DIR = _REPO / "06_gloria_mapping/data/predictions_country"

# ── Stage 05 — eval-phase model outputs ──
HARM_EVAL_DIR = _REPO / "05_harm_model/results/eval"          # per-crop models + summaries
EVAL_PRED_DIR = HARM_EVAL_DIR / "eval_predictions"            # {crop}_eval_predictions.parquet
SHAP_ROOT = HARM_EVAL_DIR / "shap_outputs"                   # SHAP arrays (fig3_shap_computation output)

# ── Stage 04 / 05 — training feature tables (fig3 SHAP computation) ──
TRAIN_DATA_ROOT = _REPO / "04_dataset_preparation/data/train"
TRAIN_DATA_ROOT_KOPPEN = _REPO / "05_harm_model/data/train_koppen"

# ── Country boundaries (shipped with Stage 06) ──
SHAPEFILE = _REPO / "06_gloria_mapping/country_shape/ne_10m_admin_0_countries.shp"

# ── si2 comparison inputs (place the two CSVs here; see README) ──
COMPARISON_DATA_DIR = _FIGS / "data"

# ── Figure output root ──
OUTPUT_DIR = _FIGS / "output"
