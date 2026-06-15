# 04_dataset_preparation — Dataset Assembly + Feature Engineering

> **Stage 04 of 7** in the HARM crop water-stress pipeline (see the [top-level README](../README.md)). This stage joins the outputs of Stages 01–03, adds large-scale climate indices, and applies feature engineering to produce the per-crop train/predict tables that the HARM model consumes.

## Role in the pipeline

This is where the model's training data is assembled. It consumes the three upstream pipelines and turns them into the final per-crop parquet tables:

- the merged environmental predictors from Stage 01 (ERA5, MODIS, SMIA, TWSA);
- the crop water-stress (CWS) target from Stage 02;
- the monthly and annual irrigated harvested area from Stage 03 (the `ha_irrigated` feature).

It also adds the one predictor source that does not come from the upstream pipelines: eight large-scale **NOAA climate-oscillation indices** (Niño, NAO, the East Atlantic pattern, PNA, the West Pacific pattern, PDO, the East Pacific–North Pacific pattern, and the Southern Oscillation Index), which capture interannual climate variability and are merged onto every grid cell by month. Feature engineering then builds the lagged and rolling-mean predictors, the cyclic temporal encodings, the spatial-coordinate transforms, the agro-climatic region join, and the TWSA gap-filling.

The result is the **74-predictor** matrix described in the paper: one row per grid cell × month, with the `waterstress` (CWS) target attached for the training tables and omitted for the 2019–2024 prediction tables. These feed the HARM model in Stage 05.

```
01_data_acquisition/          (ERA5 + MODIS + SMIA + TWSA merged NC)
02_water_stress_computation/     (CWS target, per crop per year)
03_iha_computation/                 (monthly + annual irrigated HA)
                   |
                   v
          04_dataset_preparation/
                (this folder)
                   |
                   v
              Train / Predict parquets
```

## Steps

1. **Setup** (one-time):
   - `setup-cropmask`  build per-crop monthly masks
   - `setup-coarse-grid`  build 50 km grid-id look-up
   - `setup-region9`  build 9-class agro-climatic region look-up
2. **NOAA indices**: `noaa-download` (one-time; shipped CSVs in `noaa/` already)
3. **Assembly**:
   - `build-train`   raw training parquets (merged NC + WS target)
   - `build-predict` raw prediction parquets (merged NC only)
   - `add-ha`        add `ha_irrigated` feature (monthly + annual)
4. **Feature engineering**: `feature-engineering` adds lag / rolling /
   time / location encoding / NOAA merge / region join / TWSA imputation.

All steps are wired under `main.py`; see `python main.py --help`.

## Project Structure

```
04_dataset_preparation/
├── README.md
├── requirements.txt
├── main.py                   # CLI
├── config/
│   ├── __init__.py           # dataclass config loader
│   └── config.yaml           # paths + year ranges + crops + FE config
├── src/
│   ├── __init__.py
│   ├── setup_cropmask.py
│   ├── setup_coarse_grid.py
│   ├── setup_region9.py
│   ├── noaa_download.py
│   ├── build_train_parquet.py
│   ├── build_predict_parquet.py
│   ├── add_irrigated_ha.py
│   └── feature_engineering.py
├── reference/                # setup outputs
│   ├── cropmasks/{crop}.nc
│   ├── coarse50km_gridid.parquet
│   └── region9.parquet
├── noaa/                     # NOAA climate index CSVs
├── data/                     # parquet I/O
│   ├── base_train/{crop}/{year}.parquet
│   ├── base_predict/{crop}/{year}.parquet
│   ├── train/{crop}/{crop}.parquet
│   └── predict/{crop}/{crop}.parquet
└── logs/
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# One-time setup
python main.py setup-cropmask
python main.py setup-coarse-grid
python main.py setup-region9 \
    --thz-tif ../some/dir/thz_class_CRUTS32_Hist_8110_100_avg.tif \
    --mst-tif ../some/dir/mst_class_CRUTS32_Hist_8110_100_avg.tif

# Refresh NOAA indices (optional — CSVs in noaa/ are shipped)
python main.py noaa-download

# Build the raw parquets
python main.py build-train
python main.py build-predict

# Add iHA feature
python main.py add-ha --mode both

# Apply feature engineering
python main.py feature-engineering --mode both

# Or run all assembly + FE steps in order
python main.py full-pipeline
```

## Config (`config/config.yaml`)

Key paths point at the sibling pipelines:

| Key                       | Default                                           |
| ------------------------- | ------------------------------------------------- |
| `water_stress.ws_dir`   | `../02_water_stress_computation/waterstress`       |
| `predictors.merged_dir` | `../01_data_acquisition/data/merged`            |
| `iha.monthly_dir`       | `../03_iha_computation/monthly`                       |
| `iha.annual_dir`        | `../03_iha_computation/annual`                        |
| `reference.template_nc` | `../01_data_acquisition/data/reference/2002.nc` |

Adjust these if your folder layout differs. Feature engineering knobs
(lag vars, rolling windows, NOAA merge, etc.) are all in the YAML.

## Notes

- **`region9.parquet`** is shipped; `setup-region9` is only needed if
  you want to rebuild it.
- **`add-ha` overwrites in place**: the BASE parquet is rewritten with
  a new `ha_irrigated` column. Re-running is safe — the column gets
  recomputed.
- **`feature_engineering --mode predict`** automatically pre-pends the
  tail of the training data so that early-year lag / rolling features
  are correct, then trims back to the target year range.
- Wheat and maize use a finer 30-class Köppen–Geiger region partition
  in place of the 9-class `region9`; that swap is applied later, in
  Stage 05 (`05_harm_model`), and does not change the parquets produced
  here.
