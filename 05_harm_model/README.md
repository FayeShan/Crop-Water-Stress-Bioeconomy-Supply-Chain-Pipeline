# 05_harm_model — HARM (Hierarchical Additive Residual Model)

> **Stage 05 of 7** in the HARM crop water-stress pipeline (see the [top-level README](../README.md)). This is the modelling core: a three-layer XGBoost model that predicts monthly crop water stress for 27 crops at 5-arcminute resolution, extending the physically grounded reference to recent years.

## Role in the pipeline

HARM is where the predictions are made. It takes the per-crop feature tables from Stage 04 (≈74 predictors plus the crop water-stress target), learns crop-specific relationships between environmental conditions and crop water stress (CWS), and predicts monthly CWS on the 5-arcminute grid. The physically grounded reference from Stage 02 covers 1995–2018; HARM is trained on the 2000–2018 portion of that record (the period with MODIS vegetation predictors, which begin in 2000) and is then used to extend the series to **2019–2024**. The resulting country-year CWS predictions feed the GLORIA supply-chain attribution in Stage 06.

The model decomposes monthly CWS into three additive layers,
`y = f_global(X) + f_regional(X) + f_extreme(X)`:

- **Global** (G) — crop-specific, XGBoost on the full pooled sample;
  the common predictive backbone.
- **Regional** (R) — residual correction within agro-ecological regions
  (9-class THZ × MST for most crops; 30-class Köppen–Geiger for wheat
  and maize);
- **Extreme** (E) — per-region detector + bounded corrector for
  high-stress outliers that the G+R stage systematically
  underestimates, adopted only on (crop × region) pairs that pass a
  cross-validated acceptance test.

An optional **weighted** variant (GW / GW+RW) reweights training
samples towards the high-stress tail; per-crop adoption of the
weighted pipeline is decided by country-level R² (Supplementary
Table S3.2).

## Two-phase workflow

The pipeline runs in two phases, selected via ``--phase eval|final``:

| Phase  | Years            | Purpose                                                    | Result dir            |
|--------|------------------|------------------------------------------------------------|-----------------------|
| `eval` | 2000–2016 train, 2017–2018 held-out eval | Develop, tune (Optuna), and evaluate the model; lock down best-params and adoption decisions. | `./results/eval/`    |
| `final`| 2000–2018 train, 2019–2024 predict        | Retrain on the full historical record with fixed best-params from the eval phase and generate prediction outputs. | `./results/final/`   |

All metric values reported in the paper come from the eval phase on
2017–2018; the final phase exists only to deliver forecasts for
2019–2024.

## Project structure

```
05_harm_model/
├── README.md
├── main.py                          # Unified CLI entry point
├── config.yaml                      # Unified config (eval + final phases)
├── requirements.txt
├── harm/                            # Core library (imported by all scripts)
│   ├── __init__.py
│   ├── config.py                    # YAML loader with --phase overlay
│   ├── constants.py                 # ALL_CROPS / WEIGHTED_CROPS / KOPPEN_CROPS
│   ├── global_layer.py              # Global-layer training + CV
│   ├── regional_layer.py            # Regional-layer residual correction
│   ├── extreme_layer.py             # Detector + corrector + adoption gate
│   ├── training.py                  # Shared training utilities
│   ├── metrics.py                   # R², RMSE, NRMSE, PICP, CRPS, etc.
│   └── utils.py
└── src/
    ├── train/                       # Training orchestrators
    │   ├── run_pipeline.py          # G → R → E main orchestrator
    │   ├── run_global.py
    │   ├── run_regional.py
    │   ├── run_extreme.py           # Extreme layer (add --weighted for GW+RW base)
    │   ├── run_retrain.py           # Final retrain (add --weighted for GW+RW+E cascade)
    │   ├── _retrain_unweighted_impl.py  # internal: unweighted retrain implementation
    │   └── _retrain_weighted_impl.py    # internal: weighted retrain implementation
    ├── predict/                     # Prediction orchestrators
    │   ├── run_predict.py           # Retrained model → 2019–2024
    │   └── run_eval_predictions.py  # Eval model → 2017–2018 held-out
    ├── evaluate/                    # Post-hoc diagnostics
    │   ├── compute_multilevel.py    # Metrics at 4 aggregation levels
    │   ├── country_eval.py          # Country-level metrics
    │   └── build_metric_tables.py   # Paper-ready metric tables
    └── utils/
        └── replace_regiontype_koppen.py   # Swap 9-region THZ × MST
                                            # for 30-class Köppen (wh, mai)
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick start

The unified CLI in `main.py` dispatches to each stage. Standard paper
workflow:

```bash
# 1. Eval phase — develop the model, decide adoption per crop
python main.py train        --phase eval  --crops wh --layers all
python main.py eval-predict --phase eval  --crops wh

# 2. Final phase — retrain on 2000-2018, predict 2019-2024
python main.py retrain --phase final --crops wh --layers all
python main.py predict --phase final --crops wh

# 2b. Export predictions as flat {crop}.csv for Stages 06 and 08
python main.py export-predictions --phase final

# 3. Diagnostics (feed Supplementary Tables S3.1–S3.3, S5.1, S6.4, S6.6)
python main.py multilevel
python main.py country-eval
python main.py metric-tables
```

Individual layers can be re-run in isolation:

```bash
python main.py train-global    --phase eval --crops wh
python main.py train-regional  --phase eval --crops wh
python main.py train-extreme   --phase eval --crops cot sgc              # G+R base
python main.py train-extreme   --phase eval --crops bar bea --weighted   # GW+RW base

# Final retrain (unweighted cascade) vs (weighted cascade)
python main.py retrain --phase final --crops soy sor sgb rap sgc
python main.py retrain --phase final --crops bar bea ckp --weighted
```

Köppen region swap for wheat / maize (pre-processing, run once):

```bash
python main.py koppen-regions
```

## Inputs

- **Training parquets** (from `04_dataset_preparation/`):
  `../04_dataset_preparation/data/train/{crop}/{crop}.parquet`
  (one row per grid cell × month; features ≈ 74 plus the `waterstress`
  target; see Supplementary Table S1).
- **Prediction parquets**:
  `../04_dataset_preparation/data/predict/{crop}/{crop}.parquet`
  (2019–2024 rows, no target; prediction input for the final phase).
- **Köppen region variants**: `./data/train_koppen/` and
  `./data/predict_koppen/` are generated by `koppen-regions` for
  wheat and maize and keep all columns except that `regiontype` is
  replaced by the 30-class Köppen–Geiger classification.
- **Filtering resources** under `./data/`:
  `harvest_area/` (per-crop monthly HA masks from
  `03_iha_computation/`), `spam_2020/` (SPAM 2020 irrigated HA, used
  as an additional spatial mask), `reference/` (country shapefile plus
  the Köppen–Geiger NC).

## Outputs

Under ``results/{eval|final}/{crop}/``:

```
global/              # G models  (.pkl per quantile + metrics.json)
regional/            # G+R models + residual-correction metadata
extreme/             # Extreme on G+R; adoption decision + corrector
global_weighted/     # GW models (if weighting helps this crop)
regional_weighted/
extreme_weighted/
val_predictions.parquet        # (eval phase only) 2017-2018 predictions
predictions/{crop}/predictions.parquet   # (final phase only) 2019-2024 predictions
```

`export-predictions` then writes flat per-crop CSVs to
``results/final/predictions_csv/{crop}.csv`` (columns `time, lat, lon,
year, y_pred`) — the input consumed by Stage 06 (`06_gloria_mapping/`)
and Stage 08 (`08_final_dataset/`).

## Configuration

All runtime settings live in ``config.yaml``. Shared sections (paths,
crops, features, layer hyperparameters) apply to both phases;
phase-specific settings are under ``phases.{eval|final}``:

| Key                                    | eval              | final             |
|----------------------------------------|-------------------|-------------------|
| `result_root`                          | `./results/eval`  | `./results/final` |
| `train_start` / `train_end`            | 2000 / 2016       | 2000 / 2018       |
| `eval_start` / `eval_end`              | 2017 / 2018       | 2019 / 2019 (placeholder) |
| `global_tuning_method`                 | `optuna`          | `fixed` (reuses eval best-params) |
| `include_config_comparison` / `save_latex` | true           | false             |

The ``paths`` block resolves cross-module dependencies (inputs under
`../04_dataset_preparation/data/`; reference data under `./data/`).

## Reproducibility notes

- All reported metrics in the main text and the Supplementary
  Information come from the **eval phase** on the 2017–2018 held-out
  period. The final-phase models are applied only to generate the
  2019–2024 prediction outputs consumed by `06_gloria_mapping/`.
- The pipeline is deterministic given a fixed ``common.random_state``
  (default 42); Optuna tuning uses the same seed.
- Köppen classification of wheat and maize (Table S3.1 footnote /
  §S4) is handled inside the pipeline by invoking
  `python main.py koppen-regions` once; this is not a manual data
  edit and does not modify the raw inputs under
  `../04_dataset_preparation/`.
