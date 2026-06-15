# 06_gloria_mapping — CWS → GLORIA hybrid satellite account

> **Stage 06 of 7** in the HARM crop water-stress pipeline (see the [top-level README](../README.md)). This stage turns the model's per-crop crop water-stress predictions into a per-year GLORIA satellite account — the environmental extension that the supply-chain attribution (Stage 07) runs on.

## Role in the pipeline

Stages 01–05 produce crop water stress (CWS) on the production side: how much scarcity-weighted irrigation pressure each crop generates, and where. To trace that pressure through international supply chains to the goods and regions that ultimately drive it, it first has to be expressed in the structure of the global economy. This stage is that bridge.

It takes the per-country, per-crop CWS predictions from the HARM model (Stage 05), maps the study's 27 crop categories onto the 14 GLORIA crop-growing sectors (the 27→14 concordance of Supplementary Methods S7, with country-specific split weights for the three composite categories oac / pec / pdc), and assembles a per-year **hybrid satellite account** on the GLORIA 164-region × 120-sector layout: the first 14 sectors of each region carry our CWS estimates, and the remaining sectors carry GLORIA's own water-stress accounts. The result is the environmental extension consumed by the GLORIA MRIO decomposition in Stage 07.

The accounts are built per year; the code can cover 1990–2024, while the paper's attribution analysis uses 1995–2024.

```
05_harm_model (Stage 05)
    |  per-crop CWS predictions
    v
HARM per-crop CSVs ({crop}.csv: time, lat, lon, y_pred)
    |
    v  ┌──────────────────────────────────────────────┐
       │ agg-to-country                               │
       │   -> per-crop country-month CSVs             │
       │ merge-hist-pred                              │
       │   -> crops_27_1990_2024.csv                  │
       │ weights-computation   (oac / pdc / pec only) │
       │   -> split_weights_rolling.csv + _recent.csv │
       │ mapping-27-to-14                             │
       │   -> water_stress_164x14_long.csv            │
       │ satellite-preparation                        │
       │   -> data/satellite/satellite_{year}.npy     │
       │ gloria-unzip-satellite                       │
       │   -> data/gloria_satellite/{year}/TQ_YQ.csv  │
       │ build-hybrid-satellite                       │
       │   -> data/satellite_hybrid/{year}.npy  ★     │
       └──────────────────────────────────────────────┘
    |
    v
07_gloria_mrio (Stage 07)
```

## Project structure

```
06_gloria_mapping/
├── README.md
├── requirements.txt
├── main.py                      # CLI
├── config/
│   ├── __init__.py
│   └── config.yaml
├── src/
│   ├── __init__.py
│   ├── agg_to_country.py           # grid-level predictions -> country
│   ├── merge_hist_pred.py          # hist + pred -> crops_27_1990_2024.csv
│   ├── sector_mapping.py           # 175 FAO crops -> 27 CWS -> 14 sectors
│   ├── weights_computation.py      # split weights for oac/pdc/pec
│   ├── mapping_27_to_14.py         # produces water_stress_164x14_long.csv
│   ├── satellite_preparation.py    # long CSV -> per-year .npy (14-sector block)
│   ├── gloria_unzip_satellite.py   # unzip satellite zips -> TQ/YQ CSVs
│   ├── gloria_unzip_mrio.py        # unzip MRIO zips -> SUT CSVs
│   ├── gloria_sut_to_mrio.py       # SUT CSVs -> MRIO zarr (optional)
│   └── build_hybrid_satellite.py   # merge our 14 + GLORIA 15-120 -> final .npy
├── reference/                   # shipped reference tables
│   ├── GLORIA_ReadMe_060.xlsx
│   ├── GLORIA_ReleaseNotes_060.pdf
│   ├── aggregate_region_mapping.json
│   ├── country_mapping_175_to_14.csv
│   ├── country_mapping_27_to_14.csv
│   ├── crop_mapping_175_to_27_to_14.csv
│   └── crops_175.csv
├── data/
│   ├── predictions/             # optional local {crop}.csv (default input is the Stage 05 export)
│   ├── predictions_country/     # output of agg-to-country
│   ├── crops_27.csv             # shipped 1990-2018 historical ground truth
│   ├── crops_27_1990_2024.csv   # shipped example of merge-hist-pred output
│   ├── weights/                 # output of weights-computation
│   ├── water_stress_164x14_long.csv   # output of mapping-27-to-14
│   ├── satellite/               # output of satellite-preparation (14-sector block)
│   ├── gloria/                  # USER INPUT: GLORIA release zips
│   │   ├── mrio_zips/           # GLORIA_MRIOs_60_{year}.zip
│   │   └── sat_zips/            # GLORIA_SatelliteAccounts_060_{year}.zip
│   ├── gloria_mrio/             # output of gloria-sut-to-mrio (zarr)
│   ├── gloria_satellite/        # output of gloria-unzip-satellite (TQ/YQ CSVs)
│   └── satellite_hybrid/        # FINAL: per-year hybrid satellite .npy
└── logs/
```

## Installation

```bash
pip install -r requirements.txt
```

## Inputs the user must supply

| Path | Contents | Where to get |
|------|----------|--------------|
| `../05_harm_model/results/final/predictions_csv/{crop}.csv` | HARM predictions, columns: `time, lat, lon, year, y_pred` | Produced by Stage 05 `python main.py export-predictions` (override `harm_predictions.input_dir` to use your own) |
| `country_shape/ne_10m_admin_0_countries.shp` | NaturalEarth country boundaries (with sidecar files) | included in repo (`country_shape/`); source https://www.naturalearthdata.com/ |
| `data/gloria/sat_zips/GLORIA_SatelliteAccounts_060_{year}.zip` | GLORIA v060 satellite accounts (1990-2025) | https://ielab.info/resources/gloria (Part III) |
| `data/gloria/mrio_zips/GLORIA_MRIOs_60_{year}.zip` | GLORIA v060 MRIO SUT (optional, for `gloria-sut-to-mrio`) | Same portal (Part I) |

## Usage

The core CWS → hybrid satellite chain:

```bash
# 1. HARM per-crop predictions -> country-level totals
python main.py agg-to-country

# 2. combine historical + predictions into crops_27_1990_2024.csv
python main.py merge-hist-pred

# 3. compute split weights for oac / pdc / pec
python main.py weights-computation

# 4. map 27 crops -> 14 GLORIA sectors
python main.py mapping-27-to-14

# 5. build our 14-sector satellite blocks
python main.py satellite-preparation

# 6. unzip GLORIA satellite accounts (once per GLORIA release)
python main.py gloria-unzip-satellite --start-year 1990 --end-year 2025

# 7. merge our 14 sectors + GLORIA 15-120 -> final hybrid .npy
python main.py build-hybrid-satellite --start-year 1990 --end-year 2024

# Or: one shot
python main.py full-pipeline
```

Optional — convert GLORIA SUT CSVs to MRIO zarr for the downstream
MRIO pipeline:

```bash
python main.py gloria-unzip-mrio --start-year 2010 --end-year 2024
python main.py gloria-sut-to-mrio --start-year 2010 --end-year 2024
```

## Final output

`data/satellite_hybrid/satellite_{year}.npy` — shape `(1, 19680)`,
dtype float64, all values in **million m³ world-eq**. First 14 columns of
each 120-sector region block use the HARM model estimates; the
remaining 106 columns use GLORIA's satellite accounts (agricultural
water stress only by default; set `satellite.include_non_ag: true` to
add non-agricultural water stress).

## Key config knobs

| Key | Default | Meaning |
|-----|---------|---------|
| `satellite.ag_row_index` | 390 | Row index in GLORIA TQ CSV for agricultural water stress |
| `satellite.non_ag_row_index` | 391 | Row index for non-agricultural water stress |
| `satellite.include_non_ag` | `false` | Sectors 15-120 = ag + non-ag if true, else ag only (paper uses ag only) |
| `satellite.unit_scale` | `1e6` | Our data (m³) / scale → million m³ |
| `weights.rolling_window` | 5 | Centered window for historical split weights |
| `weights.recent_period` | `[2015, 2018]` | Base for forecast-year split weights |

## Notes

- The shipped `reference/crop_mapping_175_to_27_to_14.csv` is the output
  of `sector-mapping`. Rerunning `sector-mapping` is only needed if you
  modify the master FAO-crop → CWS → GLORIA taxonomy in
  `src/sector_mapping.py`.
- The shipped `reference/crops_175.csv` (12 MB) is the FAOSTAT 175-crop
  historical production dataset used by `weights-computation` to derive
  country-specific split weights for `oac / pdc / pec`.
- `data/crops_27.csv` contains the 1990-2018 country-aggregated
  ground-truth water-stress values used as the historical leg of
  `merge-hist-pred`. It ships with the folder so a user without the
  historical HARM runs can still reproduce `crops_27_1990_2024.csv`.
