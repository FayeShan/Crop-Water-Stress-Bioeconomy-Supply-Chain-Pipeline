# Crop Water Stress Bioeconomy Supply Chain Pipeline (1995, 2024)

> Code and analysis for *"Global crop water stress is locally concentrated and redistributed through the bioeconomy."*
>
> The pipeline reconstructs global crop water stress at monthly, 5-arcminute (~9.3 km) resolution for 27 crops, predicts it through 2024 with a three-layer machine-learning model, and traces how its economic responsibility is reorganized through bioeconomy supply chains.

## Overview

This repository contains the code and analysis behind the paper. From the abstract:

> The bioeconomy is key to climate targets, but crop irrigation for biomass production accounts for over 90% of global water stress. Yet, a global mapping of water stress that combines up-to-date, quantitative estimates with crop-resolved, temporally explicit supply chain attribution is missing. We develop a monitoring and attribution framework linking where crop water stress is generated to where it is consumed. We use a machine learning model that combines satellite, climate, and crop-specific data to predict monthly crop water stress for 27 crop categories at 5-arcminute resolution through 2024. The model reaches yearly country-level R² above 0.7 for 22 of the 27 crop categories, which together account for 94% of the global crop water stress. The resulting record shows that crop water stress is highly concentrated: 1% of irrigated cropland drives over a quarter of the 2024 total. By integrating these predictions into state-of-the-art multi-regional input–output analysis, we find that international trade accounts for 22% of global crop water stress and has grown faster than the global total since 1995. Our framework allows targeting interventions in water-scarce regions and tracing crop water stress along bioeconomy supply chains, and can be regularly updated as new observations become available.

The seven-stage pipeline below reproduces this analysis end to end, from the raw environmental inputs to the supply-chain attribution.

### Names in the code vs the paper

The code keeps two internal names that the paper words differently; they map one-to-one:

| Code / repository                                                                                                         | Paper                                                         |
| ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| **HARM** (Hierarchical Additive Residual Model) — folder `05_harm_model/`, `f_global + f_regional + f_extreme` | *a machine learning model*                                  |
| the model's**regional** layer — `f_regional`, the `regiontype` label, "agro-climatic regions"                  | the**zonal** component — agro-climatic **zones** |

The code names are kept for stability; only the manuscript wording changed. Elsewhere in this README, "HARM" and "regional layer" refer to these code names.

## Pipeline overview

The code is organised as seven sequential core stages that follow the logic of the paper, plus a packaging stage (08). Stages 1–4 build the physical foundation: they download the environmental and remote-sensing inputs, compute the scarcity-weighted CWS target, reconstruct irrigated harvested area, and assemble per-crop feature matrices. Stage 5 is the modelling core, where HARM learns crop-specific relationships and predicts monthly CWS through 2024. Stages 6–7 make the supply-chain connection: they map the 27-crop predictions onto the GLORIA crop-growing sectors as an extended environmental satellite account, then perform the double-counting-free MRIO analysis that attributes water stress across producing regions, traded flows, consuming regions and final goods. Stage 8 branches from Stage 5 to compile the predictions into the published NetCDF, GeoTIFF and country-CSV dataset (the Zenodo release and the Earth Engine dashboard), independently of the MRIO analysis.

| #  | Module                           | Role                                                                                                                                                                                                                                       | Main CLI                                        |
| -- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------- |
| 01 | `01_data_acquisition/`         | Download the raw environmental and remote-sensing inputs (ERA5-Land, MODIS, SMIA, TWSA and related ancillary data) that drive both the water-stress target and the HARM predictors.                                                        | `python main.py …`                           |
| 02 | `02_water_stress_computation/` | Compute the crop water-stress (CWS) target — basin-level AWARE scarcity factors × crop-specific blue-water footprints — the physically grounded quantity the rest of the pipeline reconstructs and attributes.                          | `python main.py …`                           |
| 03 | `03_iha_computation/`          | Reconstruct irrigated harvested-area (iHA) fields from LUH2 + CROPGRIDS + FAOSTAT (Method B); aggregate oac/pec/pdc and downscale 24 ACEA crops to monthly.                                                                                | `python main.py full-pipeline`                |
| 04 | `04_dataset_preparation/`      | Assemble per-crop parquet feature matrices (74 predictors +`waterstress` target) from the outputs of Stages 1–3 — the training and prediction tables for HARM.                                                                         | `python main.py …`                           |
| 05 | `05_harm_model/`               | Train and apply the HARM three-layer pipeline (Global → Regional → Extreme, with optional stress-aware weighting) in two phases:`eval` (2000–2016 train / 2017–2018 evaluate) and `final` (2000–2018 train / 2019–2024 predict). | `python main.py train/retrain/predict …`     |
| 06 | `06_gloria_mapping/`           | Map the 27 HARM crop predictions to 14 GLORIA crop-growing sectors and construct the extended satellite account (our 14 sectors + GLORIA agriculture 15–20) used as GLORIA's environmental extension.                                     | `python main.py full-pipeline`                |
| 07 | `07_gloria_mrio/`              | Perform the double-counting-free MRIO supply-chain analysis, using the extended satellite as the environmental extension of GLORIA v060 to attribute water stress to traded flows, consuming regions and produced goods.                   | `python run.py --config configs/default.json` |
| 08 | `08_final_dataset/`            | Package the HARM predictions into the published gridded dataset (NetCDF, GeoTIFF, country CSV) for the Zenodo release and the Earth Engine dashboard. Branches from Stage 5; independent of the MRIO analysis.                             | `python main.py full-pipeline`                |

Each stage has its own `README.md` describing its inputs, outputs, configuration and individual CLI. This top-level README describes only the cross-stage interface and the scientific context.

## Key concepts

- **Crop water stress (CWS)** — the blue-water footprint of irrigated crop production multiplied by the basin-level AWARE characterization factor; a scarcity-weighted measure of irrigation pressure, expressed in m³ world-equivalent.
- **AWARE** — Available WAter REmaining; the life-cycle-assessment factor that converts water consumption into a scarcity-weighted impact based on local, monthly water availability.
- **HARM** — Hierarchical Additive Residual Model (the *machine learning model* of the paper); predicts monthly CWS as an additive sum of a Global layer, a Regional/zonal residual correction and an Extreme-tail correction: `y = f_global(X) + f_regional(X) + f_extreme(X)`.
- **GLORIA** — the Global Resource Input–Output Assessment MRIO database (release 060), used to trace production-side water stress through international supply chains to final demand.
- **Extended satellite account** — our crop water-stress vector mapped onto GLORIA's crop-growing sectors and appended as an environmental extension, enabling consumption-based and trade attribution without double counting.

## Installation

Install everything at once (recommended):

```bash
pip install -r requirements.txt
```

Or install per-stage (useful if you only need a subset):

```bash
pip install -r 01_data_acquisition/requirements.txt
pip install -r 03_iha_computation/requirements.txt
# ... etc.
```

The C++ kernel for Stage 7 is optional. See `07_gloria_mrio/README.md` for the build instructions; the numpy fallback is already fast on modern workstations.

## Paper

The manuscript, Supplementary Methods and Supplementary Tables are **not** included in this repository. The paper is available from the journal and archived on Zenodo (see its *Data and code availability*). The scripts that regenerate the supplementary tables are kept under `supplementary_tables/code/`.

## Data and outputs

Raw inputs, intermediate artefacts, and pipeline outputs are **not** stored in this repository — see the root-level `.gitignore` for the ignored directories. Each stage's `README.md` documents how to download or regenerate its data:

| Stage | What the user needs to provide / download                                                                                                |
| ----- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| 01    | CDS API credentials (ERA5), NASA EarthData login (MODIS); Stage 01 handles the rest.                                                     |
| 03    | LUH2 state/management NCs, CROPGRIDS v1.08 per-crop NCs, FAOSTAT QCL (the Stage 03`download-data` sub-command automates these).        |
| 06    | GLORIA v060 satellite-account zips from[https://ielab.info/resources/gloria](https://ielab.info/resources/gloria) (30 files, 1995–2024). |
| 07    | Only needs Stage 6 outputs; nothing external.                                                                                            |

Once all inputs are in place, the full end-to-end pipeline can be reproduced by running each stage's main CLI in order (Stages 01 → 07).

### Large data not included in this repository

Because of their size, the raw inputs, intermediate artefacts and pipeline outputs listed below are **not uploaded to GitHub** (they are excluded via the root `.gitignore`). Each item can be regenerated by running the corresponding stage from the public sources noted; the headline datasets — the gridded crop water-stress product and the MRIO attribution results — are archived on Zenodo (see *Data availability* in the paper). The full set of intermediate files is also **available from the corresponding author on request**.

| Stage | Excluded path                                                               | Contents                                                                                                                 | Approx. size  | How to obtain                                                                                |
| ----- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ------------- | -------------------------------------------------------------------------------------------- |
| 01    | `01_data_acquisition/data/`                                               | Raw ERA5-Land / MODIS / SMIA / TWSA downloads, merged per-year predictor NCs, and the`reference/2002.nc` grid template | ~0.7 GB+      | Re-download via Stage 01 CLI (needs CDS + NASA EarthData credentials); or on request         |
| 02    | `02_water_stress_computation/aware/aware_gridcell.nc`                     | Pre-gridded monthly AWARE characterization weights                                                                       | 854 MB        | Rebuild with`src/build_aware_grid.py` from the shipped `AWARE.xlsx`; or on request       |
| 02    | `02_water_stress_computation/wf/`, `waterstress/`                       | ACEA blue-water-footprint inputs; computed CWS reference NCs                                                             | large         | ACEA inputs from 4TU.ResearchData (Mialyk et al. 2024); outputs regenerated by`compute-ws` |
| 03    | `03_iha_computation/data/luh2/`, `data/cropgrids/`                      | LUH2 (8.3 GB) and CROPGRIDS v1.08 (0.9 GB) raw inputs                                                                    | ~9 GB         | `python main.py download-data` (Zenodo / Figshare)                                         |
| 03    | `03_iha_computation/output/`, `annual/`, `monthly/`                   | Regenerated annual + monthly irrigated-harvested-area NCs                                                                | ~46 GB        | Regenerated by`python main.py full-pipeline`                                               |
| 03    | `03_iha_computation/spam2020_reference/`                                  | SPAM2020 V2r0 validation rasters                                                                                         | 4.8 GB        | IFPRI Harvard Dataverse`10.7910/DVN/SWPENT`                                                |
| 04    | `04_dataset_preparation/data/`                                            | Assembled per-crop train / predict feature parquets                                                                      | large         | Regenerated by Stage 04`full-pipeline`                                                     |
| 05    | `05_harm_model/data/`, `results/`                                       | HARM inputs (Köppen variants, masks, shapefile) and trained models + predictions                                        | large         | Regenerated by Stage 05`train` / `retrain` / `predict`                                 |
| 06    | `06_gloria_mapping/data/gloria/`, `gloria_satellite/`, `gloria_mrio/` | GLORIA v060 release zips and unpacked satellite / MRIO tables                                                            | ~15 GB+       | GLORIA v060 from[ielab.info/resources/gloria](https://ielab.info/resources/gloria)            |
| 06    | `06_gloria_mapping/data/satellite/`, `satellite_hybrid/`                | Regenerated 14-sector and hybrid satellite blocks                                                                        | small–medium | Regenerated by Stage 06                                                                      |
| 07    | `07_gloria_mrio/output/`                                                  | MRIO attribution result arrays                                                                                           | large         | Regenerated by`python run.py`; headline results on Zenodo                                  |

Small reference and concordance tables (AWARE source tables, FAOSTAT scaling factors, crop calendars, crop/region concordances, GLORIA documentation, the country shapefile, and example aggregated CSVs) **are** included, so the mapping and attribution steps can be inspected and rerun without the bulk data.

## Repository layout

```
.
├── README.md                         # this file
├── requirements.txt                  # unified Python requirements
├── .gitignore                        # excludes all large data dirs
│
├── 01_data_acquisition/
├── 02_water_stress_computation/
├── 03_iha_computation/
├── 04_dataset_preparation/
├── 05_harm_model/
├── 06_gloria_mapping/
├── 07_gloria_mrio/
├── 08_final_dataset/
│
└── supplementary_tables/
    └── code/                         # scripts that generate the SI tables
```

Reviewers who want to rerun only one stage can do so by reading that stage's `README.md`; the repository is designed so that each stage's inputs are explicitly wired in its own `config/config.yaml` (or `configs/default.json` for Stage 7) and can be overridden from the command line without touching the others.

## Citation

If you use this code or the accompanying data, please cite the paper (citation information will be added after publication).
