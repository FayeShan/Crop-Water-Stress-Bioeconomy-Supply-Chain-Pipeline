# WaterStress Data Pipeline

> **Stage 01 of 7** in the HARM crop water-stress pipeline (see the [top-level README](../README.md)). This stage downloads, harmonizes and merges the environmental and remote-sensing data that become HARM's predictors — the variables that let the model learn where and when irrigated crop water stress occurs.

## Role in the pipeline

Crop water stress (CWS) depends jointly on **local water scarcity** and on **where and when irrigated crop production takes place**. This stage assembles the fields that describe the physical conditions behind that picture: climate and land-surface state from ERA5-Land, vegetation condition from MODIS, soil-moisture anomalies from SMIA, and terrestrial water-storage anomalies from TWSA (GRACE/GRACE-FO). Each source is downloaded, converted to NetCDF, corrected where needed, and resampled to a common **monthly, 5-arcminute (~9.3 km) global grid** — the same grid as the crop water-stress reference dataset — so that every input aligns in space and time before modelling.

These harmonized fields are the raw material for HARM's predictors. Together with the lagged and rolling features built from them, the irrigated harvested area added in Stage 3, and the crop water-stress target computed in Stage 2, they form the **74-variable predictor set** that the three-layer model uses to predict monthly CWS for 27 crops over 2019–2024.

## The Challenge

Environmental and climate research often requires integrating data from multiple sources — such as meteorological reanalysis (ERA5), satellite vegetation indices (MODIS), soil-moisture indices (SMIA) and terrestrial water-storage anomalies (TWSA). These datasets typically arrive in:

- Different formats: NetCDF, HDF4, GeoTIFF, etc.
- Different spatial resolutions: ranging from 0.05° to 0.25°
- Different temporal coverages: some starting from 1990, others from 2000 or 2002
- Different coordinate systems and grid structures

Harmonizing these heterogeneous datasets by hand is time-consuming and error-prone, and stands between the researcher and the actual analysis.

## Our Solution

This pipeline automates the entire workflow — downloading, format conversion, the ERA5 accumulated-flux correction, spatial resampling, and merging — to produce analysis-ready, unified NetCDF files with consistent resolution and structure, one file per year. The same machinery is general-purpose: although it is built here for crop water-stress prediction, it can be reused to harmonize ERA5-, MODIS- or Copernicus-based inputs for other agricultural, hydrological, climate-impact or vegetation-monitoring analyses that need a single, gridded, monthly archive.

## Supported Data Sources

The four base sources below provide the environmental state behind crop water stress: **ERA5-Land** supplies the atmospheric and land-surface drivers (temperature, radiation, precipitation, soil water, evaporation, wind, pressure); **MODIS** contributes vegetation condition (NDVI/EVI and surface reflectance); **SMIA** adds soil-moisture anomalies; and **TWSA** carries the GRACE/GRACE-FO terrestrial water-storage signal. They are downloaded and harmonized here; the lagged and rolling features derived from them, together with irrigated harvested area (Stage 3), make up HARM's predictor set (see the paper's Methods for the full variable list and importances).

SMIA, TWSA, and other Global Drought Observatory indicators can be downloaded manually from: https://drought.emergency.copernicus.eu/tumbo/gdo/download/

### Base sources (enabled by default)

| Source      | Product / Description                      | Native resolution | Temporal range |
| ----------- | ------------------------------------------ | ----------------- | -------------- |
| **ERA5-Land** | CDS `reanalysis-era5-land-monthly-means`  | 0.1° × 0.1°       | 1950-present   |
| **MODIS**   | MOD13C2 v061 vegetation indices            | 0.05° × 0.05°     | 2000-present   |
| **SMIA**    | Copernicus EDO Soil Moisture Index Anomaly | 0.1° × 0.1°       | varies         |
| **TWSA**    | Copernicus EDO Terrestrial Water Storage Anomaly (GRACE/GRACE-FO) | ~0.25°    | 2002-present   |

> **Resolution note:** SMIA is delivered at a 0.1° native resolution and processed at 0.05° before being resampled — like every source — onto the common 5-arcmin grid.

### Additional MODIS products (opt-in)

| Source              | Product                                  | Native resolution       | Temporal | Notes |
| ------------------- | ---------------------------------------- | ----------------------- | -------- | ----- |
| **MODIS Green**     | MOD09CMG v061 Band 4 (Green reflectance) | 0.05° × 0.05° CMG       | monthly mean from daily | Aggregated locally from daily HDFs |
| **MODIS Landcover** | MCD12Q2 v061 Land Cover Dynamics (phenology) | 500 m SIN → 0.05° reproj | yearly | 25 variables (12 phenology × 2 modes + NumCycles) |
| **MODIS Landuse**   | MCD12C1 v061 Land Cover Type (IGBP)      | 0.05° × 0.05° CMG       | yearly | `LC_Type1` (IGBP majority class) |

Yearly products (landcover/landuse) are **broadcast to monthly** during the merge step (the same yearly value repeats for each of the 12 months). Enable them by setting `merge.sources.modis_green/landcover/landuse: true` in `config.yaml`. These products are optional and sit outside the default predictor set used in the paper.

## Project Structure

```
01_data_acquisition/
├── config/
│   ├── __init__.py      # Config loader (dataclasses)
│   └── config.yaml      # Main configuration file
├── src/
│   ├── utils.py         # Utility functions
│   ├── downloaders/
│   │   ├── era5.py              # ERA5 download & unzip
│   │   ├── era5_fix.py          # ERA5 accumulated-variable correction
│   │   ├── modis.py             # MOD13C2 vegetation indices
│   │   ├── modis_green.py       # MOD09CMG Green reflectance (monthly)
│   │   ├── modis_landcover.py   # MCD12Q2 phenology (yearly)
│   │   └── modis_landuse.py     # MCD12C1 IGBP (yearly)
│   └── processors/
│       ├── smia.py      # SMIA TIFF→NC conversion
│       ├── twsa.py      # TWSA processing with month filling
│       ├── resampling.py # Spatial resampling (incl. expver dim handling)
│       └── merger.py    # Data merging
├── main.py              # CLI entry point
├── tutorial.ipynb       # Jupyter notebook tutorial
├── requirements.txt     # Dependencies
└── README.md
```

## Installation

```bash
# Clone or copy the project
cd 01_data_acquisition

# Install dependencies
pip install -r requirements.txt

# For ERA5: Configure CDS API
# See: https://cds.climate.copernicus.eu/api-how-to

# For MODIS: Login to NASA Earthaccess (interactive)
python -c "import earthaccess; earthaccess.login()"
```

## Quick Start

### Command Line Interface

```bash
# View all available commands
python main.py --help

# Download and process ERA5
python main.py era5-download --start-year 2020 --end-year 2024
python main.py era5-unzip    --start-year 2020 --end-year 2024
python main.py era5-fix      --start-year 2020 --end-year 2024  # required for 2022-2024 (see below)
python main.py resample-era5 --start-year 2020 --end-year 2024

# Download and process MODIS
python main.py modis-download --start-year 2020 --end-year 2024
python main.py modis-hdftonc  --start-year 2020 --end-year 2024
python main.py resample-modis --start-year 2020 --end-year 2024

# Process SMIA (assumes TIFF files are in place)
python main.py smia-tifftonc  --start-year 2020 --end-year 2024
python main.py resample-smia  --start-year 2020 --end-year 2024

# Process TWSA (with automatic month filling)
python main.py twsa-resample  --start-year 2020 --end-year 2024

# Merge all data sources
python main.py merge          --start-year 2020 --end-year 2024

# Run complete pipeline (includes era5-fix automatically)
python main.py full-pipeline  --start-year 2020 --end-year 2024 --continue-on-error
```

### Available Commands

| Command            | Description                         |
| ------------------ | ----------------------------------- |
| `era5-download`             | Download ERA5 monthly-means data from CDS |
| `era5-unzip`                | Unzip downloaded ERA5 files         |
| `era5-fix`                  | Apply ECMWF workaround for the Sep 2022 – Feb 2024 accumulated-variable bug (see below) |
| `modis-download`            | Download MOD13C2 vegetation indices via Earthaccess |
| `modis-hdftonc`             | Convert MOD13C2 HDF to NetCDF       |
| `modis-green`               | Download + process MOD09CMG Green reflectance (opt-in) |
| `modis-landcover`           | Download + process MCD12Q2 phenology (opt-in) |
| `modis-landuse`             | Download + process MCD12C1 IGBP (opt-in) |
| `smia-tifftonc`             | Convert SMIA TIFF to NetCDF         |
| `twsa-resample`             | Resample TWSA (with missing-month fill) |
| `resample-era5`             | Resample ERA5 to reference grid     |
| `resample-modis`            | Resample MOD13C2 to reference grid  |
| `resample-modis-green`      | Resample MOD09CMG Green reflectance |
| `resample-modis-landcover`  | Resample MCD12Q2 phenology (nearest-neighbour) |
| `resample-modis-landuse`    | Resample MCD12C1 IGBP (nearest-neighbour) |
| `resample-smia`             | Resample SMIA to reference grid     |
| `merge`                     | Merge all enabled data sources      |
| `full-pipeline`             | Run complete pipeline end-to-end    |

## Important: ERA5 correction for September 2022 – February 2024

ECMWF have documented a bug affecting the CDS product `reanalysis-era5-land-monthly-means` (product type `monthly_averaged_reanalysis`): between **September 2022 and February 2024**, the accumulated flux variables `ssr`, `ssrd`, `tp`, and `ro` are returned at approximately 50 % of their correct values. Reference: https://forum.ecmwf.int/t/issue-affecting-era5-land-monthly-averaged-reanalysis-for-the-period-september-2022-to-february-2024/2370

The pipeline addresses this via the `era5-fix` command, which:

1. Re-downloads the 4 affected variables using the ECMWF-recommended workaround product `monthly_averaged_reanalysis_by_hour_of_day` at 00:00 UTC (which returns correct values).
2. Rewrites each `data/era5/processed/{year}.nc` in place, replacing only the 4 affected variables while preserving all other variables and coordinates.

`full-pipeline` invokes `era5-fix` automatically; for years outside 2022-2024 the command is a safe no-op. The `evabs` and `evavt` variables are not returned by the workaround product and are left unchanged — see paper Methods for discussion.

## Configuration

### config/config.yaml

```yaml
# Year range
date_range:
  train_start: 1990
  train_end: 2018
  predict_start: 2019
  predict_end: 2025

# ERA5 variables to download
era5:
  variables:
    - "2m_temperature"
    - "total_precipitation"
    # ... add more as needed

# MODIS settings
modis:
  var_mapping:
    "CMG 0.05 Deg Monthly NDVI": "NDVI"
    "CMG 0.05 Deg Monthly EVI": "EVI"
  selected_vars: []  # Empty = all variables

# Merge configuration
merge:
  sources:
    era5: true
    modis: true
    smia: true
    twsa: true
  extra_files: []  # Add paths to extra NC files
```

### Customizing Merge Sources

To merge only specific sources, modify `config.yaml`:

```yaml
merge:
  sources:
    era5: true
    modis: true
    smia: false   # Exclude SMIA
    twsa: false   # Exclude TWSA
  extra_files:
    - "./data/extra/elevation.nc"
    - "./data/extra/land_cover.nc"
```

## Data Processing Pipeline

```
1. Download Raw Data
   ├── ERA5: CDS API → ZIP files
   ├── MODIS: Earthaccess → HDF files
   ├── SMIA: Local TIFF files
   └── TWSA: Local NetCDF files

2. Format Conversion
   ├── ERA5: ZIP → NetCDF (unzip)
   ├── MODIS: HDF → NetCDF
   ├── SMIA: TIFF → NetCDF
   └── TWSA: Process & fill missing months

3. ERA5 Correction (Sep 2022 – Feb 2024)
   └── Rewrite ssr/ssrd/tp/ro in place via by_hour_of_day product

4. Spatial Resampling
   └── All sources → Common reference grid

5. Data Merging
   └── Merge by year → Single NetCDF per year
```

## Key Features

### TWSA Month Filling

TWSA data may have missing months (e.g., 2025 data only until November). The pipeline automatically fills missing months using forward-fill:

```
Input:  Jan, Feb, Mar, ..., Sep, Nov (missing Oct, Dec)
Output: Jan, Feb, Mar, ..., Sep, Oct*, Nov, Dec*
        * Oct filled with Sep data
        * Dec filled with Nov data
```

To disable:

```bash
python main.py twsa-resample --start-year 2025 --end-year 2025 --no-fill
```

### Flexible MODIS Variables

Select specific variables to process:

```yaml
modis:
  selected_vars: ["NDVI", "EVI"]  # Only process these
```

Or process all available variables:

```yaml
modis:
  selected_vars: []  # Empty = process all
```

## Python API Usage

```python
from config import load_config, create_directories
from src.downloaders import download_era5, download_modis
from src.processors import resample_era5, merge_data

# Load configuration
config = load_config()
create_directories(config)

# Download ERA5
download_era5(
    output_dir="./data/era5/raw",
    variables=config.era5.variables,
    months=config.era5.months,
    start_year=2020,
    end_year=2024
)

# Resample ERA5
resample_era5(
    input_dir="./data/era5/processed",
    output_dir="./data/era5/resampled",
    reference_grid_path="./data/reference/2002.nc",
    start_year=2020,
    end_year=2024
)

# Merge with custom sources
merge_data(
    era5_dir="./data/era5/resampled",
    modis_dir="./data/modis/resampled",
    smia_dir="./data/smia/resampled",
    twsa_dir="./data/twsa/processed",
    output_dir="./data/merged",
    start_year=2020,
    end_year=2024,
    sources_enabled={'era5': True, 'modis': True, 'smia': False, 'twsa': False}
)
```

## Prerequisites

### ERA5 (CDS API)

1. Register at [CDS](https://cds.climate.copernicus.eu/)
2. Create `~/.cdsapirc` with your credentials

### MODIS (NASA Earthaccess)

1. Register at [NASA Earthdata](https://urs.earthdata.nasa.gov/)
2. Run `earthaccess.login()` interactively

### Reference Grid

Provide a reference NetCDF file with target `lon` and `lat` coordinates for resampling.

## Logs

All operations are logged to `./logs/`:

- `era5_download.log`
- `era5_process.log`
- `modis_download.log`
- `modis_process.log`
- `smia_process.log`
- `twsa_process.log`
- `*_resample.log`
- `data_merge.log`

## Troubleshooting

### "File not found" during merge

Check that the paths in `config.yaml` match where your processed files are saved.

### MODIS download fails

Ensure you've logged in to Earthaccess:

```python
import earthaccess
earthaccess.login()
```

### ERA5 download fails

Check your CDS API configuration in `~/.cdsapirc`.

## License

MIT License
