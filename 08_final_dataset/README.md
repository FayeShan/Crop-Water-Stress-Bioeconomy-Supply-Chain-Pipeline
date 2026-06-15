# 08_final_dataset — Published dataset packaging

> **Stage 08 of the HARM crop water-stress pipeline** (see the [top-level README](../README.md)). This stage packages the HARM predictions into the published gridded dataset — NetCDF, GeoTIFF and country-level CSV — for the Zenodo data release and the Google Earth Engine interactive dashboard.

## Role in the pipeline

This stage is a **data-product branch off Stage 05**, parallel to the GLORIA MRIO branch (Stages 06–07) and independent of it. It takes the per-crop crop water-stress (CWS) predictions exported by Stage 05 and reshapes them into the publication formats:

- **NetCDF** — one multi-crop file per year (27 crops as variables), monthly and yearly-aggregated, on the 5-arcmin grid;
- **GeoTIFF** — monthly and yearly rasters with 27 bands (one per crop), LZW-compressed, EPSG:4326;
- **country CSV** — global yearly totals and per-crop per-country yearly totals (via a Natural Earth point-in-polygon join).

```
05_harm_model (Stage 05)
    |  export-predictions -> results/final/predictions_csv/{crop}.csv
    v
08_final_dataset (this folder)
    |  build-netcdf -> build-geotiff -> build-country-csv
    v
final_dataset/{netcdf,geotiff,csv}/   ->  Zenodo release + GEE dashboard
```

The published dataset spans 1995–2024. This stage builds the **prediction years (2019–2024)** from the HARM output; the historical years (1995–2018) come from the physically grounded reference (Stage 02) and are placed alongside in the same `final_dataset/netcdf/monthly/` layout. The `build-country-csv` and yearly-aggregate steps then operate over whatever years are present.

## Project structure

```
08_final_dataset/
├── README.md
├── requirements.txt
├── main.py                    # CLI
├── config/
│   ├── __init__.py
│   └── config.yaml
├── src/
│   ├── __init__.py
│   ├── constants.py           # 27 crop codes, names, GeoTIFF band order + transform
│   ├── build_netcdf.py        # {crop}.csv -> monthly + yearly NetCDF
│   ├── build_geotiff.py       # monthly NetCDF -> GeoTIFF (27 bands)
│   ├── build_country_csv.py   # NetCDF -> global + per-country CSV
│   └── recompress_netcdf.py   # zlib recompression (optional)
├── final_dataset/             # OUTPUT (large — NOT in repo)
│   ├── netcdf/{monthly,yearly}/
│   ├── geotiff/{monthly,yearly}/
│   └── csv/
└── logs/
```

## Installation

```bash
pip install -r requirements.txt
```

## Inputs

| Path | Contents | Source |
|------|----------|--------|
| `../05_harm_model/results/final/predictions_csv/{crop}.csv` | HARM predictions (`time, lat, lon, year, y_pred`) | Stage 05 `export-predictions` |
| `../01_data_acquisition/data/reference/2002.nc` | 5-arcmin reference grid (lat/lon only) | Stage 01 |
| `../06_gloria_mapping/country_shape/ne_10m_admin_0_countries.shp` | Country boundaries | shipped in Stage 06 |

## Usage

```bash
# 1. Predictions -> monthly + yearly NetCDF
python main.py build-netcdf

# 2. NetCDF -> GeoTIFF (monthly + yearly, 27 bands)
python main.py build-geotiff

# 3. NetCDF -> country-level CSVs
python main.py build-country-csv

# Optional: recompress NetCDF in place (zlib)
python main.py recompress

# Or run steps 1-3 in order (add --recompress to also recompress)
python main.py full-pipeline
```

Override the year range with `--start-year` / `--end-year`.

## Output

`final_dataset/` (not stored in the repository — see the root `.gitignore`):

- `netcdf/monthly/water_stress_{year}.nc` — 27 crop variables, `(time=12, lat, lon)`
- `netcdf/yearly/yearly_by_crop_{y0}_{y1}.nc`, `yearly_total_{y0}_{y1}.nc`
- `geotiff/monthly/water_stress_{year}_{mm}.tif`, `geotiff/yearly/water_stress_{year}.tif` — 27 bands
- `csv/global_yearly_totals.csv`, `csv/all_crops_country_yearly.csv`

## Key config knobs

| Key | Default | Meaning |
|-----|---------|---------|
| `input.predictions_dir` | `../05_harm_model/results/final/predictions_csv` | Stage 05 exported predictions |
| `input.reference_grid_nc` | `../01_data_acquisition/data/reference/2002.nc` | grid template (lat/lon) |
| `input.shapefile` | `../06_gloria_mapping/country_shape/...shp` | country boundaries |
| `years.start` / `years.end` | 2019 / 2024 | prediction years to build |
| `netcdf.compression_level` | 4 | zlib level for `recompress` |
