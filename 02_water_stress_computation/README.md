# Water Stress Computation

> **Stage 02 of 7** in the HARM crop water-stress pipeline (see the [top-level README](../README.md)). This stage builds the physically grounded reference dataset: the scarcity-weighted crop water-stress target that the HARM model is trained to reproduce.

## Role in the pipeline

Crop water stress (CWS) is the quantity the whole study is built around. It is the blue-water footprint of irrigated crop production, weighted by how scarce water is in the basin and month where that consumption occurs — so that the same cubic metre counts for more in a stressed basin than where water is abundant. This scarcity weighting is what turns a volume of irrigation water into an environmental burden, and it is why the target is *crop water stress* rather than simply water use.

This stage computes CWS directly from physical inputs, with no economic proxies, producing the monthly, per-crop, 5-arcminute reference dataset that serves as the **ground-truth target** for the HARM model (Stages 04–05) and the historical baseline for the supply-chain attribution (Stages 06–07). The paper uses the 1995–2018 portion of this record as its physically grounded reference period.

For each crop, grid cell and month, CWS is the product of the blue-water footprint and the matching AWARE characterization factor — the paper's `CWS = BWF × AWARE`:

```
waterstress(t, lat, lon) = wfp_blue_ir(t, lat, lon) * aware_weight(month, lat, lon)
```

The output variable is named `waterstress` in the NetCDF files and corresponds to CWS throughout the paper, in m³ world-equivalent. Country-level aggregates used downstream (e.g. `crops_27_1990_2024.csv`) are derived from these per-grid files in Stage 06.

## Method

- **Water footprint**: ACEA (AquaCrop-Earth@lternatives) monthly blue-water footprint for irrigated agriculture, at 5 arcmin (~8.3 km) global resolution. ACEA is a process-based global gridded crop model that simulates daily crop growth and the vertical soil-water balance, separating green water, blue water from capillary rise, and blue water from irrigation; the irrigation component is used here. Variable: `wfp_blue_ir`.
- **Characterization factor**: AWARE
  ([Boulay et al., 2018](https://doi.org/10.1007/s11367-017-1333-8))
  monthly weights, nearest-neighbour matched onto the 5-arcmin grid.
  The pre-gridded weights are shipped in `aware/aware_gridcell.nc`.

## Project Structure

```
02_water_stress_computation/
├── README.md
├── main.py                       # CLI entry point
├── config/
│   ├── __init__.py              # Dataclass-based config loader
│   └── config.yaml              # Paths, filename regex, year range
├── src/
│   ├── __init__.py
│   ├── compute_water_stress.py  # Core: WF * AWARE -> NC
│   └── build_aware_grid.py      # One-time: Excel -> aware_gridcell.nc (reference only)
├── aware/                        # Shipped AWARE data
│   ├── AWARE.xlsx
│   ├── Placemarks_Data.csv
│   ├── AWARE_with_geo.csv
│   └── aware_gridcell.nc        # gridded monthly weights (~854 MB — NOT in repo; rebuild, see below)
├── wf/                           # INPUT: monthly WFP NC files (user-supplied)
│   └── README.md
├── waterstress/                  # OUTPUT: {crop}/{year}.nc
│   └── README.md
└── logs/                         # Log files from each run
```

## Installation

```bash
pip install xarray numpy pandas pyyaml tqdm h5netcdf scipy
```

`scipy` is only needed if you plan to regenerate `aware_gridcell.nc`
(see `src/build_aware_grid.py`).

## Usage

### 1. Provide input data

Place your ACEA water-footprint NC files in `./wf/`. See
[`wf/README.md`](wf/README.md) for the expected filename pattern.

### 2. Run the computation

```bash
# All crops, all years found in wf/
python main.py compute-ws

# Subset of crops
python main.py compute-ws --crops wh ri1 mai

# Restrict year range
python main.py compute-ws --start-year 2000 --end-year 2020
```

### 3. Check the output

```
waterstress/{crop}/{year}.nc   (variable: waterstress, dims: time=12, lat, lon)
```

## Configuration

See `config/config.yaml`. Defaults usually work; override at the command
line with `-c path/to/another_config.yaml` if needed.

Key knobs:

| Key | Default | Meaning |
|-----|---------|---------|
| `wf.input_dir` | `./wf` | Where to find monthly WFP NC files |
| `wf.filename_regex` | `acea_5arc_(\w+)_wfp_blue_ir` | Captures crop code; supports 2-letter (e.g. `wh`) and codes with digits (`ri1`, `ri2`) |
| `aware.weights_nc` | `./aware/aware_gridcell.nc` | AWARE gridded weights (not in repo — rebuild, see below) |
| `output.waterstress_dir` | `./waterstress` | Where to save results |
| `output.compression_level` | 5 | zlib level for output NC |
| `date_range.start_year` / `end_year` | `null` | Optional year filter (null = all) |

## Notes on the fix

The previous standalone script split outputs across two directories
(`./src/{crop}/` for most crops and `./src/ws_in_crops/{crop}/` for
`wh`, `ri1`, `ri2`) because the filename regex required exactly three
alphabetic characters. The new regex `acea_5arc_(\w+)_wfp_blue_ir`
handles all crop codes uniformly, so all outputs now land in a single
`waterstress/{crop}/` tree.

## `aware_gridcell.nc` (not included in the repository)

The gridded monthly AWARE weights `aware/aware_gridcell.nc` are **not
included** in the repository because of their size (~854 MB). Regenerate
them from the shipped `AWARE.xlsx` + `Placemarks_Data.csv` using
nearest-neighbour assignment onto the 5-arcmin global grid: the
construction code lives (commented) in `src/build_aware_grid.py`
(requires `scipy`). Uncomment and run it to rebuild the weights — e.g.
before the first `compute-ws` run, or if you change the target grid
resolution.
