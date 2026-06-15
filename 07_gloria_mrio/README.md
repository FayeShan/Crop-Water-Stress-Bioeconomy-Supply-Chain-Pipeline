# 07_gloria_mrio — MRIO Supply-Chain Attribution

> **Stage 07 of 7** in the HARM crop water-stress pipeline (see the [top-level README](../README.md)). The final stage: a double-counting-free multi-regional input–output (MRIO) decomposition that attributes crop water stress along global supply chains to the goods and regions that ultimately drive it.

## Role in the pipeline

This is the last stage. It takes the per-year **hybrid satellite account** from Stage 06 (`06_gloria_mapping/`) — crop water stress (CWS) expressed on GLORIA's 164-region × 120-sector layout — together with the GLORIA MRIO system, and decomposes the production-side water stress through international supply chains. For each year it constructs a six-dimensional impact array linking the producing region where the stress occurs, the producing sector responsible for it, the target sector through which the stress is uniquely attributed, the final-supply sector through which flows reach final demand, the consuming (final-demand) region, and a direct/indirect split. This is what lets the scarcity-weighted irrigation pressure generated on farms be read off by *who* ultimately consumes it and *through which produced goods*.

The decomposition is **double-counting-free**: each unit of water stress is attributed to exactly one target sector along its supply chain, so summing across targets reproduces the global total without overlap (Supplementary Methods S8). The method extends the supply-chain impact-mapping (SCIM) framework of Cabernard, Pfister & Hellweg (2019); unlike the earlier MATLAB implementation — which introduced sectoral/regional aggregation during attribution and could take on the order of a month per indicator — this reimplementation preserves the full region-to-region and sector-to-sector detail of the GLORIA system throughout the computation.

The single indicator carried here is crop water stress. The target sectors are the paper's bioeconomy goods (Crops, FoodProcessing, AnimalRaising, Textile, BiomassProducts, BioChemicals, Plastics), defined in `configs/default.json`.

## Project Structure

```
07_gloria_mrio/
├── configs/
│   └── default.json          # Default configuration
├── scripts/
│   └── (analysis scripts)
├── src/
│   └── gloria_mrio/
│       ├── __init__.py
│       ├── config.py         # Configuration management
│       ├── utils.py          # Utility functions
│       ├── index_builder.py  # Index management
│       ├── mrio_loader.py    # MRIO data loader
│       ├── satellite_loader.py # Satellite data loader
│       ├── calculator.py     # Core footprint calculator
│       └── runner.py         # Main orchestrator
├── run.py                    # CLI entry point
└── README.md
```

## Installation

```bash
pip install numpy xarray zarr tqdm pandas openpyxl pyarrow
```

## Quick Start

### 1. Prepare Configuration

`configs/default.json` ships with paths relative to this folder that
resolve to Stage 06 outputs under `06_gloria_mapping/`:

```json
{
    "gloria_mrio_path": "../06_gloria_mapping/data/gloria_mrio/",
    "satellite_path":   "../06_gloria_mapping/data/satellite_hybrid/",
    "output_path":      "./output/",
    "years": [1990, 1991, ...],
    "target_sectors": {
        "Crops":            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        "FoodProcessing":   [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55],
        "AnimalRaising":    [15, 16, 17, 18, 19, 21],
        "Textile":          [56, 57],
        "BiomassProducts":  [51],
        "BioChemicals":     [63, 67, 68],
        "Plastics":         [71]
    },
    "region_grouping": "individual",
    "indicators": [0]
}
```

If your project layout differs, point `gloria_mrio_path` and
`satellite_path` at wherever the Stage 06 outputs live. `indicators`
is `[0]` because the hybrid satellite carries a single indicator
(crop water stress).

### 2. Run Analysis

```bash
# Run with config file
python run.py --config configs/default.json

# Override years
python run.py --config configs/default.json --years 1990 1991 1992

# Override targets (names must match keys in target_sectors)
python run.py --config configs/default.json --targets Crops Textile
```

### 3. Python API

```python
from gloria_mrio import GloriaMRIO, Config

config = Config.from_json("configs/default.json")
analyzer = GloriaMRIO(config)
results = analyzer.run()
```

## Input Data

### MRIO Data (Required)

Directory containing GLORIA MRIO zarr files (from Stage 06):
```
gloria_mrio/
├── 1990.zarr
├── 1991.zarr
└── ...
```

Each zarr should contain:
- `T`: Transaction matrix (164, 120, 164, 120)
- `Y`: Final demand (164, 120, 164)

### Satellite Data (Required)

Directory containing the hybrid satellite account files (from Stage 06):
```
satellite_hybrid/
├── satellite_1990.npy    # or .csv, .xlsx, .parquet, .zarr
├── satellite_1991.npy
├── Q_Y_1990.csv          # Optional: household direct emissions
└── ...
```

**Q format**: `(n_indicators, n_regions * n_sectors)` or `(n_indicators, n_regions, n_sectors)`

**Q_Y format**: `(n_indicators, n_regions)`

## Output Structure

```
output/
└── Crops_1990_ind0/
    ├── target.zarr      # (Preg, Psec, n_target, FSsec, FDreg, 2)
    │                    # Slot: Direct (0) + Indirect (1)
    ├── remaining.zarr   # (Preg, Psec, FSsec, FDreg)
    ├── household.zarr   # (Preg, FDreg)
    └── metadata.json    # Configuration and metadata
```

One such directory is produced per (target sector × year × indicator).

### Output Dimensions

| Dimension | Size | Meaning |
|-----------|------|---------|
| Preg | 164 | Production Region |
| Psec | 121 | Production Sector (120 + Household) |
| Tsec | n_target | Target Sector (only selected targets) |
| FSsec | 121 | Final Supply Sector |
| FDreg | n_groups | Final Demand Region Group |
| Slot | 2 | 0=Direct, 1=Indirect |

## GLORIA Sector Reference

The target sectors used in this study (0-based indices into the
GLORIA 120-sector classification) correspond to the bioeconomy goods
in `configs/default.json`:

| Index | Sector group |
|-------|--------------|
| 0–14 | Crop-growing sectors |
| 15–21 | Animal raising / livestock |
| 40–55 | Food & beverage processing |
| 56–57 | Textiles |
| 63, 67–68 | Bio-based chemicals |
| 71 | Plastics / rubber |

See `GLORIA_ReadMe_060.xlsx` (shipped in `06_gloria_mapping/reference/`)
for the full 120-sector list.

## Method Reference

The attribution method (target-sector decomposition, double-counting-free)
is documented in Supplementary Methods S8 and extends:

> Cabernard, L., Pfister, S., & Hellweg, S. (2019).
> A new method for analyzing sustainability performance of global supply chains and its application to material resources.
> *Science of The Total Environment*, 684, 164-177.

## License

MIT
