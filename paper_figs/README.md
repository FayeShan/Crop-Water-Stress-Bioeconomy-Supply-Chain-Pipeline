# paper_figs — figure reproduction code

Scripts that produce the main-text figures (1–5) and the SI figures from
the pipeline outputs. Python scripts (`.py`) cover Figures 1–3 and the SI
figures; R scripts (`.r`) cover Figures 4–5.

## Paths

All input/output paths are centralised:

- `config.py` — for the Python scripts (resolved relative to this folder,
  so they run from any working directory).
- `config.R` — for the R scripts (`source("config.R")`; **run R scripts
  from this `paper_figs/` directory** so the relative paths resolve).

Edit those two files if your outputs live elsewhere. The large inputs are
**not** stored in the repository (see the root `.gitignore`); regenerate
them by running the relevant stage, or fetch the published dataset from
Zenodo.

## Inputs by figure

| Figure | Script(s) | Input | Produced by |
|--------|-----------|-------|-------------|
| 1b, 1c–f | `fig1b.py`, `fig1cdef.py` | `yearly_total_1990_2024.nc` | Stage 08 |
| 2a, 2b | `fig2a.py`, `fig2b.py` | `yearly_by_crop_1990_2024.nc` | Stage 08 |
| 2 (monthly bars) | `fig2month.py` | `{crop}_country.csv` | Stage 06 (`agg-to-country`) |
| 3 (SHAP precompute) | `fig3_shap_computation.py` | eval models + train tables | Stage 05 + Stage 04 |
| 3a, 3b | `fig3a.py`, `fig3b.py` | `{crop}_eval_predictions.parquet`, country shapefile | Stage 05; shapefile shipped in Stage 06 |
| 3f, 3g | `fig3f.py`, `fig3g.py` | `shap_outputs/` | `fig3_shap_computation.py` |
| 4 | `fig4sankeydata.r`, `fig4sankey.r`, `fig4stacked.r` | `aggregated_all_years.csv` | Stage 07 (`dashboard_data_generator.py`) |
| 5 | `fig5.r`, `fig5extra.r`, `fig5stacked.r` | `aggregated_all_years.csv` | Stage 07 |
| SI 1 | `si1.py` | `yearly_total_1990_2024.nc` | Stage 08 |
| SI 2 | `si2.py` | `data/gloria_agri_water_stress_by_country_yearly.csv`, `data/ours_icws_by_country_yearly.csv` | see note below |

Figures render into `output/` (gitignored).

## Order

1. Run Stages 05–08 so the inputs above exist.
2. Figure 3 only: run `fig3_shap_computation.py` once to precompute the
   SHAP arrays before `fig3f.py` / `fig3g.py`.
3. Run each figure script.

## si2 inputs

`si2.py` reads the GLORIA-vs-ours country comparison from
`paper_figs/data/`. Place the two CSVs there:

- `data/gloria_agri_water_stress_by_country_yearly.csv`
- `data/ours_icws_by_country_yearly.csv`

(they are the outputs of the `gloria_vs_ours` comparison; see the paper's
Discussion). Both are small and are kept in the repository.

## Environments

```bash
# Python (Fig 1-3, SI)
pip install numpy pandas xarray matplotlib geopandas shap xgboost pyarrow

# R (Fig 4-5)
# install.packages(c("ggplot2", "dplyr", "tidyr", "ggalluvial", "scales"))
```
