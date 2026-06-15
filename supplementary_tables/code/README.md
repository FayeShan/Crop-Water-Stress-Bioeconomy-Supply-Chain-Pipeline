# supplementary_tables/code — table generators

`generate_tables.py` regenerates the main-text and supplementary tables in
`../Supplementary_Tables.xlsx` from the pipeline outputs. Each table is an
independent, input-guarded function, so missing inputs are skipped rather
than fatal.

```bash
pip install pandas numpy scipy openpyxl
python generate_tables.py        # writes ./tables/*.csv
```

Edit the `PATHS` block at the top of `generate_tables.py` to point at your
outputs. The defaults resolve to the in-repo stage outputs where they
exist:

| Table(s) | Produces | Main inputs |
|----------|----------|-------------|
| S1 | grid-level metrics (G vs GW) | `05_harm_model/results/eval/{crop}/{global,global_weighted}/summary.json` |
| S2 | regional / extreme adoption | `…/eval/{crop}/{regional,…}/summary.json` |
| S3 | weighted vs unweighted (country) | `…/eval/country_eval_*.csv` |
| S6 | feature importance | `…/eval/{crop}/{global*}/feature_importance.csv` |
| S7 | Köppen (30) vs THZ×MST (9) | `…/eval/{wh,mai}/regional*/…/region_summary.json` |
| 2, 3 | grid / country performance | `…/eval/country_eval_*.csv` |
| 4, S4, S5 | prediction summary, per-country, transition | `06_gloria_mapping/data/crops_27_1990_2024.csv` (+ `BASE_AGG_DIR`) |
| S8 | predicted vs historical mean | `05_harm_model/.../predictions_csv/{crop}.csv` + `BASE_AGG_DIR` |
| S12 | monthly seasonality | `BASE_AGG_DIR` + per-crop country predictions |
| 5, 6, S9, S10 | ranking, climate events, temporal trend, short-window | `VALIDATION_DIR/dir*.csv` |

`BASE_AGG_DIR` (historical country-aggregated reference, one file per
year/crop) and `VALIDATION_DIR` (`dir1a`/`dir1b`/`dir2a`/`dir2b`/`dir3`
diagnostics) are analysis intermediates that are not part of the public
pipeline outputs; point the `PATHS` block at your own copies to enable the
tables that depend on them.
