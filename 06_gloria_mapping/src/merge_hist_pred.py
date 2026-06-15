"""Merge the historical country-level CSV (year, country, crop, value) with the per-crop prediction CSVs into one unified crops_27 CSV (year, country, crop, value).

On overlapping years, ``overlap_rule='keep_historical'`` (default) keeps
historical rows and drops the prediction-side overlap.
"""

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd


def merge_historical_and_predictions(
    historical_csv: str,
    pred_country_dir: str,
    crops: List[str],
    output_csv: str,
    overlap_rule: str = "keep_historical",
    logger: Optional[logging.Logger] = None,
) -> str:
    logger = logger or logging.getLogger(__name__)
    historical_csv = Path(historical_csv)
    pred_country_dir = Path(pred_country_dir)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("  Merge historical + prediction into unified country-level CSV")
    logger.info("=" * 70)

    logger.info(f"  historical_csv: {historical_csv}")
    df_hist = pd.read_csv(historical_csv)
    logger.info(f"    {len(df_hist):,} rows, years "
                f"{df_hist['year'].min()}-{df_hist['year'].max()}, "
                f"{df_hist['country'].nunique()} countries, "
                f"{df_hist['crop'].nunique()} crops")

    logger.info(f"  prediction dir: {pred_country_dir}")
    pred_data = []
    for crop in crops:
        f = pred_country_dir / f"{crop}_country.csv"
        if not f.exists():
            logger.warning(f"    {crop}: not found")
            continue
        df = pd.read_csv(f)
        agg = (df.groupby(["year", "country"])["waterstress"]
               .sum().reset_index())
        agg.columns = ["year", "country", "value"]
        agg["crop"] = crop
        pred_data.append(agg)
        logger.info(f"    {crop}: {len(agg):,} rows")

    if not pred_data:
        raise FileNotFoundError("No prediction files found")

    df_pred = pd.concat(pred_data, ignore_index=True)
    df_pred = df_pred[["year", "country", "crop", "value"]]
    logger.info(f"  prediction total: {len(df_pred):,} rows, "
                f"years {df_pred['year'].min()}-{df_pred['year'].max()}")

    overlap = sorted(set(df_hist["year"].unique()) & set(df_pred["year"].unique()))
    if overlap:
        if overlap_rule == "keep_historical":
            logger.info(f"  overlap years {overlap} -> keeping historical, dropping pred")
            df_pred = df_pred[~df_pred["year"].isin(overlap)]
        elif overlap_rule == "keep_prediction":
            logger.info(f"  overlap years {overlap} -> keeping prediction, dropping hist")
            df_hist = df_hist[~df_hist["year"].isin(overlap)]
        else:
            raise ValueError(f"Unknown overlap_rule: {overlap_rule}")

    df_all = pd.concat([df_hist, df_pred], ignore_index=True)
    df_all = df_all.sort_values(["year", "country", "crop"]).reset_index(drop=True)
    df_all.to_csv(output_csv, index=False)
    logger.info(f"  saved: {output_csv}")
    logger.info(f"    {len(df_all):,} rows, years "
                f"{df_all['year'].min()}-{df_all['year'].max()}, "
                f"{df_all['country'].nunique()} countries, "
                f"{df_all['crop'].nunique()} crops")

    yearly = df_all.groupby("year")["value"].sum()
    hist_max = df_hist["year"].max()
    if (hist_max in yearly.index) and ((hist_max + 1) in yearly.index):
        change = (yearly[hist_max + 1] - yearly[hist_max]) / yearly[hist_max] * 100
        logger.info(f"    {hist_max}->{hist_max+1} global total change: {change:+.1f}%")
    return str(output_csv)
