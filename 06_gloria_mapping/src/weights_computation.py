"""Compute country-specific split weights for the 3 multi-sector crop categories (oac -> 6 sectors, pdc -> 2, pec -> 3).

Historical years use a rolling centered window (default 5 years); forecast
years use fixed weights from the recent-period mean (default 2015-2018).
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_SPLIT_CROPS = ("oac", "pdc", "pec")
DEFAULT_ROLLING_WINDOW = 5
DEFAULT_RECENT_PERIOD = (2015, 2018)


def _load_data(
    crops_175_csv: str, crop_mapping_csv: str, country_mapping_csv: str,
    split_crops, years_175: Tuple[int, int], logger: logging.Logger,
):
    crop_mapping = pd.read_csv(crop_mapping_csv)
    country_mapping = pd.read_csv(country_mapping_csv)
    crops_175 = pd.read_csv(crops_175_csv)

    crops_175 = crops_175[
        (crops_175["year"] >= years_175[0]) & (crops_175["year"] <= years_175[1])
    ].copy()

    split_mapping = crop_mapping[crop_mapping["crop_27"].isin(split_crops)].copy()
    crops_175 = crops_175[crops_175["crop"].isin(split_mapping["crop_175"])].copy()

    logger.info(f"  crop_mapping (split only): {len(split_mapping)} rows "
                f"({split_mapping['crop_27'].nunique()} crop_27 categories)")
    logger.info(f"  crops_175 (split only):    {len(crops_175)} rows, "
                f"years {int(crops_175['year'].min())}-{int(crops_175['year'].max())}")
    return crops_175, split_mapping, country_mapping


def _build_aggregated(crops_175, split_mapping, country_mapping, logger):
    df = (
        crops_175
        .merge(split_mapping, left_on="crop", right_on="crop_175", how="inner")
        .merge(country_mapping, left_on="country", right_on="country_175", how="inner")
    )
    grouped = (
        df.groupby(["country_14_code", "crop_27", "sector_14", "year"])["value"]
        .sum()
        .reset_index()
    )
    logger.info(f"  aggregated table: {len(grouped)} rows")
    return grouped


def _compute_weights_from_values(df_values: pd.DataFrame) -> pd.DataFrame:
    total = df_values.groupby(["country_14_code", "crop_27"])["value"].transform("sum")
    df_values = df_values.copy()
    df_values["weight"] = np.where(total > 0, df_values["value"] / total, 0.0)
    return df_values[["country_14_code", "crop_27", "sector_14", "weight"]]


def _calculate_rolling_weights(aggregated: pd.DataFrame, window: int) -> pd.DataFrame:
    years = sorted(aggregated["year"].unique())
    min_yr, max_yr = int(min(years)), int(max(years))
    half = window // 2

    results = []
    for year in range(min_yr, max_yr + 1):
        w_start = max(min_yr, year - half)
        w_end = min(max_yr, year + half)
        window_data = aggregated[
            (aggregated["year"] >= w_start) & (aggregated["year"] <= w_end)
        ]
        avg = (
            window_data.groupby(["country_14_code", "crop_27", "sector_14"])["value"]
            .mean()
            .reset_index()
        )
        w = _compute_weights_from_values(avg)
        w["year"] = year
        results.append(w)
    return pd.concat(results, ignore_index=True)


def _calculate_recent_weights(
    aggregated: pd.DataFrame, period: Tuple[int, int],
) -> pd.DataFrame:
    recent = aggregated[
        (aggregated["year"] >= period[0]) & (aggregated["year"] <= period[1])
    ]
    avg = (
        recent.groupby(["country_14_code", "crop_27", "sector_14"])["value"]
        .mean()
        .reset_index()
    )
    return _compute_weights_from_values(avg)


def _validate(rolling: pd.DataFrame, recent: pd.DataFrame, logger):
    sums = rolling.groupby(["country_14_code", "crop_27", "year"])["weight"].sum()
    n_bad = ((sums < 0.999) | (sums > 1.001)).sum()
    logger.info(f"  rolling weights validation: {n_bad}/{len(sums)} groups not summing to 1")
    sums_r = recent.groupby(["country_14_code", "crop_27"])["weight"].sum()
    n_bad_r = ((sums_r < 0.999) | (sums_r > 1.001)).sum()
    logger.info(f"  recent weights validation:  {n_bad_r}/{len(sums_r)} groups not summing to 1")


def compute_weights(
    crops_175_csv: str,
    crop_mapping_csv: str,
    country_mapping_csv: str,
    rolling_csv: str,
    recent_csv: str,
    years_175: Tuple[int, int] = (1990, 2018),
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    recent_period: Tuple[int, int] = DEFAULT_RECENT_PERIOD,
    split_crops: List[str] = DEFAULT_SPLIT_CROPS,
    logger: Optional[logging.Logger] = None,
) -> Tuple[str, str]:
    """Compute and save the rolling + recent split weights."""
    logger = logger or logging.getLogger(__name__)
    logger.info("=" * 70)
    logger.info("  Split weights for oac / pdc / pec")
    logger.info("=" * 70)

    logger.info("[1] loading data (filtered to split crops only)...")
    crops_175, split_mapping, country_mapping = _load_data(
        crops_175_csv, crop_mapping_csv, country_mapping_csv,
        tuple(split_crops), years_175, logger,
    )

    logger.info("[2] aggregating by (country, crop_27, sector, year)...")
    aggregated = _build_aggregated(crops_175, split_mapping, country_mapping, logger)

    logger.info(f"[3] rolling weights (window={rolling_window})...")
    rolling = _calculate_rolling_weights(aggregated, window=rolling_window)
    logger.info(f"    result: {len(rolling)} rows, "
                f"years {int(rolling['year'].min())}-{int(rolling['year'].max())}")

    logger.info(f"[4] recent-period weights ({recent_period[0]}-{recent_period[1]})...")
    recent = _calculate_recent_weights(aggregated, period=tuple(recent_period))
    logger.info(f"    result: {len(recent)} rows")

    logger.info("[5] validating...")
    _validate(rolling, recent, logger)

    Path(rolling_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(recent_csv).parent.mkdir(parents=True, exist_ok=True)
    rolling.to_csv(rolling_csv, index=False)
    recent.to_csv(recent_csv, index=False)
    logger.info(f"  saved: {rolling_csv}")
    logger.info(f"  saved: {recent_csv}")
    return str(rolling_csv), str(recent_csv)
