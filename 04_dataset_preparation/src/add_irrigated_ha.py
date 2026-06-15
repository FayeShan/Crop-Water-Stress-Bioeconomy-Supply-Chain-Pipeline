"""Add ``ha_irrigated`` feature to raw per-crop parquets.

Individual crops use monthly HA NCs looked up per (lat, lon, month);
aggregate crops (oac, pec, pdc) use annual HA NCs. If a year's monthly NC
is missing, fall back up to 3 years earlier; for annual NC, fall back to
the last available year. Remaining NaNs are filled with 0 at save time.
"""

import gc
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm


def _load_ha_monthly(monthly_dir: Path, crop: str, year: int):
    nc_path = monthly_dir / f"{crop}_{year}.nc"
    if nc_path.exists():
        ds = xr.open_dataset(nc_path)
        return ds["harvested_area_irrigated"], year

    for fb in range(year - 1, year - 4, -1):
        fb_path = monthly_dir / f"{crop}_{fb}.nc"
        if fb_path.exists():
            ds = xr.open_dataset(fb_path)
            return ds["harvested_area_irrigated"], fb
    return None, None


def _extract_monthly(ha_da, lons, lats, months):
    n = len(lons)
    values = np.full(n, np.nan, dtype=np.float32)
    for m in np.unique(months):
        m_mask = months == m
        if not m_mask.any():
            continue
        try:
            month_data = ha_da.isel(time=m - 1)
        except IndexError:
            continue
        try:
            vals = month_data.sel(
                lat=xr.DataArray(lats[m_mask], dims="points"),
                lon=xr.DataArray(lons[m_mask], dims="points"),
                method="nearest",
            ).values
            values[m_mask] = vals
        except Exception:
            pass
    return values


def _load_ha_annual(annual_dir: Path, crop: str):
    nc_path = annual_dir / f"{crop}.nc"
    if not nc_path.exists():
        return None
    return xr.open_dataset(nc_path)


def _extract_annual(ha_ds, year, lons, lats):
    n = len(lons)
    values = np.full(n, np.nan, dtype=np.float32)

    years_in_nc = ha_ds["time"].values
    year_vals = np.array([int(y) for y in years_in_nc])
    actual_year = year if year in year_vals else int(year_vals[-1])
    t_idx = np.where(year_vals == actual_year)[0][0]
    year_data = ha_ds["harvested_area_irrigated"].isel(time=t_idx)

    try:
        vals = year_data.sel(
            lat=xr.DataArray(lats, dims="points"),
            lon=xr.DataArray(lons, dims="points"),
            method="nearest",
        ).values
        values[:] = vals
    except Exception:
        pass
    return values, actual_year


def process_parquet(
    parquet_path: Path,
    output_path: Path,
    crop: str,
    monthly_dir: Path,
    annual_dir: Path,
    aggregate_crops: List[str],
    logger: logging.Logger,
) -> bool:
    logger.info(f"  loading {parquet_path}")
    df = pd.read_parquet(parquet_path)
    if "time" not in df.columns:
        logger.error(f"  no 'time' column in {parquet_path}")
        return False

    df["time"] = pd.to_datetime(df["time"])
    logger.info(f"  rows: {len(df):,}")
    df["ha_irrigated"] = np.nan

    years = sorted(df["time"].dt.year.unique())
    is_aggregate = crop in aggregate_crops

    if is_aggregate:
        ha_ds = _load_ha_annual(annual_dir, crop)
        if ha_ds is None:
            logger.warning(f"  no annual NC for {crop}, setting ha_irrigated=0")
            df["ha_irrigated"] = 0.0
        else:
            for year in years:
                year_mask = df["time"].dt.year == year
                year_df = df.loc[year_mask]
                values, actual_year = _extract_annual(
                    ha_ds, year, year_df["lon"].values, year_df["lat"].values)
                df.loc[year_mask, "ha_irrigated"] = values
                fb_note = f" (fallback from {actual_year})" if actual_year != year else ""
                logger.info(
                    f"    {year}: {year_mask.sum():,} rows"
                    f" [annual{fb_note}, mean={np.nanmean(values):.1f} ha]"
                )
            ha_ds.close()
    else:
        for year in years:
            year_mask = df["time"].dt.year == year
            year_df = df.loc[year_mask]

            ha_da, actual_year = _load_ha_monthly(monthly_dir, crop, year)
            if ha_da is None:
                logger.warning(f"    {year}: monthly NC not found, leaving NaN")
                continue
            values = _extract_monthly(
                ha_da,
                year_df["lon"].values,
                year_df["lat"].values,
                year_df["time"].dt.month.values,
            )
            df.loc[year_mask, "ha_irrigated"] = values
            ha_da.close()
            fb_note = f" (fallback {actual_year})" if actual_year != year else ""
            logger.info(
                f"    {year}: {year_mask.sum():,} rows"
                f" [monthly{fb_note}, mean={np.nanmean(values):.1f} ha]"
            )

    n_nan = df["ha_irrigated"].isna().sum()
    if n_nan > 0:
        df["ha_irrigated"] = df["ha_irrigated"].fillna(0.0)
        logger.info(f"  filled {n_nan:,} NaN with 0")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"  saved: {output_path}")
    del df
    gc.collect()
    return True


def add_irrigated_ha(
    crops: List[str],
    input_dir: str,
    output_dir: str,
    monthly_dir: str,
    annual_dir: str,
    aggregate_crops: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Add ``ha_irrigated`` column to each raw per-crop parquet.

    Args:
        crops: Crop codes to process.
        input_dir: Directory with raw parquets ``{crop}/{year}.parquet``.
        output_dir: Destination with the same layout; ``ha_irrigated``
            column added.
        monthly_dir: iHA monthly NCs ``{acea}_{year}.nc``.
        annual_dir: iHA annual NCs ``{acea}.nc`` (for aggregate crops).
        aggregate_crops: Crops that should be looked up via annual NC
            (default: ``['oac', 'pec', 'pdc']``).
    """
    logger = logger or logging.getLogger(__name__)
    aggregate_crops = aggregate_crops or ["oac", "pec", "pdc"]

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    monthly_dir = Path(monthly_dir)
    annual_dir = Path(annual_dir)

    logger.info("=" * 60)
    logger.info("  Add ha_irrigated to per-crop parquets")
    logger.info("=" * 60)
    logger.info(f"  Input:        {input_dir}")
    logger.info(f"  Output:       {output_dir}")
    logger.info(f"  Monthly HA:   {monthly_dir}")
    logger.info(f"  Annual HA:    {annual_dir}")
    logger.info(f"  Aggregate:    {aggregate_crops}")
    logger.info(f"  Crops:        {len(crops)}")

    for crop in tqdm(crops, desc="Adding ha_irrigated"):
        logger.info(f"\n  === {crop} ===")
        in_dir = input_dir / crop
        out_dir = output_dir / crop
        if not in_dir.exists():
            logger.warning(f"  {crop}: input folder missing at {in_dir}")
            continue
        parquet_files = sorted(in_dir.glob("*.parquet"))
        if not parquet_files:
            logger.warning(f"  {crop}: no parquet files in {in_dir}")
            continue
        for pq in parquet_files:
            process_parquet(
                pq, out_dir / pq.name, crop,
                monthly_dir, annual_dir, aggregate_crops, logger,
            )

    logger.info("Done.")
