"""Shared feature engineering for training and prediction datasets.

Applies cleaning, TWSA imputation, time/location encoding, NOAA merge,
grid-id/region joins, and lag/rolling features, writing one parquet per
crop. Predict mode prepends the tail of the training data so early-year
lag/rolling features use a continuous time series, then trims back to the
target year range.
"""

import gc
import logging
import os
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


def load_crop_parquets(
    data_root: Path, crop: str, start_year: int, end_year: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    dfs = []
    for year in range(start_year, end_year + 1):
        path = data_root / crop / f"{year}.parquet"
        if path.exists():
            dfs.append(pd.read_parquet(path))
    if not dfs:
        raise FileNotFoundError(
            f"No parquet files for {crop} in {data_root} ({start_year}-{end_year})"
        )
    df = pd.concat(dfs, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"])
    logger.info(f"  loaded {len(dfs)} files, {len(df):,} rows")
    return df


def basic_cleaning(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    core = ["d2m", "t2m", "swvl1", "ssr", "evabs", "v10", "sp", "NDVI", "EVI"]
    core = [c for c in core if c in df.columns]
    before = len(df)
    df = df.dropna(subset=core)
    logger.info(f"  cleaning: {before:,} -> {len(df):,}")
    return df


def impute_twsan(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    if "twsan" not in df.columns:
        return df
    df = df.copy()
    n_before = df["twsan"].isna().sum()
    df["twsan"] = df.groupby(["lat", "lon"])["twsan"].transform(lambda x: x.fillna(x.median()))
    df["twsan"] = df.groupby(["year", "month"])["twsan"].transform(lambda x: x.fillna(x.median()))
    df["twsan"] = df["twsan"].fillna(df["twsan"].median())
    n_after = df["twsan"].isna().sum()
    logger.info(f"  TWSA imputation: {n_before:,} -> {n_after:,} NaN")
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df["day_of_year"] = df["time"].dt.dayofyear
    df["quarter"] = df["time"].dt.quarter
    df["half_year"] = (df["month"] > 6).astype(int)
    df["cos_doy"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    df["sin_doy"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["decade"] = (df["year"] // 10) * 10
    return df


def add_location_encoding(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["lat_sin"] = np.sin(np.radians(df["lat"]))
    df["lat_cos"] = np.cos(np.radians(df["lat"]))
    df["lon_sin"] = np.sin(np.radians(df["lon"]))
    df["lon_cos"] = np.cos(np.radians(df["lon"]))
    return df


def assign_grid_id(df: pd.DataFrame, grid50_path: str,
                   logger: logging.Logger, precision: int = 6) -> pd.DataFrame:
    grid = pd.read_parquet(grid50_path)
    grid["key"] = (grid["lon"].round(precision).astype(str) + "_"
                   + grid["lat"].round(precision).astype(str))
    df = df.copy()
    df["key"] = (df["lon"].round(precision).astype(str) + "_"
                 + df["lat"].round(precision).astype(str))
    merged = df.merge(
        grid[["key", "grid_id"]].rename(columns={"grid_id": "grid50_id"}),
        on="key", how="left",
    )
    miss = merged["grid50_id"].isna().mean()
    logger.info(f"  grid id assigned ({miss:.1%} unmatched)")
    return merged.drop(columns="key")


def assign_region(df: pd.DataFrame, region_path: str,
                  logger: logging.Logger, precision: int = 6) -> pd.DataFrame:
    region = pd.read_parquet(region_path)
    region["key"] = (region["lon"].round(precision).astype(str) + "_"
                     + region["lat"].round(precision).astype(str))
    df = df.copy()
    df["key"] = (df["lon"].round(precision).astype(str) + "_"
                 + df["lat"].round(precision).astype(str))
    merged = df.merge(region[["key", "regiontype"]], on="key", how="left")
    miss = merged["regiontype"].isna().mean()
    logger.info(f"  region assigned ({miss:.1%} unmatched)")
    return merged.drop(columns="key")


def merge_noaa_indices(df: pd.DataFrame, noaa_dir: str,
                       logger: logging.Logger) -> pd.DataFrame:
    n_merged = 0
    for fname in sorted(os.listdir(noaa_dir)):
        if not fname.lower().endswith(".csv"):
            continue
        file_path = os.path.join(noaa_dir, fname)
        index_name = re.sub(r"_?index\.csv$", "", fname, flags=re.IGNORECASE).rsplit(".", 1)[0]
        temp = pd.read_csv(file_path)
        temp.columns = [c.lower() for c in temp.columns]
        if {"year", "month"}.issubset(temp.columns):
            val_cols = list(set(temp.columns) - {"year", "month"})
            if not val_cols:
                continue
            val = val_cols[0]
            temp2 = temp[["year", "month", val]].rename(columns={val: index_name})
        elif "year" in temp.columns:
            month_cols = [c for c in temp.columns if re.search(r"\d+", c)]
            if not month_cols:
                continue
            temp2 = temp.melt(id_vars="year", value_vars=month_cols,
                              var_name="month", value_name=index_name)
            temp2["month"] = temp2["month"].str.extract(r"(\d+)").astype(int)
        else:
            continue
        df = df.merge(temp2, on=["year", "month"], how="left")
        n_merged += 1
    logger.info(f"  merged {n_merged} NOAA indices")
    return df


def add_lag_features(df, variables, lag_range, logger):
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values(["lat", "lon", "time"])
    n_new = 0
    for var in variables:
        if var not in df.columns:
            continue
        for lag in lag_range:
            df[f"{var}_lag{lag}"] = df.groupby(["lat", "lon"])[var].shift(lag)
            n_new += 1
    logger.info(f"  added {n_new} lag features")
    return df


def add_rolling_features(df, variables, windows, logger):
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values(["lat", "lon", "time"])
    n_new = 0
    for var in variables:
        if var not in df.columns:
            continue
        for w in windows:
            df[f"{var}_roll{w}"] = (
                df.groupby(["lat", "lon"])[var]
                .transform(lambda x: x.rolling(window=w, min_periods=1).mean())
            )
            n_new += 1
    logger.info(f"  added {n_new} rolling features")
    return df


def process_crop(
    crop: str,
    data_root: str,
    output_root: str,
    year_range: tuple,
    grid50_path: str,
    region_path: str,
    noaa_dir: str,
    lag_vars: List[str],
    lag_range,
    roll_vars: List[str],
    roll_windows: List[int],
    prepend_root: Optional[str] = None,
    prepend_year_range: Optional[tuple] = None,
    overwrite: bool = False,
    logger: Optional[logging.Logger] = None,
) -> None:
    logger = logger or logging.getLogger(__name__)
    data_root = Path(data_root)
    output_root = Path(output_root)

    start_year, end_year = year_range
    out_dir = output_root / crop
    out_file = out_dir / f"{crop}.parquet"
    if out_file.exists() and not overwrite:
        logger.info(f"  {crop}: already exists -> {out_file}, skip")
        return

    logger.info("")
    logger.info(f"  === {crop}: loading {start_year}-{end_year} from {data_root} ===")
    try:
        df = load_crop_parquets(data_root, crop, start_year, end_year, logger)
    except FileNotFoundError as e:
        logger.warning(f"  {crop}: {e}")
        return

    prepended = False
    if prepend_root is not None and prepend_year_range is not None:
        try:
            df_prepend = load_crop_parquets(
                Path(prepend_root), crop, *prepend_year_range, logger)
            cutoff = pd.Timestamp(f"{start_year}-01-01") - pd.DateOffset(months=6)
            df_prepend = df_prepend[df_prepend["time"] >= cutoff]
            if len(df_prepend) > 0:
                logger.info(
                    f"  prepending {len(df_prepend):,} training rows "
                    f"({df_prepend['time'].min():%Y-%m} – "
                    f"{df_prepend['time'].max():%Y-%m}) for lag continuity"
                )
                df = pd.concat([df_prepend, df], ignore_index=True)
                df = df.drop_duplicates(subset=["lat", "lon", "time"], keep="last")
                prepended = True
            del df_prepend
        except FileNotFoundError:
            logger.warning(f"  {crop}: prepend data not found, early-month lag may have NaN")

    df = basic_cleaning(df, logger)
    df = add_time_features(df)
    df = add_location_encoding(df)
    df = merge_noaa_indices(df, noaa_dir, logger)
    df = assign_grid_id(df, grid50_path, logger)
    df = impute_twsan(df, logger)
    df = add_lag_features(df, lag_vars, lag_range, logger)
    df = add_rolling_features(df, roll_vars, roll_windows, logger)
    df = assign_region(df, region_path, logger)

    if prepended:
        before_trim = len(df)
        df = df[df["time"].dt.year >= start_year].reset_index(drop=True)
        logger.info(f"  trimmed prepended rows: {before_trim:,} -> {len(df):,}")

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_file, index=False)
    logger.info(f"  {crop}: saved -> {out_file} "
                f"({len(df):,} rows, {len(df.columns)} cols)")
    del df
    gc.collect()


def run_feature_engineering(
    mode: str,
    crops: List[str],
    train_input: str,
    predict_input: str,
    train_output: str,
    predict_output: str,
    train_year_range: tuple,
    predict_year_range: tuple,
    grid50_path: str,
    region_path: str,
    noaa_dir: str,
    lag_vars: List[str],
    lag_range,
    roll_vars: List[str],
    roll_windows: List[int],
    prepend_years_from_train: tuple,
    overwrite: bool = False,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Run feature engineering for ``mode`` in {train, predict, both}."""
    logger = logger or logging.getLogger(__name__)

    if mode in ("train", "both"):
        logger.info("=" * 60)
        logger.info("  TRAINING DATA: feature engineering")
        logger.info(f"  input:  {train_input}")
        logger.info(f"  output: {train_output}")
        logger.info(f"  years:  {train_year_range[0]}-{train_year_range[1]}")
        logger.info(f"  crops:  {len(crops)}")
        logger.info("=" * 60)
        for crop in tqdm(crops, desc="Train FE"):
            process_crop(
                crop=crop,
                data_root=train_input,
                output_root=train_output,
                year_range=train_year_range,
                grid50_path=grid50_path,
                region_path=region_path,
                noaa_dir=noaa_dir,
                lag_vars=lag_vars, lag_range=lag_range,
                roll_vars=roll_vars, roll_windows=roll_windows,
                overwrite=overwrite, logger=logger,
            )

    if mode in ("predict", "both"):
        logger.info("=" * 60)
        logger.info("  PREDICTION DATA: feature engineering")
        logger.info(f"  input:    {predict_input}")
        logger.info(f"  output:   {predict_output}")
        logger.info(f"  years:    {predict_year_range[0]}-{predict_year_range[1]}")
        logger.info(f"  prepend:  {train_input} {prepend_years_from_train}")
        logger.info(f"  crops:    {len(crops)}")
        logger.info("=" * 60)
        for crop in tqdm(crops, desc="Predict FE"):
            process_crop(
                crop=crop,
                data_root=predict_input,
                output_root=predict_output,
                year_range=predict_year_range,
                grid50_path=grid50_path,
                region_path=region_path,
                noaa_dir=noaa_dir,
                lag_vars=lag_vars, lag_range=lag_range,
                roll_vars=roll_vars, roll_windows=roll_windows,
                prepend_root=train_input,
                prepend_year_range=tuple(prepend_years_from_train),
                overwrite=overwrite, logger=logger,
            )

    logger.info("Done.")
