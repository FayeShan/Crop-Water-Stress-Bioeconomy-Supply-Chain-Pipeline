"""Build raw per-crop prediction parquets from merged predictor NCs.

No target variable is included; feature engineering is applied later by
``feature_engineering.py``.
"""

import gc
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr
from tqdm import tqdm


def convert_crop_predict(
    crop: str,
    predictors_dir: str,
    cropmasks_dir: str,
    output_dir: str,
    predictor_vars: List[str],
    years: range,
    overwrite: bool = False,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Convert merged NC -> prediction parquet for one crop over all years."""
    logger = logger or logging.getLogger(__name__)

    predictors_dir = Path(predictors_dir)
    cropmasks_dir = Path(cropmasks_dir)
    output_dir = Path(output_dir) / crop
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_path = cropmasks_dir / f"{crop}.nc"
    if not mask_path.exists():
        logger.warning(f"  {crop}: mask not found at {mask_path}, skip")
        return
    mask_ds = xr.open_dataset(mask_path)
    mask_var = f"{crop}_monthly_mask"
    mask_da = mask_ds[mask_var]

    base_cols = ["time", "lat", "lon"] + predictor_vars

    for yr in years:
        out_file = output_dir / f"{yr}.parquet"
        if out_file.exists() and not overwrite:
            continue

        pred_nc = predictors_dir / f"{yr}_merged.nc"
        if not pred_nc.exists():
            logger.warning(f"  {crop}/{yr}: {pred_nc.name} missing, skip")
            continue

        ds_year = xr.open_dataset(pred_nc)
        writer, slices = None, 0

        for t in ds_year.time.values:
            ts = pd.to_datetime(str(t))
            month_idx = ts.month - 1
            if month_idx >= mask_da.sizes["time"]:
                continue

            mask_slice = mask_da.isel(time=month_idx)
            df = (
                ds_year.sel(time=t)
                .where(mask_slice)
                .to_dataframe()
                .reset_index()
            )
            df = df.dropna(how="all", subset=[v for v in predictor_vars if v in df.columns])
            if df.empty:
                continue

            df["time"] = ts
            cols_now = [c for c in base_cols if c in df.columns]
            table = pa.Table.from_pandas(df[cols_now], preserve_index=False)

            if writer is None:
                writer = pq.ParquetWriter(str(out_file), table.schema, compression="zstd")
            writer.write_table(table)
            slices += 1

            del df, table
            gc.collect()

        if writer:
            writer.close()
            logger.info(f"  {crop}/{yr}: {slices} months written -> {out_file.name}")
        else:
            logger.info(f"  {crop}/{yr}: no data")

        ds_year.close()
        del ds_year
        gc.collect()

    mask_ds.close()
    del mask_ds, mask_da
    gc.collect()


def build_predict_parquets(
    crops: List[str],
    predictors_dir: str,
    cropmasks_dir: str,
    output_dir: str,
    predictor_vars: List[str],
    start_year: int,
    end_year: int,
    overwrite: bool = False,
    logger: Optional[logging.Logger] = None,
) -> None:
    logger = logger or logging.getLogger(__name__)
    years = range(start_year, end_year + 1)

    logger.info("=" * 60)
    logger.info("  Build raw prediction parquets")
    logger.info("=" * 60)
    logger.info(f"  Crops:      {len(crops)}")
    logger.info(f"  Years:      {start_year}-{end_year}")
    logger.info(f"  Predictors: {predictors_dir}")
    logger.info(f"  Masks:      {cropmasks_dir}")
    logger.info(f"  Output:     {output_dir}")

    for crop in tqdm(crops, desc="Predict parquets"):
        logger.info(f"\n  === {crop} ===")
        convert_crop_predict(
            crop=crop,
            predictors_dir=predictors_dir,
            cropmasks_dir=cropmasks_dir,
            output_dir=output_dir,
            predictor_vars=predictor_vars,
            years=years,
            overwrite=overwrite,
            logger=logger,
        )

    logger.info("Done.")
