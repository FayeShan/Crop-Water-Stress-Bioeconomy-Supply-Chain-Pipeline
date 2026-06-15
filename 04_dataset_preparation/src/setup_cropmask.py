"""Build per-crop boolean monthly masks from water-stress NC files.

For each crop, records the union of (month, lat, lon) cells with finite
``waterstress`` data into ``{crop}_monthly_mask`` of shape (12, lat, lon).
Consumed by ``build_train_parquet`` / ``build_predict_parquet`` to restrict
rows to crop-relevant cells.
"""

import gc
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm


def generate_cropmask(
    crop: str,
    ws_dir: str,
    output_dir: str,
    years: range,
    logger: Optional[logging.Logger] = None,
) -> Optional[str]:
    """Build a monthly mask for one crop."""
    logger = logger or logging.getLogger(__name__)

    ws_dir = Path(ws_dir) / crop
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    savepath = output_dir / f"{crop}.nc"
    if savepath.exists():
        logger.info(f"  {crop}: mask already exists, skip")
        return str(savepath)

    # Use first available year to learn the grid
    template = None
    template_year = None
    for year in years:
        p = ws_dir / f"{year}.nc"
        if p.exists():
            with xr.open_dataset(p) as ds:
                if "waterstress" not in ds.data_vars:
                    continue
                template = ds["waterstress"].isel(time=0).copy()
                template_year = year
            break

    if template is None:
        logger.warning(f"  {crop}: no NC files found in {ws_dir}, skip")
        return None

    lat_name, lon_name = template.dims[-2:]
    time_coord = pd.date_range("2000-01-01", periods=12, freq="MS")

    mask = xr.DataArray(
        data=np.zeros((12,) + template.shape, dtype=bool),
        coords={
            "time": time_coord,
            lat_name: template[lat_name],
            lon_name: template[lon_name],
        },
        dims=("time", lat_name, lon_name),
        name=f"{crop}_monthly_mask",
    )

    for year in years:
        p = ws_dir / f"{year}.nc"
        if not p.exists():
            continue
        ds = xr.open_dataset(p)
        if "waterstress" not in ds.data_vars:
            ds.close()
            continue
        data = ds["waterstress"]
        for m in range(1, 13):
            month_data = data.sel(time=data["time.month"] == m)
            if month_data.sizes["time"] == 0:
                continue
            month_mask = ~np.isnan(month_data).all(dim="time")
            mask.loc[dict(time=time_coord[m - 1])] |= month_mask
            del month_data, month_mask
            gc.collect()
        ds.close()
        del ds, data
        gc.collect()

    encoding = {mask.name: {"zlib": True, "complevel": 5}}
    mask.to_netcdf(savepath, engine="h5netcdf", encoding=encoding)
    logger.info(f"  {crop}: saved -> {savepath}")
    del mask
    gc.collect()
    return str(savepath)


def generate_cropmasks(
    crops: List[str],
    ws_dir: str,
    output_dir: str,
    start_year: int,
    end_year: int,
    logger: Optional[logging.Logger] = None,
) -> List[str]:
    """Generate masks for all listed crops."""
    logger = logger or logging.getLogger(__name__)
    years = range(start_year, end_year + 1)

    logger.info("=" * 60)
    logger.info("  Generate per-crop monthly masks")
    logger.info("=" * 60)
    logger.info(f"  Crops:    {len(crops)}")
    logger.info(f"  Years:    {start_year}-{end_year}")
    logger.info(f"  WS dir:   {ws_dir}")
    logger.info(f"  Output:   {output_dir}")

    outputs = []
    for crop in tqdm(crops, desc="Crop masks"):
        try:
            out = generate_cropmask(crop, ws_dir, output_dir, years, logger)
            if out:
                outputs.append(out)
        except Exception as e:
            logger.error(f"  {crop}: failed — {e}")
    logger.info(f"Done — {len(outputs)} masks written")
    return outputs
