"""Build the 50 km coarse grid-ID look-up parquet (lon, lat, grid_id).

Each 6x6 block of 0.05-degree cells gets one integer grid_id. Joined onto
parquets by ``feature_engineering.py`` to provide the ``grid50_id`` feature.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr


def generate_coarse_grid(
    template_nc: str,
    output_parquet: str,
    coarse_factor: int = 6,
    logger: Optional[logging.Logger] = None,
) -> str:
    """Build the 50 km grid-id parquet from a 0.05-degree template NC.

    Args:
        template_nc: Path to a 0.05-degree NC file (only lon/lat are read).
        output_parquet: Where to save the output parquet.
        coarse_factor: Number of fine cells per coarse cell per axis
            (6 ~ 50 km, 12 ~ 100 km).

    Returns:
        Path to the saved parquet.
    """
    logger = logger or logging.getLogger(__name__)

    ds = xr.open_dataset(template_nc)
    ds0 = ds.drop_vars(list(ds.data_vars), errors="ignore")
    if "time" in ds0.dims:
        ds0 = ds0.isel(time=0, drop=True)

    orig_nlon = ds0.sizes["lon"]
    orig_nlat = ds0.sizes["lat"]
    lon = ds0["lon"].values
    lat = ds0["lat"].values
    ds.close()

    lon2d, lat2d = np.meshgrid(lon, lat)

    ncoarse_lon = orig_nlon // coarse_factor
    idx_lon = np.arange(orig_nlon) // coarse_factor
    idx_lat = np.arange(orig_nlat) // coarse_factor
    idx_lon2d, idx_lat2d = np.meshgrid(idx_lon, idx_lat)
    grid_id = idx_lat2d * ncoarse_lon + idx_lon2d

    df = pd.DataFrame({
        "lon": lon2d.ravel(),
        "lat": lat2d.ravel(),
        "grid_id": grid_id.ravel().astype(int),
    })

    out = Path(output_parquet)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    n_coarse = ncoarse_lon * (orig_nlat // coarse_factor)
    logger.info(f"  template:      {template_nc}")
    logger.info(f"  fine grid:     {orig_nlat} x {orig_nlon} (0.05 deg)")
    logger.info(f"  coarse factor: {coarse_factor} -> {n_coarse} coarse cells")
    logger.info(f"  saved:         {out} ({len(df):,} rows)")
    return str(out)
