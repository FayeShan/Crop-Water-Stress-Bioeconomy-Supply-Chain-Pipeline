"""Build the 9-class agro-climatic-region parquet (lat, lon, regiontype).

Combines GAEZ thermal zone (thz) and moisture zone (mst) rasters into a
9-class code, reprojected onto a template grid. Joined by
``feature_engineering.py`` to provide the ``regiontype`` feature.

Region mapping (thz, mst) -> class
  thz <= 3, mst <= 2        -> 1   cold / dry
  thz <= 3, mst 3-5         -> 2   cold / moist
  thz <= 3, mst >= 6        -> 3   cold / wet
  thz 4-7, mst <= 2         -> 4   temperate / dry
  thz 4-7, mst 3-5          -> 5   temperate / moist
  thz 4-7, mst >= 6         -> 6   temperate / wet
  thz >= 8, mst <= 2        -> 7   tropical / dry
  thz >= 8, mst 3-5         -> 8   tropical / moist
  thz >= 8, mst >= 6        -> 9   tropical / wet
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr


def _to_9region(thz: np.ndarray, mst: np.ndarray) -> np.ndarray:
    reg = np.full_like(thz, np.nan, dtype=np.float32)
    reg[(thz <= 3) & (mst <= 2)] = 1
    reg[(thz <= 3) & (mst >= 3) & (mst <= 5)] = 2
    reg[(thz <= 3) & (mst >= 6)] = 3
    reg[(thz >= 4) & (thz <= 7) & (mst <= 2)] = 4
    reg[(thz >= 4) & (thz <= 7) & (mst >= 3) & (mst <= 5)] = 5
    reg[(thz >= 4) & (thz <= 7) & (mst >= 6)] = 6
    reg[(thz >= 8) & (mst <= 2)] = 7
    reg[(thz >= 8) & (mst >= 3) & (mst <= 5)] = 8
    reg[(thz >= 8) & (mst >= 6)] = 9
    return reg


def generate_region9(
    thz_tif: str,
    mst_tif: str,
    template_nc: str,
    output_parquet: str,
    logger: Optional[logging.Logger] = None,
) -> str:
    """Build the 9-class region parquet from GAEZ thz/mst rasters.

    Args:
        thz_tif: Path to GAEZ thermal zone GeoTIFF.
        mst_tif: Path to GAEZ moisture zone GeoTIFF.
        template_nc: Path to a 0.05-deg NC for the target grid (any file
            containing ``lat`` / ``lon`` coordinates works).
        output_parquet: Where to save ``lat, lon, regiontype`` parquet.

    Returns:
        Path to the saved parquet.
    """
    logger = logger or logging.getLogger(__name__)

    logger.info(f"  loading thz: {thz_tif}")
    thz = rxr.open_rasterio(thz_tif, masked=True).squeeze()
    logger.info(f"  loading mst: {mst_tif}")
    mst = rxr.open_rasterio(mst_tif, masked=True).squeeze()

    region9 = _to_9region(thz.values, mst.values)
    region9_da = xr.DataArray(
        region9,
        coords=thz.coords,
        dims=thz.dims,
        name="Region_class",
    ).rio.write_crs("EPSG:4326")

    logger.info(f"  loading template: {template_nc}")
    ds = xr.open_dataset(template_nc)
    dummy = xr.DataArray(
        np.zeros((len(ds["lat"]), len(ds["lon"]))),
        coords={"lat": ds["lat"], "lon": ds["lon"]},
        dims=["lat", "lon"],
    )
    dummy.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    dummy.rio.write_crs("EPSG:4326", inplace=True)
    ds.close()

    logger.info("  reprojecting onto target grid...")
    region9_aligned = region9_da.rio.reproject_match(dummy)
    region9_aligned = region9_aligned.rename({"x": "lon", "y": "lat"})
    region9_aligned = region9_aligned.drop_vars(["band", "spatial_ref"], errors="ignore")

    df = region9_aligned.stack(z=("lat", "lon")).to_series().reset_index()
    df.columns = ["lat", "lon", "regiontype"]
    df = df.dropna()

    out = Path(output_parquet)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    logger.info(f"  saved: {out} ({len(df):,} rows)")
    return str(out)
