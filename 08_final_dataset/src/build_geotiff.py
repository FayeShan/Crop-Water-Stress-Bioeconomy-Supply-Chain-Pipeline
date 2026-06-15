"""Write GeoTIFFs (monthly + yearly, 27 bands) from the monthly NetCDF files."""

import gc
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from tqdm import tqdm

from src.constants import GEOTIFF_BAND_ORDER, GEOTIFF_TRANSFORM


def _profile():
    return {
        "driver": "GTiff", "dtype": "float32", "crs": "EPSG:4326",
        "transform": GEOTIFF_TRANSFORM, "nodata": np.nan,
        "compress": "lzw", "interleave": "pixel",
    }


def save_monthly_geotiff(year, nc_dir, out_dir):
    ds = xr.open_dataset(Path(nc_dir) / f"water_stress_{year}.nc")
    n_lat, n_lon = len(ds["lat"]), len(ds["lon"])
    for month in range(1, 13):
        out_path = Path(out_dir) / f"water_stress_{year}_{month:02d}.tif"
        with rasterio.open(out_path, "w", height=n_lat, width=n_lon,
                           count=len(GEOTIFF_BAND_ORDER), **_profile()) as dst:
            for band_idx, crop in enumerate(GEOTIFF_BAND_ORDER, start=1):
                dst.write(ds[crop].isel(time=month - 1).values, band_idx)
                dst.set_band_description(band_idx, crop)
    ds.close()


def save_yearly_geotiff(year, nc_dir, out_dir):
    ds = xr.open_dataset(Path(nc_dir) / f"water_stress_{year}.nc")
    n_lat, n_lon = len(ds["lat"]), len(ds["lon"])
    out_path = Path(out_dir) / f"water_stress_{year}.tif"
    with rasterio.open(out_path, "w", height=n_lat, width=n_lon,
                       count=len(GEOTIFF_BAND_ORDER), **_profile()) as dst:
        for band_idx, crop in enumerate(GEOTIFF_BAND_ORDER, start=1):
            dst.write(ds[crop].sum(dim="time", skipna=True).values, band_idx)
            dst.set_band_description(band_idx, crop)
    ds.close()


def build_geotiff(nc_dir, monthly_out, yearly_out, year_start, year_end, logger):
    Path(monthly_out).mkdir(parents=True, exist_ok=True)
    Path(yearly_out).mkdir(parents=True, exist_ok=True)

    for year in tqdm(range(year_start, year_end + 1), desc="GeoTIFF"):
        nc = Path(nc_dir) / f"water_stress_{year}.nc"
        if not nc.exists():
            logger.warning(f"  {nc} not found, skipping {year}")
            continue
        save_monthly_geotiff(year, nc_dir, monthly_out)
        save_yearly_geotiff(year, nc_dir, yearly_out)
        gc.collect()
    logger.info("Done building GeoTIFF.")
