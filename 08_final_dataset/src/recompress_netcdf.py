"""Recompress the dataset NetCDF files in place with zlib (resumable)."""

from pathlib import Path

import xarray as xr
from tqdm import tqdm


def _is_compressed(nc_path, skip_threshold_mb):
    return nc_path.stat().st_size / 1024 ** 2 < skip_threshold_mb


def recompress_one(nc_path, complevel, logger):
    tmp_path = nc_path.with_suffix(".nc.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    old_mb = nc_path.stat().st_size / 1024 ** 2
    ds = xr.open_dataset(nc_path)
    encoding = {v: {"zlib": True, "complevel": complevel} for v in ds.data_vars}
    ds.to_netcdf(tmp_path, engine="netcdf4", encoding=encoding)
    ds.close()

    try:
        xr.open_dataset(tmp_path).close()
    except Exception as e:
        logger.warning(f"  verify failed for {tmp_path.name}: {e}")
        tmp_path.unlink()
        return

    tmp_path.replace(nc_path)
    new_mb = nc_path.stat().st_size / 1024 ** 2
    logger.info(f"  {nc_path.name}: {old_mb:.1f} MB -> {new_mb:.1f} MB")


def recompress_netcdf(monthly_dir, yearly_dir, complevel, skip_threshold_mb, logger):
    files = sorted(Path(monthly_dir).glob("*.nc")) + sorted(Path(yearly_dir).glob("*.nc"))
    todo = [f for f in files if not _is_compressed(f, skip_threshold_mb)]
    logger.info(f"  {len(files)} files, {len(todo)} to recompress (complevel={complevel})")
    for nc in tqdm(todo, desc="Recompressing"):
        recompress_one(nc, complevel, logger)
    logger.info("Done recompressing.")
