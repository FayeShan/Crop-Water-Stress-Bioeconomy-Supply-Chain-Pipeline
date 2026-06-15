"""Workaround for the ECMWF bug affecting ERA5-Land monthly accumulated fields
(ssr, ssrd, tp, ro) for Sep 2022 - Feb 2024, returned at ~50% of correct values.

Fix: re-download the affected variables via the ``monthly_averaged_reanalysis_by_hour_of_day``
product at 00:00 UTC (correct values) and overwrite them in the existing processed
monthly NC files in place; instantaneous variables are left unchanged. Idempotent.

See: https://forum.ecmwf.int/t/issue-affecting-era5-land-monthly-averaged-
     reanalysis-for-the-period-september-2022-to-february-2024/2370
"""

import gc
import os
import tempfile
from pathlib import Path
from typing import List, Optional

import cdsapi
import logging
import xarray as xr
from tqdm import tqdm

from src.utils import (
    setup_logger,
    get_netcdf_encoding,
    standardize_coordinates,
    validate_era5_data,
)

# The accumulated flux variables returned by the by-hour-of-day product
# (evabs and evavt are not available in this product; they are left as-is.)
DEFAULT_ACCUM_VARIABLES = [
    "surface_net_solar_radiation",
    "surface_solar_radiation_downwards",
    "total_precipitation",
    "runoff",
]


class ERA5FixDownloader:
    """Download corrected accumulated variables via the by-hour-of-day product."""

    def __init__(
        self,
        output_dir: str,
        variables: Optional[List[str]] = None,
        months: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.output_dir = output_dir
        self.variables = variables or DEFAULT_ACCUM_VARIABLES
        self.months = months or [f"{m:02d}" for m in range(1, 13)]
        self.logger = logger or logging.getLogger(__name__)
        self.client = cdsapi.Client()
        os.makedirs(output_dir, exist_ok=True)

    def download_year(self, year: int, force: bool = False) -> Optional[str]:
        """Download corrected accumulated variables for one year."""
        filename = os.path.join(self.output_dir, f"{year}_fixed.nc")
        if os.path.exists(filename) and not force:
            self.logger.info(f"  {year}: corrected file already exists, skipping")
            return filename

        request = {
            "product_type": "monthly_averaged_reanalysis_by_hour_of_day",
            "variable": self.variables,
            "year": str(year),
            "month": self.months,
            "time": ["00:00"],
            "data_format": "netcdf",
            "download_format": "unarchived",
        }

        self.logger.info(
            f"  {year}: downloading {len(self.variables)} accumulated variables "
            f"(product: monthly_averaged_reanalysis_by_hour_of_day, time: 00:00)"
        )
        try:
            self.client.retrieve(
                "reanalysis-era5-land-monthly-means", request
            ).download(filename)
            size_mb = os.path.getsize(filename) / (1024 * 1024)
            self.logger.info(f"  {year}: saved -> {filename} ({size_mb:.0f} MB)")
            return filename
        except Exception as e:
            self.logger.error(f"  {year}: download failed -- {e}")
            return None

    def download_range(
        self, start_year: int, end_year: int, force: bool = False
    ) -> List[str]:
        """Download corrected data for a range of years."""
        files = []
        years = list(range(start_year, end_year + 1))
        for year in tqdm(years, desc="Downloading corrected ERA5"):
            path = self.download_year(year, force=force)
            if path:
                files.append(path)
        return files


def patch_processed_files(
    processed_dir: str,
    fixed_dir: str,
    start_year: int,
    end_year: int,
    logger: Optional[logging.Logger] = None,
) -> List[str]:
    """Rewrite each ``processed_dir/{year}.nc`` in place, replacing the affected
    accumulated variables with values from ``fixed_dir/{year}_fixed.nc`` and
    preserving all other variables, coordinates, and attributes.
    """
    logger = logger or logging.getLogger(__name__)
    processed_dir = Path(processed_dir)
    fixed_dir = Path(fixed_dir)

    patched_files = []

    for year in range(start_year, end_year + 1):
        existing_path = processed_dir / f"{year}.nc"
        fixed_path = fixed_dir / f"{year}_fixed.nc"

        if not existing_path.exists():
            logger.warning(
                f"  {year}: processed file not found ({existing_path}) — "
                f"run era5-download and era5-unzip first"
            )
            continue
        if not fixed_path.exists():
            logger.warning(f"  {year}: corrected file not found: {fixed_path}")
            continue

        logger.info(f"  {year}: patching in place ...")

        ds_existing = xr.open_dataset(existing_path).load()
        ds_existing = standardize_coordinates(ds_existing)

        ds_fixed = xr.open_dataset(fixed_path).load()
        ds_fixed = standardize_coordinates(ds_fixed)

        # Drop expver / number from fixed data if present
        for v in ["expver", "number"]:
            if v in ds_fixed.dims:
                ds_fixed = ds_fixed.isel({v: 0}).drop_vars(v, errors="ignore")
            elif v in ds_fixed.coords or v in ds_fixed.data_vars:
                ds_fixed = ds_fixed.drop_vars(v, errors="ignore")

        # Align time coordinates to match existing file
        ds_fixed["time"] = ds_existing["time"]

        replaced = []
        for var in ds_fixed.data_vars:
            if var not in ds_existing.data_vars:
                continue

            if ds_fixed[var].shape != ds_existing[var].shape:
                logger.info(f"    {var}: interpolating onto existing grid")
                ds_fixed[var] = ds_fixed[var].interp(
                    lat=ds_existing.lat, lon=ds_existing.lon, method="linear"
                )

            old_mean = float(ds_existing[var].mean())
            new_mean = float(ds_fixed[var].mean())
            ratio = new_mean / old_mean if abs(old_mean) > 1e-15 else float("nan")

            ds_existing[var] = ds_fixed[var]
            replaced.append(var)
            logger.info(
                f"    {var}: old_mean={old_mean:.4g} -> new_mean={new_mean:.4g} "
                f"(ratio={ratio:.2f})"
            )

        logger.info(f"    Replaced {len(replaced)} variables: {replaced}")

        validate_era5_data(ds_existing, year, logger)

        ds_existing_close = ds_existing  # keep ref for closing after write
        ds_fixed.close()

        # Rewrite in place via a temporary file (safe atomic overwrite)
        encoding = get_netcdf_encoding(list(ds_existing.data_vars))
        with tempfile.NamedTemporaryFile(
            dir=str(processed_dir), suffix=".nc.tmp", delete=False
        ) as tmp:
            tmp_path = tmp.name
        try:
            ds_existing.to_netcdf(tmp_path, encoding=encoding)
            ds_existing_close.close()
            os.replace(tmp_path, existing_path)
            logger.info(f"  {year}: rewritten in place -> {existing_path}")
            patched_files.append(str(existing_path))
        except Exception as e:
            logger.error(f"  {year}: failed to rewrite ({e})")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        del ds_existing, ds_fixed
        gc.collect()

    return patched_files


def fix_era5_accumulated(
    processed_dir: str,
    start_year: int,
    end_year: int,
    fixed_download_dir: Optional[str] = None,
    variables: Optional[List[str]] = None,
    log_dir: str = "./logs",
    force: bool = False,
) -> List[str]:
    """End-to-end ERA5 accumulated-variable fix: download corrected data, then
    overwrite the affected variables in the processed monthly NC files in place.

    The effective year range is clipped to the corruption window (2022-2024);
    a wider range is safe and is a no-op for years outside that window.
    """
    logger = setup_logger("era5_fix", log_dir, "era5_fix.log")

    # Clip to the ECMWF-documented corruption window (2022-2024)
    effective_start = max(start_year, 2022)
    effective_end = min(end_year, 2024)

    if effective_start > effective_end:
        logger.info(
            f"  Year range [{start_year}-{end_year}] is outside the 2022-2024 "
            f"corruption window; nothing to fix."
        )
        return []

    processed_dir = Path(processed_dir)
    if fixed_download_dir is None:
        fixed_download_dir = str(processed_dir.parent / "_fixed_tmp")
    os.makedirs(fixed_download_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  ERA5 Accumulated-Variable Fix")
    logger.info("=" * 60)
    logger.info(f"  Year range requested: {start_year}-{end_year}")
    logger.info(f"  Effective year range: {effective_start}-{effective_end}")
    logger.info(f"  Product: monthly_averaged_reanalysis_by_hour_of_day @ 00:00")
    logger.info(f"  Processed dir (input & output): {processed_dir}")
    logger.info(f"  Temp download dir: {fixed_download_dir}")

    logger.info("\n--- Step 1: downloading corrected data ---")
    dl = ERA5FixDownloader(fixed_download_dir, variables=variables, logger=logger)
    dl.download_range(effective_start, effective_end, force=force)

    logger.info("\n--- Step 2: rewriting processed files in place ---")
    patched = patch_processed_files(
        str(processed_dir), fixed_download_dir,
        effective_start, effective_end, logger,
    )

    logger.info(f"\nDone. {len(patched)} processed files rewritten.")
    return patched
