"""
MODIS MOD09CMG Green reflectance (Band 4) — monthly mean from daily data.

Downloads MOD09CMG daily HDF files, extracts Band 4 (Green, 545-565 nm),
and aggregates to monthly mean. Output: ``{year}.nc`` with shape
``(time=12, lat=3600, lon=7200)`` on the 0.05° CMG grid.
"""

import calendar
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import earthaccess
import numpy as np
import xarray as xr
from pyhdf.SD import SD, SDC
from tqdm import tqdm

from src.utils import setup_logger


# Product-intrinsic constants
SHORT_NAME = "MOD09CMG"
VERSION = "061"
GREEN_VAR = "Coarse Resolution Surface Reflectance Band 4"  # 545-565 nm
FILL_VALUE = -28672
SCALE_FACTOR = 0.0001

# Output grid: 0.05° CMG (3600 × 7200)
OUT_LAT = np.linspace(89.975, -89.975, 3600)
OUT_LON = np.linspace(-179.975, 179.975, 7200)


class ModisGreenDownloader:
    """Download MOD09CMG daily HDFs via NASA Earthaccess."""

    def __init__(
        self,
        raw_dir: str,
        nc_dir: str,
        keep_hdf: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        self.raw_dir = Path(raw_dir)
        self.nc_dir = Path(nc_dir)
        self.keep_hdf = keep_hdf
        self.logger = logger or logging.getLogger(__name__)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.nc_dir.mkdir(parents=True, exist_ok=True)
        self._login()

    def _login(self):
        try:
            self.auth = earthaccess.login()
            self.logger.info("Logged in to NASA Earthaccess")
        except Exception as e:
            self.logger.error(f"Earthaccess login failed: {e}")
            self.logger.info("Run 'earthaccess.login()' interactively once")
            raise

    def _read_green(self, filepath: Path) -> Optional[np.ndarray]:
        """Read Band 4 from one MOD09CMG HDF. Return float32 (3600, 7200) or None."""
        try:
            hdf = SD(str(filepath), SDC.READ)
            sds = hdf.select(GREEN_VAR)
            data = sds[:].astype(np.float32)
            del sds, hdf

            data[data == FILL_VALUE] = np.nan
            data[data < -100] = np.nan
            data[data > 16000] = np.nan
            data = data * SCALE_FACTOR
            return data
        except Exception as e:
            self.logger.warning(f"Failed to read {filepath.name}: {e}")
            return None

    def _download_month(self, year: int, month: int) -> Optional[np.ndarray]:
        """Download and aggregate one (year, month). Returns monthly mean (3600, 7200)."""
        n_days = calendar.monthrange(year, month)[1]
        start = f"{year}-{month:02d}-01"
        end = f"{year}-{month:02d}-{n_days:02d}"

        self.logger.info(f"  {year}-{month:02d}: searching granules ({start} - {end})")
        results = earthaccess.search_data(
            short_name=SHORT_NAME, version=VERSION, temporal=(start, end)
        )
        if not results:
            self.logger.warning(f"  {year}-{month:02d}: no granules found")
            return None

        month_dir = self.raw_dir / f"{year}" / f"{month:02d}"
        month_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"  {year}-{month:02d}: downloading {len(results)} granules")
        downloaded = earthaccess.download(results, str(month_dir))
        if not downloaded:
            self.logger.warning(f"  {year}-{month:02d}: download failed")
            return None

        daily_arrays = []
        for f in sorted(month_dir.glob("*.hdf")):
            arr = self._read_green(f)
            if arr is not None:
                daily_arrays.append(arr)

        if not daily_arrays:
            self.logger.warning(f"  {year}-{month:02d}: no valid daily data")
            return None

        monthly_mean = np.nanmean(np.stack(daily_arrays, axis=0), axis=0).astype(np.float32)
        self.logger.info(
            f"  {year}-{month:02d}: monthly mean from {len(daily_arrays)} daily files"
        )

        if not self.keep_hdf:
            for f in month_dir.glob("*.hdf"):
                f.unlink()
            try:
                month_dir.rmdir()
                (self.raw_dir / f"{year}").rmdir()
            except OSError:
                pass

        return monthly_mean

    def process_year(self, year: int) -> Optional[str]:
        """Download + aggregate all 12 months of one year, save to {nc_dir}/{year}.nc."""
        output_path = self.nc_dir / f"{year}.nc"
        if output_path.exists():
            self.logger.info(f"  {year}: {output_path} already exists, skipping")
            return str(output_path)

        self.logger.info(f"Processing MOD09CMG Green reflectance for {year}")

        now = datetime.now()
        start_month = 2 if year == 2000 else 1  # product starts Feb 2000
        if year > now.year:
            self.logger.warning(f"  {year}: future year, skipping")
            return None
        end_month = now.month - 1 if year == now.year else 12

        monthly_data = []
        for month in range(1, 13):
            if month < start_month or month > end_month:
                monthly_data.append(
                    np.full((3600, 7200), np.nan, dtype=np.float32)
                )
                continue
            result = self._download_month(year, month)
            monthly_data.append(
                result if result is not None
                else np.full((3600, 7200), np.nan, dtype=np.float32)
            )

        data_stack = np.stack(monthly_data, axis=0)
        time_coords = np.array(
            [f"{year}-{m:02d}-01" for m in range(1, 13)], dtype="datetime64[ns]"
        )

        ds = xr.Dataset(
            {
                "Green_reflectance": (
                    ["time", "lat", "lon"],
                    data_stack,
                    {
                        "long_name": "Monthly Mean Green Reflectance (Band 4)",
                        "units": "reflectance",
                        "scale_factor_applied": SCALE_FACTOR,
                        "source": f"{SHORT_NAME}.{VERSION} Band 4 (545-565nm)",
                        "aggregation": "monthly_mean",
                    },
                )
            },
            coords={
                "time": time_coords,
                "lat": ("lat", OUT_LAT, {"units": "degrees_north", "long_name": "Latitude"}),
                "lon": ("lon", OUT_LON, {"units": "degrees_east", "long_name": "Longitude"}),
            },
            attrs={
                "title": f"MOD09CMG Green Reflectance Monthly Mean - {year}",
                "source": f"NASA MODIS {SHORT_NAME}.{VERSION}",
                "variable": "Band 4 Green Reflectance (545-565 nm)",
                "spatial_resolution": "0.05 degree CMG",
                "temporal_resolution": "monthly mean from daily data",
                "created": datetime.now().isoformat(),
            },
        )

        encoding = {
            "Green_reflectance": {
                "dtype": "float32",
                "zlib": True,
                "complevel": 4,
                "_FillValue": np.nan,
            }
        }
        ds.to_netcdf(output_path, encoding=encoding)
        ds.close()
        self.logger.info(f"  {year}: saved -> {output_path}")
        return str(output_path)

    def process_range(self, start_year: int, end_year: int) -> List[str]:
        paths = []
        for year in tqdm(range(start_year, end_year + 1), desc="MODIS Green"):
            try:
                p = self.process_year(year)
                if p:
                    paths.append(p)
            except Exception as e:
                self.logger.error(f"  {year}: failed — {e}")
                continue
        return paths


def process_modis_green(
    raw_dir: str,
    nc_dir: str,
    start_year: int,
    end_year: int,
    keep_hdf: bool = False,
    log_dir: str = "./logs",
) -> List[str]:
    """End-to-end: download + aggregate MOD09CMG Green reflectance to yearly NC files."""
    logger = setup_logger("modis_green", log_dir, "modis_green.log")
    dl = ModisGreenDownloader(raw_dir, nc_dir, keep_hdf=keep_hdf, logger=logger)
    return dl.process_range(start_year, end_year)
