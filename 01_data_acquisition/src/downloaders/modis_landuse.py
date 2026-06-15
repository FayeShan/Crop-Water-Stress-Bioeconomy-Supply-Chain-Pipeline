"""
MODIS MCD12C1 Land Cover Type (IGBP classification), yearly.

Downloads one MCD12C1 HDF per year and extracts the IGBP
majority-class variable (``Majority_Land_Cover_Type_1``).
Output: ``{year}.nc`` with shape ``(lat=3600, lon=7200)`` on the
0.05° CMG grid (native, no reprojection needed).
"""

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
SHORT_NAME = "MCD12C1"
VERSION = "061"
LC_VAR = "Majority_Land_Cover_Type_1"  # IGBP classification
FILL_VALUE = 255

# Output grid: 0.05° CMG global (3600 × 7200)
OUT_LAT = np.linspace(89.975, -89.975, 3600)
OUT_LON = np.linspace(-179.975, 179.975, 7200)

# IGBP class code → name
IGBP_CLASSES = {
    0: "Water Bodies",
    1: "Evergreen Needleleaf Forests",
    2: "Evergreen Broadleaf Forests",
    3: "Deciduous Needleleaf Forests",
    4: "Deciduous Broadleaf Forests",
    5: "Mixed Forests",
    6: "Closed Shrublands",
    7: "Open Shrublands",
    8: "Woody Savannas",
    9: "Savannas",
    10: "Grasslands",
    11: "Permanent Wetlands",
    12: "Croplands",
    13: "Urban and Built-up Lands",
    14: "Cropland/Natural Vegetation Mosaics",
    15: "Permanent Snow and Ice",
    16: "Barren",
    255: "Unclassified",
}


class ModisLanduseDownloader:
    """Download MCD12C1 IGBP land cover and save as yearly NC."""

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
            raise

    def _read_lc(self, filepath: Path) -> Optional[np.ndarray]:
        """Read Majority_Land_Cover_Type_1 from one MCD12C1 HDF."""
        try:
            hdf = SD(str(filepath), SDC.READ)
            sds = hdf.select(LC_VAR)
            data = sds[:].astype(np.uint8)
            del sds, hdf
            return data
        except Exception as e:
            self.logger.warning(f"Failed to read {filepath.name}: {e}")
            return None

    def process_year(self, year: int) -> Optional[str]:
        output_path = self.nc_dir / f"{year}.nc"
        if output_path.exists():
            self.logger.info(f"  {year}: {output_path} already exists, skipping")
            return str(output_path)

        self.logger.info(f"Processing MCD12C1 IGBP for {year}")

        results = earthaccess.search_data(
            short_name=SHORT_NAME, version=VERSION,
            temporal=(f"{year}-01-01", f"{year}-12-31"),
        )
        if not results:
            self.logger.warning(f"  {year}: no granules found")
            return None

        self.logger.info(f"  {year}: downloading {len(results)} granule(s)")
        year_dir = self.raw_dir / f"{year}"
        year_dir.mkdir(parents=True, exist_ok=True)
        downloaded = earthaccess.download(results, str(year_dir))
        if not downloaded:
            self.logger.warning(f"  {year}: download failed")
            return None

        data = None
        for f in sorted(year_dir.glob("*.hdf")):
            data = self._read_lc(f)
            if data is not None:
                break
        if data is None:
            self.logger.warning(f"  {year}: no valid data")
            return None

        ds = xr.Dataset(
            {
                "LC_Type1": (
                    ["lat", "lon"],
                    data,
                    {
                        "long_name": "Land Cover Type 1 (IGBP)",
                        "units": "class",
                        "fill_value": FILL_VALUE,
                        "source": f"{SHORT_NAME}.{VERSION}",
                        "classification": "IGBP",
                        "class_names": str(IGBP_CLASSES),
                    },
                )
            },
            coords={
                "lat": ("lat", OUT_LAT, {"units": "degrees_north", "long_name": "Latitude"}),
                "lon": ("lon", OUT_LON, {"units": "degrees_east", "long_name": "Longitude"}),
            },
            attrs={
                "title": f"MCD12C1 Land Cover Type (IGBP) - {year}",
                "source": f"NASA MODIS {SHORT_NAME}.{VERSION}",
                "variable": "Majority_Land_Cover_Type_1 (IGBP)",
                "spatial_resolution": "0.05 degree CMG",
                "temporal_resolution": "yearly",
                "year": year,
                "created": datetime.now().isoformat(),
            },
        )
        encoding = {
            "LC_Type1": {
                "dtype": "uint8", "zlib": True, "complevel": 4,
                "_FillValue": np.uint8(FILL_VALUE),
            }
        }
        ds.to_netcdf(output_path, encoding=encoding)
        ds.close()
        self.logger.info(f"  {year}: saved -> {output_path}")

        if not self.keep_hdf:
            for f in year_dir.glob("*.hdf"):
                f.unlink()
            try:
                year_dir.rmdir()
            except OSError:
                pass

        return str(output_path)

    def process_range(self, start_year: int, end_year: int) -> List[str]:
        paths = []
        for year in tqdm(range(start_year, end_year + 1), desc="MODIS Landuse"):
            try:
                p = self.process_year(year)
                if p:
                    paths.append(p)
            except Exception as e:
                self.logger.error(f"  {year}: failed — {e}")
                continue
        return paths


def process_modis_landuse(
    raw_dir: str,
    nc_dir: str,
    start_year: int,
    end_year: int,
    keep_hdf: bool = False,
    log_dir: str = "./logs",
) -> List[str]:
    """End-to-end: download + save MCD12C1 IGBP land cover."""
    logger = setup_logger("modis_landuse", log_dir, "modis_landuse.log")
    dl = ModisLanduseDownloader(raw_dir, nc_dir, keep_hdf=keep_hdf, logger=logger)
    return dl.process_range(start_year, end_year)
