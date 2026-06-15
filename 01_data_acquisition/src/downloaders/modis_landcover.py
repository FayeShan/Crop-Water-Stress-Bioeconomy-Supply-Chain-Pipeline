"""
MODIS MCD12Q2 Land Cover Dynamics (phenology), yearly.

Downloads all 500m sinusoidal tiles for a year, reprojects each tile onto
a global 0.05° regular grid via nearest-neighbour index mapping, and
mosaics all tiles into a single NetCDF per year.

Output: ``{year}.nc`` with shape ``(lat=3600, lon=7200)`` and 25 variables:
    12 phenology variables × 2 modes  (Num_Modes_01 / Num_Modes_02)
    + NumCycles (no mode suffix)
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import earthaccess
import numpy as np
import xarray as xr
from pyhdf.SD import SD, SDC
from tqdm import tqdm

from src.utils import setup_logger


# Product-intrinsic constants
SHORT_NAME = "MCD12Q2"
VERSION = "061"
FILL_VALUE = 32767

# Variables with (2400, 2400, 2) shape — 3rd dim is phenology mode 1/2
PHENO_VARS = [
    "Greenup",
    "MidGreenup",
    "Maturity",
    "Peak",
    "Senescence",
    "MidGreendown",
    "Dormancy",
    "EVI_Minimum",
    "EVI_Amplitude",
    "EVI_Area",
    "QA_Detailed",
    "QA_Overall",
]

# Variables with (2400, 2400) shape
OTHER_VARS = ["NumCycles"]

# MODIS sinusoidal tile grid constants
EARTH_R = 6371007.181  # metres
TILE_SIZE = 1111950.0  # ~10° tile in metres
GLOBAL_XMIN = -20015109.354
GLOBAL_YMAX = 10007554.677
NPIX = 2400  # pixels per tile (500 m product)
PIXEL_SIZE = TILE_SIZE / NPIX

# Output grid: 0.05° global
OUT_LAT = np.linspace(89.975, -89.975, 3600)
OUT_LON = np.linspace(-179.975, 179.975, 7200)


# Variable metadata
_VAR_META = {
    "Greenup":       ("Date of onset greenness increase", "day_of_year"),
    "MidGreenup":    ("Date of mid-point greenness increase", "day_of_year"),
    "Maturity":      ("Date of onset greenness maximum", "day_of_year"),
    "Peak":          ("Date of mid-point greenness maximum", "day_of_year"),
    "Senescence":    ("Date of onset greenness decrease", "day_of_year"),
    "MidGreendown":  ("Date of mid-point greenness decrease", "day_of_year"),
    "Dormancy":      ("Date of onset greenness minimum", "day_of_year"),
    "EVI_Minimum":   ("Segment minimum EVI2 value", "EVI2 (scale=0.0001)"),
    "EVI_Amplitude": ("Segment max minus min EVI2", "EVI2 (scale=0.0001)"),
    "EVI_Area":      ("Sum of daily interpolated EVI2 Greenup to Dormancy",
                      "EVI2 (scale=0.1)"),
    "QA_Detailed":   ("Bit-packed SDS-specific QA codes", "bit_field"),
    "QA_Overall":    ("QA code for entire segment", "class"),
    "NumCycles":     ("Number of valid vegetation cycles", "count"),
}


def _var_attrs(var_name: str):
    """Return (long_name, units) for one variable name (with or without mode suffix)."""
    if "_Num_Modes_" in var_name:
        base, mode = var_name.rsplit("_Num_Modes_", 1)
        suffix = f" (Num_Modes_{mode})"
    else:
        base, suffix = var_name, ""
    long_name, units = _VAR_META.get(base, (base, "unknown"))
    return long_name + suffix, units


def _parse_tile_id(filename: str) -> Optional[str]:
    """Extract tile ID (e.g. 'h08v05') from an MCD12Q2 filename."""
    for part in filename.split("."):
        if part.startswith("h") and "v" in part and len(part) == 6:
            return part
    return None


def _compute_grid_indices(tile_id: str):
    """Map each (2400, 2400) tile pixel to an index in the 0.05° global grid."""
    h = int(tile_id[1:3])
    v = int(tile_id[4:6])

    xmin_tile = GLOBAL_XMIN + h * TILE_SIZE
    ymax_tile = GLOBAL_YMAX - v * TILE_SIZE

    x_1d = np.arange(NPIX) * PIXEL_SIZE + xmin_tile + PIXEL_SIZE / 2.0
    y_1d = ymax_tile - np.arange(NPIX) * PIXEL_SIZE - PIXEL_SIZE / 2.0
    xx, yy = np.meshgrid(x_1d, y_1d)

    # Inverse sinusoidal projection → lat, lon
    lat_rad = yy / EARTH_R
    lat_deg = np.degrees(lat_rad)
    cos_lat = np.cos(lat_rad)
    cos_lat = np.where(np.abs(cos_lat) < 1e-10, 1e-10, cos_lat)
    lon_deg = np.degrees(xx / (EARTH_R * cos_lat))
    lon_deg = np.clip(lon_deg, -180.0, 180.0)

    lat_idx = np.round((89.975 - lat_deg) / 0.05).astype(np.int32)
    lon_idx = np.round((lon_deg + 179.975) / 0.05).astype(np.int32)
    lat_idx = np.clip(lat_idx, 0, 3599)
    lon_idx = np.clip(lon_idx, 0, 7199)
    return lat_idx, lon_idx


class ModisLandcoverDownloader:
    """Download MCD12Q2 tiles and mosaic to global 0.05° grid."""

    def __init__(
        self,
        raw_dir: str,
        nc_dir: str,
        keep_hdf: bool = False,
        tiles: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.raw_dir = Path(raw_dir)
        self.nc_dir = Path(nc_dir)
        self.keep_hdf = keep_hdf
        self.tiles = tiles  # optional filter
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

    def _read_tile(self, filepath: Path) -> Optional[Dict[str, np.ndarray]]:
        """Read all target variables from one MCD12Q2 tile. PHENO_VARS split into _01/_02."""
        try:
            hdf = SD(str(filepath), SDC.READ)
            datasets = hdf.datasets()
            data_dict: Dict[str, np.ndarray] = {}

            for var in PHENO_VARS:
                if var in datasets:
                    sds = hdf.select(var)
                    arr = sds[:].astype(np.int16)  # (2400, 2400, 2)
                    del sds
                    data_dict[f"{var}_Num_Modes_01"] = arr[:, :, 0]
                    data_dict[f"{var}_Num_Modes_02"] = arr[:, :, 1]

            for var in OTHER_VARS:
                if var in datasets:
                    sds = hdf.select(var)
                    data_dict[var] = sds[:].astype(np.int16)  # (2400, 2400)
                    del sds

            del hdf
            return data_dict
        except Exception as e:
            self.logger.warning(f"Failed to read {filepath.name}: {e}")
            return None

    def process_year(self, year: int) -> Optional[str]:
        output_path = self.nc_dir / f"{year}.nc"
        if output_path.exists():
            self.logger.info(f"  {year}: {output_path} already exists, skipping")
            return str(output_path)

        self.logger.info(f"Processing MCD12Q2 phenology for {year}")

        # 1) Search + download tiles
        results = earthaccess.search_data(
            short_name=SHORT_NAME, version=VERSION,
            temporal=(f"{year}-01-01", f"{year}-12-31"),
        )
        if not results:
            self.logger.warning(f"  {year}: no granules found")
            return None
        self.logger.info(f"  {year}: found {len(results)} granules")

        if self.tiles is not None:
            filtered = []
            for r in results:
                links = r.data_links()
                name = os.path.basename(links[0]) if links else ""
                if _parse_tile_id(name) in self.tiles:
                    filtered.append(r)
            results = filtered
            self.logger.info(f"  {year}: {len(results)} granules after tile filter")
            if not results:
                return None

        year_dir = self.raw_dir / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        downloaded = earthaccess.download(results, str(year_dir))
        if not downloaded:
            self.logger.warning(f"  {year}: download failed")
            return None

        # 2) Read + mosaic
        hdf_files = sorted(year_dir.glob("*.hdf"))
        if not hdf_files:
            self.logger.warning(f"  {year}: no HDF files in {year_dir}")
            return None

        # Discover variable names from first readable tile
        sample = None
        for f in hdf_files:
            sample = self._read_tile(f)
            if sample is not None:
                break
        if sample is None:
            self.logger.warning(f"  {year}: cannot read any tile")
            return None

        var_names = list(sample.keys())
        global_grids = {
            v: np.full((3600, 7200), FILL_VALUE, dtype=np.int16) for v in var_names
        }
        index_cache: Dict[str, tuple] = {}

        for i, fpath in enumerate(hdf_files):
            tile_id = _parse_tile_id(fpath.name)
            if tile_id is None:
                continue
            if (i + 1) % 50 == 0 or i == 0:
                self.logger.info(
                    f"  {year}: mosaicking tile {i+1}/{len(hdf_files)} ({tile_id})"
                )
            if tile_id not in index_cache:
                index_cache[tile_id] = _compute_grid_indices(tile_id)
            lat_idx, lon_idx = index_cache[tile_id]

            data_dict = sample if i == 0 else self._read_tile(fpath)
            if data_dict is None:
                continue
            for vname, arr in data_dict.items():
                if vname not in global_grids:
                    continue
                valid = arr != FILL_VALUE
                if np.any(valid):
                    global_grids[vname][lat_idx[valid], lon_idx[valid]] = arr[valid]

        # 3) Build dataset + save
        data_vars = {}
        encoding = {}
        for vname in var_names:
            long_name, units = _var_attrs(vname)
            data_vars[vname] = (
                ["lat", "lon"],
                global_grids[vname],
                {"long_name": long_name, "units": units, "original_name": vname},
            )
            encoding[vname] = {
                "dtype": "int16", "zlib": True, "complevel": 4,
                "_FillValue": np.int16(FILL_VALUE),
            }

        ds = xr.Dataset(
            data_vars,
            coords={
                "lat": ("lat", OUT_LAT, {"units": "degrees_north", "long_name": "Latitude"}),
                "lon": ("lon", OUT_LON, {"units": "degrees_east", "long_name": "Longitude"}),
            },
            attrs={
                "title": f"MCD12Q2 Land Cover Dynamics - {year}",
                "source": f"NASA MODIS {SHORT_NAME}.{VERSION}",
                "product": "Land Cover Dynamics Yearly L3 Global 500m SIN Grid",
                "spatial_resolution": "0.05 degree (reprojected from 500m sinusoidal)",
                "temporal_resolution": "yearly",
                "year": year,
                "projection": "WGS84 geographic (EPSG:4326)",
                "reproject_method": "nearest neighbor (500m -> 0.05deg)",
                "created": datetime.now().isoformat(),
            },
        )
        ds.to_netcdf(output_path, encoding=encoding)
        file_mb = os.path.getsize(output_path) / 1e6
        self.logger.info(f"  {year}: saved -> {output_path} ({file_mb:.1f} MB)")
        ds.close()
        del global_grids, ds

        if not self.keep_hdf:
            for f in year_dir.glob("*.hdf"):
                f.unlink()
            for f in year_dir.glob("*.hdf.xml"):
                f.unlink()
            try:
                year_dir.rmdir()
            except OSError:
                pass

        return str(output_path)

    def process_range(self, start_year: int, end_year: int) -> List[str]:
        paths = []
        for year in tqdm(range(start_year, end_year + 1), desc="MODIS Landcover"):
            try:
                p = self.process_year(year)
                if p:
                    paths.append(p)
            except Exception as e:
                self.logger.error(f"  {year}: failed — {e}")
                import traceback; traceback.print_exc()
                continue
        return paths


def process_modis_landcover(
    raw_dir: str,
    nc_dir: str,
    start_year: int,
    end_year: int,
    keep_hdf: bool = False,
    tiles: Optional[List[str]] = None,
    log_dir: str = "./logs",
) -> List[str]:
    """End-to-end: download + mosaic MCD12Q2 phenology."""
    logger = setup_logger("modis_landcover", log_dir, "modis_landcover.log")
    dl = ModisLandcoverDownloader(
        raw_dir, nc_dir, keep_hdf=keep_hdf, tiles=tiles, logger=logger
    )
    return dl.process_range(start_year, end_year)
