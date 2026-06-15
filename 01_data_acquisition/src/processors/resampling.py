"""Spatial resampling of ERA5, MODIS, SMIA, and TWSA data to a common reference grid."""

import os
import xarray as xr
import numpy as np
import logging
from typing import Optional, List, Tuple
from tqdm import tqdm

from src.utils import (
    setup_logger,
    get_netcdf_encoding,
    standardize_coordinates,
    normalize_longitude,
    convert_to_float32,
    validate_era5_data
)


class DataResampler:
    """Generic data resampler for NetCDF files."""
    
    def __init__(
        self,
        reference_grid_path: str,
        method: str = "linear",
        logger: Optional[logging.Logger] = None
    ):
        """Initialize data resampler (method: linear, nearest, etc.)."""
        self.reference_grid_path = reference_grid_path
        self.method = method
        self.logger = logger or logging.getLogger(__name__)

        self._load_reference_grid()
    
    def _load_reference_grid(self):
        """Load target coordinates from reference grid."""
        from src.utils import get_reference_grid
    
        self.target_lon, self.target_lat = get_reference_grid(self.reference_grid_path)
        
        self.logger.info(
            f"Loaded reference grid: {len(self.target_lon)} lon x {len(self.target_lat)} lat"
        )
    
    def resample_file(
        self,
        input_path: str,
        output_path: str,
        preprocess_era5: bool = False
    ) -> bool:
        """Resample a single NetCDF file to the reference grid; returns success flag."""
        try:
            self.logger.info(f"Loading: {input_path}")
            ds = xr.open_dataset(input_path)

            ds = standardize_coordinates(ds)

            if preprocess_era5:
                ds = self._preprocess_era5(ds)

            ds = normalize_longitude(ds)

            self.logger.info("Interpolating to target resolution...")
            ds_interp = ds.interp(
                lon=self.target_lon,
                lat=self.target_lat,
                method=self.method
            )

            ds_interp = convert_to_float32(ds_interp)

            encoding = get_netcdf_encoding(list(ds_interp.data_vars))
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            self.logger.info(f"Saving to {output_path}...")
            ds_interp.to_netcdf(output_path, encoding=encoding)
            
            self.logger.info(f"Done: {output_path} with shape {ds_interp.dims}")
            
            ds.close()
            ds_interp.close()
            del ds, ds_interp
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error resampling {input_path}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _preprocess_era5(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Apply ERA5-specific preprocessing.

        Handles the expver dimension that appears when CDS returns data
        containing both ERA5 final (expver=1) and ERA5T near-real-time
        (expver=5) data for overlapping months. For each month, only one
        expver has valid data; the other is NaN. We merge by taking the
        first non-NaN value (preferring ERA5 final over ERA5T).
        """
        # Handle expver dimension (ERA5 vs ERA5T overlap)
        if 'expver' in ds.dims:
            n_expver = ds.dims['expver']
            self.logger.info(
                f"Found expver as dimension with {n_expver} values — "
                f"merging ERA5 final and ERA5T slices..."
            )
            # Prefer ERA5 final (expver=1), fill gaps with ERA5T (expver=5)
            expver_values = ds.expver.values
            ds_primary = ds.sel(expver=expver_values[0])
            ds_secondary = ds.sel(expver=expver_values[-1])
            ds = ds_primary.combine_first(ds_secondary)
            # Drop expver if it remains as a scalar coordinate
            if 'expver' in ds.coords:
                ds = ds.drop_vars('expver')
            self.logger.info("expver dimension merged successfully")
        elif 'expver' in ds.data_vars or 'expver' in ds.coords:
            self.logger.info("Dropping scalar expver coordinate")
            ds = ds.drop_vars('expver')

        # Drop 'number' dimension/coordinate if present
        if 'number' in ds.dims:
            ds = ds.isel(number=0).drop_vars('number', errors='ignore')
        elif 'number' in ds.data_vars or 'number' in ds.coords:
            ds = ds.drop_vars('number')

        return ds


class ERA5Resampler(DataResampler):
    """ERA5-specific resampler."""
    
    def resample_year(
        self,
        input_dir: str,
        output_dir: str,
        year: int
    ) -> Optional[str]:
        """Resample ERA5 data for a specific year."""
        input_path = os.path.join(input_dir, f"{year}.nc")
        output_path = os.path.join(output_dir, f"{year}.nc")

        if not os.path.exists(input_path):
            self.logger.warning(f"Skipping {year}: input file not found")
            return None

        os.makedirs(output_dir, exist_ok=True)

        success = self.resample_file(input_path, output_path, preprocess_era5=True)

        if success:
            # Validate accumulated flux variables after resampling
            ds_check = xr.open_dataset(output_path)
            valid = validate_era5_data(ds_check, year, self.logger)
            ds_check.close()
            if not valid:
                self.logger.warning(
                    f"Year {year}: ERA5 validation failed! "
                    f"Check expver handling or re-download the source data."
                )

        return output_path if success else None
    
    def resample_range(
        self,
        input_dir: str,
        output_dir: str,
        start_year: int,
        end_year: int
    ) -> List[str]:
        """Resample ERA5 data for a range of years."""
        output_files = []
        
        for year in tqdm(range(start_year, end_year + 1), desc="Resampling ERA5"):
            filepath = self.resample_year(input_dir, output_dir, year)
            if filepath:
                output_files.append(filepath)
        
        return output_files


class MODISResampler(DataResampler):
    """MODIS-specific resampler."""
    
    def resample_year(
        self,
        input_dir: str,
        output_dir: str,
        year: int
    ) -> Optional[str]:
        """Resample MODIS data for a specific year."""
        input_path = os.path.join(input_dir, f"{year}.nc")
        output_path = os.path.join(output_dir, f"{year}.nc")
        
        if not os.path.exists(input_path):
            self.logger.warning(f"Skipping {year}: input file not found")
            return None
        
        os.makedirs(output_dir, exist_ok=True)
        
        success = self.resample_file(input_path, output_path)
        return output_path if success else None
    
    def resample_range(
        self,
        input_dir: str,
        output_dir: str,
        start_year: int,
        end_year: int
    ) -> List[str]:
        """Resample MODIS data for a range of years."""
        # MODIS starts from 2000
        start_year = max(start_year, 2000)
        output_files = []
        
        for year in tqdm(range(start_year, end_year + 1), desc="Resampling MODIS"):
            filepath = self.resample_year(input_dir, output_dir, year)
            if filepath:
                output_files.append(filepath)
        
        return output_files


class SMIAResampler(DataResampler):
    """SMIA-specific resampler."""
    
    def resample_year(
        self,
        input_dir: str,
        output_dir: str,
        year: int
    ) -> Optional[str]:
        """Resample SMIA data for a specific year."""
        # Try different filename patterns
        patterns = [f"{year}_SMIA.nc", f"{year}.nc"]
        
        input_path = None
        for pattern in patterns:
            path = os.path.join(input_dir, pattern)
            if os.path.exists(path):
                input_path = path
                break
        
        if input_path is None:
            self.logger.warning(f"Skipping {year}: input file not found")
            return None
        
        output_path = os.path.join(output_dir, f"{year}.nc")
        os.makedirs(output_dir, exist_ok=True)
        
        success = self.resample_file(input_path, output_path)
        return output_path if success else None
    
    def resample_range(
        self,
        input_dir: str,
        output_dir: str,
        start_year: int,
        end_year: int
    ) -> List[str]:
        """Resample SMIA data for a range of years."""
        output_files = []
        
        for year in tqdm(range(start_year, end_year + 1), desc="Resampling SMIA"):
            filepath = self.resample_year(input_dir, output_dir, year)
            if filepath:
                output_files.append(filepath)
        
        return output_files


class ModisGreenResampler(MODISResampler):
    """Resampler for MOD09CMG Green reflectance (monthly (time, lat, lon))."""
    pass


class _YearlyResampler(DataResampler):
    """Shared logic for yearly (lat, lon) MODIS products (landcover / landuse).

    Categorical variables — forces nearest-neighbour interpolation regardless
    of the resampler's default method.
    """

    def resample_year(
        self,
        input_dir: str,
        output_dir: str,
        year: int
    ) -> Optional[str]:
        input_path = os.path.join(input_dir, f"{year}.nc")
        output_path = os.path.join(output_dir, f"{year}.nc")

        if not os.path.exists(input_path):
            self.logger.warning(f"Skipping {year}: input file not found")
            return None

        os.makedirs(output_dir, exist_ok=True)

        # Force nearest-neighbour for categorical data
        original_method = self.method
        self.method = "nearest"
        try:
            success = self.resample_file(input_path, output_path)
        finally:
            self.method = original_method
        return output_path if success else None

    def resample_range(
        self,
        input_dir: str,
        output_dir: str,
        start_year: int,
        end_year: int
    ) -> List[str]:
        output_files = []
        desc = f"Resampling {self.__class__.__name__}"
        for year in tqdm(range(start_year, end_year + 1), desc=desc):
            filepath = self.resample_year(input_dir, output_dir, year)
            if filepath:
                output_files.append(filepath)
        return output_files


class ModisLandcoverResampler(_YearlyResampler):
    """Resampler for MCD12Q2 phenology (yearly (lat, lon), categorical)."""
    pass


class ModisLanduseResampler(_YearlyResampler):
    """Resampler for MCD12C1 IGBP (yearly (lat, lon), categorical)."""
    pass


def resample_era5(
    input_dir: str,
    output_dir: str,
    reference_grid_path: str,
    start_year: int,
    end_year: int,
    method: str = "linear",
    log_dir: str = "./logs"
) -> List[str]:
    """Resample ERA5 data."""
    logger = setup_logger("era5_resample", log_dir, "era5_resample.log")
    resampler = ERA5Resampler(reference_grid_path, method, logger)
    return resampler.resample_range(input_dir, output_dir, start_year, end_year)


def resample_modis(
    input_dir: str,
    output_dir: str,
    reference_grid_path: str,
    start_year: int,
    end_year: int,
    method: str = "linear",
    log_dir: str = "./logs"
) -> List[str]:
    """Resample MODIS data."""
    logger = setup_logger("modis_resample", log_dir, "modis_resample.log")
    resampler = MODISResampler(reference_grid_path, method, logger)
    return resampler.resample_range(input_dir, output_dir, start_year, end_year)


def resample_smia(
    input_dir: str,
    output_dir: str,
    reference_grid_path: str,
    start_year: int,
    end_year: int,
    method: str = "linear",
    log_dir: str = "./logs"
) -> List[str]:
    """Resample SMIA data."""
    logger = setup_logger("smia_resample", log_dir, "smia_resample.log")
    resampler = SMIAResampler(reference_grid_path, method, logger)
    return resampler.resample_range(input_dir, output_dir, start_year, end_year)


def resample_modis_green(
    input_dir: str,
    output_dir: str,
    reference_grid_path: str,
    start_year: int,
    end_year: int,
    method: str = "linear",
    log_dir: str = "./logs"
) -> List[str]:
    """Resample MOD09CMG Green reflectance."""
    logger = setup_logger("modis_green_resample", log_dir, "modis_green_resample.log")
    resampler = ModisGreenResampler(reference_grid_path, method, logger)
    return resampler.resample_range(input_dir, output_dir, start_year, end_year)


def resample_modis_landcover(
    input_dir: str,
    output_dir: str,
    reference_grid_path: str,
    start_year: int,
    end_year: int,
    log_dir: str = "./logs"
) -> List[str]:
    """Resample MCD12Q2 phenology (always uses nearest-neighbour)."""
    logger = setup_logger(
        "modis_landcover_resample", log_dir, "modis_landcover_resample.log"
    )
    resampler = ModisLandcoverResampler(reference_grid_path, "nearest", logger)
    return resampler.resample_range(input_dir, output_dir, start_year, end_year)


def resample_modis_landuse(
    input_dir: str,
    output_dir: str,
    reference_grid_path: str,
    start_year: int,
    end_year: int,
    log_dir: str = "./logs"
) -> List[str]:
    """Resample MCD12C1 IGBP (always uses nearest-neighbour)."""
    logger = setup_logger(
        "modis_landuse_resample", log_dir, "modis_landuse_resample.log"
    )
    resampler = ModisLanduseResampler(reference_grid_path, "nearest", logger)
    return resampler.resample_range(input_dir, output_dir, start_year, end_year)
