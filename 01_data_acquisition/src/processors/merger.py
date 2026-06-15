"""Merge ERA5, MODIS, SMIA, TWSA, and additional MODIS products into one dataset per year.

Annual products (landcover / landuse) are broadcast from yearly to monthly time.
"""

import os
import xarray as xr
import numpy as np
import pandas as pd
import logging
import gc
from typing import Optional, List, Dict
from tqdm import tqdm

from src.utils import setup_logger, load_dataset_safe, get_netcdf_encoding


# Sources that store yearly data (lat, lon) — need broadcasting to monthly time
YEARLY_SOURCES = {"modis_landcover", "modis_landuse"}

# Year lower bounds for data-availability checks
SOURCE_START_YEAR = {
    "modis": 2000,
    "twsa": 2002,
    "modis_green": 2000,
    "modis_landcover": 2001,
    "modis_landuse": 2001,
}


def _broadcast_yearly_to_monthly(ds: xr.Dataset, year: int) -> xr.Dataset:
    """Broadcast a yearly (lat, lon) dataset to (time=12, lat, lon).

    The same yearly value is repeated for each month, producing a monthly
    time coordinate matching the other monthly sources.
    """
    if "time" in ds.dims:
        return ds  # already has time dim
    time_coords = pd.date_range(f"{year}-01-01", periods=12, freq="MS")
    ds_expanded = ds.expand_dims(time=time_coords)
    # Ensure `time` is the leading dimension
    for var in ds_expanded.data_vars:
        ds_expanded[var] = ds_expanded[var].transpose("time", "lat", "lon")
    return ds_expanded


class DataMerger:
    """Merge multiple data sources into single datasets."""
    
    def __init__(
        self,
        output_dir: str,
        source_paths: Dict[str, str],
        sources_enabled: Dict[str, bool],
        extra_files: Optional[List[str]] = None,
        compression_level: int = 6,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize data merger.

        source_paths maps source names to directories; sources_enabled toggles each.
        """
        self.output_dir = output_dir
        self.source_paths = source_paths
        self.sources_enabled = sources_enabled
        self.extra_files = extra_files or []
        self.compression_level = compression_level
        self.logger = logger or logging.getLogger(__name__)

        os.makedirs(output_dir, exist_ok=True)

        enabled = [k for k, v in sources_enabled.items() if v]
        disabled = [k for k, v in sources_enabled.items() if not v]
        self.logger.info(f"Enabled sources: {enabled}")
        if disabled:
            self.logger.info(f"Disabled sources: {disabled}")
        if self.extra_files:
            self.logger.info(f"Extra files to merge: {self.extra_files}")
    
    def _load_year_data(self, year: int) -> Dict[str, Optional[xr.Dataset]]:
        """Load all enabled data sources for a specific year."""
        data = {}

        for source_name, source_dir in self.source_paths.items():
            if not self.sources_enabled.get(source_name, False):
                continue

            min_year = SOURCE_START_YEAR.get(source_name)
            if min_year is not None and year < min_year:
                self.logger.info(
                    f"Skipping {source_name} for {year} (data starts from {min_year})"
                )
                continue

            file_path = os.path.join(source_dir, f"{year}.nc")
            ds = load_dataset_safe(file_path, self.logger)

            # Broadcast yearly (lat, lon) sources to monthly time
            if ds is not None and source_name in YEARLY_SOURCES:
                self.logger.info(
                    f"Broadcasting {source_name} {year} from yearly to monthly"
                )
                ds = _broadcast_yearly_to_monthly(ds, year)

            data[source_name] = ds

        return data
    
    def _load_extra_files(self) -> List[xr.Dataset]:
        """Load extra files."""
        extra_datasets = []
        
        for file_path in self.extra_files:
            if os.path.exists(file_path):
                ds = load_dataset_safe(file_path, self.logger)
                if ds is not None:
                    extra_datasets.append(ds)
                    self.logger.info(f"Loaded extra file: {file_path}")
            else:
                self.logger.warning(f"Extra file not found: {file_path}")
        
        return extra_datasets
    
    def merge_year(
        self, 
        year: int,
        require_all: bool = False
    ) -> Optional[str]:
        """Merge all enabled data sources for a year (require_all skips on any gap)."""
        self.logger.info(f"Processing year {year}")

        data = self._load_year_data(year)

        available = {k: v for k, v in data.items() if v is not None}
        missing = [k for k, v in data.items() if v is None]

        if missing:
            self.logger.warning(f"Missing data for {year}: {missing}")

        if require_all and missing:
            self.logger.warning(f"Skipping {year} due to missing required data")
            for ds in available.values():
                ds.close()
            gc.collect()
            return None
        
        if not available:
            self.logger.error(f"No data available for year {year}")
            return None
        
        try:
            extra_datasets = self._load_extra_files()

            all_datasets = list(available.values()) + extra_datasets

            self.logger.info(f"Merging {len(all_datasets)} datasets for year {year}")
            merged_ds = xr.merge(all_datasets, join="outer")

            for ds in available.values():
                ds.close()
            for ds in extra_datasets:
                ds.close()
            del available, extra_datasets
            gc.collect()

            vars_to_drop = [v for v in ["number", "expver"] if v in merged_ds.data_vars]
            if vars_to_drop:
                merged_ds = merged_ds.drop_vars(vars_to_drop)

            encoding = {
                var: {
                    'zlib': True,
                    'complevel': self.compression_level
                }
                for var in merged_ds.data_vars
            }

            output_file = os.path.join(self.output_dir, f"{year}_merged.nc")
            self.logger.info(f"Saving merged data to {output_file}")
            merged_ds.to_netcdf(output_file, encoding=encoding)

            self.logger.info(f"Merged variables: {list(merged_ds.data_vars)}")
            self.logger.info(f"Dimensions: {dict(merged_ds.dims)}")
            
            merged_ds.close()
            del merged_ds
            gc.collect()
            
            self.logger.info(f"Successfully merged year {year}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error merging year {year}: {e}")
            import traceback
            traceback.print_exc()
            gc.collect()
            return None
    
    def merge_range(
        self,
        start_year: int,
        end_year: int,
        require_all: bool = False
    ) -> List[str]:
        """Merge data for a range of years."""
        output_files = []
        
        for year in tqdm(range(start_year, end_year + 1), desc="Merging data"):
            filepath = self.merge_year(year, require_all)
            if filepath:
                output_files.append(filepath)
        
        self.logger.info(f"Merging complete! Created {len(output_files)} files.")
        return output_files


def merge_data(
    era5_dir: str,
    modis_dir: str,
    smia_dir: str,
    twsa_dir: str,
    output_dir: str,
    start_year: int,
    end_year: int,
    sources_enabled: Optional[Dict[str, bool]] = None,
    extra_files: Optional[List[str]] = None,
    compression_level: int = 6,
    require_all: bool = False,
    log_dir: str = "./logs",
    modis_green_dir: Optional[str] = None,
    modis_landcover_dir: Optional[str] = None,
    modis_landuse_dir: Optional[str] = None,
) -> List[str]:
    """Convenience function to merge all data sources.

    sources_enabled keys: era5, modis, smia, twsa, modis_green, modis_landcover,
    modis_landuse. If None, the four base sources are enabled and the rest disabled.
    """
    logger = setup_logger("data_merge", log_dir, "data_merge.log")

    # Default: only the 4 base sources enabled
    if sources_enabled is None:
        sources_enabled = {
            'era5': True,
            'modis': True,
            'smia': True,
            'twsa': True,
            'modis_green': False,
            'modis_landcover': False,
            'modis_landuse': False,
        }

    source_paths = {
        'era5': era5_dir,
        'modis': modis_dir,
        'smia': smia_dir,
        'twsa': twsa_dir,
    }
    if modis_green_dir is not None:
        source_paths['modis_green'] = modis_green_dir
    if modis_landcover_dir is not None:
        source_paths['modis_landcover'] = modis_landcover_dir
    if modis_landuse_dir is not None:
        source_paths['modis_landuse'] = modis_landuse_dir

    merger = DataMerger(
        output_dir=output_dir,
        source_paths=source_paths,
        sources_enabled=sources_enabled,
        extra_files=extra_files,
        compression_level=compression_level,
        logger=logger
    )

    return merger.merge_range(start_year, end_year, require_all)
