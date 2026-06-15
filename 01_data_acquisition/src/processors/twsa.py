"""TWSA (Terrestrial Water Storage Anomaly) processing and resampling.

Missing months are filled by forward-fill (previous available month).
"""

import os
import re
import glob
import xarray as xr
import numpy as np
import pandas as pd
import logging
from typing import Optional, List
from tqdm import tqdm

from src.utils import setup_logger, get_netcdf_encoding


class TWSAProcessor:
    """TWSA data processor."""
    
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        fill_missing_months: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize TWSA processor."""
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.fill_missing_months = fill_missing_months
        self.logger = logger or logging.getLogger(__name__)
        
        os.makedirs(output_dir, exist_ok=True)
    
    def _find_twsa_file(self, year: int) -> Optional[str]:
        """Find TWSA file for a year by flexible pattern matching.

        Supports ``twsan_m_gdo_{year}*_m.nc`` (any date range) and plain ``{year}.nc``.
        """
        # Pattern 1: glob for twsan_m_gdo_{year}*_m.nc
        glob_pattern = os.path.join(self.input_dir, f"twsan_m_gdo_{year}*_m.nc")
        matches = glob.glob(glob_pattern)
        
        if matches:
            # If multiple matches, prefer the one with most months (longest date range)
            if len(matches) > 1:
                self.logger.info(f"Found {len(matches)} files for {year}, selecting best match")
            return sorted(matches)[-1]  # Usually the latest/most complete
        
        # Pattern 2: simple {year}.nc
        simple_path = os.path.join(self.input_dir, f"{year}.nc")
        if os.path.exists(simple_path):
            return simple_path
        
        return None
    
    def _fill_missing_months(
        self, 
        ds: xr.DataArray, 
        year: int
    ) -> xr.DataArray:
        """Fill missing months to a complete 12-month series via forward-fill
        (previous available month), with backward-fill for any leading gaps.
        """
        existing_times = pd.to_datetime(ds.time.values)
        existing_months = set(existing_times.month)

        self.logger.info(f"Existing months: {sorted(existing_months)}")

        if len(existing_months) == 12:
            self.logger.info("All 12 months present, no filling needed")
            return ds

        all_months = set(range(1, 13))
        missing_months = all_months - existing_months
        self.logger.info(f"Missing months: {sorted(missing_months)}")

        full_times = pd.date_range(
            start=f"{year}-01-01",
            end=f"{year}-12-01",
            freq="MS"  # Month Start
        )
        
        # Reindex to full 12 months (creates NaN for missing)
        ds_full = ds.reindex(time=full_times)

        # Forward fill: e.g. if Oct missing use Sep; if Dec missing use Nov
        ds_filled = ds_full.ffill(dim="time")

        # If leading months are missing (rare), backward fill
        if ds_filled.isnull().any():
            ds_filled = ds_filled.bfill(dim="time")

        for month in sorted(missing_months):
            month_idx = month - 1
            if month_idx > 0:
                source_month = month - 1
                while source_month not in existing_months and source_month > 0:
                    source_month -= 1
                if source_month in existing_months:
                    self.logger.info(f"  Month {month:02d} filled with data from month {source_month:02d}")
                else:
                    # Had to use backward fill
                    source_month = min(m for m in existing_months if m > month)
                    self.logger.info(f"  Month {month:02d} filled with data from month {source_month:02d} (backfill)")
        
        return ds_filled
    
    def process_year(
        self, 
        year: int,
        target_lon: Optional[xr.DataArray] = None,
        target_lat: Optional[xr.DataArray] = None
    ) -> Optional[str]:
        """Process TWSA data for a year; returns output path or None on failure."""
        self.logger.info(f"Processing TWSA for year {year}")

        input_path = self._find_twsa_file(year)
        
        if input_path is None:
            self.logger.warning(f"Skipping {year}: input file not found")
            return None
        
        try:
            self.logger.info(f"Loading: {input_path}")
            ds = xr.open_dataset(input_path)

            if 'twsan' in ds.data_vars:
                da = ds.drop_vars(
                    [v for v in ["band"] if v in ds.coords]
                ).squeeze()['twsan']
            else:
                # Fallback: if twsan not found, use the first data variable
                var_name = list(ds.data_vars)[0]
                self.logger.warning(f"'twsan' not found, using '{var_name}'")
                da = ds[var_name].squeeze()

            ds.close()

            if self.fill_missing_months:
                da = self._fill_missing_months(da, year)

            if target_lon is not None and target_lat is not None:
                self.logger.info(f"Interpolating {year} to target resolution...")
                da_interp = da.interp(
                    lon=target_lon, 
                    lat=target_lat, 
                    method="linear"
                )
            else:
                da_interp = da

            da_interp = da_interp.astype("float32")

            ds_out = da_interp.to_dataset(name="twsan")

            output_path = os.path.join(self.output_dir, f"{year}.nc")
            encoding = get_netcdf_encoding(["twsan"])

            self.logger.info(f"Saving to {output_path}...")
            ds_out.to_netcdf(output_path, encoding=encoding)

            n_times = len(ds_out.time)
            self.logger.info(f"Done: {year}.nc with {n_times} time steps, shape {dict(ds_out.dims)}")
            
            del da, da_interp, ds_out
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error processing {year}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_range(
        self,
        start_year: int,
        end_year: int,
        reference_grid_path: Optional[str] = None
    ) -> List[str]:
        """Process TWSA data for a range of years (start clamped to >= 2002)."""
        # TWSA starts from 2002
        start_year = max(start_year, 2002)

        target_lon, target_lat = None, None
        if reference_grid_path and os.path.exists(reference_grid_path):
            ref = xr.open_dataset(reference_grid_path)
            target_lon = ref.lon
            target_lat = ref.lat
            ref.close()
            self.logger.info(f"Loaded reference grid: {len(target_lon)} x {len(target_lat)}")
        
        output_files = []
        
        for year in tqdm(range(start_year, end_year + 1), desc="Processing TWSA"):
            try:
                filepath = self.process_year(year, target_lon, target_lat)
                if filepath:
                    output_files.append(filepath)
            except Exception as e:
                self.logger.error(f"Error processing year {year}: {e}")
        
        return output_files


def process_twsa(
    input_dir: str,
    output_dir: str,
    start_year: int,
    end_year: int,
    reference_grid_path: Optional[str] = None,
    fill_missing_months: bool = True,
    log_dir: str = "./logs"
) -> List[str]:
    """Convenience function to process TWSA data."""
    logger = setup_logger("twsa_process", log_dir, "twsa_process.log")
    
    processor = TWSAProcessor(
        input_dir=input_dir,
        output_dir=output_dir,
        fill_missing_months=fill_missing_months,
        logger=logger
    )
    
    return processor.process_range(start_year, end_year, reference_grid_path)
