"""SMIA (Soil Moisture Index Anomaly) TIFF-to-NetCDF conversion, per year."""

import os
import numpy as np
import xarray as xr
import pandas as pd
import rasterio
import logging
from typing import List, Optional
from tqdm import tqdm

from src.utils import setup_logger, get_netcdf_encoding


class SMIAProcessor:
    """SMIA TIFF to NetCDF converter."""
    
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        days: List[str],
        logger: Optional[logging.Logger] = None
    ):
        """Initialize SMIA processor (days as zero-padded strings, e.g. ["01", "11", "21"])."""
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.days = days
        self.logger = logger or logging.getLogger(__name__)

        os.makedirs(output_dir, exist_ok=True)

        self.months = [f"{m:02d}" for m in range(1, 13)]
    
    def process_year(self, year: int) -> Optional[str]:
        """Process SMIA TIFF files for a year to NetCDF; returns path or None on failure."""
        self.logger.info(f"Processing year: {year}")
        
        year_dir = os.path.join(self.input_dir, str(year))
        
        if not os.path.exists(year_dir):
            self.logger.warning(f"Skipping year {year}: directory does not exist.")
            return None

        data_dict = {day: [] for day in self.days}
        time_coords = []

        # Get spatial dimensions from sample file
        sample_file = os.path.join(
            year_dir, 
            f"smang_m_gdo_{year}0101_t_300_z01.tif"
        )
        
        if not os.path.exists(sample_file):
            # Try alternative naming pattern
            sample_file = os.path.join(year_dir, f"{year}0101.tiff")
        
        if not os.path.exists(sample_file):
            self.logger.warning(f"Skipping year {year}: no sample file found")
            return None
        
        with rasterio.open(sample_file) as dataset:
            height, width = dataset.shape
            lon = np.linspace(dataset.bounds.left, dataset.bounds.right, width)
            lat = np.linspace(dataset.bounds.top, dataset.bounds.bottom, height)
        
        self.logger.info(f"Year {year}: Loaded spatial dimensions {height}x{width}")

        for month in self.months:
            self.logger.info(f"Processing month: {month} of year {year}")
            
            for day in self.days:
                # Try different filename patterns
                file_patterns = [
                    f"smang_m_gdo_{year}{month}{day}_t_300_z01.tif",
                    f"{year}{month}{day}.tiff"
                ]
                
                file_found = False
                for pattern in file_patterns:
                    file_path = os.path.join(year_dir, pattern)
                    if os.path.exists(file_path):
                        file_found = True
                        break
                
                if file_found:
                    with rasterio.open(file_path) as dataset:
                        data = dataset.read(1).astype(np.float32)
                        data[data >= 1e+20] = np.nan
                        data_dict[day].append(data)
                else:
                    self.logger.warning(f"Missing file for {year}-{month}-{day}")
                    data_dict[day].append(np.full((height, width), np.nan))
            
            time_coords.append(
                np.datetime64(f"{year}-{month}-01T00:00:00.000000000")
            )
        
        self.logger.info(f"Finished collecting data for year {year}")

        for day in self.days:
            data_dict[day] = np.stack(data_dict[day], axis=0)

        ds = xr.Dataset(
            {
                f"smia_{day}": (["time", "lat", "lon"], data_dict[day])
                for day in self.days
            },
            coords={
                "time": time_coords,
                "lat": lat,
                "lon": lon,
            },
        )
        
        self.logger.info(f"Dataset created for year {year}")

        output_file = os.path.join(self.output_dir, f"{year}.nc")
        encoding = get_netcdf_encoding(list(ds.data_vars))
        ds.to_netcdf(output_file, encoding=encoding)

        self.logger.info(f"NetCDF file saved: {output_file}")

        del ds, data_dict, time_coords
        
        return output_file
    
    def process_range(
        self, 
        start_year: int, 
        end_year: int
    ) -> List[str]:
        """Process SMIA data for a range of years."""
        output_files = []
        
        for year in tqdm(range(start_year, end_year + 1), desc="Processing SMIA"):
            try:
                filepath = self.process_year(year)
                if filepath:
                    output_files.append(filepath)
            except Exception as e:
                self.logger.error(f"Error processing year {year}: {e}")
        
        return output_files


def process_smia(
    input_dir: str,
    output_dir: str,
    days: List[str],
    start_year: int,
    end_year: int,
    log_dir: str = "./logs"
) -> List[str]:
    """Convenience function to process SMIA data."""
    logger = setup_logger("smia_process", log_dir, "smia_process.log")
    
    processor = SMIAProcessor(
        input_dir=input_dir,
        output_dir=output_dir,
        days=days,
        logger=logger
    )
    
    return processor.process_range(start_year, end_year)
