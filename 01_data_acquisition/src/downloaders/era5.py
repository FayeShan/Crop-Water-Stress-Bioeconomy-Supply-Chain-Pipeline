"""ERA5 download (CDS API), unzip, and conversion to NetCDF."""

import os
import zipfile
import cdsapi
from tqdm import tqdm
from typing import List, Optional
import logging

from src.utils import setup_logger


class ERA5Downloader:
    """ERA5 data downloader using CDS API."""
    
    def __init__(
        self,
        output_dir: str,
        variables: List[str],
        months: List[str],
        logger: Optional[logging.Logger] = None
    ):
        """Initialize ERA5 downloader (months as zero-padded strings, e.g. "01")."""
        self.output_dir = output_dir
        self.variables = variables
        self.months = months
        self.logger = logger or logging.getLogger(__name__)
        self.client = cdsapi.Client()
        
        os.makedirs(output_dir, exist_ok=True)
    
    def download_year(self, year: int, force: bool = False) -> str:
        """Download ERA5 data for a specific year; returns the saved file path."""
        filename = os.path.join(self.output_dir, f"{year}.zip")
        
        if os.path.exists(filename) and not force:
            self.logger.info(f"File {filename} already exists, skipping...")
            return filename
        
        request = {
            "product_type": "monthly_averaged_reanalysis",
            "variable": self.variables,
            "year": [str(year)],
            "month": self.months,
            "time": ["00:00"],
            "data_format": "netcdf",
            "download_format": "zip"
        }
        
        self.logger.info(f"Downloading ERA5 data for year {year}...")
        self.client.retrieve(
            "reanalysis-era5-land-monthly-means", 
            request
        ).download(filename)
        
        self.logger.info(f"Downloaded: {filename}")
        return filename
    
    def download_range(
        self, 
        start_year: int, 
        end_year: int, 
        force: bool = False
    ) -> List[str]:
        """Download ERA5 data for an inclusive range of years."""
        files = []
        years = list(range(start_year, end_year + 1))
        
        for year in tqdm(years, desc="Downloading ERA5 data"):
            try:
                filepath = self.download_year(year, force)
                files.append(filepath)
            except Exception as e:
                self.logger.error(f"Error downloading year {year}: {e}")
        
        self.logger.info("All downloads completed!")
        return files


class ERA5Processor:
    """ERA5 data processor for unzipping and converting files."""
    
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize ERA5 processor."""
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.logger = logger or logging.getLogger(__name__)
        
        os.makedirs(output_dir, exist_ok=True)
    
    def unzip_year(
        self, 
        year: int, 
        delete_zip: bool = False
    ) -> Optional[str]:
        """Unzip ERA5 data for a year; returns extracted NetCDF path or None on failure."""
        zip_path = os.path.join(self.input_dir, f"{year}.zip")
        
        if not os.path.exists(zip_path):
            self.logger.warning(f"{zip_path} does not exist.")
            return None
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                nc_files = [
                    name for name in zf.namelist() 
                    if name.lower().endswith('.nc')
                ]
                
                if not nc_files:
                    self.logger.warning(f"{year}.zip does not contain .nc file")
                    return None
                
                original_nc = nc_files[0]
                self.logger.info(f"Extracting {original_nc} to {year}.nc")
                
                zf.extract(original_nc, self.output_dir)
                
                original_path = os.path.join(self.output_dir, original_nc)
                new_path = os.path.join(self.output_dir, f"{year}.nc")
                
                if os.path.abspath(original_path) != os.path.abspath(new_path):
                    os.replace(original_path, new_path)
                
                if delete_zip:
                    os.remove(zip_path)
                    self.logger.info(f"Deleted {zip_path}")
                
                return new_path
                
        except Exception as e:
            self.logger.error(f"Error processing {year}: {e}")
            return None
    
    def unzip_range(
        self, 
        start_year: int, 
        end_year: int,
        delete_zip: bool = False
    ) -> List[str]:
        """Unzip ERA5 data for an inclusive range of years."""
        files = []
        years = range(start_year, end_year + 1)
        
        for year in tqdm(years, desc="Unzipping ERA5 data"):
            filepath = self.unzip_year(year, delete_zip)
            if filepath:
                files.append(filepath)
        
        return files


def download_era5(
    output_dir: str,
    variables: List[str],
    months: List[str],
    start_year: int,
    end_year: int,
    log_dir: str = "./logs",
    force: bool = False
) -> List[str]:
    """Convenience function to download ERA5 data."""
    logger = setup_logger("era5_download", log_dir, "era5_download.log")
    
    downloader = ERA5Downloader(
        output_dir=output_dir,
        variables=variables,
        months=months,
        logger=logger
    )
    
    return downloader.download_range(start_year, end_year, force)


def process_era5(
    input_dir: str,
    output_dir: str,
    start_year: int,
    end_year: int,
    log_dir: str = "./logs",
    delete_zip: bool = False
) -> List[str]:
    """Convenience function to process (unzip) ERA5 data."""
    logger = setup_logger("era5_process", log_dir, "era5_process.log")
    
    processor = ERA5Processor(
        input_dir=input_dir,
        output_dir=output_dir,
        logger=logger
    )
    
    return processor.unzip_range(start_year, end_year, delete_zip)
