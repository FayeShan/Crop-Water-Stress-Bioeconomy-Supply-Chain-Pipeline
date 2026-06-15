"""MODIS MOD13C2 download (NASA Earthaccess) and HDF4-to-NetCDF conversion (pyhdf)."""

import os
import glob
import warnings
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict
import logging

import earthaccess
import xarray as xr
from pyhdf.SD import SD, SDC

from src.utils import setup_logger, get_netcdf_encoding

warnings.filterwarnings("ignore")


# Default variable mapping for MODIS MOD13C2
# Users can customize this in config.yaml
DEFAULT_VAR_MAPPING = {
    "CMG 0.05 Deg Monthly NDVI": "NDVI",
    "CMG 0.05 Deg Monthly EVI": "EVI",
    "CMG 0.05 Deg Monthly pixel reliability": "pixel_reliability",
    "CMG 0.05 Deg Monthly red reflectance": "red_reflectance",
    "CMG 0.05 Deg Monthly NIR reflectance": "NIR_reflectance",
    "CMG 0.05 Deg Monthly blue reflectance": "blue_reflectance",
    "CMG 0.05 Deg Monthly MIR reflectance": "MIR_reflectance",
    "CMG 0.05 Deg Monthly NDVI std dev": "NDVI_std_dev",
    "CMG 0.05 Deg Monthly VI Quality": "VI_Quality",
    "CMG 0.05 Deg Monthly EVI std dev": "EVI_std_dev",
}

# MODIS VI fill value (fallback if not found in SDS attributes)
DEFAULT_FILL_VALUE = -3000


class MODISDownloader:
    """MODIS data downloader using NASA Earthaccess."""
    
    def __init__(
        self,
        output_dir: str,
        short_name: str = "MOD13C2",
        version: str = "061",
        logger: Optional[logging.Logger] = None
    ):
        """Initialize MODIS downloader."""
        self.output_dir = Path(output_dir)
        self.short_name = short_name
        self.version = version
        self.logger = logger or logging.getLogger(__name__)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._login()
    
    def _login(self):
        """Login to NASA Earthaccess."""
        try:
            self.auth = earthaccess.login()
            self.logger.info("Successfully logged in to NASA Earthaccess")
        except Exception as e:
            self.logger.error(f"Failed to login to Earthaccess: {e}")
            self.logger.info("Please run 'earthaccess.login()' interactively first")
            raise
    
    def download_year(self, year: int) -> List[str]:
        """Download MODIS files for a specific year."""
        self.logger.info(f"Searching MODIS data for year {year}...")
        
        temporal = (f"{year}-01-01", f"{year}-12-31")
        
        try:
            results = earthaccess.search_data(
                short_name=self.short_name,
                version=self.version,
                temporal=temporal
            )
            
            self.logger.info(f"Found {len(results)} granules for year {year}")
            
            if not results:
                self.logger.warning(f"No data found for year {year}")
                return []

            self.logger.info(f"Downloading to {self.output_dir}...")
            downloaded = earthaccess.download(results, str(self.output_dir))
            
            self.logger.info(f"Downloaded {len(downloaded)} files for year {year}")
            return downloaded
            
        except Exception as e:
            self.logger.error(f"Error downloading year {year}: {e}")
            return []
    
    def download_range(
        self, 
        start_year: int, 
        end_year: int
    ) -> List[str]:
        """Download MODIS data for a range of years (start clamped to >= 2000)."""
        # MODIS data starts from 2000
        start_year = max(start_year, 2000)

        all_files = []
        for year in range(start_year, end_year + 1):
            files = self.download_year(year)
            all_files.extend(files)
        
        self.logger.info(f"Total downloaded: {len(all_files)} files")
        return all_files
    
    def download_temporal(
        self,
        start_date: str,
        end_date: str
    ) -> List[str]:
        """Download MODIS data for a temporal range (dates as YYYY-MM-DD)."""
        self.logger.info(f"Searching MODIS data from {start_date} to {end_date}...")
        
        try:
            results = earthaccess.search_data(
                short_name=self.short_name,
                version=self.version,
                temporal=(start_date, end_date)
            )
            
            self.logger.info(f"Found {len(results)} granules")
            
            if not results:
                return []
            
            downloaded = earthaccess.download(results, str(self.output_dir))
            self.logger.info(f"Downloaded {len(downloaded)} files")
            return downloaded
            
        except Exception as e:
            self.logger.error(f"Error downloading: {e}")
            return []


class MODISProcessor:
    """MODIS HDF4 to NetCDF converter using pyhdf."""
    
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        var_mapping: Optional[Dict[str, str]] = None,
        selected_vars: Optional[List[str]] = None,
        fill_value: int = DEFAULT_FILL_VALUE,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize MODIS processor.

        fill_value is a fallback used only when an SDS lacks a _FillValue attribute.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.var_mapping = var_mapping or DEFAULT_VAR_MAPPING
        self.selected_vars = selected_vars
        self.fill_value = fill_value
        self.logger = logger or logging.getLogger(__name__)

        os.makedirs(output_dir, exist_ok=True)

        if self.selected_vars:
            self.var_mapping = {
                k: v for k, v in self.var_mapping.items()
                if v in self.selected_vars
            }
            self.logger.info(f"Processing selected variables: {self.selected_vars}")
    
    def process_year(self, year: int) -> Optional[str]:
        """Process MODIS HDF files for a year to NetCDF; returns path or None on failure."""
        self.logger.info(f"Processing year: {year}")
        
        pattern = os.path.join(self.input_dir, f"MOD13C2.A{year}*.hdf")
        hdf_files = sorted(glob.glob(pattern))
        
        if not hdf_files:
            self.logger.warning(f"No files found for year {year}")
            return None
        
        all_data = {}
        processed_dates = set()
        
        for hdf_file in hdf_files:
            try:
                basename = os.path.basename(hdf_file)
                self.logger.info(f"Processing: {basename}")
                
                # Extract date from filename
                julian_str = basename.split('.')[1][5:8]
                date = datetime(year, 1, 1) + timedelta(days=int(julian_str) - 1)
                
                # Skip duplicate months
                month_key = (date.year, date.month)
                if month_key in processed_dates:
                    self.logger.info(f"  Skipping: duplicate month {date.strftime('%Y-%m')}")
                    continue
                processed_dates.add(month_key)
                
                # Open HDF file
                hdf = SD(hdf_file, SDC.READ)
                
                for orig_var, new_name in self.var_mapping.items():
                    if orig_var not in hdf.datasets():
                        continue
                    
                    self.logger.info(f"  Reading: {orig_var} -> {new_name}")

                    sds = hdf.select(orig_var)
                    data = sds[:].astype(np.float64)
                    attrs = sds.attributes()
                    sds.endaccess()
                    
                    # Read QC metadata from SDS attributes
                    # (each variable carries its own valid_range, scale_factor, _FillValue)
                    fill = attrs.get('_FillValue', self.fill_value)
                    valid = attrs.get('valid_range', None)
                    scale = attrs.get('scale_factor', None)
                    offset = attrs.get('add_offset', 0.0)
                    
                    self.logger.info(
                        f"    SDS attrs: fill={fill}, valid_range={valid}, "
                        f"scale_factor={scale}, add_offset={offset}"
                    )
                    
                    # Step 1: Mask fill value
                    data = np.where(data == fill, np.nan, data)
                    
                    # Step 2: Mask values outside valid range (if available)
                    if valid is not None:
                        vmin, vmax = valid[0], valid[1]
                        data = np.where(
                            (data < vmin) | (data > vmax),
                            np.nan,
                            data
                        )
                    
                    # Step 3: Apply scale factor and offset
                    # MODIS HDF4 convention: physical = (raw - add_offset) / scale_factor
                    # This differs from NetCDF/CF convention (physical = raw * scale + offset)
                    if scale is not None:
                        data = (data - offset) / scale
                    
                    data = data.astype(np.float32)
                    
                    self.logger.info(
                        f"    After QC: range [{np.nanmin(data):.4f}, {np.nanmax(data):.4f}], "
                        f"NaN ratio: {np.isnan(data).sum() / data.size:.2%}"
                    )
                    
                    # MODIS CMG 0.05 deg grid (pixel centres)
                    lat = np.linspace(90 - 0.025, -90 + 0.025, data.shape[0])
                    lon = np.linspace(-180 + 0.025, 180 - 0.025, data.shape[1])

                    da = xr.DataArray(
                        data,
                        dims=["lat", "lon"],
                        coords={"lat": lat, "lon": lon}
                    )
                    da = da.expand_dims(time=[date])
                    
                    if new_name not in all_data:
                        all_data[new_name] = []
                    all_data[new_name].append(da)
                
                hdf.end()
                
            except Exception as e:
                self.logger.error(f"Error reading {hdf_file}: {e}")
                continue
        
        if not all_data:
            self.logger.warning(f"No data collected for year {year}")
            return None
        
        data_vars = {
            var: xr.concat(dalist, dim="time")
            for var, dalist in all_data.items()
        }
        ds = xr.Dataset(data_vars)
        ds.attrs["crs"] = "EPSG:4326"

        encoding = get_netcdf_encoding(list(ds.data_vars))

        output_path = os.path.join(self.output_dir, f"{year}.nc")
        ds.to_netcdf(output_path, encoding=encoding)
        
        self.logger.info(f"Saved {output_path} with {len(processed_dates)} time steps")
        
        del ds
        return output_path
    
    def process_range(
        self, 
        start_year: int, 
        end_year: int
    ) -> List[str]:
        """Process MODIS data for a range of years (start clamped to >= 2000)."""
        # MODIS data starts from 2000
        start_year = max(start_year, 2000)

        output_files = []
        for year in range(start_year, end_year + 1):
            try:
                filepath = self.process_year(year)
                if filepath:
                    output_files.append(filepath)
            except Exception as e:
                self.logger.error(f"Error processing year {year}: {e}")
        
        return output_files


def download_modis(
    output_dir: str,
    start_year: int,
    end_year: int,
    short_name: str = "MOD13C2",
    version: str = "061",
    log_dir: str = "./logs"
) -> List[str]:
    """Convenience function to download MODIS data."""
    logger = setup_logger("modis_download", log_dir, "modis_download.log")
    
    downloader = MODISDownloader(
        output_dir=output_dir,
        short_name=short_name,
        version=version,
        logger=logger
    )
    
    return downloader.download_range(start_year, end_year)


def process_modis(
    input_dir: str,
    output_dir: str,
    start_year: int,
    end_year: int,
    var_mapping: Optional[Dict[str, str]] = None,
    selected_vars: Optional[List[str]] = None,
    fill_value: int = DEFAULT_FILL_VALUE,
    log_dir: str = "./logs"
) -> List[str]:
    """Convenience function to process MODIS HDF to NetCDF.

    QC metadata (valid_range, scale_factor, add_offset) is read automatically
    from each HDF SDS's attributes.
    """
    logger = setup_logger("modis_process", log_dir, "modis_process.log")
    
    processor = MODISProcessor(
        input_dir=input_dir,
        output_dir=output_dir,
        var_mapping=var_mapping,
        selected_vars=selected_vars,
        fill_value=fill_value,
        logger=logger
    )
    
    return processor.process_range(start_year, end_year)
