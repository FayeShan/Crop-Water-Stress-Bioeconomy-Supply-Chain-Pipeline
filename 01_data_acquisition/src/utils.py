"""
Utility functions for the water data pipeline.
"""

import os
import sys
import shutil
import logging
import xarray as xr
import numpy as np
from pathlib import Path
from typing import Optional, List, Union
from datetime import datetime


def setup_logger(
    name: str,
    log_dir: str,
    log_file: str,
    level: str = "INFO"
) -> logging.Logger:
    """Set up a logger with file and console handlers."""
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    logger.handlers.clear()

    fh = logging.FileHandler(os.path.join(log_dir, log_file), mode='w')
    fh.setLevel(getattr(logging, level.upper()))

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, level.upper()))

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def check_dataset_variables(ds: xr.Dataset, logger: Optional[logging.Logger] = None) -> dict:
    """Check whether all variables in a dataset can be successfully loaded.

    Returns a dict of {variable: {"status", "error"}}.
    """
    results = {}
    log_func = logger.info if logger else print
    error_func = logger.error if logger else print
    
    log_func("Checking variable loadability...")
    
    for var in ds.data_vars:
        try:
            ds[var].load()
            results[var] = {"status": "success", "error": None}
            log_func(f"✅ {var}: successfully loaded")
        except Exception as e:
            results[var] = {"status": "failed", "error": str(e)}
            error_func(f"❌ {var}: failed to load — {type(e).__name__}: {e}")
    
    return results


def load_dataset_safe(
    path: str,
    logger: Optional[logging.Logger] = None
) -> Optional[xr.Dataset]:
    """Load a NetCDF dataset with error handling; returns None on failure."""
    log_func = logger.warning if logger else print
    error_func = logger.error if logger else print
    
    try:
        return xr.open_dataset(path)
    except FileNotFoundError:
        log_func(f"File not found: {path}")
        return None
    except Exception as e:
        error_func(f"Error loading {path}: {str(e)}")
        return None


def get_reference_grid(ref_path: str) -> tuple:
    """Load reference grid coordinates (target_lon, target_lat) for resampling.

    Accepts a NetCDF (.nc/.nc4), a GeoTIFF (.tif/.tiff), or a resolution string
    (e.g. "0.05") that generates a global grid at that resolution.
    """
    import numpy as np

    # A numeric ref_path is treated as a target resolution rather than a file
    try:
        resolution = float(ref_path)
        target_lon = xr.DataArray(
            np.arange(-180 + resolution/2, 180, resolution),
            dims=['lon']
        )
        target_lat = xr.DataArray(
            np.arange(90 - resolution/2, -90, -resolution),
            dims=['lat']
        )
        return target_lon, target_lat
    except ValueError:
        pass  # Not a resolution value, treat as file path

    ref_path_lower = ref_path.lower()

    if ref_path_lower.endswith('.nc') or ref_path_lower.endswith('.nc4'):
        ref = xr.open_dataset(ref_path)

        # Try common coordinate names
        lon_names = ['lon', 'longitude', 'x', 'X']
        lat_names = ['lat', 'latitude', 'y', 'Y']
        
        target_lon = None
        target_lat = None
        
        for name in lon_names:
            if name in ref.coords or name in ref.dims:
                target_lon = ref[name]
                break
        
        for name in lat_names:
            if name in ref.coords or name in ref.dims:
                target_lat = ref[name]
                break
        
        ref.close()
        
        if target_lon is None or target_lat is None:
            raise ValueError(f"Could not find lon/lat coordinates in {ref_path}")
        
        return target_lon, target_lat
    
    elif ref_path_lower.endswith('.tif') or ref_path_lower.endswith('.tiff'):
        import rasterio

        with rasterio.open(ref_path) as src:
            bounds = src.bounds
            height, width = src.height, src.width

            # Coordinates at cell centers
            res_x = (bounds.right - bounds.left) / width
            res_y = (bounds.top - bounds.bottom) / height
            
            lon_values = np.linspace(
                bounds.left + res_x/2, 
                bounds.right - res_x/2, 
                width
            )
            lat_values = np.linspace(
                bounds.top - res_y/2, 
                bounds.bottom + res_y/2, 
                height
            )
            
            target_lon = xr.DataArray(lon_values, dims=['lon'])
            target_lat = xr.DataArray(lat_values, dims=['lat'])
        
        return target_lon, target_lat
    
    else:
        raise ValueError(
            f"Unsupported reference file format: {ref_path}\n"
            f"Supported formats: .nc, .nc4, .tif, .tiff, or resolution value (e.g., '0.05')"
        )


def get_netcdf_encoding(
    variables: List[str],
    dtype: str = "float32",
    fill_value: float = -9999.0,
    zlib: bool = True,
    complevel: int = 5
) -> dict:
    """Generate a NetCDF encoding dictionary for the given variables."""
    return {
        var: {
            "zlib": zlib,
            "complevel": complevel,
            "dtype": dtype,
            "_FillValue": fill_value
        }
        for var in variables
    }


def delete_path(path: str) -> str:
    """Delete a file or folder; returns a status message."""
    try:
        if os.path.isfile(path):
            os.remove(path)
            return f"File deleted: {path}"
        elif os.path.isdir(path):
            shutil.rmtree(path)
            return f"Folder deleted: {path}"
        else:
            return f"Path does not exist: {path}"
    except Exception as e:
        return f"Error deleting path: {e}"


def get_size(path: str) -> str:
    """Get size of a file or folder in human-readable format."""
    try:
        if not os.path.exists(path):
            return "Path not found"

        if os.path.isfile(path):
            total_size = os.path.getsize(path)
        else:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if os.path.exists(fp):
                        total_size += os.path.getsize(fp)

        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if total_size < 1024:
                return f"{total_size:.2f} {unit}"
            total_size /= 1024
            
        return f"{total_size:.2f} PB"
    except Exception as e:
        return f"Error: {e}"


def convert_to_float32(ds: xr.Dataset) -> xr.Dataset:
    """Convert all floating-point variables in a dataset to float32."""
    for var in ds.data_vars:
        if np.issubdtype(ds[var].dtype, np.floating):
            ds[var] = ds[var].astype("float32")
    return ds


def standardize_coordinates(ds: xr.Dataset) -> xr.Dataset:
    """Standardize coordinate names to lon, lat, time."""
    rename_dict = {}

    for name in ['longitude', 'x', 'X', 'LON', 'Longitude']:
        if name in ds.dims or name in ds.coords:
            rename_dict[name] = 'lon'
            break

    for name in ['latitude', 'y', 'Y', 'LAT', 'Latitude']:
        if name in ds.dims or name in ds.coords:
            rename_dict[name] = 'lat'
            break

    for name in ['valid_time', 'TIME', 'Time']:
        if name in ds.dims or name in ds.coords:
            rename_dict[name] = 'time'
            break
    
    if rename_dict:
        ds = ds.rename(rename_dict)
    
    return ds


def normalize_longitude(ds: xr.Dataset) -> xr.Dataset:
    """Normalize longitude to the -180 to 180 range."""
    if 'lon' in ds.coords:
        ds = ds.assign_coords(lon=((ds.lon + 180) % 360) - 180)
        ds = ds.sortby("lon")
        
        # Remove duplicate at 180
        if float(ds.lon.max()) >= 180.0 - 1e-9:
            ds = ds.sel(lon=ds.lon < 180.0)
    
    return ds


def validate_era5_data(
    ds: xr.Dataset,
    year: int,
    logger: Optional[logging.Logger] = None
) -> bool:
    """Validate ERA5 accumulated flux variables (ssr, ssrd, tp, ro, evabs, evavt)
    fall within expected magnitude ranges. Catches the common expver mis-handling
    bug where values are scaled to ~50% of their true values.

    Returns True if all checks pass, False if any variable looks suspicious.
    """
    log_func = logger.info if logger else print
    warn_func = logger.warning if logger else print

    # Expected global-mean magnitude ranges for monthly-averaged ERA5-Land
    # (order-of-magnitude checks; actual values vary by region/season)
    EXPECTED_RANGES = {
        'ssr':   (1e5,  3e7),     # J/m² — net solar radiation
        'ssrd':  (1e5,  4e7),     # J/m² — solar radiation downwards
        'tp':    (1e-6, 0.02),    # m    — total precipitation
        'ro':    (1e-7, 0.01),    # m    — runoff
        'evabs': (-0.005, -1e-7), # m    — bare-soil evaporation (negative)
        'evavt': (-0.01, -1e-7),  # m    — vegetation transpiration (negative)
    }

    issues = []
    checked = 0

    for var in ds.data_vars:
        if var not in EXPECTED_RANGES:
            continue
        checked += 1

        lo, hi = EXPECTED_RANGES[var]
        actual_mean = float(ds[var].mean(skipna=True))

        # Allow 3x headroom beyond documented range
        range_lo = min(lo, hi) * 3 if min(lo, hi) > 0 else min(lo, hi) * 3
        range_hi = max(lo, hi) * 3 if max(lo, hi) > 0 else max(lo, hi) / 3

        out_of_range = actual_mean < range_lo or actual_mean > range_hi
        if out_of_range:
            issues.append(
                f"{var}: global mean = {actual_mean:.6g}, "
                f"expected order-of-magnitude [{lo:.2g}, {hi:.2g}]"
            )

    if checked == 0:
        log_func(f"Year {year}: no accumulated flux variables found to validate")
        return True

    if issues:
        warn_func(f"Year {year} ERA5 validation FAILED ({len(issues)} issues):")
        for issue in issues:
            warn_func(f"  ⚠ {issue}")
        return False
    else:
        log_func(f"Year {year} ERA5 validation PASSED ({checked} variables checked)")
        return True


def print_dataset_info(ds: xr.Dataset, name: str = "Dataset"):
    """Print summary information about a dataset."""
    print(f"\n{'='*50}")
    print(f"{name} Information")
    print(f"{'='*50}")
    print(f"Dimensions: {dict(ds.dims)}")
    print(f"Variables: {list(ds.data_vars)}")
    print(f"Coordinates: {list(ds.coords)}")
    if 'time' in ds.dims:
        print(f"Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
    print(f"{'='*50}\n")
