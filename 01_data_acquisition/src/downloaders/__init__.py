"""
Data downloaders package.
"""

from .era5 import ERA5Downloader, ERA5Processor, download_era5, process_era5
from .era5_fix import (
    ERA5FixDownloader,
    fix_era5_accumulated,
    patch_processed_files,
    DEFAULT_ACCUM_VARIABLES,
)
from .modis import (
    MODISDownloader,
    MODISProcessor,
    download_modis,
    process_modis,
    DEFAULT_VAR_MAPPING,
    DEFAULT_FILL_VALUE,
)
from .modis_green import ModisGreenDownloader, process_modis_green
from .modis_landcover import ModisLandcoverDownloader, process_modis_landcover
from .modis_landuse import ModisLanduseDownloader, process_modis_landuse

__all__ = [
    # ERA5
    "ERA5Downloader",
    "ERA5Processor",
    "ERA5FixDownloader",
    "download_era5",
    "process_era5",
    "fix_era5_accumulated",
    "patch_processed_files",
    "DEFAULT_ACCUM_VARIABLES",
    # MODIS MOD13C2 (vegetation indices)
    "MODISDownloader",
    "MODISProcessor",
    "download_modis",
    "process_modis",
    "DEFAULT_VAR_MAPPING",
    "DEFAULT_FILL_VALUE",
    # MODIS additional products
    "ModisGreenDownloader",
    "process_modis_green",
    "ModisLandcoverDownloader",
    "process_modis_landcover",
    "ModisLanduseDownloader",
    "process_modis_landuse",
]
