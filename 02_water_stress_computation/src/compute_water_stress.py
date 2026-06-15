"""Compute per-crop per-year water stress as wfp_blue_ir * aware_monthly_weight."""

import gc
import logging
import os
import re
from pathlib import Path
from typing import List, Optional

import xarray as xr
from tqdm import tqdm


def compute_water_stress(
    wf_dir: str,
    weights_nc: str,
    output_dir: str,
    filename_regex: str = r"acea_5arc_(\w+)_wfp_blue_ir",
    wf_variable: str = "wfp_blue_ir",
    weights_variable: str = "monthly_weight",
    crops: Optional[List[str]] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    compression_level: int = 5,
    logger: Optional[logging.Logger] = None,
) -> List[str]:
    """Compute per-crop per-year water stress and save as NC.

    filename_regex must have one capture group = crop code. start_year/end_year
    are inclusive year bounds. Returns the list of written NC file paths.
    """
    logger = logger or logging.getLogger(__name__)
    wf_dir = Path(wf_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading AWARE weights from {weights_nc}")
    weights_ds = xr.open_dataset(weights_nc).chunk({"month": 1})
    weights = weights_ds[weights_variable]

    pattern = re.compile(filename_regex)
    files = sorted(p for p in wf_dir.iterdir()
                   if p.suffix == ".nc" and not p.name.startswith("._"))
    logger.info(f"Found {len(files)} .nc files in {wf_dir}")

    written: List[str] = []

    for filepath in tqdm(files, desc="Computing water stress"):
        m = pattern.search(filepath.name)
        if not m:
            logger.warning(f"  {filepath.name}: crop not matched, skipping")
            continue
        crop_code = m.group(1)

        if crops is not None and crop_code not in crops:
            continue

        logger.info(f"  Processing {filepath.name} -> crop={crop_code}")

        ds = xr.open_dataset(filepath).chunk({"time": 12})
        if wf_variable not in ds.data_vars:
            logger.error(f"    Variable '{wf_variable}' not in file, skipping")
            ds.close()
            continue

        crop_out_dir = output_dir / crop_code
        crop_out_dir.mkdir(parents=True, exist_ok=True)

        available_years = sorted(set(ds["time"].dt.year.values))
        target_years = [
            y for y in available_years
            if (start_year is None or y >= start_year)
            and (end_year is None or y <= end_year)
        ]
        logger.info(f"    Years available: {available_years[0]}-{available_years[-1]}, "
                    f"processing: {target_years[0] if target_years else 'none'}"
                    f"-{target_years[-1] if target_years else 'none'}")

        for year in target_years:
            out_path = crop_out_dir / f"{year}.nc"
            if out_path.exists():
                logger.info(f"    {year}: already exists, skipping")
                continue

            yearly_ds = ds.sel(time=ds["time"].dt.year == year)
            time_months_year = yearly_ds["time"].dt.month - 1

            weights_expanded = weights.isel(month=time_months_year)
            weighted_wfp = yearly_ds[wf_variable] * weights_expanded
            weighted_wfp = (
                weighted_wfp.rename("waterstress")
                .drop_vars("month", errors="ignore")
            )

            encoding = {
                "waterstress": {"zlib": True, "complevel": compression_level}
            }
            weighted_wfp.to_netcdf(
                out_path, engine="h5netcdf", encoding=encoding, compute=True
            )
            written.append(str(out_path))
            logger.info(f"    {year}: saved -> {out_path}")

            del yearly_ds, weights_expanded, weighted_wfp
            gc.collect()

        ds.close()
        del ds
        gc.collect()

    weights_ds.close()
    logger.info(f"Done. {len(written)} files written.")
    return written
