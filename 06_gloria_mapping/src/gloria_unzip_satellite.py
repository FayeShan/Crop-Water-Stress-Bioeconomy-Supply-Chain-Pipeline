"""Unzip GLORIA satellite-account zips, keeping only the TQ/YQ-Results CSVs in a per-year layout ({output_dir}/{year}/TQ_{year}.csv, YQ_{year}.csv)."""

import glob
import logging
import os
import shutil
import zipfile
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm


ZIP_NAME_TEMPLATE = "GLORIA_SatelliteAccounts_060_{year}.zip"


def unzip_satellite_zips(
    zip_dir: str,
    output_dir: str,
    start_year: int,
    end_year: int,
    logger: Optional[logging.Logger] = None,
) -> List[str]:
    """Unzip the GLORIA satellite-account zips into a clean per-year layout."""
    logger = logger or logging.getLogger(__name__)
    zip_dir = Path(zip_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("  GLORIA satellite-account unzip")
    logger.info("=" * 70)
    logger.info(f"  zip_dir:    {zip_dir}")
    logger.info(f"  output_dir: {output_dir}")

    extracted = []
    for year in tqdm(range(start_year, end_year + 1), desc="unzipping", leave=False):
        zip_path = zip_dir / ZIP_NAME_TEMPLATE.format(year=year)
        if not zip_path.exists():
            logger.warning(f"  {year}: ZIP not found: {zip_path}")
            continue

        out_year_dir = output_dir / str(year)
        out_year_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = output_dir / f"_tmp_{year}"

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmp_dir)

        for f in glob.glob(os.path.join(tmp_dir, "**", "*.csv"), recursive=True):
            fname = os.path.basename(f)
            if "TQ-Results" in fname:
                dest = out_year_dir / f"TQ_{year}.csv"
                shutil.move(f, dest)
                logger.info(f"  {year}: TQ -> {dest}")
                extracted.append(str(dest))
            elif "YQ-Results" in fname:
                dest = out_year_dir / f"YQ_{year}.csv"
                shutil.move(f, dest)
                logger.info(f"  {year}: YQ -> {dest}")
                extracted.append(str(dest))

        shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info(f"Done. {len(extracted)} files extracted.")
    return extracted
