"""Unzip GLORIA MRIO zips (Part I) into per-year CSV folders for ``gloria_sut_to_mrio``.

CSVs are extracted without renaming; the SUT -> MRIO converter identifies
T/Y/V CSVs by substring.
"""

import logging
import shutil
import zipfile
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm


ZIP_NAME_TEMPLATE = "GLORIA_MRIOs_60_{year}.zip"


def unzip_mrio_zips(
    zip_dir: str,
    output_dir: str,
    start_year: int,
    end_year: int,
    logger: Optional[logging.Logger] = None,
) -> List[str]:
    logger = logger or logging.getLogger(__name__)
    zip_dir = Path(zip_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("  GLORIA MRIO unzip")
    logger.info("=" * 70)
    logger.info(f"  zip_dir:    {zip_dir}")
    logger.info(f"  output_dir: {output_dir}")

    extracted_years = []
    for year in tqdm(range(start_year, end_year + 1), desc="unzipping", leave=False):
        zip_path = zip_dir / ZIP_NAME_TEMPLATE.format(year=year)
        if not zip_path.exists():
            logger.warning(f"  {year}: ZIP not found: {zip_path}")
            continue

        out_year_dir = output_dir / str(year)
        if out_year_dir.exists() and any(out_year_dir.glob("*.csv")):
            logger.info(f"  {year}: {out_year_dir} already populated, skip")
            extracted_years.append(str(out_year_dir))
            continue

        out_year_dir.mkdir(parents=True, exist_ok=True)

        tmp_dir = output_dir / f"_tmp_{year}"
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmp_dir)

        csvs = list(tmp_dir.rglob("*.csv"))
        for f in csvs:
            shutil.move(str(f), out_year_dir / f.name)
        shutil.rmtree(tmp_dir, ignore_errors=True)

        logger.info(f"  {year}: {len(csvs)} CSVs -> {out_year_dir}")
        extracted_years.append(str(out_year_dir))

    logger.info(f"Done. {len(extracted_years)} year folders populated.")
    return extracted_years
