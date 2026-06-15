"""Reshape the long water-stress table into per-year satellite arrays, saved flat as ``(1, N_REGIONS * N_SECTORS_GLORIA)`` in ``{output_dir}/satellite_{year}.npy``.

The first ``N_SECTORS_OUR`` columns hold the 14 crop sectors; the rest are
left as zeros and filled later by ``build_hybrid_satellite``.
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .mapping_27_to_14 import SECTORS_14


def build_our_satellite_arrays(
    long_csv: str,
    output_dir: str,
    n_regions: int = 164,
    n_sectors_gloria: int = 120,
    logger: Optional[logging.Logger] = None,
) -> List[str]:
    """Reshape the long CSV into per-year `(1, N_REGIONS * N_SECTORS_GLORIA)` npy files."""
    logger = logger or logging.getLogger(__name__)
    long_csv = Path(long_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(long_csv)
    # Deterministic region order from the CSV (it was written in the region order
    # coming out of mapping_27_to_14, which respects the GLORIA ReadMe order).
    regions = df["country_14_code"].astype(str).drop_duplicates().tolist()
    years = sorted(df["year"].unique())
    n_sectors_our = len(SECTORS_14)

    logger.info(f"  years: {len(years)} ({years[0]}-{years[-1]})")
    logger.info(f"  regions: {len(regions)} (target n_regions={n_regions})")
    logger.info(f"  sectors: {n_sectors_our} (our) / {n_sectors_gloria} (gloria target)")

    region_to_idx = {r: i for i, r in enumerate(regions)}
    sector_to_idx = {s: i for i, s in enumerate(SECTORS_14)}

    outputs = []
    for year in years:
        df_year = df[df["year"] == year]
        matrix = np.zeros((n_regions, n_sectors_gloria), dtype=np.float64)

        ri = df_year["country_14_code"].astype(str).map(region_to_idx).values
        ci = df_year["sector_14"].astype(str).map(sector_to_idx).values
        mask = (~pd.isna(ri)) & (~pd.isna(ci))
        ri, ci = ri[mask].astype(int), ci[mask].astype(int)
        vals = df_year.loc[mask, "water_stress"].values

        # Clip to ensure no out-of-bounds for misaligned regions
        valid = (ri < n_regions) & (ci < n_sectors_our)
        matrix[ri[valid], ci[valid]] = vals[valid]

        flat = matrix.reshape(1, n_regions * n_sectors_gloria)
        out = output_dir / f"satellite_{int(year)}.npy"
        np.save(out, flat)
        outputs.append(str(out))
        logger.info(f"    {int(year)}: saved {out.name} shape={flat.shape}")

    return outputs
