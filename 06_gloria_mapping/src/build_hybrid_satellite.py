"""Build the per-year hybrid satellite account merging our 14 CWS crop sectors (m^3) with GLORIA's sectors 15-120 from the TQ CSV (million m^3); output is in million m^3.

Composition modes:
  include_non_ag=False  hybrid[:,14:] = GLORIA agriculture row only
  include_non_ag=True   hybrid[:,14:] = GLORIA agriculture + non-agriculture rows

The GLORIA TQ CSV has ``2 * N_REGIONS * N_SECTORS_GLORIA`` columns (supply
and use halves); supply-side indices are ``s + N_SECTORS_GLORIA * c * 2``
for sector ``s`` in region ``c``.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


def _supply_cols(n_regions: int, n_sectors: int) -> np.ndarray:
    cols = []
    for c in range(n_regions):
        for s in range(n_sectors):
            cols.append(s + n_sectors * c * 2)
    return np.array(cols)


def _read_gloria_row(
    gloria_path: Path, row_index: int, supply_cols: np.ndarray,
    n_regions: int, n_sectors: int,
) -> np.ndarray:
    """Read the row at `row_index` from the GLORIA TQ CSV and reshape to
    ``(n_regions, n_sectors)``.
    """
    row = pd.read_csv(gloria_path, header=None,
                      skiprows=lambda x: x != row_index)
    return row.iloc[0, supply_cols].values.astype(float).reshape(n_regions, n_sectors)


def build_hybrid_satellite(
    our_satellite_dir: str,
    gloria_satellite_dir: str,
    output_dir: str,
    start_year: int,
    end_year: int,
    ag_row_index: int = 390,
    non_ag_row_index: int = 391,
    include_non_ag: bool = True,
    n_regions: int = 164,
    n_sectors: int = 120,
    unit_scale: float = 1e6,
    logger: Optional[logging.Logger] = None,
) -> List[str]:
    """Merge our 14-sector block with GLORIA's 15-120 sectors per year."""
    logger = logger or logging.getLogger(__name__)
    our_dir = Path(our_satellite_dir)
    gloria_dir = Path(gloria_satellite_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("  Build hybrid satellite accounts")
    logger.info("=" * 70)
    logger.info(f"  our:     {our_dir}  (units: m^3, scale 1/{unit_scale:g})")
    logger.info(f"  gloria:  {gloria_dir} (units: million m^3)")
    logger.info(f"  out:     {out_dir}")
    logger.info(f"  ag_row:  {ag_row_index}   "
                f"non_ag_row: {non_ag_row_index}   include_non_ag: {include_non_ag}")

    supply_cols = _supply_cols(n_regions, n_sectors)

    outputs = []
    for year in tqdm(range(start_year, end_year + 1), desc="hybrid"):
        ours_path = our_dir / f"satellite_{year}.npy"
        if not ours_path.exists():
            logger.info(f"  {year}: our satellite not found, skip")
            continue

        gloria_path = gloria_dir / str(year) / f"TQ_{year}.csv"
        if not gloria_path.exists():
            logger.info(f"  {year}: GLORIA TQ not found at {gloria_path}, skip")
            continue

        ours = np.load(ours_path).reshape(n_regions, n_sectors)
        ours_Mm3 = ours / unit_scale

        gloria_ag = _read_gloria_row(
            gloria_path, ag_row_index, supply_cols, n_regions, n_sectors)

        hybrid = np.zeros((n_regions, n_sectors))
        hybrid[:, :14] = ours_Mm3[:, :14]
        if include_non_ag:
            gloria_non_ag = _read_gloria_row(
                gloria_path, non_ag_row_index, supply_cols, n_regions, n_sectors)
            hybrid[:, 14:] = gloria_ag[:, 14:] + gloria_non_ag[:, 14:]
        else:
            hybrid[:, 14:] = gloria_ag[:, 14:]

        out_path = out_dir / f"satellite_{year}.npy"
        np.save(out_path, hybrid.reshape(1, n_regions * n_sectors))

        ours_14 = hybrid[:, :14].sum()
        gloria_rest = hybrid[:, 14:].sum()
        total = hybrid.sum()
        logger.info(
            f"  {year}: total={total:.4e} Mm^3 | "
            f"ours(1-14)={ours_14:.4e} ({ours_14/total*100:.1f}%) | "
            f"GLORIA(15-120)={gloria_rest:.4e} ({gloria_rest/total*100:.1f}%)"
        )
        outputs.append(str(out_path))

    logger.info(f"Done. {len(outputs)} hybrid files -> {out_dir}")
    return outputs
