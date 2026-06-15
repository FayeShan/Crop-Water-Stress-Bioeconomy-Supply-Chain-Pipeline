"""Convert GLORIA Release 060 Supply-Use Table (SUT) CSVs into MRIO format, saving one ``{year}.zarr`` per year.

In the SUT a region block is 240 wide (120 industries + 120 products),
hence the ``* 240`` strides below. Only the Markup001 (basic-prices) T/Y/V
CSVs are used.

Output zarr per year:
  T        (N_REGIONS, N_SECTORS, N_REGIONS, N_SECTORS)  transaction matrix
  Y        (N_REGIONS, N_SECTORS, N_REGIONS)             final demand aggregated per destination region
  Y_full   (MRIO_SIZE, N_REGIONS * N_FD_CATEGORIES)      raw final demand
  V        (n_va_categories, MRIO_SIZE)                  value added
"""

import gc
import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import xarray as xr
from tqdm import tqdm


# GLORIA Release 060 constants
N_REGIONS = 164
N_SECTORS = 120
N_FD_CATEGORIES = 6
MRIO_SIZE = N_REGIONS * N_SECTORS  # 19,680


def _load_gloria_csv(filepath: str, dtype=np.float32) -> np.ndarray:
    return np.loadtxt(filepath, delimiter=",", dtype=dtype)


def _sut_to_mrio_T(T_sut: np.ndarray, dtype=np.float32) -> np.ndarray:
    """Convert SUT T matrix to MRIO T matrix (industries x industries)."""
    T_mrio = np.zeros((MRIO_SIZE, MRIO_SIZE), dtype=dtype)
    for c1 in range(N_REGIONS):
        for c2 in range(N_REGIONS):
            r1, r2 = c1 * N_SECTORS, (c1 + 1) * N_SECTORS
            c1_idx, c2_idx = c2 * N_SECTORS, (c2 + 1) * N_SECTORS
            sut_r1, sut_r2 = c1 * 240 + 120, c1 * 240 + 240
            sut_c1, sut_c2 = c2 * 240, c2 * 240 + 120
            T_mrio[r1:r2, c1_idx:c2_idx] = T_sut[sut_r1:sut_r2, sut_c1:sut_c2]
    return T_mrio


def _sut_to_mrio_Y(Y_sut: np.ndarray, dtype=np.float32) -> np.ndarray:
    """Convert SUT Y (final demand) to MRIO format."""
    Y_mrio = np.zeros((MRIO_SIZE, Y_sut.shape[1]), dtype=dtype)
    for c1 in range(N_REGIONS):
        r1, r2 = c1 * N_SECTORS, (c1 + 1) * N_SECTORS
        sut_r1, sut_r2 = c1 * 240 + 120, c1 * 240 + 240
        Y_mrio[r1:r2, :] = Y_sut[sut_r1:sut_r2, :]
    return Y_mrio


def _sut_to_mrio_V(V_sut: np.ndarray, dtype=np.float32) -> np.ndarray:
    """Convert SUT V (value added) to MRIO format."""
    V_mrio = np.zeros((V_sut.shape[0], MRIO_SIZE), dtype=dtype)
    for c1 in range(N_REGIONS):
        mrio_c1, mrio_c2 = c1 * N_SECTORS, (c1 + 1) * N_SECTORS
        sut_c1, sut_c2 = c1 * 240, c1 * 240 + 120
        V_mrio[:, mrio_c1:mrio_c2] = V_sut[:, sut_c1:sut_c2]
    return V_mrio


def _aggregate_Y_by_region(Y_mrio: np.ndarray, dtype=np.float32) -> np.ndarray:
    """Sum final demand across the 6 FD categories per destination region."""
    Y_agg = np.zeros((MRIO_SIZE, N_REGIONS), dtype=dtype)
    for r in range(N_REGIONS):
        col_start = r * N_FD_CATEGORIES
        col_end = col_start + N_FD_CATEGORIES
        Y_agg[:, r] = Y_mrio[:, col_start:col_end].sum(axis=1)
    return Y_agg


def _find_markup001(input_path: Path):
    """Find the T / Y / V CSV files tagged Markup001 in `input_path`."""
    t_file = y_file = v_file = None
    for f in os.listdir(input_path):
        if not f.endswith(".csv") or "Markup001" not in f:
            continue
        if "T-Results" in f:
            t_file = f
        elif "Y-Results" in f:
            y_file = f
        elif "V-Results" in f:
            v_file = f
    return t_file, y_file, v_file


def convert_year(
    year: int,
    unzipped_dir: Path,
    zarr_dir: Path,
    logger: logging.Logger,
) -> bool:
    """Convert one year's SUT CSVs into MRIO zarr."""
    input_path = unzipped_dir / str(year)
    zarr_path = zarr_dir / f"{year}.zarr"

    if zarr_path.exists():
        logger.info(f"  {year}: zarr already exists, skip")
        return True

    if not input_path.exists():
        logger.warning(f"  {year}: unzipped CSV folder not found: {input_path}")
        return False

    t_file, y_file, v_file = _find_markup001(input_path)
    if not all([t_file, y_file, v_file]):
        logger.warning(f"  {year}: missing Markup001 T/Y/V CSV in {input_path}")
        return False

    logger.info(f"  {year}: loading CSVs...")
    T_sut = _load_gloria_csv(input_path / t_file)
    Y_sut = _load_gloria_csv(input_path / y_file)
    V_sut = _load_gloria_csv(input_path / v_file)

    logger.info(f"  {year}: converting T...")
    T_mrio = _sut_to_mrio_T(T_sut); del T_sut; gc.collect()
    logger.info(f"  {year}: converting Y...")
    Y_mrio = _sut_to_mrio_Y(Y_sut); del Y_sut; gc.collect()
    logger.info(f"  {year}: converting V...")
    V_mrio = _sut_to_mrio_V(V_sut); del V_sut; gc.collect()

    logger.info(f"  {year}: aggregating Y by region...")
    Y_region = _aggregate_Y_by_region(Y_mrio)

    T_4d = T_mrio.reshape(N_REGIONS, N_SECTORS, N_REGIONS, N_SECTORS)
    Y_3d = Y_region.reshape(N_REGIONS, N_SECTORS, N_REGIONS)

    ds = xr.Dataset(
        {
            "T": xr.DataArray(
                T_4d, dims=["from_region", "from_sector", "to_region", "to_sector"],
                attrs={"description": "Transaction matrix", "unit": "1000 USD"}),
            "Y": xr.DataArray(
                Y_3d, dims=["from_region", "from_sector", "to_region"],
                attrs={"description": "Final demand (aggregated by region)", "unit": "1000 USD"}),
            "V": xr.DataArray(
                V_mrio, dims=["value_added_category", "region_sector"],
                attrs={"description": "Value added", "unit": "1000 USD"}),
            "Y_full": xr.DataArray(
                Y_mrio, dims=["region_sector", "final_demand"],
                attrs={"description": "Final demand (full, 6 categories per region)",
                       "unit": "1000 USD"}),
        },
        attrs={
            "source": "GLORIA MRIO Database Release 060",
            "year": year,
            "n_regions": N_REGIONS,
            "n_sectors": N_SECTORS,
            "valuation": "Basic prices (Markup001)",
        },
    )

    encoding = {
        "T": {"chunks": (50, N_SECTORS, 50, N_SECTORS), "dtype": "float32"},
        "Y": {"chunks": (50, N_SECTORS, N_REGIONS), "dtype": "float32"},
        "V": {"chunks": (V_mrio.shape[0], 5000), "dtype": "float32"},
        "Y_full": {"chunks": (5000, Y_mrio.shape[1]), "dtype": "float32"},
    }

    zarr_dir.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(zarr_path, mode="w", consolidated=True, encoding=encoding, zarr_format=2)
    logger.info(f"  {year}: saved -> {zarr_path}")

    del T_mrio, Y_mrio, V_mrio, Y_region, T_4d, Y_3d, ds
    gc.collect()
    return True


def convert_sut_to_mrio(
    unzipped_dir: str,
    zarr_dir: str,
    start_year: int,
    end_year: int,
    logger: Optional[logging.Logger] = None,
) -> List[str]:
    """Convert GLORIA SUT CSVs for each year in [start_year, end_year] to zarr."""
    logger = logger or logging.getLogger(__name__)
    unzipped_dir = Path(unzipped_dir)
    zarr_dir = Path(zarr_dir)

    logger.info("=" * 70)
    logger.info(f"  GLORIA SUT -> MRIO zarr ({start_year}-{end_year})")
    logger.info("=" * 70)
    logger.info(f"  unzipped_dir: {unzipped_dir}")
    logger.info(f"  zarr_dir:     {zarr_dir}")

    results = []
    for year in tqdm(range(start_year, end_year + 1), desc="converting", leave=False):
        if convert_year(year, unzipped_dir, zarr_dir, logger):
            results.append(str(zarr_dir / f"{year}.zarr"))
    logger.info(f"Done. {len(results)} years processed.")
    return results
