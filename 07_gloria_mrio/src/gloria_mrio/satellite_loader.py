"""Satellite account loader supporting csv, xlsx, parquet, npy, and zarr formats."""

import numpy as np
import xarray as xr
from pathlib import Path
from typing import Dict, Optional


def load_satellite_data(path: str, year: int, 
                        n_regions: int, n_sectors: int) -> Dict[str, Optional[np.ndarray]]:
    """
    Load satellite account data
    
    Supported formats:
    - .csv
    - .xlsx / .xls
    - .parquet
    - .npy
    - .zarr
    
    Expected data format:
    - Q: (n_indicators, n_regions, n_sectors) or (n_indicators, n_regions * n_sectors)
    - Q_Y: (n_indicators, n_regions) [optional, for household direct emissions]
    
    File naming convention:
    - satellite_{year}.csv / .xlsx / .parquet / .npy
    - Q_{year}.csv / .npy
    - {year}.zarr
    - Q_Y_{year}.csv / .npy (optional)
    
    Args:
        path: Directory containing satellite data files
        year: Year to load
        n_regions: Number of regions
        n_sectors: Number of sectors
    
    Returns:
        {'Q': array, 'Q_Y': array or None, 'labels': list or None}
    """
    path = Path(path)

    patterns = [
        f"satellite_{year}",
        f"Q_{year}",
        f"{year}",
    ]
    
    Q = None
    Q_Y = None
    labels = None
    
    for pattern in patterns:
        csv_file = path / f"{pattern}.csv"
        if csv_file.exists():
            print(f"  Loading satellite from CSV: {csv_file}")
            import pandas as pd
            df = pd.read_csv(csv_file, index_col=0)
            Q = df.values.astype(np.float32)
            labels = df.index.tolist() if df.index.dtype == object else None
            break

        for ext in ['.xlsx', '.xls']:
            excel_file = path / f"{pattern}{ext}"
            if excel_file.exists():
                print(f"  Loading satellite from Excel: {excel_file}")
                import pandas as pd
                df = pd.read_excel(excel_file, index_col=0)
                Q = df.values.astype(np.float32)
                labels = df.index.tolist() if df.index.dtype == object else None
                break
        if Q is not None:
            break

        parquet_file = path / f"{pattern}.parquet"
        if parquet_file.exists():
            print(f"  Loading satellite from Parquet: {parquet_file}")
            import pandas as pd
            df = pd.read_parquet(parquet_file)
            Q = df.values.astype(np.float32)
            labels = df.index.tolist() if df.index.dtype == object else None
            break

        npy_file = path / f"{pattern}.npy"
        if npy_file.exists():
            print(f"  Loading satellite from NPY: {npy_file}")
            Q = np.load(npy_file).astype(np.float32)
            break

        zarr_file = path / f"{pattern}.zarr"
        if zarr_file.exists():
            print(f"  Loading satellite from Zarr: {zarr_file}")
            ds = xr.open_zarr(zarr_file)
            Q = ds['Q'].values.astype(np.float32)
            if 'Q_Y' in ds:
                Q_Y = ds['Q_Y'].values.astype(np.float32)
            if 'labels' in ds.attrs:
                labels = ds.attrs['labels']
            break
    
    if Q is None:
        raise FileNotFoundError(
            f"No satellite data found for year {year} in {path}\n"
            f"Expected file patterns: {patterns}"
        )
    
    # Reshape Q if needed
    if Q.ndim == 2:
        n_ind = Q.shape[0]
        expected_cols = n_regions * n_sectors
        if Q.shape[1] == expected_cols:
            Q = Q.reshape(n_ind, n_regions, n_sectors)
        else:
            raise ValueError(
                f"Unexpected Q shape: {Q.shape}\n"
                f"Expected (n_ind, {expected_cols}) or (n_ind, {n_regions}, {n_sectors})"
            )
    
    print(f"    Q shape: {Q.shape}")
    
    # Try to load Q_Y if not already loaded
    if Q_Y is None:
        qy_patterns = [f"Q_Y_{year}", f"satellite_Y_{year}"]
        for pattern in qy_patterns:
            qy_file = path / f"{pattern}.csv"
            if qy_file.exists():
                import pandas as pd
                Q_Y = pd.read_csv(qy_file, index_col=0).values.astype(np.float32)
                print(f"    Q_Y shape: {Q_Y.shape}")
                break
            
            qy_file = path / f"{pattern}.npy"
            if qy_file.exists():
                Q_Y = np.load(qy_file).astype(np.float32)
                print(f"    Q_Y shape: {Q_Y.shape}")
                break
    
    if Q_Y is None:
        print(f"    Q_Y: not found (household emissions will be zero)")
    
    return {'Q': Q, 'Q_Y': Q_Y, 'labels': labels}
