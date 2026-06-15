"""
MRIO Data Loader
"""

import numpy as np
import xarray as xr
import gc
from .config import Config
from .utils import inverse_with_progress


class MRIOLoader:
    """
    MRIO data loader for GLORIA zarr files
    """
    
    def __init__(self, zarr_path: str, config: Config):
        """
        Args:
            zarr_path: Path to zarr file (e.g., gloria_mrio/2022.zarr)
            config: Configuration object
        """
        self.path = zarr_path
        self.config = config
        self.n_regions = config.n_regions
        self.n_sectors = config.n_sectors
        self.n_tot = self.n_regions * self.n_sectors
        self.dtype = getattr(np, config.dtype)
        
        self._load()
    
    def _load(self):
        """Load data from zarr file"""
        print(f"  Loading MRIO from: {self.path}")
        
        # Try consolidated first, fallback to non-consolidated
        try:
            ds = xr.open_zarr(self.path, consolidated=True)
        except KeyError:
            ds = xr.open_zarr(self.path, consolidated=False)
        
        # T: (from_region, from_sector, to_region, to_sector) -> (n_tot, n_tot)
        T_4d = ds['T'].values
        self.Z = T_4d.reshape(self.n_tot, self.n_tot).astype(self.dtype)
        
        # Y: (from_region, from_sector, to_region) -> (n_tot, n_regions)
        Y_3d = ds['Y'].values
        self.Y = Y_3d.reshape(self.n_tot, self.n_regions).astype(self.dtype)

        self.x = self.Z.sum(axis=1) + self.Y.sum(axis=1)
        
        print(f"    Z shape: {self.Z.shape}")
        print(f"    Y shape: {self.Y.shape}")
        
        ds.close()
        del T_4d, Y_3d
        gc.collect()
    
    def compute_A(self):
        """Compute technical coefficient matrix A = Z / x"""
        print("  Computing A matrix...")
        
        x_safe = np.where(self.x > 0, self.x, 1.0)
        self.A = self.Z / x_safe[np.newaxis, :]
        self.A = np.nan_to_num(self.A, nan=0, posinf=0, neginf=0)
        self.A = np.maximum(self.A, 0).astype(self.dtype)  # Force non-negative
        
        print(f"    A shape: {self.A.shape}")
    
    def compute_L(self):
        """Compute Leontief inverse L = (I - A)^(-1)"""
        print("  Computing Leontief inverse L...")
        
        I_minus_A = np.eye(self.n_tot, dtype=self.dtype) - self.A
        self.L = inverse_with_progress(I_minus_A, dtype=self.dtype, desc="    L inverse")
        
        print(f"    L shape: {self.L.shape}")
