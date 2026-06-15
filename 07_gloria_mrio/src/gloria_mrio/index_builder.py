"""
Index Builder for managing region and sector groupings
"""

import numpy as np
from typing import List, Tuple
from .config import Config


class IndexBuilder:
    """
    Index builder for managing region and sector groupings
    """
    
    def __init__(self, config: Config):
        self.n_regions = config.n_regions
        self.n_sectors = config.n_sectors
        self.n_tot = self.n_regions * self.n_sectors
        
        self.target_sectors = config.target_sectors
        self.region_grouping = config.region_grouping
        
        # Build index_matrix_P: (n_regions, n_sectors)
        # index_matrix_P[r, s] = r * n_sectors + s
        self.index_matrix_P = np.zeros((self.n_regions, self.n_sectors), dtype=np.int32)
        for r in range(self.n_regions):
            self.index_matrix_P[r, :] = np.arange(
                r * self.n_sectors, 
                (r + 1) * self.n_sectors
            )
        
        # Build region aggregation matrix Agg_FDreg: (n_regions, n_groups)
        self.n_fd_groups = len(self.region_grouping)
        self.Agg_FDreg = np.zeros((self.n_regions, self.n_fd_groups), dtype=np.int32)
        self.fd_group_names = list(self.region_grouping.keys())
        
        for i, (name, regions) in enumerate(self.region_grouping.items()):
            if isinstance(regions, str) and regions == 'all':
                regions = list(range(self.n_regions))
            self.Agg_FDreg[regions, i] = 1
    
    def get_target_indices(self, target_names: List[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get target sector indices
        
        Args:
            target_names: List of target sector group names, None for all defined targets
        
        Returns:
            index_t_s: Target sector indices (sector level)
            index_t: Target sector indices (expanded to all regions)
            index_o: Non-target sector indices
        """
        if target_names is None:
            target_names = list(self.target_sectors.keys())

        t_sectors = []
        for name in target_names:
            if name not in self.target_sectors:
                raise KeyError(f"Unknown target sector group: {name}")
            t_sectors.extend(self.target_sectors[name])
        
        index_t_s = np.array(sorted(set(t_sectors)), dtype=np.int32)
        
        # Expand to all regions
        index_t = []
        for r in range(self.n_regions):
            for s in index_t_s:
                index_t.append(r * self.n_sectors + s)
        index_t = np.array(sorted(index_t), dtype=np.int32)
        
        # Non-target indices
        index_all = np.arange(self.n_tot, dtype=np.int32)
        index_o = np.setdiff1d(index_all, index_t)
        
        return index_t_s, index_t, index_o
    
    def get_sector_rows(self, sector_idx: int) -> np.ndarray:
        """Get row indices for a sector across all regions"""
        return self.index_matrix_P[:, sector_idx].astype(np.int32)
